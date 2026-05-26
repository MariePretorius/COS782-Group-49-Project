"""Generic transformer runner. Parameterised version of run_enhanced_afroxlmr.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# windows DLL ordering: sentencepiece must come before numpy/pandas
import sentencepiece  # noqa: F401

import numpy as np
import pandas as pd

from error_analysis import run_error_analysis
from evaluate_enhanced import evaluate_predictions
from utils import (
    ENHANCED_VALID_SPLITS,
    RANDOM_SEED,
    compute_metrics,
    save_json,
    validate_experiment_dataframe,
    validate_split_sets,
)

TRAIN_PATH = Path("data/processed/enhanced/train.csv")
VALIDATION_PATH = Path("data/processed/enhanced/validation.csv")
TEST_PATH = Path("data/processed/enhanced/test.csv")
COMPARISON_OUTPUT_DIR = Path("outputs/comparison")

MODEL_NAME = "Davlan/afro-xlmr-base-76L_script"
TAG = "afroxlmr"
LOSS_TYPE = "weighted_ce"
FOCAL_GAMMA = 2.0
MIN_WORDS = 0
MAX_WORDS = 0

MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "128"))
TRAIN_BATCH_SIZE = int(os.environ.get("COS760_TRAIN_BATCH_SIZE", "16"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "32"))
NUM_TRAIN_EPOCHS = float(os.environ.get("COS760_NUM_TRAIN_EPOCHS", "3"))
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = int(os.environ.get("COS760_EARLY_STOPPING_PATIENCE", "1"))
LEARNING_RATES = [
    float(value)
    for value in os.environ.get("COS760_LEARNING_RATES", "2e-5,3e-5").split(",")
    if value.strip()
]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def output_dirs() -> tuple[Path, Path, Path]:
    base = Path("outputs/enhanced")
    return (base / TAG, base / f"{TAG}_prior_calibrated", base / f"{TAG}_pairwise_exploratory")


def apply_length_filter(dataframe: pd.DataFrame, name: str) -> pd.DataFrame:
    if MIN_WORDS <= 0 and MAX_WORDS <= 0:
        return dataframe
    words = dataframe["text"].astype(str).str.split().str.len()
    mask = pd.Series(True, index=dataframe.index)
    if MIN_WORDS > 0:
        mask &= words >= MIN_WORDS
    if MAX_WORDS > 0:
        mask &= words <= MAX_WORDS
    filtered = dataframe.loc[mask].copy()
    print(f"  [{name}] length filter min={MIN_WORDS} max={MAX_WORDS}: "
          f"{len(dataframe)} -> {len(filtered)} rows")
    return filtered


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enhanced {split_name} set: {path.resolve()}. "
            "Run python src/build_enhanced_dataset.py first."
        )
    dataframe = pd.read_csv(path)
    validate_experiment_dataframe(dataframe, str(path), ENHANCED_VALID_SPLITS)
    return apply_length_filter(dataframe, split_name)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_split(TRAIN_PATH, "train")
    validation_df = load_split(VALIDATION_PATH, "validation")
    test_df = load_split(TEST_PATH, "test")
    validate_split_sets({"train": train_df, "validation": validation_df, "test": test_df})
    return train_df, validation_df, test_df


class TextClassificationDataset:
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_length: int) -> None:
        import torch
        self.encodings = tokenizer(texts.astype(str).tolist(), truncation=True, max_length=max_length)
        self.labels = labels.astype(int).tolist()
        self.torch = torch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: self.torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = self.torch.tensor(self.labels[index])
        return item


def compute_class_weights(train_labels: pd.Series):
    import torch
    counts = train_labels.astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    weights = total / (len(counts) * counts.replace(0, 1))
    return torch.tensor(weights.to_numpy(dtype=np.float32))


def make_trainer_class(class_weights, loss_type: str, focal_gamma: float):
    import torch
    from transformers import Trainer

    class CustomLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = class_weights.to(logits.device) if class_weights is not None else None
            if loss_type == "focal":
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, model.config.num_labels), labels.view(-1),
                    weight=weight, reduction="none",
                )
                pt = torch.exp(-ce)
                loss = ((1.0 - pt) ** focal_gamma * ce).mean()
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return CustomLossTrainer


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predictions_from_probabilities(dataframe, probabilities, threshold):
    predicted_labels = (probabilities[:, 1] >= threshold).astype(int)
    return pd.DataFrame({
        "id": dataframe["id"], "text": dataframe["text"], "language": dataframe["language"],
        "language_name": dataframe["language_name"], "true_label": dataframe["label"].astype(int),
        "predicted_label": predicted_labels, "human_probability": probabilities[:, 0],
        "machine_probability": probabilities[:, 1], "source": dataframe["source"],
        "generator": dataframe["generator"], "split": dataframe["split"],
    })


def train_one_learning_rate(learning_rate, train_df, validation_df, tokenizer, model_output_dir):
    import torch
    from transformers import (
        AutoModelForSequenceClassification, DataCollatorWithPadding,
        EarlyStoppingCallback, TrainingArguments, set_seed,
    )
    set_seed(RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "human", 1: "machine"}, label2id={"human": 0, "machine": 1},
    )
    train_dataset = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer, MAX_LENGTH)
    validation_dataset = TextClassificationDataset(validation_df["text"], validation_df["label"], tokenizer, MAX_LENGTH)
    class_weights = None if LOSS_TYPE == "none" else compute_class_weights(train_df["label"])
    trainer_class = make_trainer_class(class_weights, LOSS_TYPE, FOCAL_GAMMA)

    use_cpu = not torch.cuda.is_available()
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(model_output_dir), learning_rate=learning_rate,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        seed=RANDOM_SEED, use_cpu=use_cpu, fp16=use_fp16,
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        dataloader_num_workers=0, dataloader_pin_memory=False, report_to=[],
    )

    def trainer_metrics(eval_prediction):
        logits, labels = eval_prediction
        return compute_metrics(labels, np.argmax(logits, axis=1))

    trainer = trainer_class(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()
    validation_output = trainer.predict(validation_dataset)
    return trainer, softmax(validation_output.predictions)


def tune_threshold(validation_df, probabilities):
    rows = []
    y_true = validation_df["label"].astype(int).to_numpy()
    machine_probabilities = pd.Series(probabilities[:, 1])
    quantile_thresholds = [
        float(machine_probabilities.quantile(q))
        for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ]
    thresholds = sorted(set(THRESHOLDS + quantile_thresholds))
    for threshold in thresholds:
        predicted_labels = (probabilities[:, 1] >= threshold).astype(int)
        rows.append({"threshold": threshold, **compute_metrics(y_true, predicted_labels)})
    return pd.DataFrame(rows)


def prior_calibrated_predictions(dataframe, probabilities, calibration_df):
    predictions = []
    thresholds = {}
    probability_df = dataframe.copy()
    probability_df["human_probability"] = probabilities[:, 0]
    probability_df["machine_probability"] = probabilities[:, 1]
    for language, language_df in probability_df.groupby("language"):
        calibration_language_df = calibration_df.loc[calibration_df["language"] == language]
        positive_prior = float(calibration_language_df["label"].astype(int).mean())
        threshold = float(language_df["machine_probability"].quantile(1 - positive_prior))
        thresholds[language] = threshold
        language_predictions = language_df.copy()
        language_predictions["predicted_label"] = (language_predictions["machine_probability"] >= threshold).astype(int)
        predictions.append(language_predictions)
    calibrated = pd.concat(predictions).sort_index()
    output = pd.DataFrame({
        "id": calibrated["id"], "text": calibrated["text"], "language": calibrated["language"],
        "language_name": calibrated["language_name"], "true_label": calibrated["label"].astype(int),
        "predicted_label": calibrated["predicted_label"].astype(int),
        "human_probability": calibrated["human_probability"], "machine_probability": calibrated["machine_probability"],
        "source": calibrated["source"], "generator": calibrated["generator"], "split": calibrated["split"],
    })
    return output, thresholds


def pairwise_exploratory_predictions(predictions, direction):
    if direction not in {"lower", "higher"}:
        raise ValueError("Pairwise direction must be 'lower' or 'higher'.")
    pairwise = predictions.copy()
    pairwise["base_id"] = pairwise.apply(
        lambda row: str(row["id"]).replace("machine_", "") if int(row["true_label"]) == 1 else str(row["id"]),
        axis=1,
    )
    pairwise["predicted_label"] = 0
    for _, group in pairwise.groupby("base_id"):
        if len(group) != 2:
            continue
        index = group["machine_probability"].idxmin() if direction == "lower" else group["machine_probability"].idxmax()
        pairwise.loc[index, "predicted_label"] = 1
    return pairwise.drop(columns=["base_id"])


def update_comparison_outputs(model_name, output_dir, metrics, per_language_metrics, selected_by):
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARISON_OUTPUT_DIR / "enhanced_model_comparison.csv"
    per_language_path = COMPARISON_OUTPUT_DIR / "enhanced_per_language_metrics.csv"
    new_comparison_row = pd.DataFrame(
        [{"model": model_name, **metrics, "selected_by": selected_by,
          "best_params_path": str(output_dir / "best_config.json")}]
    )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison.loc[comparison["model"] != model_name]
        comparison = pd.concat([comparison, new_comparison_row], ignore_index=True)
    else:
        comparison = new_comparison_row
    comparison.to_csv(comparison_path, index=False)
    per_language = per_language_metrics.copy()
    per_language.insert(0, "model", model_name)
    if per_language_path.exists():
        existing = pd.read_csv(per_language_path)
        existing = existing.loc[existing["model"] != model_name]
        per_language = pd.concat([existing, per_language], ignore_index=True)
    per_language.to_csv(per_language_path, index=False)
    try:
        from run_enhanced_experiments import write_basic_vs_enhanced_summary, write_experiment_summary
        basic_vs_enhanced = write_basic_vs_enhanced_summary(comparison)
        write_experiment_summary(comparison, per_language, basic_vs_enhanced)
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: Could not refresh comparison summary: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generic enhanced transformer runner.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path.")
    parser.add_argument("--tag", required=True, help="Label for output dirs and comparison rows.")
    parser.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "focal", "none"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    global MODEL_NAME, TAG, LOSS_TYPE, FOCAL_GAMMA, MIN_WORDS, MAX_WORDS
    args = parse_args()
    MODEL_NAME, TAG, LOSS_TYPE = args.model, args.tag, args.loss
    FOCAL_GAMMA, MIN_WORDS, MAX_WORDS = args.focal_gamma, args.min_words, args.max_words
    output_dir, prior_dir, pairwise_dir = output_dirs()
    try:
        import torch
        from transformers import AutoTokenizer
        train_df, validation_df, test_df = load_datasets()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model: {MODEL_NAME}  tag: {TAG}  loss: {LOSS_TYPE}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"Train/val/test rows: {len(train_df)}/{len(validation_df)}/{len(test_df)}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        sweep_rows = []
        best_trainer = best_probabilities = best_threshold_results = None
        best_learning_rate = None
        best_macro_f1 = -1.0
        for learning_rate in LEARNING_RATES:
            print(f"\nTraining {TAG} with learning_rate={learning_rate}")
            trainer, validation_probabilities = train_one_learning_rate(
                learning_rate, train_df, validation_df, tokenizer, output_dir / f"lr_{learning_rate:g}")
            threshold_results_for_lr = tune_threshold(validation_df, validation_probabilities)
            best_threshold_for_lr = threshold_results_for_lr.sort_values("macro_f1", ascending=False).iloc[0]
            threshold_metrics = {key: float(best_threshold_for_lr[key])
                                 for key in ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"]}
            sweep_rows.append({"learning_rate": learning_rate,
                               "best_threshold": float(best_threshold_for_lr["threshold"]),
                               **{f"threshold_{k}": v for k, v in threshold_metrics.items()}})
            if threshold_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = threshold_metrics["macro_f1"]
                best_learning_rate = learning_rate
                best_trainer = trainer
                best_probabilities = validation_probabilities
                best_threshold_results = threshold_results_for_lr
        assert best_trainer is not None and best_threshold_results is not None
        pd.DataFrame(sweep_rows).to_csv(output_dir / "learning_rate_sweep.csv", index=False)
        threshold_results = best_threshold_results
        threshold_results.to_csv(output_dir / "threshold_tuning.csv", index=False)
        best_threshold_row = threshold_results.sort_values("macro_f1", ascending=False).iloc[0]
        best_threshold = float(best_threshold_row["threshold"])
        save_json(output_dir / "best_config.json", {
            "model_name": MODEL_NAME, "tag": TAG, "loss": LOSS_TYPE,
            "min_words": MIN_WORDS, "max_words": MAX_WORDS,
            "learning_rate": best_learning_rate, "threshold": best_threshold,
            "validation_macro_f1_for_threshold": float(best_threshold_row["macro_f1"]),
            "max_length": MAX_LENGTH, "train_batch_size": TRAIN_BATCH_SIZE,
            "num_train_epochs": NUM_TRAIN_EPOCHS})
        test_dataset = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)
        test_probabilities = softmax(best_trainer.predict(test_dataset).predictions)
        predictions = predictions_from_probabilities(test_df, test_probabilities, best_threshold)
        metrics, per_language_metrics = evaluate_predictions(predictions, output_dir, TAG, include_probabilities=True)
        best_trainer.save_model(str(output_dir / "model"))
        tokenizer.save_pretrained(str(output_dir / "model"))
        run_error_analysis(output_dir / "predictions.csv")
        update_comparison_outputs(TAG, output_dir, metrics, per_language_metrics,
                                  "validation_learning_rate_and_threshold_macro_f1")
        train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
        calibrated_predictions, calibrated_thresholds = prior_calibrated_predictions(
            test_df, test_probabilities, train_validation_df)
        prior_dir.mkdir(parents=True, exist_ok=True)
        calibrated_metrics, calibrated_per_language_metrics = evaluate_predictions(
            calibrated_predictions, prior_dir, f"{TAG}_prior_calibrated", include_probabilities=True)
        save_json(prior_dir / "best_config.json",
                  {"model_name": MODEL_NAME, "tag": f"{TAG}_prior_calibrated",
                   "prior_calibrated_thresholds": calibrated_thresholds})
        run_error_analysis(prior_dir / "predictions.csv")
        update_comparison_outputs(f"{TAG}_prior_calibrated", prior_dir, calibrated_metrics,
                                  calibrated_per_language_metrics,
                                  "validation_learning_rate_plus_unlabeled_prior_calibration")
        pairwise_direction = os.environ.get("COS760_PAIRWISE_DIRECTION", "lower")
        pairwise_predictions = pairwise_exploratory_predictions(predictions, direction=pairwise_direction)
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        pairwise_metrics, pairwise_per_language_metrics = evaluate_predictions(
            pairwise_predictions, pairwise_dir, f"{TAG}_pairwise_exploratory", include_probabilities=True)
        save_json(pairwise_dir / "best_config.json",
                  {"model_name": MODEL_NAME, "tag": f"{TAG}_pairwise_exploratory",
                   "pairwise_direction": pairwise_direction,
                   "warning": "Exploratory paired-sample post-processing; report separately."})
        run_error_analysis(pairwise_dir / "predictions.csv")
        update_comparison_outputs(f"{TAG}_pairwise_exploratory", pairwise_dir, pairwise_metrics,
                                  pairwise_per_language_metrics,
                                  "exploratory_pairwise_postprocessing_no_threshold")
        print(f"\nDone. strict={metrics['macro_f1']:.3f} "
              f"prior_cal={calibrated_metrics['macro_f1']:.3f} "
              f"pairwise={pairwise_metrics['macro_f1']:.3f}")
        print(f"Outputs in: {output_dir.resolve()}")
        return 0
    except ImportError as exc:
        print(f"ERROR: Transformer deps missing. Install requirements.txt. Details: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
