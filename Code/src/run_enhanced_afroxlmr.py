"""Enhanced AfroXLMR experiment with validation selection and threshold tuning."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
OUTPUT_DIR = Path("outputs/enhanced/afroxlmr")
COMPARISON_OUTPUT_DIR = Path("outputs/comparison")

MODEL_NAME = "Davlan/afro-xlmr-base-76L_script"
MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "256"))
TRAIN_BATCH_SIZE = int(os.environ.get("COS760_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "8"))
NUM_TRAIN_EPOCHS = float(os.environ.get("COS760_NUM_TRAIN_EPOCHS", "5"))
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATES = [1e-5, 2e-5, 3e-5]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Load and validate one enhanced split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enhanced {split_name} set: {path.resolve()}. "
            "Run python src/build_enhanced_dataset.py first."
        )
    dataframe = pd.read_csv(path)
    validate_experiment_dataframe(dataframe, str(path), ENHANCED_VALID_SPLITS)
    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test data."""
    train_df = load_split(TRAIN_PATH, "train")
    validation_df = load_split(VALIDATION_PATH, "validation")
    test_df = load_split(TEST_PATH, "test")
    validate_split_sets({"train": train_df, "validation": validation_df, "test": test_df})
    return train_df, validation_df, test_df


class TextClassificationDataset:
    """Small PyTorch Dataset wrapper for tokenized text classification."""

    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_length: int) -> None:
        import torch

        self.encodings = tokenizer(
            texts.astype(str).tolist(),
            truncation=True,
            max_length=max_length,
        )
        self.labels = labels.astype(int).tolist()
        self.torch = torch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {
            key: self.torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = self.torch.tensor(self.labels[index])
        return item


def compute_class_weights(train_labels: pd.Series):
    """Compute inverse-frequency class weights from training labels only."""
    import torch

    counts = train_labels.astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    weights = total / (len(counts) * counts.replace(0, 1))
    return torch.tensor(weights.to_numpy(dtype=np.float32))


def make_weighted_trainer_class(class_weights):
    """Create a Trainer subclass that applies class-weighted CrossEntropyLoss."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable NumPy softmax for model logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predictions_from_probabilities(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Format thresholded transformer predictions."""
    predicted_labels = (probabilities[:, 1] >= threshold).astype(int)
    return pd.DataFrame(
        {
            "id": dataframe["id"],
            "text": dataframe["text"],
            "language": dataframe["language"],
            "language_name": dataframe["language_name"],
            "true_label": dataframe["label"].astype(int),
            "predicted_label": predicted_labels,
            "human_probability": probabilities[:, 0],
            "machine_probability": probabilities[:, 1],
            "source": dataframe["source"],
            "generator": dataframe["generator"],
            "split": dataframe["split"],
        }
    )


def train_one_learning_rate(
    learning_rate: float,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer,
    model_output_dir: Path,
):
    """Train one learning-rate candidate and return trainer plus validation probabilities."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "human", 1: "machine"},
        label2id={"human": 0, "machine": 1},
    )

    train_dataset = TextClassificationDataset(
        train_df["text"],
        train_df["label"],
        tokenizer,
        MAX_LENGTH,
    )
    validation_dataset = TextClassificationDataset(
        validation_df["text"],
        validation_df["label"],
        tokenizer,
        MAX_LENGTH,
    )
    class_weights = compute_class_weights(train_df["label"])
    trainer_class = make_weighted_trainer_class(class_weights)

    use_cpu = not torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        seed=RANDOM_SEED,
        use_cpu=use_cpu,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],
    )

    def trainer_metrics(eval_prediction):
        logits, labels = eval_prediction
        predicted_labels = np.argmax(logits, axis=1)
        return compute_metrics(labels, predicted_labels)

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()
    validation_output = trainer.predict(validation_dataset)
    validation_probabilities = softmax(validation_output.predictions)
    return trainer, validation_probabilities


def tune_threshold(validation_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate fixed thresholds on validation probabilities."""
    rows = []
    y_true = validation_df["label"].astype(int).to_numpy()
    for threshold in THRESHOLDS:
        predicted_labels = (probabilities[:, 1] >= threshold).astype(int)
        rows.append({"threshold": threshold, **compute_metrics(y_true, predicted_labels)})
    return pd.DataFrame(rows)


def update_comparison_outputs(metrics: dict[str, float], per_language_metrics: pd.DataFrame) -> None:
    """Add/replace the AfroXLMR row in enhanced comparison outputs."""
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARISON_OUTPUT_DIR / "enhanced_model_comparison.csv"
    per_language_path = COMPARISON_OUTPUT_DIR / "enhanced_per_language_metrics.csv"

    new_comparison_row = pd.DataFrame(
        [
            {
                "model": "afroxlmr",
                **metrics,
                "selected_by": "validation_macro_f1_and_threshold",
                "best_params_path": str(OUTPUT_DIR / "best_config.json"),
            }
        ]
    )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison.loc[comparison["model"] != "afroxlmr"]
        comparison = pd.concat([comparison, new_comparison_row], ignore_index=True)
    else:
        comparison = new_comparison_row
    comparison.to_csv(comparison_path, index=False)

    per_language = per_language_metrics.copy()
    per_language.insert(0, "model", "afroxlmr")
    if per_language_path.exists():
        existing_per_language = pd.read_csv(per_language_path)
        existing_per_language = existing_per_language.loc[
            existing_per_language["model"] != "afroxlmr"
        ]
        per_language = pd.concat([existing_per_language, per_language], ignore_index=True)
    per_language.to_csv(per_language_path, index=False)

    try:
        from run_enhanced_experiments import (
            write_basic_vs_enhanced_summary,
            write_experiment_summary,
        )

        basic_vs_enhanced = write_basic_vs_enhanced_summary(comparison)
        write_experiment_summary(comparison, per_language, basic_vs_enhanced)
    except Exception as exc:  # pragma: no cover - summary refresh should not break training.
        print(f"WARNING: Could not refresh comparison summary: {exc}")


def main() -> int:
    """Run enhanced AfroXLMR with LR sweep and validation threshold tuning."""
    try:
        import torch
        from transformers import AutoTokenizer

        train_df, validation_df, test_df = load_datasets()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        device_hint = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device available to Trainer: {device_hint}")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(validation_df)}")
        print(f"Test rows: {len(test_df)}")
        print(f"max_length={MAX_LENGTH}, train_batch_size={TRAIN_BATCH_SIZE}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        sweep_rows = []
        best_trainer = None
        best_probabilities = None
        best_learning_rate = None
        best_macro_f1 = -1.0

        for learning_rate in LEARNING_RATES:
            print(f"\nTraining AfroXLMR with learning_rate={learning_rate}")
            trainer, validation_probabilities = train_one_learning_rate(
                learning_rate,
                train_df,
                validation_df,
                tokenizer,
                OUTPUT_DIR / f"lr_{learning_rate:g}",
            )
            validation_predictions = np.argmax(validation_probabilities, axis=1)
            metrics = compute_metrics(
                validation_df["label"].astype(int).to_numpy(),
                validation_predictions,
            )
            sweep_rows.append({"learning_rate": learning_rate, **metrics})
            print(pd.Series(metrics).to_string())

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_learning_rate = learning_rate
                best_trainer = trainer
                best_probabilities = validation_probabilities

        assert best_trainer is not None and best_probabilities is not None
        learning_rate_sweep = pd.DataFrame(sweep_rows)
        learning_rate_sweep.to_csv(OUTPUT_DIR / "learning_rate_sweep.csv", index=False)

        threshold_results = tune_threshold(validation_df, best_probabilities)
        threshold_results.to_csv(OUTPUT_DIR / "threshold_tuning.csv", index=False)
        best_threshold_row = threshold_results.sort_values("macro_f1", ascending=False).iloc[0]
        best_threshold = float(best_threshold_row["threshold"])

        save_json(
            OUTPUT_DIR / "best_config.json",
            {
                "model_name": MODEL_NAME,
                "learning_rate": best_learning_rate,
                "threshold": best_threshold,
                "validation_macro_f1_for_learning_rate": best_macro_f1,
                "validation_macro_f1_for_threshold": float(best_threshold_row["macro_f1"]),
                "max_length": MAX_LENGTH,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "num_train_epochs": NUM_TRAIN_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
        )

        test_dataset = TextClassificationDataset(
            test_df["text"],
            test_df["label"],
            tokenizer,
            MAX_LENGTH,
        )
        test_output = best_trainer.predict(test_dataset)
        test_probabilities = softmax(test_output.predictions)
        predictions = predictions_from_probabilities(test_df, test_probabilities, best_threshold)

        metrics, per_language_metrics = evaluate_predictions(
            predictions,
            OUTPUT_DIR,
            "afroxlmr",
            include_probabilities=True,
        )
        best_trainer.save_model(str(OUTPUT_DIR / "model"))
        tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))
        run_error_analysis(OUTPUT_DIR / "predictions.csv")
        update_comparison_outputs(metrics, per_language_metrics)

        print(f"Wrote outputs to: {OUTPUT_DIR.resolve()}")
        return 0
    except ImportError as exc:
        print(
            "ERROR: Transformer dependencies are not installed. "
            "Install requirements.txt first. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
