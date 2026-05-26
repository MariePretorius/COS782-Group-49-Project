"""Cross-lingual transfer experiment (leave-one-out and train-on-one)."""

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

from evaluate_enhanced import evaluate_predictions
from utils import (
    ENHANCED_VALID_SPLITS,
    LANGUAGE_NAMES,
    RANDOM_SEED,
    compute_metrics,
    save_json,
    validate_experiment_dataframe,
)

TRAIN_PATH = Path("data/processed/enhanced/train.csv")
VALIDATION_PATH = Path("data/processed/enhanced/validation.csv")
TEST_PATH = Path("data/processed/enhanced/test.csv")

MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "128"))
TRAIN_BATCH_SIZE = int(os.environ.get("COS760_TRAIN_BATCH_SIZE", "16"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "32"))
NUM_TRAIN_EPOCHS = float(os.environ.get("COS760_NUM_TRAIN_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("COS760_LEARNING_RATE", "3e-5"))


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enhanced {split_name} set: {path.resolve()}. "
            "Run python src/build_enhanced_dataset.py first.")
    dataframe = pd.read_csv(path)
    validate_experiment_dataframe(dataframe, str(path), ENHANCED_VALID_SPLITS)
    return dataframe


class TextClassificationDataset:
    def __init__(self, texts, labels, tokenizer, max_length):
        import torch
        self.encodings = tokenizer(texts.astype(str).tolist(), truncation=True, max_length=max_length)
        self.labels = labels.astype(int).tolist()
        self.torch = torch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: self.torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = self.torch.tensor(self.labels[index])
        return item


def compute_class_weights(train_labels):
    import torch
    counts = train_labels.astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    weights = total / (len(counts) * counts.replace(0, 1))
    return torch.tensor(weights.to_numpy(dtype=np.float32))


def make_weighted_trainer_class(class_weights):
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


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_model(model_name, train_df, validation_df, tokenizer, out_dir):
    import torch
    from transformers import (
        AutoModelForSequenceClassification, DataCollatorWithPadding,
        EarlyStoppingCallback, TrainingArguments, set_seed,
    )
    set_seed(RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "human", 1: "machine"}, label2id={"human": 0, "machine": 1})
    train_dataset = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer, MAX_LENGTH)
    validation_dataset = TextClassificationDataset(validation_df["text"], validation_df["label"], tokenizer, MAX_LENGTH)
    class_weights = compute_class_weights(train_df["label"])
    trainer_class = make_weighted_trainer_class(class_weights)
    training_args = TrainingArguments(
        output_dir=str(out_dir), learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS, weight_decay=0.01, warmup_ratio=0.1,
        seed=RANDOM_SEED, use_cpu=not torch.cuda.is_available(), fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        dataloader_num_workers=0, dataloader_pin_memory=False, report_to=[])

    def trainer_metrics(eval_prediction):
        logits, labels = eval_prediction
        return compute_metrics(labels, np.argmax(logits, axis=1))

    trainer = trainer_class(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()
    return trainer


def predict_df(trainer, tokenizer, eval_df):
    dataset = TextClassificationDataset(eval_df["text"], eval_df["label"], tokenizer, MAX_LENGTH)
    probabilities = softmax(trainer.predict(dataset).predictions)
    predicted_labels = np.argmax(probabilities, axis=1)
    return pd.DataFrame({
        "id": eval_df["id"], "text": eval_df["text"], "language": eval_df["language"],
        "language_name": eval_df["language_name"], "true_label": eval_df["label"].astype(int),
        "predicted_label": predicted_labels, "human_probability": probabilities[:, 0],
        "machine_probability": probabilities[:, 1], "source": eval_df["source"],
        "generator": eval_df["generator"], "split": eval_df["split"]})


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-lingual transfer experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--protocol", default="leave-one-out", choices=["leave-one-out", "train-on-one"])
    parser.add_argument("--source", default="eng", choices=sorted(LANGUAGE_NAMES),
                        help="Source language for train-on-one protocol.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path("outputs/crosslingual") / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer
        train_df = load_split(TRAIN_PATH, "train")
        validation_df = load_split(VALIDATION_PATH, "validation")
        test_df = load_split(TEST_PATH, "test")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        summary_rows = []

        if args.protocol == "leave-one-out":
            languages = sorted(LANGUAGE_NAMES)
            for held_out in languages:
                sources = [lang for lang in languages if lang != held_out]
                tr = train_df.loc[train_df["language"].isin(sources)]
                va = validation_df.loc[validation_df["language"].isin(sources)]
                te = test_df.loc[test_df["language"] == held_out]
                print(f"\n=== Hold out {held_out}: train on {sources} ===")
                setting_dir = out_root / f"holdout_{held_out}"
                trainer = train_model(args.model, tr, va, tokenizer, setting_dir / "train")
                predictions = predict_df(trainer, tokenizer, te)
                metrics, _ = evaluate_predictions(predictions, setting_dir, f"{args.tag}_holdout_{held_out}",
                                                  include_probabilities=True)
                summary_rows.append({"protocol": "leave-one-out", "trained_on": "+".join(sources),
                                     "evaluated_on": held_out, "transfer": "zero-shot", **metrics})
        else:
            source = args.source
            tr = train_df.loc[train_df["language"] == source]
            va = validation_df.loc[validation_df["language"] == source]
            print(f"\n=== Train on {source} only ===")
            trainer = train_model(args.model, tr, va, tokenizer, out_root / f"train_{source}" / "train")
            for target in sorted(LANGUAGE_NAMES):
                te = test_df.loc[test_df["language"] == target]
                setting_dir = out_root / f"train_{source}" / f"eval_{target}"
                predictions = predict_df(trainer, tokenizer, te)
                metrics, _ = evaluate_predictions(predictions, setting_dir, f"{args.tag}_{source}_to_{target}",
                                                  include_probabilities=True)
                summary_rows.append({"protocol": "train-on-one", "trained_on": source, "evaluated_on": target,
                                     "transfer": "in-language" if target == source else "zero-shot", **metrics})

        summary = pd.DataFrame(summary_rows)
        summary_path = out_root / "crosslingual_summary.csv"
        summary.to_csv(summary_path, index=False)
        save_json(out_root / "config.json", {"model": args.model, "protocol": args.protocol,
                                              "source": args.source, "learning_rate": LEARNING_RATE,
                                              "num_train_epochs": NUM_TRAIN_EPOCHS})
        print("\nSummary:")
        print(summary.to_string(index=False))
        print(f"\nWrote: {summary_path.resolve()}")
        return 0
    except ImportError as exc:
        print(f"ERROR: Transformer deps missing. Install requirements.txt. Details: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
