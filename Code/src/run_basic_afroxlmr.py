"""Optional fixed AfroXLMR baseline for the COS760 basic experiment.

This script uses only train and test data. It does not use validation,
hyperparameter tuning, early stopping, or threshold tuning.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Conservative defaults for macOS/Anaconda stability. The user can set
# COS760_USE_MPS=1 to try MPS explicitly, but CPU is safer for this small run.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from evaluate_basic import evaluate_predictions
from utils import RANDOM_SEED, validate_basic_dataframe, validate_train_test_sets


TRAIN_PATH = Path("data/processed/basic/train.csv")
TEST_PATH = Path("data/processed/basic/test.csv")
OUTPUT_DIR = Path("outputs/basic/afroxlmr_basic")

MODEL_NAME = "Davlan/afro-xlmr-base-76L_script"
MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "256"))
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = int(os.environ.get("COS760_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "8"))
NUM_TRAIN_EPOCHS = float(os.environ.get("COS760_NUM_TRAIN_EPOCHS", "3"))
WEIGHT_DECAY = 0.01


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed train/test CSVs created by build_basic_dataset.py."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing train set: {TRAIN_PATH.resolve()}. "
            "Run python src/build_basic_dataset.py first."
        )
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing test set: {TEST_PATH.resolve()}. "
            "Run python src/build_basic_dataset.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    validate_basic_dataframe(train_df, str(TRAIN_PATH))
    validate_basic_dataframe(test_df, str(TEST_PATH))
    validate_train_test_sets(train_df, test_df)

    return train_df, test_df


class TextClassificationDataset:
    """Tiny PyTorch Dataset wrapper around tokenized texts and labels."""

    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer) -> None:
        import torch

        self.encodings = tokenizer(
            texts.astype(str).tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
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


def make_predictions(test_df: pd.DataFrame, predicted_labels: np.ndarray) -> pd.DataFrame:
    """Format transformer predictions to match the basic pipeline outputs."""
    return pd.DataFrame(
        {
            "id": test_df["id"],
            "text": test_df["text"],
            "language": test_df["language"],
            "language_name": test_df["language_name"],
            "true_label": test_df["label"].astype(int),
            "predicted_label": predicted_labels.astype(int),
            "source": test_df["source"],
            "generator": test_df["generator"],
            "split": test_df["split"],
        }
    )


def main() -> int:
    """Fine-tune AfroXLMR with fixed settings and evaluate once on test."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        train_df, test_df = load_train_test()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        set_seed(RANDOM_SEED)

        use_mps = (
            os.environ.get("COS760_USE_MPS") == "1"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        use_cpu = not torch.cuda.is_available() and not use_mps
        device_hint = "cuda" if torch.cuda.is_available() else "mps" if use_mps else "cpu"
        print(f"Using device for Trainer: {device_hint}")
        print(f"Train rows: {len(train_df)}")
        print(f"Test rows: {len(test_df)}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        )
        test_dataset = TextClassificationDataset(
            test_df["text"],
            test_df["label"],
            tokenizer,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "trainer"),
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            seed=RANDOM_SEED,
            use_cpu=use_cpu,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=25,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        raw_predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(raw_predictions.predictions, axis=1)
        predictions = make_predictions(test_df, predicted_labels)

        evaluate_predictions(predictions, OUTPUT_DIR, "afroxlmr_basic")
        trainer.save_model(str(OUTPUT_DIR / "model"))
        tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))

        print(f"Wrote outputs to: {OUTPUT_DIR.resolve()}")
        return 0
    except ImportError as exc:
        print(
            "ERROR: Missing transformer dependencies. Install requirements.txt first. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
