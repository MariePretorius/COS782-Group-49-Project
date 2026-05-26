"""Evaluate a saved fine-tuned model on any enhanced-format CSV."""

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

EVAL_REQUIRED = ["id", "text", "label", "language", "language_name", "source", "generator", "split"]
MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "128"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "32"))


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class TextClassificationDataset:
    def __init__(self, texts, tokenizer, max_length):
        import torch
        self.encodings = tokenizer(texts.astype(str).tolist(), truncation=True, max_length=max_length)
        self.torch = torch
        self.length = len(texts)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: self.torch.tensor(values[index]) for key, values in self.encodings.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on an enhanced CSV.")
    parser.add_argument("--model-dir", required=True, help="Folder with a saved HF sequence-classification model.")
    parser.add_argument("--eval-csv", required=True, help="Enhanced-format CSV to evaluate on.")
    parser.add_argument("--tag", required=True, help="Label for output dir and metrics.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Machine-class probability threshold; default uses argmax.")
    parser.add_argument("--out-dir", default=None, help="Defaults to outputs/transfer/<tag>.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_path = Path(args.eval_csv)
    if not eval_path.exists():
        print(f"ERROR: Missing eval CSV: {eval_path.resolve()}", file=sys.stderr)
        return 1
    eval_df = pd.read_csv(eval_path)
    missing = [c for c in EVAL_REQUIRED if c not in eval_df.columns]
    if missing:
        print(f"ERROR: eval CSV missing columns: {missing}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/transfer") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        dataset = TextClassificationDataset(eval_df["text"], tokenizer, MAX_LENGTH)
        trainer = Trainer(model=model, tokenizer=tokenizer)
        probabilities = softmax(trainer.predict(dataset).predictions)
        if args.threshold is None:
            predicted_labels = np.argmax(probabilities, axis=1)
        else:
            predicted_labels = (probabilities[:, 1] >= args.threshold).astype(int)
        predictions = pd.DataFrame({
            "id": eval_df["id"], "text": eval_df["text"], "language": eval_df["language"],
            "language_name": eval_df["language_name"], "true_label": eval_df["label"].astype(int),
            "predicted_label": predicted_labels, "human_probability": probabilities[:, 0],
            "machine_probability": probabilities[:, 1], "source": eval_df["source"],
            "generator": eval_df["generator"], "split": eval_df["split"]})
        metrics, per_language_metrics = evaluate_predictions(
            predictions, out_dir, args.tag, include_probabilities=True)
        run_error_analysis(out_dir / "predictions.csv")
        print(f"Model: {args.model_dir}")
        print(f"Eval:  {args.eval_csv}  (threshold={args.threshold})")
        print(pd.Series(metrics).to_string())
        print("\nPer language:")
        print(per_language_metrics.to_string(index=False))
        print(f"\nWrote: {out_dir.resolve()}")
        return 0
    except ImportError as exc:
        print(f"ERROR: Transformer deps missing. Install requirements.txt. Details: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
