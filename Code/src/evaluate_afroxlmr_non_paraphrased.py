"""Evaluate the fine-tuned AfroXLMR model on non-paraphrased machine text."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from error_analysis import run_error_analysis
from evaluate_enhanced import evaluate_predictions
from run_enhanced_afroxlmr import (
    TextClassificationDataset,
    pairwise_exploratory_predictions,
    prior_calibrated_predictions,
    softmax,
    update_comparison_outputs,
)
from utils import save_json, validate_experiment_dataframe


MODEL_PATH = Path("outputs/enhanced/afroxlmr/model")
TEST_PATH = Path("data/processed/non_paraphrased/test.csv")
TRAIN_PATH = Path("data/processed/enhanced/train.csv")
VALIDATION_PATH = Path("data/processed/enhanced/validation.csv")
OUTPUT_DIR = Path("outputs/enhanced/afroxlmr_non_paraphrased")
PRIOR_OUTPUT_DIR = Path("outputs/enhanced/afroxlmr_non_paraphrased_prior_calibrated")
PAIRWISE_OUTPUT_DIR = Path("outputs/enhanced/afroxlmr_non_paraphrased_pairwise_exploratory")
MAX_LENGTH = int(os.environ.get("COS760_MAX_LENGTH", "128"))
EVAL_BATCH_SIZE = int(os.environ.get("COS760_EVAL_BATCH_SIZE", "8"))


def predictions_from_probabilities(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Format thresholded model probabilities for evaluation."""
    return pd.DataFrame(
        {
            "id": dataframe["id"],
            "text": dataframe["text"],
            "language": dataframe["language"],
            "language_name": dataframe["language_name"],
            "true_label": dataframe["label"].astype(int),
            "predicted_label": (probabilities[:, 1] >= threshold).astype(int),
            "human_probability": probabilities[:, 0],
            "machine_probability": probabilities[:, 1],
            "source": dataframe["source"],
            "generator": dataframe["generator"],
            "split": dataframe["split"],
        }
    )


def load_test_set() -> pd.DataFrame:
    """Load the non-paraphrased test set."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing non-paraphrased test set: {TEST_PATH.resolve()}. "
            "Run src/build_non_paraphrased_test_dataset.py first."
        )
    dataframe = pd.read_csv(TEST_PATH)
    validate_experiment_dataframe(dataframe, str(TEST_PATH), {"test"})
    return dataframe


def main() -> int:
    """Evaluate saved AfroXLMR model on non-paraphrased test rows."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing fine-tuned AfroXLMR model: {MODEL_PATH.resolve()}. "
                "Run src/run_enhanced_afroxlmr.py first."
            )

        test_df = load_test_set()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        dataset = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "trainer"),
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            use_cpu=not torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
        )
        trainer = Trainer(model=model, args=args)
        raw_predictions = trainer.predict(dataset)
        probabilities = softmax(raw_predictions.predictions)

        threshold = 0.5
        config_path = Path("outputs/enhanced/afroxlmr/best_config.json")
        if config_path.exists():
            config = pd.read_json(config_path, typ="series")
            threshold = float(config.get("threshold", 0.5))

        predictions = predictions_from_probabilities(test_df, probabilities, threshold)
        metrics, per_language = evaluate_predictions(
            predictions,
            OUTPUT_DIR,
            "afroxlmr_non_paraphrased",
            include_probabilities=True,
        )
        save_json(OUTPUT_DIR / "best_config.json", {"threshold": threshold, "source_model": str(MODEL_PATH)})
        run_error_analysis(OUTPUT_DIR / "predictions.csv")
        update_comparison_outputs(
            "afroxlmr_non_paraphrased",
            OUTPUT_DIR,
            metrics,
            per_language,
            "saved_afroxlmr_on_non_paraphrased_test",
        )

        calibration_df = pd.concat(
            [pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)],
            ignore_index=True,
        )
        prior_predictions, prior_thresholds = prior_calibrated_predictions(
            test_df,
            probabilities,
            calibration_df,
        )
        prior_metrics, prior_per_language = evaluate_predictions(
            prior_predictions,
            PRIOR_OUTPUT_DIR,
            "afroxlmr_non_paraphrased_prior_calibrated",
            include_probabilities=True,
        )
        save_json(
            PRIOR_OUTPUT_DIR / "best_config.json",
            {
                "prior_calibrated_thresholds": prior_thresholds,
                "source_model": str(MODEL_PATH),
            },
        )
        run_error_analysis(PRIOR_OUTPUT_DIR / "predictions.csv")
        update_comparison_outputs(
            "afroxlmr_non_paraphrased_prior_calibrated",
            PRIOR_OUTPUT_DIR,
            prior_metrics,
            prior_per_language,
            "saved_afroxlmr_on_non_paraphrased_prior_calibrated",
        )

        pairwise_predictions = pairwise_exploratory_predictions(
            predictions,
            direction=os.environ.get("COS760_PAIRWISE_DIRECTION", "lower"),
        )
        pairwise_metrics, pairwise_per_language = evaluate_predictions(
            pairwise_predictions,
            PAIRWISE_OUTPUT_DIR,
            "afroxlmr_non_paraphrased_pairwise_exploratory",
            include_probabilities=True,
        )
        save_json(
            PAIRWISE_OUTPUT_DIR / "best_config.json",
            {
                "warning": "Exploratory pairing over balanced but unpaired non-paraphrased rows.",
                "source_model": str(MODEL_PATH),
            },
        )
        run_error_analysis(PAIRWISE_OUTPUT_DIR / "predictions.csv")
        update_comparison_outputs(
            "afroxlmr_non_paraphrased_pairwise_exploratory",
            PAIRWISE_OUTPUT_DIR,
            pairwise_metrics,
            pairwise_per_language,
            "saved_afroxlmr_on_non_paraphrased_pairwise_exploratory",
        )

        print(f"Wrote outputs to: {OUTPUT_DIR.resolve()}")
        print(f"Wrote prior-calibrated outputs to: {PRIOR_OUTPUT_DIR.resolve()}")
        print(f"Wrote pairwise outputs to: {PAIRWISE_OUTPUT_DIR.resolve()}")
        return 0
    except (ImportError, FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
