"""Create calibrated Monolingual Afrikaans BERT outputs from an existing predictions.csv.

This is useful when transformer training has already taken hours and you only
want to try the faster post-processing strategies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from error_analysis import run_error_analysis
from evaluate_enhanced import evaluate_predictions
from run_enhanced_afrikaans_bert import (
    PAIRWISE_OUTPUT_DIR,
    PRIOR_CALIBRATED_OUTPUT_DIR,
    OUTPUT_DIR,
    pairwise_exploratory_predictions,
    prior_calibrated_predictions,
    update_comparison_outputs,
)
from utils import save_json


TRAIN_PATH = Path("data/processed/enhanced/train.csv")
VALIDATION_PATH = Path("data/processed/enhanced/validation.csv")
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

def load_existing_predictions() -> pd.DataFrame:
    """Load strict monolingual Afrikaans RoBERTa predictions."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing Monolingual Afrikaans BERT predictions: {PREDICTIONS_PATH.resolve()}. "
            "Run src/run_enhanced_afrikaans_bert.py first."
        )
    return pd.read_csv(PREDICTIONS_PATH)


def load_calibration_data() -> pd.DataFrame:
    """Load train+validation rows for prior calibration."""
    if not TRAIN_PATH.exists() or not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "Missing enhanced train/validation data. "
            "Run src/build_enhanced_dataset.py first."
        )
    return pd.concat(
        [pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)],
        ignore_index=True,
    )


def predictions_to_dataset_like(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add columns expected by prior_calibrated_predictions."""
    converted = predictions.copy()
    converted["label"] = converted["true_label"]
    return converted


def write_prior_calibrated(predictions: pd.DataFrame, calibration_df: pd.DataFrame) -> None:
    """Write prior-calibrated outputs from existing probabilities."""
    probabilities = predictions[["human_probability", "machine_probability"]].to_numpy()
    calibrated_predictions, thresholds = prior_calibrated_predictions(
        predictions_to_dataset_like(predictions),
        probabilities,
        calibration_df,
    )

    PRIOR_CALIBRATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, per_language_metrics = evaluate_predictions(
        calibrated_predictions,
        PRIOR_CALIBRATED_OUTPUT_DIR,
        "afrikaans_roberta_prior_calibrated",
        include_probabilities=True,
    )
    save_json(
        PRIOR_CALIBRATED_OUTPUT_DIR / "best_config.json",
        {
            "prior_calibrated_thresholds": thresholds,
            "calibration_source": "existing_afrikaans_roberta_predictions_plus_train_validation_label_priors",
        },
    )
    run_error_analysis(PRIOR_CALIBRATED_OUTPUT_DIR / "predictions.csv")
    update_comparison_outputs(
        "afrikaans_roberta_prior_calibrated",
        PRIOR_CALIBRATED_OUTPUT_DIR,
        metrics,
        per_language_metrics,
        "postprocess_unlabeled_prior_calibration",
    )


def write_pairwise(predictions: pd.DataFrame) -> None:
    """Write exploratory pairwise outputs from existing probabilities."""
    direction = os.environ.get("COS760_AFRIKAANS_BERT_PAIRWISE_DIRECTION", "lower")
    pairwise_predictions = pairwise_exploratory_predictions(predictions, direction=direction)

    PAIRWISE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, per_language_metrics = evaluate_predictions(
        pairwise_predictions,
        PAIRWISE_OUTPUT_DIR,
        "afrikaans_roberta_pairwise_exploratory",
        include_probabilities=True,
    )
    save_json(
        PAIRWISE_OUTPUT_DIR / "best_config.json",
        {
            "pairwise_direction": direction,
            "warning": (
                "Exploratory paired-sample post-processing. This assumes each "
                "human test row has exactly one machine paraphrase partner and "
                "should be reported separately from ordinary classification."
            ),
        },
    )
    run_error_analysis(PAIRWISE_OUTPUT_DIR / "predictions.csv")
    update_comparison_outputs(
        "afrikaans_roberta_pairwise_exploratory",
        PAIRWISE_OUTPUT_DIR,
        metrics,
        per_language_metrics,
        "postprocess_exploratory_pairwise",
    )


def main() -> int:
    """Create post-processed Monolingual Afrikaans BERT outputs without retraining."""
    try:
        predictions = load_existing_predictions()
        calibration_df = load_calibration_data()
        calibration_df = calibration_df.loc[calibration_df["language"] == "afr"].copy()
        write_prior_calibrated(predictions, calibration_df)
        write_pairwise(predictions)
        print(f"Wrote prior-calibrated outputs to: {PRIOR_CALIBRATED_OUTPUT_DIR.resolve()}")
        print(f"Wrote pairwise exploratory outputs to: {PAIRWISE_OUTPUT_DIR.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
