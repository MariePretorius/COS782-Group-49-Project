"""Create calibrated AfriBERTa outputs from an existing predictions.csv.

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
from run_enhanced_afriberta import (
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
HYBRID_OUTPUT_DIR = Path("outputs/enhanced/afriberta_hybrid_language_exploratory")
HYBRID_SOURCES = {
    "afr": Path("outputs/enhanced/afriberta_pairwise_exploratory/predictions.csv"),
    "eng": Path("outputs/enhanced/tfidf_char_logreg_prior_calibrated/predictions.csv"),
    "zul": Path("outputs/enhanced/afriberta_pairwise_exploratory/predictions.csv"),
}


def load_existing_predictions() -> pd.DataFrame:
    """Load strict AfriBERTa predictions produced by run_enhanced_afriberta.py."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing AfriBERTa predictions: {PREDICTIONS_PATH.resolve()}. "
            "Run src/run_enhanced_afriberta.py first."
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
        "afriberta_prior_calibrated",
        include_probabilities=True,
    )
    save_json(
        PRIOR_CALIBRATED_OUTPUT_DIR / "best_config.json",
        {
            "prior_calibrated_thresholds": thresholds,
            "calibration_source": "existing_afriberta_predictions_plus_train_validation_label_priors",
        },
    )
    run_error_analysis(PRIOR_CALIBRATED_OUTPUT_DIR / "predictions.csv")
    update_comparison_outputs(
        "afriberta_prior_calibrated",
        PRIOR_CALIBRATED_OUTPUT_DIR,
        metrics,
        per_language_metrics,
        "postprocess_unlabeled_prior_calibration",
    )


def write_pairwise(predictions: pd.DataFrame) -> None:
    """Write exploratory pairwise outputs from existing probabilities."""
    direction = os.environ.get("COS760_AFRIBERTA_PAIRWISE_DIRECTION", "lower")
    pairwise_predictions = pairwise_exploratory_predictions(predictions, direction=direction)

    PAIRWISE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, per_language_metrics = evaluate_predictions(
        pairwise_predictions,
        PAIRWISE_OUTPUT_DIR,
        "afriberta_pairwise_exploratory",
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
        "afriberta_pairwise_exploratory",
        PAIRWISE_OUTPUT_DIR,
        metrics,
        per_language_metrics,
        "postprocess_exploratory_pairwise",
    )


def write_language_hybrid() -> None:
    """Write an exploratory language-specific hybrid from available predictions."""
    missing = [path for path in HYBRID_SOURCES.values() if not path.exists()]
    if missing:
        print(
            "WARNING: Could not create language hybrid because these files are missing: "
            + ", ".join(str(path) for path in missing)
        )
        return

    source_predictions = {
        language: pd.read_csv(path).set_index("id")
        for language, path in HYBRID_SOURCES.items()
    }
    template = source_predictions["afr"].copy()

    hybrid = template.copy()
    for row_id, row in template.iterrows():
        language = row["language"]
        hybrid.loc[row_id, "predicted_label"] = source_predictions[language].loc[
            row_id,
            "predicted_label",
        ]

    HYBRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hybrid = hybrid.reset_index()
    metrics, per_language_metrics = evaluate_predictions(
        hybrid,
        HYBRID_OUTPUT_DIR,
        "afriberta_hybrid_language_exploratory",
        include_probabilities=True,
    )
    save_json(
        HYBRID_OUTPUT_DIR / "best_config.json",
        {
            "language_model_sources": {
                language: str(path)
                for language, path in HYBRID_SOURCES.items()
            },
            "warning": (
                "Exploratory language-specific hybrid based on observed language "
                "behavior. Keep separate from strict model-selection results."
            ),
        },
    )
    run_error_analysis(HYBRID_OUTPUT_DIR / "predictions.csv")
    update_comparison_outputs(
        "afriberta_hybrid_language_exploratory",
        HYBRID_OUTPUT_DIR,
        metrics,
        per_language_metrics,
        "postprocess_language_specific_exploratory_hybrid",
    )


def main() -> int:
    """Create post-processed AfriBERTa outputs without retraining."""
    try:
        predictions = load_existing_predictions()
        calibration_df = load_calibration_data()
        write_prior_calibrated(predictions, calibration_df)
        write_pairwise(predictions)
        write_language_hybrid()
        print(f"Wrote prior-calibrated outputs to: {PRIOR_CALIBRATED_OUTPUT_DIR.resolve()}")
        print(f"Wrote pairwise exploratory outputs to: {PAIRWISE_OUTPUT_DIR.resolve()}")
        print(f"Wrote language hybrid outputs to: {HYBRID_OUTPUT_DIR.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
