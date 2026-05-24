"""Evaluation helpers for enhanced COS760 experiment outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils import (
    compute_metrics,
    compute_per_language_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_json,
)


BASE_PREDICTION_COLUMNS = [
    "id",
    "text",
    "language",
    "language_name",
    "true_label",
    "predicted_label",
    "source",
    "generator",
    "split",
]

TRANSFORMER_PREDICTION_COLUMNS = [
    "id",
    "text",
    "language",
    "language_name",
    "true_label",
    "predicted_label",
    "human_probability",
    "machine_probability",
    "source",
    "generator",
    "split",
]


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    include_probabilities: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Save metrics, per-language metrics, confusion matrix, and report."""
    required_columns = (
        TRANSFORMER_PREDICTION_COLUMNS if include_probabilities else BASE_PREDICTION_COLUMNS
    )
    missing_columns = [column for column in required_columns if column not in predictions.columns]
    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()

    metrics = compute_metrics(y_true, y_pred)
    per_language_metrics = compute_per_language_metrics(predictions, y_true, y_pred)

    save_json(output_dir / "metrics.json", {"model": model_name, **metrics})
    predictions[required_columns].to_csv(output_dir / "predictions.csv", index=False)
    per_language_metrics.to_csv(output_dir / "per_language_metrics.csv", index=False)
    save_confusion_matrix(output_dir / "confusion_matrix.png", y_true, y_pred, model_name)
    save_classification_report(output_dir / "classification_report.txt", y_true, y_pred)

    return metrics, per_language_metrics


def main() -> int:
    """Evaluate an enhanced predictions.csv file from the command line."""
    if len(sys.argv) != 2:
        print("ERROR: Expected one argument: path to predictions.csv", file=sys.stderr)
        return 1

    prediction_path = Path(sys.argv[1])
    if not prediction_path.exists():
        print(f"ERROR: Missing predictions file: {prediction_path.resolve()}", file=sys.stderr)
        return 1

    try:
        predictions = pd.read_csv(prediction_path)
        include_probabilities = {
            "human_probability",
            "machine_probability",
        }.issubset(predictions.columns)
        metrics, _ = evaluate_predictions(
            predictions,
            prediction_path.parent,
            prediction_path.parent.name,
            include_probabilities=include_probabilities,
        )
        print(f"Evaluated: {prediction_path}")
        print(pd.Series(metrics).to_string())
        return 0
    except (ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
