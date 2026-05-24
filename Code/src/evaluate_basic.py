"""Evaluation helpers and CLI for the COS760 basic experiment outputs."""

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


PREDICTION_COLUMNS = [
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


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Save standard metrics, reports, and confusion matrix for predictions."""
    missing_columns = [
        column for column in PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions["true_label"].astype(int).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()

    metrics = compute_metrics(y_true, y_pred)
    per_language_metrics = compute_per_language_metrics(
        predictions,
        y_true,
        y_pred,
    )

    save_json(output_dir / "metrics.json", {"model": model_name, **metrics})
    predictions[PREDICTION_COLUMNS].to_csv(output_dir / "predictions.csv", index=False)
    per_language_metrics.to_csv(output_dir / "per_language_metrics.csv", index=False)
    save_confusion_matrix(
        output_dir / "confusion_matrix.png",
        y_true,
        y_pred,
        title=model_name,
    )
    save_classification_report(
        output_dir / "classification_report.txt",
        y_true,
        y_pred,
    )

    return metrics, per_language_metrics


def main() -> int:
    """Evaluate an existing predictions.csv file.

    Usage:
        python src/evaluate_basic.py outputs/basic/tfidf_word_logreg/predictions.csv
    """
    if len(sys.argv) != 2:
        print(
            "ERROR: Expected one argument: path to predictions.csv",
            file=sys.stderr,
        )
        return 1

    prediction_path = Path(sys.argv[1])
    if not prediction_path.exists():
        print(f"ERROR: Missing predictions file: {prediction_path.resolve()}", file=sys.stderr)
        return 1

    try:
        predictions = pd.read_csv(prediction_path)
        output_dir = prediction_path.parent
        model_name = output_dir.name
        metrics, _ = evaluate_predictions(predictions, output_dir, model_name)
        print(f"Evaluated: {prediction_path}")
        print(pd.Series(metrics).to_string())
        return 0
    except (ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
