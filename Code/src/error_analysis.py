"""Create error analysis artifacts for enhanced model predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS_PATH = Path("outputs/enhanced/afroxlmr/predictions.csv")
REQUIRED_COLUMNS = [
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
ERROR_COLUMNS = [
    "id",
    "text",
    "language",
    "language_name",
    "true_label",
    "predicted_label",
    "machine_probability",
    "source",
    "generator",
    "split",
    "error_type",
    "text_length_words",
]


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions CSV and validate the columns needed for error analysis."""
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path.resolve()}")

    predictions = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(
            "Predictions file is missing required columns: "
            f"{', '.join(missing)}"
        )

    if "machine_probability" not in predictions.columns:
        predictions["machine_probability"] = pd.NA

    return predictions


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per false positive/false negative."""
    analysis = predictions.copy()
    analysis["text_length_words"] = analysis["text"].astype(str).str.split().str.len()
    analysis["error_type"] = ""
    analysis.loc[
        (analysis["true_label"] == 0) & (analysis["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"
    analysis.loc[
        (analysis["true_label"] == 1) & (analysis["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    errors = analysis.loc[analysis["error_type"] != ""].copy()
    return errors[ERROR_COLUMNS]


def format_examples(title: str, examples: pd.DataFrame) -> list[str]:
    """Format compact example snippets for the text summary."""
    lines = [title]
    if examples.empty:
        lines.append("- None")
        return lines

    for _, row in examples.iterrows():
        snippet = str(row["text"]).replace("\n", " ")[:220]
        lines.append(
            f"- {row['id']} | {row['language']} | {row['error_type']} | "
            f"{row['text_length_words']} words | {snippet}"
        )
    return lines


def write_error_summary(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    summary_path: Path,
) -> None:
    """Write a report-friendly text summary of model errors."""
    predictions = predictions.copy()
    predictions["text_length_words"] = predictions["text"].astype(str).str.split().str.len()
    correct = predictions.loc[predictions["true_label"] == predictions["predicted_label"]]

    false_positives = errors.loc[errors["error_type"] == "false_positive"]
    false_negatives = errors.loc[errors["error_type"] == "false_negative"]

    lines = [
        "Enhanced AfroXLMR Error Summary",
        "==============================",
        "",
        f"Total test samples: {len(predictions)}",
        f"Total errors: {len(errors)}",
        f"Total false positives: {len(false_positives)}",
        f"Total false negatives: {len(false_negatives)}",
        "",
        "Errors per language:",
        errors["language"].value_counts().sort_index().to_string()
        if not errors.empty
        else "None",
        "",
        "Errors per generator:",
        errors["generator"].value_counts().sort_index().to_string()
        if not errors.empty
        else "None",
        "",
        "Average text length:",
        f"- Correctly classified: {correct['text_length_words'].mean():.2f} words",
        f"- Misclassified: {errors['text_length_words'].mean():.2f} words"
        if not errors.empty
        else "- Misclassified: n/a",
        "",
    ]

    shortest = errors.sort_values("text_length_words").head(5)
    longest = errors.sort_values("text_length_words", ascending=False).head(5)
    lines.extend(format_examples("Five shortest misclassified examples:", shortest))
    lines.append("")
    lines.extend(format_examples("Five longest misclassified examples:", longest))
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_error_analysis(predictions_path: Path = DEFAULT_PREDICTIONS_PATH) -> tuple[Path, Path]:
    """Create error_analysis.csv and error_summary.txt next to predictions.csv."""
    predictions = load_predictions(predictions_path)
    errors = build_error_analysis(predictions)
    output_dir = predictions_path.parent
    analysis_path = output_dir / "error_analysis.csv"
    summary_path = output_dir / "error_summary.txt"

    errors.to_csv(analysis_path, index=False)
    write_error_summary(predictions, errors, summary_path)

    return analysis_path, summary_path


def main() -> int:
    """Run error analysis for enhanced AfroXLMR by default, or a provided CSV."""
    predictions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH

    try:
        analysis_path, summary_path = run_error_analysis(predictions_path)
        print(f"Wrote error analysis: {analysis_path.resolve()}")
        print(f"Wrote error summary: {summary_path.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
