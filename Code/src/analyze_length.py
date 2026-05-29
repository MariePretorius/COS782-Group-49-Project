"""Length-stratified error analysis. CPU only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/.cache/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import compute_metrics

DEFAULT_BINS = [0, 5, 10, 20, 40, 80, 1_000_000]
DEFAULT_LABELS = ["1-4", "5-9", "10-19", "20-39", "40-79", "80+"]


def bin_metrics(frame: pd.DataFrame) -> dict:
    y_true = frame["true_label"].astype(int).to_numpy()
    y_pred = frame["predicted_label"].astype(int).to_numpy()
    metrics = compute_metrics(y_true, y_pred)
    machine = frame.loc[frame["true_label"].astype(int) == 1]
    false_negatives = int(((machine["predicted_label"].astype(int)) == 0).sum())
    metrics["machine_support"] = int(len(machine))
    metrics["false_negatives"] = false_negatives
    metrics["false_negative_rate"] = (false_negatives / len(machine)) if len(machine) else float("nan")
    metrics["support"] = int(len(frame))
    return metrics


def stratify(predictions: pd.DataFrame, by_language: bool) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["text_length_words"] = predictions["text"].astype(str).str.split().str.len()
    predictions["length_bin"] = pd.cut(
        predictions["text_length_words"], bins=DEFAULT_BINS, labels=DEFAULT_LABELS, right=False)
    rows = []
    groups = ["__all__"] + (sorted(predictions["language"].unique()) if by_language else [])
    for language in groups:
        subset = predictions if language == "__all__" else predictions.loc[predictions["language"] == language]
        for length_bin in DEFAULT_LABELS:
            bin_df = subset.loc[subset["length_bin"] == length_bin]
            if bin_df.empty:
                continue
            rows.append({"language": language, "length_bin": length_bin, **bin_metrics(bin_df)})
    return pd.DataFrame(rows)


def plot_curves(table: pd.DataFrame, tag: str, out_dir: Path) -> Path:
    overall = table.loc[table["language"] == "__all__"].set_index("length_bin").reindex(DEFAULT_LABELS).dropna(how="all")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(overall.index, overall["recall"], marker="o", label="machine recall")
    ax.plot(overall.index, overall["macro_f1"], marker="s", label="macro F1")
    ax.plot(overall.index, overall["false_negative_rate"], marker="^", label="false-negative rate")
    ax.set_xlabel("Text length (words)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Detection vs text length: {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / f"length_curves_{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def parse_args():
    parser = argparse.ArgumentParser(description="Length-stratified error analysis.")
    parser.add_argument("--predictions", required=True, help="Path to a predictions.csv.")
    parser.add_argument("--tag", required=True, help="Label for output files.")
    parser.add_argument("--out-dir", default="outputs/length_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_path = Path(args.predictions)
    if not prediction_path.exists():
        print(f"ERROR: Missing predictions file: {prediction_path.resolve()}", file=sys.stderr)
        return 1
    predictions = pd.read_csv(prediction_path)
    required = {"text", "true_label", "predicted_label", "language"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        print(f"ERROR: predictions missing columns: {missing}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = stratify(predictions, by_language=True)
    table_path = out_dir / f"length_metrics_{args.tag}.csv"
    table.to_csv(table_path, index=False)
    plot_path = plot_curves(table, args.tag, out_dir)
    print(table.loc[table["language"] == "__all__"].to_string(index=False))
    print(f"\nWrote: {table_path.resolve()}")
    print(f"Wrote: {plot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
