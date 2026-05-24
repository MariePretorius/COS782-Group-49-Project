"""Shared helpers for the COS760 experiment pipelines."""

from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE_DIR = Path("outputs/.cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RANDOM_SEED = 42

LABEL_NAMES = {
    0: "human",
    1: "machine",
}

LANGUAGE_NAMES = {
    "afr": "Afrikaans",
    "eng": "English",
    "zul": "isiZulu",
}

VALID_LANGUAGES = set(LANGUAGE_NAMES)
VALID_SPLITS = {"train", "test"}
ENHANCED_VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = set(LABEL_NAMES)


def ensure_columns(dataframe: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """Raise a helpful error when a CSV is missing required columns."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def validate_experiment_dataframe(
    dataframe: pd.DataFrame,
    name: str,
    valid_splits: set[str],
) -> None:
    """Validate common experiment dataset fields."""
    ensure_columns(
        dataframe,
        ["id", "text", "label", "label_name", "language", "language_name", "split"],
        name,
    )

    empty_text = dataframe["text"].isna() | dataframe["text"].astype(str).str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"{name} contains {int(empty_text.sum())} rows with empty text.")

    labels = set(dataframe["label"].dropna().astype(int).unique())
    unexpected_labels = sorted(labels - VALID_LABELS)
    if unexpected_labels:
        raise ValueError(f"{name} has unexpected label values: {unexpected_labels}")

    label_name_mismatches = dataframe.loc[
        dataframe["label"].astype(int).map(LABEL_NAMES) != dataframe["label_name"]
    ]
    if not label_name_mismatches.empty:
        raise ValueError(
            f"{name} contains {len(label_name_mismatches)} rows where label_name "
            "does not match label."
        )

    languages = set(dataframe["language"].dropna().astype(str).unique())
    unexpected_languages = sorted(languages - VALID_LANGUAGES)
    if unexpected_languages:
        raise ValueError(f"{name} has unexpected language values: {unexpected_languages}")

    splits = set(dataframe["split"].dropna().astype(str).unique())
    unexpected_splits = sorted(splits - valid_splits)
    if unexpected_splits:
        raise ValueError(f"{name} has unexpected split values: {unexpected_splits}")


def validate_basic_dataframe(dataframe: pd.DataFrame, name: str) -> None:
    """Validate the common train/test experiment dataset fields."""
    validate_experiment_dataframe(dataframe, name, VALID_SPLITS)


def validate_train_test_sets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Check that the final train and test sets are suitable for a basic experiment."""
    if train_df.empty:
        raise ValueError("Train set is empty.")
    if test_df.empty:
        raise ValueError("Test set is empty.")

    for name, dataframe in [("train set", train_df), ("test set", test_df)]:
        labels = set(dataframe["label"].astype(int).unique())
        missing_labels = sorted(VALID_LABELS - labels)
        if missing_labels:
            raise ValueError(f"{name} is missing label values: {missing_labels}")

        languages = set(dataframe["language"].astype(str).unique())
        missing_languages = sorted(VALID_LANGUAGES - languages)
        if missing_languages:
            raise ValueError(f"{name} is missing languages: {missing_languages}")


def validate_split_sets(split_frames: dict[str, pd.DataFrame]) -> None:
    """Check that train/validation/test splits all have both labels and languages."""
    for split, dataframe in split_frames.items():
        if dataframe.empty:
            raise ValueError(f"{split} set is empty.")

        labels = set(dataframe["label"].astype(int).unique())
        missing_labels = sorted(VALID_LABELS - labels)
        if missing_labels:
            raise ValueError(f"{split} set is missing label values: {missing_labels}")

        languages = set(dataframe["language"].astype(str).unique())
        missing_languages = sorted(VALID_LANGUAGES - languages)
        if missing_languages:
            raise ValueError(f"{split} set is missing languages: {missing_languages}")


def compute_metrics(y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int]) -> dict[str, float]:
    """Compute the fixed metric set used by all basic experiments."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_per_language_metrics(
    dataframe: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    model_name: str | None = None,
) -> pd.DataFrame:
    """Compute metrics separately for Afrikaans, English, and isiZulu."""
    metrics_rows = []
    eval_df = dataframe.copy()
    eval_df["true_label"] = y_true
    eval_df["predicted_label"] = y_pred

    for language, language_name in LANGUAGE_NAMES.items():
        language_df = eval_df.loc[eval_df["language"] == language]
        if language_df.empty:
            continue

        row = {
            "language": language,
            "language_name": language_name,
            **compute_metrics(
                language_df["true_label"].to_numpy(),
                language_df["predicted_label"].to_numpy(),
            ),
            "support": len(language_df),
        }
        if model_name is not None:
            row = {"model": model_name, **row}

        metrics_rows.append(row)

    return pd.DataFrame(metrics_rows)


def save_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_classification_report(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a human/machine classification report as text."""
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["human", "machine"],
        zero_division=0,
    )
    path.write_text(report, encoding="utf-8")


def save_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """Save a simple confusion matrix PNG."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["human", "machine"])
    ax.set_yticks([0, 1], labels=["human", "machine"])

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
