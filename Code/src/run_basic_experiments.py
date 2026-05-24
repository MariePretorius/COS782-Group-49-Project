"""Run fixed traditional baselines for human vs machine text detection."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluate_basic import evaluate_predictions
from utils import validate_basic_dataframe, validate_train_test_sets


RANDOM_SEED = 42

TRAIN_PATH = Path("data/processed/basic/train.csv")
TEST_PATH = Path("data/processed/basic/test.csv")
OUTPUT_ROOT = Path("outputs/basic")

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


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the train/test CSVs created by build_basic_dataset.py."""
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


def build_models() -> dict[str, Pipeline]:
    """Create the four fixed baseline pipelines."""
    return {
        "tfidf_word_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=50000,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "tfidf_word_svm": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=50000,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "tfidf_char_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=50000,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "tfidf_char_svm": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=50000,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def make_predictions(test_df: pd.DataFrame, predicted_labels: list[int]) -> pd.DataFrame:
    """Format model predictions for downstream inspection."""
    predictions = pd.DataFrame(
        {
            "id": test_df["id"],
            "text": test_df["text"],
            "language": test_df["language"],
            "language_name": test_df["language_name"],
            "true_label": test_df["label"].astype(int),
            "predicted_label": predicted_labels,
            "source": test_df["source"],
            "generator": test_df["generator"],
            "split": test_df["split"],
        }
    )
    return predictions[PREDICTION_COLUMNS]


def run_model(
    model_name: str,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one fixed model and save its evaluation outputs."""
    print(f"\nRunning model: {model_name}")
    model_output_dir = OUTPUT_ROOT / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(int))
    predicted_labels = pipeline.predict(test_df["text"].astype(str))
    predictions = make_predictions(test_df, predicted_labels)

    metrics, per_language_metrics = evaluate_predictions(
        predictions,
        model_output_dir,
        model_name,
    )
    joblib.dump(pipeline, model_output_dir / "model.joblib")

    print(pd.Series(metrics).to_string())
    return {"model": model_name, **metrics}, per_language_metrics


def write_comparison_outputs(
    comparison_rows: list[dict[str, object]],
    per_language_frames: list[pd.DataFrame],
) -> None:
    """Save cross-model comparison CSVs."""
    comparison = pd.DataFrame(
        comparison_rows,
        columns=["model", "accuracy", "precision", "recall", "macro_f1", "weighted_f1"],
    )
    comparison.to_csv(OUTPUT_ROOT / "basic_model_comparison.csv", index=False)

    per_language = pd.concat(per_language_frames, ignore_index=True)
    per_language.to_csv(OUTPUT_ROOT / "basic_per_language_metrics.csv", index=False)

    print(f"\nWrote comparison: {(OUTPUT_ROOT / 'basic_model_comparison.csv').resolve()}")
    print(
        "Wrote per-language comparison: "
        f"{(OUTPUT_ROOT / 'basic_per_language_metrics.csv').resolve()}"
    )


def main() -> int:
    """Run all four basic baseline models."""
    try:
        train_df, test_df = load_train_test()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        comparison_rows = []
        per_language_frames = []

        print(f"Train rows: {len(train_df)}")
        print(f"Test rows: {len(test_df)}")

        for model_name, pipeline in build_models().items():
            metrics_row, per_language_metrics = run_model(
                model_name,
                pipeline,
                train_df,
                test_df,
            )
            comparison_rows.append(metrics_row)
            per_language_metrics.insert(0, "model", model_name)
            per_language_frames.append(per_language_metrics)

        write_comparison_outputs(comparison_rows, per_language_frames)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
