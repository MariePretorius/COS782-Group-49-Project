"""Run enhanced traditional ML experiments with validation-based selection."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from evaluate_enhanced import evaluate_predictions
from utils import (
    ENHANCED_VALID_SPLITS,
    RANDOM_SEED,
    compute_metrics,
    save_json,
    validate_experiment_dataframe,
    validate_split_sets,
)


TRAIN_PATH = Path("data/processed/enhanced/train.csv")
VALIDATION_PATH = Path("data/processed/enhanced/validation.csv")
TEST_PATH = Path("data/processed/enhanced/test.csv")
FULL_DATASET_PATH = Path("data/processed/enhanced/full_dataset.csv")
OUTPUT_ROOT = Path("outputs/enhanced")
COMPARISON_OUTPUT_DIR = Path("outputs/comparison")

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


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Load and validate one enhanced split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enhanced {split_name} set: {path.resolve()}. "
            "Run python src/build_enhanced_dataset.py first."
        )
    dataframe = pd.read_csv(path)
    validate_experiment_dataframe(dataframe, str(path), ENHANCED_VALID_SPLITS)
    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test enhanced datasets."""
    train_df = load_split(TRAIN_PATH, "train")
    validation_df = load_split(VALIDATION_PATH, "validation")
    test_df = load_split(TEST_PATH, "test")
    validate_split_sets(
        {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
        }
    )
    return train_df, validation_df, test_df


def make_classifier(classifier_type: str, c_value: float):
    """Create a fixed classifier with only C selected on validation."""
    if classifier_type == "logreg":
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
    if classifier_type == "svm":
        return LinearSVC(
            C=c_value,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown classifier type: {classifier_type}")


def build_pipeline(family: str, classifier_type: str, params: dict[str, object]) -> Pipeline:
    """Build one candidate pipeline from a model family and parameter dict."""
    classifier = make_classifier(classifier_type, float(params["classifier_C"]))

    if family == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=params["ngram_range"],
            max_features=int(params["max_features"]),
            min_df=int(params["min_df"]),
        )
    elif family == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=params["ngram_range"],
            max_features=int(params["max_features"]),
            min_df=int(params["min_df"]),
        )
    elif family == "combined":
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=int(params["word_max_features"]),
                        min_df=int(params["word_min_df"]),
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=int(params["char_max_features"]),
                        min_df=int(params["char_min_df"]),
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    return Pipeline([("features", vectorizer), ("classifier", classifier)])


def candidate_params(family: str) -> list[dict[str, object]]:
    """Generate validation-search candidates for one model family."""
    c_values = [0.1, 1.0, 3.0]

    if family == "word":
        return [
            {
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
                "classifier_C": c_value,
            }
            for ngram_range, max_features, min_df, c_value in itertools.product(
                [(1, 1), (1, 2)],
                [30000, 50000, 100000],
                [1, 2],
                c_values,
            )
        ]

    if family == "char":
        return [
            {
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
                "classifier_C": c_value,
            }
            for ngram_range, max_features, min_df, c_value in itertools.product(
                [(3, 5), (3, 6), (4, 6)],
                [30000, 50000, 100000],
                [1, 2],
                c_values,
            )
        ]

    if family == "combined":
        return [
            {
                "word_max_features": word_max_features,
                "char_max_features": char_max_features,
                "word_min_df": 1,
                "char_min_df": 1,
                "classifier_C": c_value,
            }
            for word_max_features, char_max_features, c_value in itertools.product(
                [30000, 50000, 100000],
                [30000, 50000, 100000],
                c_values,
            )
        ]

    raise ValueError(f"Unknown model family: {family}")


def make_predictions(dataframe: pd.DataFrame, predicted_labels) -> pd.DataFrame:
    """Format predictions with the requested columns."""
    predictions = pd.DataFrame(
        {
            "id": dataframe["id"],
            "text": dataframe["text"],
            "language": dataframe["language"],
            "language_name": dataframe["language_name"],
            "true_label": dataframe["label"].astype(int),
            "predicted_label": predicted_labels,
            "source": dataframe["source"],
            "generator": dataframe["generator"],
            "split": dataframe["split"],
        }
    )
    return predictions[PREDICTION_COLUMNS]


def run_validation_search(
    model_name: str,
    family: str,
    classifier_type: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train candidates on train and select the best by validation macro_f1."""
    rows = []
    best_row = None
    best_params = None

    for index, params in enumerate(candidate_params(family), start=1):
        pipeline = build_pipeline(family, classifier_type, params)
        pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(int))
        validation_pred = pipeline.predict(validation_df["text"].astype(str))
        metrics = compute_metrics(validation_df["label"].astype(int), validation_pred)

        row = {
            "candidate": index,
            **params,
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)

        if best_row is None or metrics["macro_f1"] > best_row["validation_macro_f1"]:
            best_row = row
            best_params = params

    assert best_row is not None and best_params is not None
    validation_results = pd.DataFrame(rows)
    print(
        f"Best {model_name} validation macro_f1: "
        f"{best_row['validation_macro_f1']:.4f}"
    )
    return best_params, validation_results


def run_model_family(
    model_name: str,
    family: str,
    classifier_type: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select, retrain, and evaluate one enhanced traditional model family."""
    print(f"\nRunning enhanced model family: {model_name}")
    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params, validation_results = run_validation_search(
        model_name,
        family,
        classifier_type,
        train_df,
        validation_df,
    )
    validation_results.to_csv(output_dir / "validation_results.csv", index=False)
    save_json(output_dir / "best_params.json", best_params)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline = build_pipeline(family, classifier_type, best_params)
    final_pipeline.fit(
        train_validation_df["text"].astype(str),
        train_validation_df["label"].astype(int),
    )

    test_pred = final_pipeline.predict(test_df["text"].astype(str))
    predictions = make_predictions(test_df, test_pred)
    metrics, per_language_metrics = evaluate_predictions(predictions, output_dir, model_name)
    joblib.dump(final_pipeline, output_dir / "model.joblib")

    print(pd.Series(metrics).to_string())
    return (
        {
            "model": model_name,
            **metrics,
            "selected_by": "validation_macro_f1",
            "best_params_path": str(output_dir / "best_params.json"),
        },
        per_language_metrics.assign(model=model_name),
    )


def write_comparison_outputs(
    comparison_rows: list[dict[str, object]],
    per_language_frames: list[pd.DataFrame],
) -> None:
    """Write enhanced model comparison CSVs."""
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(
        comparison_rows,
        columns=[
            "model",
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "weighted_f1",
            "selected_by",
            "best_params_path",
        ],
    )
    comparison.to_csv(COMPARISON_OUTPUT_DIR / "enhanced_model_comparison.csv", index=False)

    per_language = pd.concat(per_language_frames, ignore_index=True)
    per_language = per_language[
        [
            "model",
            "language",
            "language_name",
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "weighted_f1",
            "support",
        ]
    ]
    per_language.to_csv(
        COMPARISON_OUTPUT_DIR / "enhanced_per_language_metrics.csv",
        index=False,
    )


def write_basic_vs_enhanced_summary(enhanced_comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare matching basic and enhanced traditional models when available."""
    basic_path = Path("outputs/basic/basic_model_comparison.csv")
    output_path = COMPARISON_OUTPUT_DIR / "basic_vs_enhanced_summary.csv"
    rows = []

    if not basic_path.exists():
        print(f"WARNING: Basic comparison file not found: {basic_path}. Skipping comparison.")
        pd.DataFrame(
            columns=[
                "model_group",
                "basic_model",
                "enhanced_model",
                "basic_macro_f1",
                "enhanced_macro_f1",
                "macro_f1_change",
                "basic_accuracy",
                "enhanced_accuracy",
                "accuracy_change",
            ]
        ).to_csv(output_path, index=False)
        return pd.DataFrame()

    basic = pd.read_csv(basic_path)
    for _, enhanced_row in enhanced_comparison.iterrows():
        enhanced_model = enhanced_row["model"]
        basic_match = basic.loc[basic["model"] == enhanced_model]
        if basic_match.empty:
            print(f"WARNING: No matching basic result for {enhanced_model}; skipping.")
            continue

        basic_row = basic_match.iloc[0]
        rows.append(
            {
                "model_group": enhanced_model.replace("tfidf_", ""),
                "basic_model": basic_row["model"],
                "enhanced_model": enhanced_model,
                "basic_macro_f1": basic_row["macro_f1"],
                "enhanced_macro_f1": enhanced_row["macro_f1"],
                "macro_f1_change": enhanced_row["macro_f1"] - basic_row["macro_f1"],
                "basic_accuracy": basic_row["accuracy"],
                "enhanced_accuracy": enhanced_row["accuracy"],
                "accuracy_change": enhanced_row["accuracy"] - basic_row["accuracy"],
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "model_group",
            "basic_model",
            "enhanced_model",
            "basic_macro_f1",
            "enhanced_macro_f1",
            "macro_f1_change",
            "basic_accuracy",
            "enhanced_accuracy",
            "accuracy_change",
        ],
    )
    summary.to_csv(output_path, index=False)
    return summary


def write_experiment_summary(
    enhanced_comparison: pd.DataFrame,
    per_language: pd.DataFrame,
    basic_vs_enhanced: pd.DataFrame,
) -> None:
    """Write a concise markdown summary for reports."""
    summary_path = OUTPUT_ROOT / "experiment_summary.md"
    dataset_counts = "Dataset counts unavailable."
    if FULL_DATASET_PATH.exists():
        full_df = pd.read_csv(FULL_DATASET_PATH)
        dataset_counts = full_df.groupby(["split", "language", "label"]).size().to_string()

    best_model = enhanced_comparison.sort_values("macro_f1", ascending=False).iloc[0]
    best_lang_rows = per_language.loc[per_language["model"] == best_model["model"]]
    best_language = best_lang_rows.sort_values("macro_f1", ascending=False).iloc[0]
    worst_language = best_lang_rows.sort_values("macro_f1").iloc[0]

    transformer_path = OUTPUT_ROOT / "afroxlmr" / "metrics.json"
    transformer_note = "Not run yet."
    if transformer_path.exists():
        transformer_note = transformer_path.read_text(encoding="utf-8")

    comparison_text = (
        basic_vs_enhanced.to_string(index=False)
        if not basic_vs_enhanced.empty
        else "No matching basic results were available for comparison."
    )

    predictions_path = OUTPUT_ROOT / best_model["model"] / "predictions.csv"
    error_note = "Error patterns unavailable."
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        errors = predictions.loc[predictions["true_label"] != predictions["predicted_label"]].copy()
        if not errors.empty:
            error_note = (
                "Most errors by language:\n"
                + errors["language"].value_counts().to_string()
                + "\n\nMost errors by generator:\n"
                + errors["generator"].value_counts().to_string()
            )
        else:
            error_note = "The best traditional model made no test-set errors."

    markdown = f"""# Enhanced Experiment Summary

## Dataset Counts

```text
{dataset_counts}
```

## Best Enhanced Traditional Model

- Model: {best_model['model']}
- Accuracy: {best_model['accuracy']:.4f}
- Macro F1: {best_model['macro_f1']:.4f}
- Selected by: {best_model['selected_by']}

## Enhanced Transformer Result

```text
{transformer_note}
```

## Basic vs Enhanced

```text
{comparison_text}
```

## Language Notes

- Best language for the best traditional model: {best_language['language_name']} ({best_language['macro_f1']:.4f} macro F1)
- Weakest language for the best traditional model: {worst_language['language_name']} ({worst_language['macro_f1']:.4f} macro F1)

## Common Error Patterns

```text
{error_note}
```
"""
    summary_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    """Run all enhanced traditional model families."""
    try:
        train_df, validation_df, test_df = load_datasets()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(validation_df)}")
        print(f"Test rows: {len(test_df)}")

        model_specs = [
            ("tfidf_word_logreg", "word", "logreg"),
            ("tfidf_word_svm", "word", "svm"),
            ("tfidf_char_logreg", "char", "logreg"),
            ("tfidf_char_svm", "char", "svm"),
            ("tfidf_combined_logreg", "combined", "logreg"),
            ("tfidf_combined_svm", "combined", "svm"),
        ]

        comparison_rows = []
        per_language_frames = []
        for model_name, family, classifier_type in model_specs:
            metrics_row, per_language_metrics = run_model_family(
                model_name,
                family,
                classifier_type,
                train_df,
                validation_df,
                test_df,
            )
            comparison_rows.append(metrics_row)
            per_language_frames.append(per_language_metrics)

        write_comparison_outputs(comparison_rows, per_language_frames)
        enhanced_comparison = pd.DataFrame(comparison_rows)
        basic_vs_enhanced = write_basic_vs_enhanced_summary(enhanced_comparison)
        per_language = pd.concat(per_language_frames, ignore_index=True)
        write_experiment_summary(enhanced_comparison, per_language, basic_vs_enhanced)

        print(f"\nWrote enhanced outputs to: {OUTPUT_ROOT.resolve()}")
        print(f"Wrote comparison outputs to: {COMPARISON_OUTPUT_DIR.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
