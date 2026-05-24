"""Build the enhanced train/validation/test dataset for COS760 experiments."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from utils import (
    ENHANCED_VALID_SPLITS,
    LANGUAGE_NAMES,
    VALID_LANGUAGES,
    ensure_columns,
    validate_experiment_dataframe,
    validate_split_sets,
)


HUMAN_PATH = Path("data/processed/human_samples.csv")
MACHINE_DIR = Path("data/machine_generated")
ENHANCED_OUTPUT_DIR = Path("data/processed/enhanced")

FULL_OUTPUT_PATH = ENHANCED_OUTPUT_DIR / "full_dataset.csv"
TRAIN_OUTPUT_PATH = ENHANCED_OUTPUT_DIR / "train.csv"
VALIDATION_OUTPUT_PATH = ENHANCED_OUTPUT_DIR / "validation.csv"
TEST_OUTPUT_PATH = ENHANCED_OUTPUT_DIR / "test.csv"

HUMAN_REQUIRED_COLUMNS = [
    "id",
    "text",
    "label",
    "label_name",
    "language",
    "language_name",
    "source",
    "generator",
    "split",
]

MACHINE_REQUIRED_COLUMNS = [
    "original_human_id",
    "machine_text",
    "language",
    "language_name",
    "split",
    "generator",
]

FINAL_COLUMNS = [
    "id",
    "original_human_id",
    "text",
    "label",
    "label_name",
    "language",
    "language_name",
    "source",
    "generator",
    "split",
    "prompt_type",
]

MACHINE_FILENAME_PATTERN = re.compile(
    r"^machine_samp(?:le|el)s_(?P<language>afr|eng|zul)_(?P<split>train|validation|test)\.csv$"
)


def find_machine_files() -> list[Path]:
    """Find machine CSVs, supporting the requested misspelling and current spelling."""
    if not MACHINE_DIR.exists():
        raise FileNotFoundError(f"Missing machine-generated folder: {MACHINE_DIR.resolve()}")

    files = sorted(MACHINE_DIR.glob("machine_sampels_*_*.csv"))
    corrected_files = sorted(MACHINE_DIR.glob("machine_samples_*_*.csv"))
    files = sorted(set(files + corrected_files))

    if not files:
        raise FileNotFoundError(
            "No machine CSV files found. Expected files matching "
            f"{MACHINE_DIR}/machine_sampels_*_*.csv"
        )

    expected = {
        (language, split)
        for language in sorted(VALID_LANGUAGES)
        for split in sorted(ENHANCED_VALID_SPLITS)
    }
    found = set()
    for path in files:
        match = MACHINE_FILENAME_PATTERN.match(path.name)
        if match:
            found.add((match.group("language"), match.group("split")))

    missing = sorted(expected - found)
    if missing:
        raise FileNotFoundError(
            "Missing validation/train/test machine files for: "
            + ", ".join(f"{language}_{split}" for language, split in missing)
        )

    return files


def load_human_rows() -> pd.DataFrame:
    """Load all human rows for train, validation, and test."""
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(f"Missing human_samples.csv: {HUMAN_PATH.resolve()}")

    human_df = pd.read_csv(HUMAN_PATH)
    ensure_columns(human_df, HUMAN_REQUIRED_COLUMNS, str(HUMAN_PATH))
    print(f"Human rows loaded: {len(human_df)}")

    human_df = human_df.loc[human_df["split"].isin(ENHANCED_VALID_SPLITS)].copy()
    human_df["original_human_id"] = ""
    human_df["prompt_type"] = ""

    validate_experiment_dataframe(human_df, "human samples", ENHANCED_VALID_SPLITS)
    return human_df[FINAL_COLUMNS]


def parse_machine_filename(path: Path) -> tuple[str | None, str | None]:
    """Parse language/split from supported generated-machine filenames."""
    match = MACHINE_FILENAME_PATTERN.match(path.name)
    if not match:
        return None, None
    return match.group("language"), match.group("split")


def convert_machine_file(path: Path) -> pd.DataFrame:
    """Convert one machine-generated CSV to the final enhanced schema."""
    filename_language, filename_split = parse_machine_filename(path)
    machine_df = pd.read_csv(path)
    ensure_columns(machine_df, MACHINE_REQUIRED_COLUMNS, str(path))

    if "prompt_type" not in machine_df.columns:
        machine_df["prompt_type"] = "paraphrase"
    machine_df["prompt_type"] = machine_df["prompt_type"].fillna("paraphrase")

    if filename_language is not None:
        csv_languages = set(machine_df["language"].dropna().astype(str).unique())
        if csv_languages and csv_languages != {filename_language}:
            print(
                f"WARNING: {path.name} language {filename_language} differs from "
                f"CSV values {sorted(csv_languages)}. Trusting CSV columns."
            )

    if filename_split is not None:
        csv_splits = set(machine_df["split"].dropna().astype(str).unique())
        if csv_splits and csv_splits != {filename_split}:
            print(
                f"WARNING: {path.name} split {filename_split} differs from "
                f"CSV values {sorted(csv_splits)}. Trusting CSV columns."
            )

    return pd.DataFrame(
        {
            "id": "machine_" + machine_df["original_human_id"].astype(str),
            "original_human_id": machine_df["original_human_id"].astype(str),
            "text": machine_df["machine_text"],
            "label": 1,
            "label_name": "machine",
            "language": machine_df["language"],
            "language_name": machine_df["language_name"],
            "source": "generated",
            "generator": machine_df["generator"],
            "split": machine_df["split"],
            "prompt_type": machine_df["prompt_type"],
        }
    )


def validate_machine_against_human(machine_df: pd.DataFrame, human_df: pd.DataFrame) -> None:
    """Warn for missing original IDs and fail if machine rows changed source split."""
    human_split_by_id = human_df.set_index("id")["split"].to_dict()
    original_ids = machine_df["original_human_id"].astype(str)
    missing_ids = sorted(set(original_ids) - set(human_split_by_id))
    if missing_ids:
        print(
            "WARNING: "
            f"{len(missing_ids)} machine original_human_id values do not exist in "
            "human_samples.csv. First few: "
            + ", ".join(missing_ids[:10])
        )

    known_machine_rows = machine_df.loc[original_ids.isin(human_split_by_id)].copy()
    expected_splits = known_machine_rows["original_human_id"].map(human_split_by_id)
    mismatches = known_machine_rows.loc[known_machine_rows["split"] != expected_splits]
    if not mismatches.empty:
        examples = mismatches[["original_human_id", "split"]].head(10).to_dict("records")
        raise ValueError(
            "Machine rows must preserve the same split as their original human row. "
            f"Found {len(mismatches)} mismatches. Examples: {examples}"
        )


def load_machine_rows(human_df: pd.DataFrame) -> pd.DataFrame:
    """Load and validate all generated machine rows for enhanced experiments."""
    machine_files = find_machine_files()
    print(f"Machine files found: {len(machine_files)}")

    frames = []
    for path in machine_files:
        converted = convert_machine_file(path)
        print(f"  - {path} ({len(converted)} rows loaded)")
        frames.append(converted)

    machine_df = pd.concat(frames, ignore_index=True)
    machine_df = machine_df.loc[machine_df["split"].isin(ENHANCED_VALID_SPLITS)].copy()
    validate_machine_against_human(machine_df, human_df)
    validate_experiment_dataframe(machine_df, "machine samples", ENHANCED_VALID_SPLITS)

    return machine_df[FINAL_COLUMNS]


def print_counts(dataframe: pd.DataFrame) -> None:
    """Print requested enhanced dataset summary counts."""
    print("\nFinal row counts by split:")
    print(dataframe["split"].value_counts().sort_index().to_string())

    print("\nFinal row counts by label:")
    print(dataframe["label"].value_counts().sort_index().to_string())

    print("\nFinal row counts by language:")
    print(dataframe["language"].value_counts().sort_index().to_string())

    print("\nFinal row counts by split/language/label:")
    print(dataframe.groupby(["split", "language", "label"]).size().to_string())


def write_outputs(dataframe: pd.DataFrame) -> None:
    """Save full, train, validation, and test enhanced CSVs."""
    ENHANCED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_frames = {
        "train": dataframe.loc[dataframe["split"] == "train"].copy(),
        "validation": dataframe.loc[dataframe["split"] == "validation"].copy(),
        "test": dataframe.loc[dataframe["split"] == "test"].copy(),
    }

    dataframe.to_csv(FULL_OUTPUT_PATH, index=False)
    split_frames["train"].to_csv(TRAIN_OUTPUT_PATH, index=False)
    split_frames["validation"].to_csv(VALIDATION_OUTPUT_PATH, index=False)
    split_frames["test"].to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nOutput paths written:")
    for path in [FULL_OUTPUT_PATH, TRAIN_OUTPUT_PATH, VALIDATION_OUTPUT_PATH, TEST_OUTPUT_PATH]:
        print(f"  - {path.resolve()}")


def main() -> int:
    """Build enhanced datasets with train/validation/test splits."""
    try:
        human_df = load_human_rows()
        machine_df = load_machine_rows(human_df)
        combined_df = pd.concat([human_df, machine_df], ignore_index=True)

        validate_experiment_dataframe(combined_df, "enhanced full dataset", ENHANCED_VALID_SPLITS)
        split_frames = {
            split: combined_df.loc[combined_df["split"] == split]
            for split in ["train", "validation", "test"]
        }
        validate_split_sets(split_frames)

        print_counts(combined_df)
        write_outputs(combined_df)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
