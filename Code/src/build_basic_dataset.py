"""Build the train/test dataset for the first COS760 basic experiment."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from utils import (
    LANGUAGE_NAMES,
    VALID_LANGUAGES,
    VALID_SPLITS,
    ensure_columns,
    validate_basic_dataframe,
    validate_train_test_sets,
)


HUMAN_PATH = Path("data/processed/human_samples.csv")
PRIMARY_MACHINE_DIR = Path("data/machine_generated")
LEGACY_MACHINE_DIR = Path("data/machine-generated")
BASIC_OUTPUT_DIR = Path("data/processed/basic")

FULL_OUTPUT_PATH = BASIC_OUTPUT_DIR / "full_basic_dataset.csv"
TRAIN_OUTPUT_PATH = BASIC_OUTPUT_DIR / "train.csv"
TEST_OUTPUT_PATH = BASIC_OUTPUT_DIR / "test.csv"

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
    r"^machine_samp(?:el|le)s_(?P<language>afr|eng|zul)_(?P<split>train|test)\.csv$"
)


def resolve_machine_dir() -> Path:
    """Prefer the correct underscore folder, with a legacy hyphen fallback."""
    if PRIMARY_MACHINE_DIR.exists():
        return PRIMARY_MACHINE_DIR
    if LEGACY_MACHINE_DIR.exists():
        print(
            f"WARNING: {PRIMARY_MACHINE_DIR} not found; using legacy folder "
            f"{LEGACY_MACHINE_DIR}."
        )
        return LEGACY_MACHINE_DIR
    raise FileNotFoundError(
        f"Missing machine-generated folder: {PRIMARY_MACHINE_DIR.resolve()}"
    )


def find_machine_files(machine_dir: Path) -> list[Path]:
    """Find generated machine files, supporting both sampels and samples spelling."""
    files = sorted(machine_dir.glob("machine_sampels_*_*.csv"))
    corrected_spelling_files = sorted(machine_dir.glob("machine_samples_*_*.csv"))

    for path in corrected_spelling_files:
        if path not in files:
            files.append(path)

    if not files:
        raise FileNotFoundError(
            "No generated machine CSV files found. Expected files matching "
            f"{machine_dir}/machine_sampels_*_*.csv"
        )

    return sorted(files)


def load_human_rows() -> pd.DataFrame:
    """Load human samples and keep only train/test rows."""
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(f"Missing human_samples.csv: {HUMAN_PATH.resolve()}")

    human_df = pd.read_csv(HUMAN_PATH)
    ensure_columns(human_df, HUMAN_REQUIRED_COLUMNS, str(HUMAN_PATH))
    print(f"Human rows loaded: {len(human_df)}")

    human_df = human_df.loc[human_df["split"].isin(VALID_SPLITS)].copy()
    print(f"Human rows kept for train/test: {len(human_df)}")

    human_df["original_human_id"] = ""
    human_df["prompt_type"] = ""
    return human_df[FINAL_COLUMNS]


def parse_machine_filename(path: Path) -> tuple[str | None, str | None]:
    """Parse language/split from the filename for diagnostics and fallback checks."""
    match = MACHINE_FILENAME_PATTERN.match(path.name)
    if not match:
        return None, None
    return match.group("language"), match.group("split")


def convert_machine_rows(path: Path) -> pd.DataFrame:
    """Load one generated CSV and convert it to the final experiment schema."""
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

    converted = pd.DataFrame(
        {
            "id": "machine_" + machine_df["original_human_id"].astype(str),
            "original_human_id": machine_df["original_human_id"],
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

    return converted


def load_machine_rows() -> pd.DataFrame:
    """Load every generated machine CSV and keep only train/test rows."""
    machine_dir = resolve_machine_dir()
    machine_files = find_machine_files(machine_dir)

    print("Machine files found:")
    converted_frames = []
    loaded_rows = 0

    for path in machine_files:
        converted = convert_machine_rows(path)
        loaded_rows += len(converted)
        print(f"  - {path} ({len(converted)} rows loaded)")
        converted_frames.append(converted)

    machine_df = pd.concat(converted_frames, ignore_index=True)
    machine_df = machine_df.loc[machine_df["split"].isin(VALID_SPLITS)].copy()
    print(f"Machine rows loaded: {loaded_rows}")
    print(f"Machine rows kept for train/test: {len(machine_df)}")

    return machine_df[FINAL_COLUMNS]


def validate_combined_dataset(dataframe: pd.DataFrame) -> None:
    """Validate final combined rows before writing train/test CSVs."""
    validate_basic_dataframe(dataframe, "full basic dataset")

    unexpected_languages = sorted(set(dataframe["language"].unique()) - VALID_LANGUAGES)
    if unexpected_languages:
        raise ValueError(f"Unexpected language values: {unexpected_languages}")

    unexpected_splits = sorted(set(dataframe["split"].unique()) - VALID_SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected split values: {unexpected_splits}")

    train_df = dataframe.loc[dataframe["split"] == "train"]
    test_df = dataframe.loc[dataframe["split"] == "test"]
    validate_train_test_sets(train_df, test_df)


def print_final_counts(dataframe: pd.DataFrame) -> None:
    """Print summary counts for the combined experiment dataset."""
    print("\nFinal row counts by split:")
    print(dataframe["split"].value_counts().sort_index().to_string())

    print("\nFinal row counts by label:")
    print(dataframe["label"].value_counts().sort_index().to_string())

    print("\nFinal row counts by language:")
    print(dataframe["language"].value_counts().sort_index().to_string())


def write_outputs(dataframe: pd.DataFrame) -> None:
    """Save the full, train, and test datasets."""
    BASIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = dataframe.loc[dataframe["split"] == "train"].copy()
    test_df = dataframe.loc[dataframe["split"] == "test"].copy()

    dataframe.to_csv(FULL_OUTPUT_PATH, index=False)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nOutput paths written:")
    print(f"  - {FULL_OUTPUT_PATH.resolve()}")
    print(f"  - {TRAIN_OUTPUT_PATH.resolve()}")
    print(f"  - {TEST_OUTPUT_PATH.resolve()}")


def main() -> int:
    """Build data/processed/basic train and test CSVs."""
    try:
        human_df = load_human_rows()
        machine_df = load_machine_rows()
        combined_df = pd.concat([human_df, machine_df], ignore_index=True)

        validate_combined_dataset(combined_df)
        print_final_counts(combined_df)
        write_outputs(combined_df)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
