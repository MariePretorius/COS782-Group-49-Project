"""Split human samples into prompt batches for machine-text generation.

Run this script from inside the Project/Code directory:

    python src/split_human_samples_for_generation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/human_samples.csv")
OUTPUT_DIR = Path("data/generated/input_batches")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
README_PATH = OUTPUT_DIR / "README.md"

REQUIRED_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "id",
    "language",
    "language_name",
    "split",
    "original_text",
    "prompt_type",
    "prompt",
]

LANGUAGES = ["eng", "afr", "zul"]
SPLITS = ["train", "validation", "test"]
LANGUAGE_NAMES = {
    "eng": "English",
    "afr": "Afrikaans",
    "zul": "isiZulu",
}

PROMPT_TEMPLATE = """Rewrite the following text in {language_name}. Preserve the meaning and topic, but write it naturally in your own words. Do not mention that you are an AI. Output only the rewritten text.

Text:
{original_text}"""

README_TEXT = """# Human Sample Input Batches

These files are grouped human samples for machine-text generation.

Each row contains a prompt based on one human-written sample.

The LLM should generate one paraphrased machine-written text for each row.

The generated output should preserve the original id, language, language_name and split.

The generated output should later be converted into machine_samples.csv.
"""


def load_input_csv() -> pd.DataFrame:
    """Load the source human samples CSV with a helpful error if it is missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH.resolve()}. "
            "Run src/build_human_dataset.py first or place human_samples.csv there."
        )

    return pd.read_csv(INPUT_PATH)


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the input CSV contains exactly the columns this task depends on."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{', '.join(missing_columns)}"
        )


def validate_values(dataframe: pd.DataFrame) -> None:
    """Validate labels, languages, splits, and text values before writing batches."""
    invalid_labels = dataframe.loc[dataframe["label"] != 0, "label"].dropna().unique()
    if len(invalid_labels) > 0:
        raise ValueError(f"Expected all labels to be 0, found: {invalid_labels}")

    invalid_label_names = (
        dataframe.loc[dataframe["label_name"] != "human", "label_name"]
        .dropna()
        .unique()
    )
    if len(invalid_label_names) > 0:
        raise ValueError(
            "Expected all label_name values to be 'human', "
            f"found: {invalid_label_names}"
        )

    invalid_languages = sorted(set(dataframe["language"].dropna()) - set(LANGUAGES))
    if invalid_languages:
        raise ValueError(
            "Expected language values to be only eng, afr, zul; "
            f"found: {invalid_languages}"
        )

    invalid_splits = sorted(set(dataframe["split"].dropna()) - set(SPLITS))
    if invalid_splits:
        raise ValueError(
            "Expected split values to be only train, validation, test; "
            f"found: {invalid_splits}"
        )

    empty_text_count = dataframe["text"].isna().sum()
    empty_text_count += dataframe["text"].astype(str).str.strip().eq("").sum()
    if empty_text_count > 0:
        raise ValueError(f"Found {empty_text_count} rows with empty text values.")


def print_counts(dataframe: pd.DataFrame) -> None:
    """Print dataset counts requested for quick inspection."""
    print(f"Total rows loaded: {len(dataframe)}")

    print("\nRows per language:")
    print(dataframe["language"].value_counts().sort_index().to_string())

    print("\nRows per split:")
    print(dataframe["split"].value_counts().sort_index().to_string())

    print("\nRows per language/split:")
    counts = dataframe.groupby(["language", "split"]).size()
    print(counts.to_string())


def make_prompt(language_name: str, original_text: str) -> str:
    """Create the paraphrase prompt for one human-written sample."""
    return PROMPT_TEMPLATE.format(
        language_name=language_name,
        original_text=original_text,
    )


def build_batch(dataframe: pd.DataFrame, language: str, split: str) -> pd.DataFrame:
    """Create one output batch for a language and split without shuffling rows."""
    batch = dataframe.loc[
        (dataframe["language"] == language) & (dataframe["split"] == split)
    ].copy()

    batch["original_text"] = batch["text"]
    batch["prompt_type"] = "paraphrase"
    batch["prompt"] = batch.apply(
        lambda row: make_prompt(row["language_name"], row["original_text"]),
        axis=1,
    )

    return batch[OUTPUT_COLUMNS]


def write_batches(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Write the nine grouped CSV files and return manifest rows."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for language in LANGUAGES:
        for split in SPLITS:
            batch = build_batch(dataframe, language, split)
            output_path = OUTPUT_DIR / f"human_samples_{language}_{split}.csv"
            batch.to_csv(output_path, index=False)

            print(f"\nWrote: {output_path}")
            print(f"  Rows written: {len(batch)}")

            manifest_rows.append(
                {
                    "file_path": str(output_path),
                    "language": language,
                    "language_name": LANGUAGE_NAMES[language],
                    "split": split,
                    "row_count": len(batch),
                }
            )

    return manifest_rows


def write_manifest(manifest_rows: list[dict[str, object]]) -> None:
    """Write the manifest that records every generated batch file."""
    manifest = pd.DataFrame(
        manifest_rows,
        columns=["file_path", "language", "language_name", "split", "row_count"],
    )
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"  Rows written: {len(manifest)}")


def write_readme() -> None:
    """Write a short README explaining how to use the generated input batches."""
    README_PATH.write_text(README_TEXT, encoding="utf-8")
    print(f"\nWrote README: {README_PATH}")


def main() -> int:
    """Validate human_samples.csv and split it into generation input batches."""
    try:
        dataframe = load_input_csv()
        validate_required_columns(dataframe)
        validate_values(dataframe)
        print_counts(dataframe)

        manifest_rows = write_batches(dataframe)
        write_manifest(manifest_rows)
        write_readme()

        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
