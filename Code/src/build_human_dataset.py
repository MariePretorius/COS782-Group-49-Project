"""Build a compact human-written sample dataset from NCHLT text corpora.

Run this script from inside the Project/Code directory:

    python src/build_human_dataset.py
"""

from __future__ import annotations

import csv
import random
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_PATH = OUTPUT_DIR / "human_samples.csv"

ZIP_LANGUAGE_CODES = {
    "af": "afr",
    "en": "eng",
    "zu": "zul",
}

LANGUAGE_NAMES = {
    "afr": "Afrikaans",
    "eng": "English",
    "zul": "isiZulu",
}

SAMPLE_SIZES = {
    "train": 500,
    "validation": 100,
    "test": 100,
}

OUTPUT_COLUMNS = [
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

ZIP_FILENAME_PATTERN = re.compile(r"^corpora\.nchlt\.(?P<language>[a-z]+)\.zip$")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply light cleaning while preserving case, punctuation, and spelling."""
    return WHITESPACE_PATTERN.sub(" ", text.strip().lstrip("\ufeff"))


def parse_zip_filename(path: Path) -> str:
    """Return the project language code detected from a corpus zip filename."""
    match = ZIP_FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    zip_language = match.group("language")
    if zip_language not in ZIP_LANGUAGE_CODES:
        raise ValueError(f"Unknown language code '{zip_language}' in filename: {path.name}")

    return ZIP_LANGUAGE_CODES[zip_language]


def is_metadata_line(text: str) -> bool:
    """Skip corpus headers and document markers, keeping actual corpus text only."""
    return (
        not text
        or text.startswith("_")
        or text.startswith("<fn>")
        or text.startswith("License:")
        or text.startswith("URL:")
        or text.startswith("Name and version:")
        or text.startswith("Attribute work to")
    )


def find_clean_corpus_member(archive: zipfile.ZipFile, zip_path: Path) -> str:
    """Find the cleaned corpus text file inside an NCHLT zip archive."""
    clean_members = [
        name
        for name in archive.namelist()
        if name.lower().endswith(".txt")
        and "clean" in name.lower()
        and ("corpus" in name.lower() or "corpora" in name.lower())
    ]

    if not clean_members:
        raise ValueError(f"No cleaned corpus text file found inside: {zip_path}")

    if len(clean_members) > 1:
        raise ValueError(
            f"Multiple cleaned corpus text files found inside {zip_path}: {clean_members}"
        )

    return clean_members[0]


def read_and_clean_zip(path: Path) -> tuple[list[str], int, str]:
    """Read the cleaned corpus member inside a zip file and return usable lines."""
    usable_lines = []
    raw_count = 0

    try:
        with zipfile.ZipFile(path) as archive:
            member_name = find_clean_corpus_member(archive, path)

            with archive.open(member_name) as member:
                for raw_line in member:
                    raw_count += 1
                    text = clean_text(raw_line.decode("utf-8", errors="replace"))

                    if is_metadata_line(text):
                        continue

                    word_count = len(text.split())
                    if 5 <= word_count <= 300:
                        usable_lines.append(text)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Could not read zip file {path}: {exc}") from exc

    return usable_lines, raw_count, member_name


def make_row(language: str, split: str, index: int, text: str) -> dict[str, object]:
    """Format one text sample for the final CSV schema."""
    return {
        "id": f"{language}_{split}_{index:06d}",
        "text": text,
        "label": 0,
        "label_name": "human",
        "language": language,
        "language_name": LANGUAGE_NAMES[language],
        "source": "NCHLT",
        "generator": "human",
        "split": split,
    }


def build_rows() -> list[dict[str, object]]:
    """Load, clean, sample, split, and format all NCHLT rows for the output CSV."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data folder: {DATA_DIR.resolve()}")

    files = sorted(DATA_DIR.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"No .zip files found in: {DATA_DIR.resolve()}")

    print("Found files:")
    for path in files:
        print(f"  - {path}")

    rng = random.Random(RANDOM_SEED)
    rows = []

    for path in files:
        language = parse_zip_filename(path)
        usable_lines, raw_count, member_name = read_and_clean_zip(path)
        rng.shuffle(usable_lines)

        cursor = 0
        print(f"\nFile: {path}")
        print(f"  Corpus member: {member_name}")
        print(f"  Raw lines: {raw_count}")
        print(f"  Usable lines after cleaning: {len(usable_lines)}")

        for split, requested_count in SAMPLE_SIZES.items():
            available_count = max(0, len(usable_lines) - cursor)
            sample_count = min(requested_count, available_count)

            if available_count < requested_count:
                print(
                    "WARNING: "
                    f"{path} has only {available_count} remaining usable rows for "
                    f"{split}; requested {requested_count}. Using all available rows."
                )

            sampled_lines = usable_lines[cursor : cursor + sample_count]
            cursor += sample_count

            print(f"  Sampled lines for {split}: {len(sampled_lines)}")

            for index, text in enumerate(sampled_lines, start=1):
                rows.append(make_row(language, split, index, text))

    return rows


def main() -> int:
    """Create data/processed/human_samples.csv."""
    try:
        rows = build_rows()

        if not rows:
            raise ValueError("Final dataset is empty; no rows were saved.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dataframe = pd.DataFrame(rows)
        dataframe.to_csv(
            OUTPUT_PATH,
            index=False,
            columns=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
        )

        print(f"\nFinal rows saved: {len(dataframe)}")
        print(f"Output path: {OUTPUT_PATH.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
