"""Build a balanced non-paraphrased test set for existing models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils import RANDOM_SEED, ensure_columns, validate_experiment_dataframe


HUMAN_PATH = Path("data/processed/human_samples.csv")
MACHINE_DIR = Path("data/non_paraphrased")
OUTPUT_DIR = Path("data/processed/non_paraphrased")
OUTPUT_PATH = OUTPUT_DIR / "test.csv"

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


def load_human_rows(machine_counts: pd.Series) -> pd.DataFrame:
    """Load a balanced number of human test rows per language."""
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(f"Missing human samples: {HUMAN_PATH.resolve()}")

    human_df = pd.read_csv(HUMAN_PATH)
    ensure_columns(human_df, HUMAN_REQUIRED_COLUMNS, str(HUMAN_PATH))
    human_df = human_df.loc[human_df["split"] == "test"].copy()
    human_df["original_human_id"] = ""
    human_df["prompt_type"] = ""

    sampled_frames = []
    for language, count in machine_counts.items():
        language_df = human_df.loc[human_df["language"] == language]
        if len(language_df) < count:
            raise ValueError(
                f"Not enough human test rows for {language}: "
                f"need {count}, found {len(language_df)}"
            )
        sampled_frames.append(
            language_df.sample(n=int(count), random_state=RANDOM_SEED)
        )

    return pd.concat(sampled_frames, ignore_index=True)[FINAL_COLUMNS]


def load_machine_rows() -> pd.DataFrame:
    """Load non-paraphrased machine rows and convert to final schema."""
    if not MACHINE_DIR.exists():
        raise FileNotFoundError(f"Missing non_paraphrased folder: {MACHINE_DIR.resolve()}")

    files = sorted(MACHINE_DIR.glob("machine_samples_*_test_unpaired.csv"))
    files = [path for path in files if "_all_" not in path.name]
    if not files:
        raise FileNotFoundError(
            f"No non-paraphrased machine files found in {MACHINE_DIR.resolve()}"
        )

    frames = []
    print("Non-paraphrased machine files found:")
    for path in files:
        dataframe = pd.read_csv(path)
        ensure_columns(dataframe, MACHINE_REQUIRED_COLUMNS, str(path))
        if "prompt_type" not in dataframe.columns:
            dataframe["prompt_type"] = "topic_controlled_unpaired"

        print(f"  - {path} ({len(dataframe)} rows)")
        frames.append(
            pd.DataFrame(
                {
                    "id": "machine_" + dataframe["original_human_id"].astype(str),
                    "original_human_id": dataframe["original_human_id"].astype(str),
                    "text": dataframe["machine_text"],
                    "label": 1,
                    "label_name": "machine",
                    "language": dataframe["language"],
                    "language_name": dataframe["language_name"],
                    "source": "generated",
                    "generator": dataframe["generator"],
                    "split": dataframe["split"],
                    "prompt_type": dataframe["prompt_type"].fillna(
                        "topic_controlled_unpaired"
                    ),
                }
            )
        )

    machine_df = pd.concat(frames, ignore_index=True)
    return machine_df[FINAL_COLUMNS]


def main() -> int:
    """Create data/processed/non_paraphrased/test.csv."""
    try:
        machine_df = load_machine_rows()
        machine_counts = machine_df.groupby("language").size()
        human_df = load_human_rows(machine_counts)
        combined = pd.concat([human_df, machine_df], ignore_index=True)

        validate_experiment_dataframe(combined, "non-paraphrased test set", {"test"})
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False)

        print("\nFinal row counts by language/label:")
        print(combined.groupby(["language", "label"]).size().to_string())
        print(f"\nWrote: {OUTPUT_PATH.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
