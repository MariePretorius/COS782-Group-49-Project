"""Build a Qwen-based test CSV in enhanced format."""
from __future__ import annotations
import pandas as pd
from pathlib import Path

HUMAN = Path("data/processed/human_samples.csv")
MACHINE_DIR = Path("data/machine_generated_qwen")
OUT = Path("data/processed/enhanced_qwen")
OUT.mkdir(parents=True, exist_ok=True)

human = pd.read_csv(HUMAN)
human = human.loc[human["split"] == "test"].copy()

rows = []
for lang in ["afr", "eng", "zul"]:
    h_lang = human.loc[human["language"] == lang].head(40).copy()
    h_lang["original_human_id"] = ""
    h_lang["prompt_type"] = ""
    rows.append(h_lang[["id", "text", "label", "label_name", "language",
                        "language_name", "source", "generator", "split", "prompt_type"]])
    m = pd.read_csv(MACHINE_DIR / f"machine_samples_{lang}_test.csv")
    m_out = pd.DataFrame({
        "id": "machine_" + m["original_human_id"].astype(str),
        "text": m["machine_text"],
        "label": 1,
        "label_name": "machine",
        "language": m["language"],
        "language_name": m["language_name"],
        "source": "generated",
        "generator": m["generator"],
        "split": m["split"],
        "prompt_type": m["prompt_type"],
    })
    rows.append(m_out)
combined = pd.concat(rows, ignore_index=True)
print(combined["language"].value_counts())
print(combined["label"].value_counts())
combined.to_csv(OUT / "test.csv", index=False)
print(f"wrote {OUT/'test.csv'} rows={len(combined)}")
