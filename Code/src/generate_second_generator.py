"""Generate a second-generator machine set."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

HUMAN_PATH = Path("data/processed/human_samples.csv")
LANGUAGE_NAMES = {"afr": "Afrikaans", "eng": "English", "zul": "isiZulu"}

PARAPHRASE_INSTRUCTION = {
    "afr": "Herskryf die volgende teks in Afrikaans. Behou die betekenis maar verander die bewoording. Gee slegs die herskryfde teks terug:",
    "eng": "Rewrite the following text in English. Preserve the meaning but change the wording. Return only the rewritten text:",
    "zul": "Bhala kabusha lo mbhalo ngesiZulu. Gcina incazelo kodwa ushintshe amagama. Buyisela umbhalo obhalwe kabusha kuphela:",
}
TOPIC_INSTRUCTION = {
    "afr": "Skryf 'n kort, formele paragraaf in Afrikaans oor dieselfde onderwerp as die volgende teks, sonder om dit te herskryf. Gee slegs die nuwe paragraaf terug:",
    "eng": "Write a short, formal paragraph in English about the same topic as the following text, without rewriting it. Return only the new paragraph:",
    "zul": "Bhala isigaba esifushane, esisemthethweni ngesiZulu ngesihloko esifanayo nalo mbhalo, ungawubhali kabusha. Buyisela isigaba esisha kuphela:",
}


def select_rows(human_df: pd.DataFrame, language: str, split: str, per_language: int) -> pd.DataFrame:
    subset = human_df.loc[(human_df["language"] == language) & (human_df["split"] == split)].copy()
    if per_language > 0:
        subset = subset.head(per_language)
    return subset


def build_hf_generator(model_name: str, load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = dict(
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    has_chat = getattr(tokenizer, "chat_template", None) is not None

    def generate(instruction: str, source_text: str) -> str:
        prompt_text = f"{instruction}\n\n{source_text}"
        if has_chat:
            messages = [{"role": "user", "content": prompt_text}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(inputs, max_new_tokens=256, do_sample=True,
                                temperature=0.8, top_p=0.95,
                                pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return generated.strip()

    return generate


def build_openai_generator(model_name: str):
    from openai import OpenAI

    client = OpenAI()

    def generate(instruction: str, source_text: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"{instruction}\n\n{source_text}"}],
            temperature=0.8)
        return (response.choices[0].message.content or "").strip()

    return generate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a second-generator machine set.")
    parser.add_argument("--backend", required=True, choices=["hf", "openai"])
    parser.add_argument("--model", required=True, help="HF model id/path or OpenAI model name.")
    parser.add_argument("--generator-name", required=True, help="Value written to the 'generator' column.")
    parser.add_argument("--mode", default="topic", choices=["paraphrase", "topic"],
                        help="paraphrase = rewrite the human text, topic = new text on the same topic.")
    parser.add_argument("--per-language", type=int, default=40,
                        help="Rows per language per split (0 = all human rows).")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--out-dir", required=True, help="Folder for machine_samples_*.csv files.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls (API rate limits).")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantize an HF model to fit 16GB VRAM.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not HUMAN_PATH.exists():
        print(f"ERROR: Missing {HUMAN_PATH.resolve()}. Run build_human_dataset.py first.", file=sys.stderr)
        return 1
    human_df = pd.read_csv(HUMAN_PATH)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        generate = (build_hf_generator(args.model, load_in_4bit=args.load_in_4bit)
                if args.backend == "hf" else build_openai_generator(args.model))
    except ImportError as exc:
        print(f"ERROR: backend deps missing: {exc}", file=sys.stderr)
        return 1

    instructions = PARAPHRASE_INSTRUCTION if args.mode == "paraphrase" else TOPIC_INSTRUCTION
    prompt_type = "paraphrase" if args.mode == "paraphrase" else "non_paraphrase"

    for language in sorted(LANGUAGE_NAMES):
        for split in args.splits:
            rows = select_rows(human_df, language, split, args.per_language)
            if rows.empty:
                print(f"WARNING: no human rows for {language}/{split}; skipping.")
                continue
            records = []
            print(f"Generating {len(rows)} {args.mode} samples for {language}/{split} with {args.generator_name}...")
            for counter, (_, human_row) in enumerate(rows.iterrows(), start=1):
                try:
                    machine_text = generate(instructions[language], str(human_row["text"]))
                except Exception as exc:
                    print(f"  WARN: generation failed for {human_row['id']}: {exc}")
                    machine_text = ""
                if not machine_text.strip():
                    continue
                records.append({
                    "original_human_id": human_row["id"],
                    "machine_text": machine_text.replace("\n", " ").strip(),
                    "language": language, "language_name": LANGUAGE_NAMES[language],
                    "split": split, "prompt_type": prompt_type, "generator": args.generator_name})
                if args.sleep:
                    time.sleep(args.sleep)
                if counter % 10 == 0:
                    print(f"  {counter}/{len(rows)}")
            out_path = out_dir / f"machine_samples_{language}_{split}.csv"
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"  wrote {len(records)} rows -> {out_path.resolve()}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
