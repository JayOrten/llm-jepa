"""Download OPUS-100 translation pairs and convert to LLM-JEPA JSONL format.

Downloads 9 language pairs (en-centric, typologically diverse) from
Helsinki-NLP/opus-100 via HuggingFace datasets, shuffles, and writes
train/eval splits in the messages format expected by llm_jepa.

Language pairs: en<->fr, de, zh, ja, ar, hi, tr, fi, ko
Total: ~900k train, ~18k eval (varies by pair availability)

Usage:
    # Full dataset (~900k train examples)
    python scripts/prepare_opus.py

    # Smaller subset (10k per language pair = ~90k total)
    python scripts/prepare_opus.py --max_per_pair 10000

    # Custom output paths
    python scripts/prepare_opus.py --output datasets/opus_train.jsonl \\
                                   --eval_output datasets/opus_eval.jsonl

Requirements:
    pip install datasets  (included in llm-jepa dependencies)

The script streams from HuggingFace Hub — no manual download needed.
First run will cache the dataset under ~/.cache/huggingface/datasets/.
"""

import argparse
import json
import os
import random

from datasets import load_dataset

# Language pairs: (opus_config_name, source_code, target_code)
# OPUS-100 names its configs alphabetically, so Arabic-English is "ar-en" not "en-ar".
LANG_PAIRS = [
    ("en-fr", "en", "fr"),  # French – high resource, Romance
    ("de-en", "de", "en"),  # German – high resource, Germanic
    ("en-zh", "en", "zh"),  # Chinese – high resource, Sinitic
    ("en-ja", "en", "ja"),  # Japanese – high resource, Japonic
    ("ar-en", "ar", "en"),  # Arabic – medium resource, Semitic (note: reversed in OPUS)
    ("en-hi", "en", "hi"),  # Hindi – medium resource, Indo-Aryan
    ("en-tr", "en", "tr"),  # Turkish – medium resource, Turkic/agglutinative
    ("en-fi", "en", "fi"),  # Finnish – medium resource, Uralic/agglutinative
    ("en-ko", "en", "ko"),  # Korean – medium resource, Koreanic/SOV
]

# Human-readable language names for system prompts
LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "fi": "Finnish",
    "ko": "Korean",
}


def make_message(src_lang, tgt_lang, src_text, tgt_text):
    """Create a 3-message conversation for a translation pair."""
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"[{src_lang}] {src_text}"},
            {"role": "assistant", "content": f"[{tgt_lang}] {tgt_text}"},
        ]
    }


def load_pair(config_name, src_code, tgt_code, max_examples=None):
    """Load one language pair from OPUS-100 and convert to messages format."""
    print(f"Loading {config_name}...")
    ds = load_dataset("Helsinki-NLP/opus-100", config_name)

    train_examples = []
    eval_examples = []

    for split_name, output_list in [
        ("train", train_examples),
        ("validation", eval_examples),
    ]:
        if split_name not in ds:
            print(f"  Warning: no {split_name} split for {config_name}")
            continue
        split = ds[split_name]
        if max_examples and split_name == "train":
            split = split.select(range(min(max_examples, len(split))))

        for row in split:
            trans = row["translation"]
            src_text = trans[src_code].strip()
            tgt_text = trans[tgt_code].strip()
            if not src_text or not tgt_text:
                continue

            # Always translate TO the non-English language.
            # For ar-en (reversed in OPUS), English is tgt, Arabic is src — flip it.
            if src_code == "en":
                output_list.append(make_message("en", tgt_code, src_text, tgt_text))
            else:
                # e.g., ar-en: source is Arabic, target is English in OPUS
                # We want en->ar, so flip
                output_list.append(make_message("en", src_code, tgt_text, src_text))

    print(f"  {config_name}: {len(train_examples)} train, {len(eval_examples)} eval")
    return train_examples, eval_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/opus_train.jsonl")
    parser.add_argument("--eval_output", default="datasets/opus_eval.jsonl")
    parser.add_argument(
        "--max_per_pair",
        type=int,
        default=None,
        help="Max training examples per language pair (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_train = []
    all_eval = []

    for config_name, src_code, tgt_code in LANG_PAIRS:
        train, eval_ = load_pair(config_name, src_code, tgt_code, args.max_per_pair)
        all_train.extend(train)
        all_eval.extend(eval_)

    random.shuffle(all_train)
    random.shuffle(all_eval)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(args.eval_output, "w") as f:
        for ex in all_eval:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_train)} train, {len(all_eval)} eval")
    print(f"Written to {args.output} and {args.eval_output}")


if __name__ == "__main__":
    main()
