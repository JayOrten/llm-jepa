"""Unified evaluation entry point.

Usage:
    python eval.py --config configs/experiments/replicate_stp_synth.toml \
        --set evaluation.model_name=./ft-stp-synth \
        --set data.eval_file=datasets/synth_test.jsonl
"""

import os

import torch

from llm_jepa.config import load_settings
from llm_jepa.evaluation.evaluate import load_model_and_tokenizer, process_dataset
from llm_jepa.utils import setup_logging


def main():
    settings = load_settings()
    setup_logging(settings.get("debug.level", "INFO"))

    eval_cfg = settings.evaluation
    model_name = settings.get("evaluation.model_name", settings.model.name)
    original = eval_cfg.original_model_name or settings.model.name
    input_file = settings.get("data.eval_file", "") or settings.get("data.train_file", "")

    # Default output into the model directory
    output_file = settings.get("evaluation.output_file", "eval_output.jsonl")
    if output_file == "eval_output.jsonl" and os.path.isdir(model_name):
        output_file = os.path.join(model_name, output_file)

    if not input_file:
        raise ValueError("Must provide data.eval_file for evaluation")

    print(f"=== LLM-JEPA Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Input: {input_file}")

    model, tokenizer = load_model_and_tokenizer(
        model_name, original,
        load_in_8bit=eval_cfg.load_in_8bit,
        load_in_4bit=eval_cfg.load_in_4bit,
        device_map=eval_cfg.device_map,
    )

    max_eval = settings.get("data.max_eval_samples", -1)

    process_dataset(
        input_file=input_file,
        output_file=output_file,
        original_model_name=original,
        model=model,
        tokenizer=tokenizer,
        max_length=eval_cfg.max_length,
        max_new_tokens=eval_cfg.max_new_tokens,
        spider_path=eval_cfg.spider_path,
        split_tune_untune=eval_cfg.split_tune_untune,
        max_examples=max_eval if max_eval > 0 else None,
        plain=settings.get("data.plain", False),
    )


if __name__ == "__main__":
    main()
