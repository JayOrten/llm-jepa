"""Main evaluation loop + generation.

Loads a fine-tuned model, generates responses on a test set,
and reports accuracy using dataset-specific metrics.
"""

import copy
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from llm_jepa.evaluation.metrics import evaluate_sample, translation_scores
from llm_jepa.models import get_adapter

logger = logging.getLogger("llm_jepa")


def load_model_and_tokenizer(model_name, original_model_name, load_in_8bit=False,
                              load_in_4bit=False, device_map="auto"):
    """Load model and tokenizer for evaluation with optional quantization."""
    adapter = get_adapter(original_model_name)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Tokenizer
    if hasattr(adapter, "load_tokenizer"):
        tokenizer = adapter.load_tokenizer(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Special tokens
    from llm_jepa.models import SPECIAL_TOKENS
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # Model
    attn_kwargs = {}
    attn_impl = adapter.get_attn_implementation()
    if attn_impl:
        attn_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        **attn_kwargs,
    )
    model.eval()

    return model, tokenizer


def format_conversation(messages, tokenizer, include_assistant=False, plain=False, similarity=False):
    """Format conversation for the model."""
    if not include_assistant:
        messages = [msg for msg in messages if msg["role"] != "assistant"]
    if plain:
        if similarity:
            return messages[0]["content"]
        return messages[1]["content"] + "<|perception|>"
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(model, tokenizer, prompt, max_length=512, max_new_tokens=128):
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if response.endswith("<|end|>"):
        response = response[:-7].strip()
    return response


def relative_probability(model, tokenizer, prompt, max_length=512):
    """HellaSwag: pick A/B/C/D by next-token probability."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    answers = ["A", "B", "C", "D"]
    token_ids = [tokenizer.convert_tokens_to_ids(a) for a in answers]
    probs = torch.softmax(logits, dim=-1)
    probs_tensor = torch.tensor([probs[tid].item() for tid in token_ids])
    return answers[torch.argmax(probs_tensor)]


def process_dataset(
    input_file, output_file, original_model_name, model, tokenizer,
    max_length=512, max_new_tokens=128, spider_path="",
    max_examples=None, split_tune_untune=False, plain=False,
    similarity=False, startswith=False, layer=-1, pooling="last",
):
    """Process dataset and report accuracy."""
    adapter = get_adapter(original_model_name)
    dataset_name = os.path.basename(input_file)

    if input_file.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=input_file)["train"]
    else:
        raise ValueError("Only JSONL files are supported")

    logger.info(f"Loaded {len(dataset)} examples from {input_file}")
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    correct = []
    incorrect = []
    all_generated = []
    all_references = []

    with open(output_file, "w") as f:
        for idx, example in enumerate(tqdm(dataset, desc="Generating")):
            messages = example["messages"]
            full_messages = adapter.get_messages(messages)
            prompt = format_conversation(full_messages, tokenizer, plain=plain)

            if dataset_name.startswith("hellaswag"):
                generated = relative_probability(model, tokenizer, prompt, max_length)
            else:
                generated = generate_response(model, tokenizer, prompt, max_length, max_new_tokens)

            is_correct = evaluate_sample(
                generated, messages, dataset_name, spider_path=spider_path,
            )

            all_generated.append(generated)
            all_references.append(messages[2]["content"])

            result = {
                "generated": generated,
                "reference": messages[2]["content"],
                "correct": is_correct,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if split_tune_untune:
                if is_correct:
                    correct.append(1)
                else:
                    incorrect.append(1)
            else:
                correct.append(1 if is_correct else 0)

    if split_tune_untune:
        total = len(correct) + len(incorrect)
        n_correct = len(correct)
    else:
        total = len(correct)
        n_correct = sum(correct)
    accuracy = n_correct / total if total > 0 else 0.0
    print(f"Success Rate: {input_file}, {accuracy:.4f}")
    print(f"Correct: {n_correct}, Incorrect: {total - n_correct}")

    # Corpus-level translation metrics
    summary = {
        "input_file": input_file,
        "total": total,
        "correct": n_correct,
        "accuracy": round(accuracy, 4),
    }
    if all_generated:
        scores = translation_scores(all_generated, all_references)
        print(f"BLEU: {scores['bleu']:.2f}, chrF: {scores['chrf']:.2f}")
        summary["bleu"] = round(scores["bleu"], 2)
        summary["chrf"] = round(scores["chrf"], 2)

    # Write summary next to the output JSONL
    summary_path = output_file.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return accuracy
