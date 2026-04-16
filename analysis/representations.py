"""Shared representation extraction utilities.

Load trained models and extract hidden states via generation or teacher forcing.
Designed for analysis scripts — simple functions, no classes, no registries.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_jepa.models import get_adapter, SPECIAL_TOKENS
from llm_jepa.evaluation.evaluate import format_conversation
from llm_jepa.evaluation.metrics import evaluate_sample


def load_model(checkpoint_path, base_model_name):
    """Load a trained model and tokenizer from checkpoint.

    Checkpoints have LoRA already merged, so this is just a straightforward
    load. We use the base model's tokenizer for the chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens to match training vocab size
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_examples(data_file, base_model_name, n=None):
    """Load dataset and format prompts for generation.

    Returns list of dicts: {prompt, reference, messages}
    """
    adapter = get_adapter(base_model_name)
    dataset = load_dataset("json", data_files=data_file)["train"]
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))

    # We need a tokenizer just for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    for row in dataset:
        messages = row["messages"]
        full_messages = adapter.get_messages(messages)
        prompt = format_conversation(full_messages, tokenizer)
        reference = messages[2]["content"]  # assistant response
        examples.append(
            {
                "prompt": prompt,
                "reference": reference,
                "messages": messages,
            }
        )
    return examples


def extract_generated(model, tokenizer, prompt, max_length=128, max_new_tokens=128):
    """Generate autoregressively and collect hidden states at every step.

    Returns dict:
        hidden_states: np.ndarray (layers, generated_tokens, hidden_dim)
            Only includes generated tokens (not prompt).
        generated_text: str
        prompt_len: int (number of prompt tokens)
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # Stitch hidden states from generate() output.
    #
    # outputs.hidden_states is a tuple of length (num_generated_tokens).
    # - hidden_states[0] is the prompt: tuple of (num_layers+1) tensors,
    #   each (1, prompt_len, hidden_dim)
    # - hidden_states[t] for t>0 is step t: tuple of (num_layers+1) tensors,
    #   each (1, 1, hidden_dim)
    #
    # We want: (num_layers, num_generated_tokens, hidden_dim)
    # Only generated tokens — skip the prompt step.

    num_layers = len(outputs.hidden_states[0])  # includes embedding layer
    num_steps = len(outputs.hidden_states) - 1  # exclude prompt step

    # Stitch prompt hidden states: (num_layers, prompt_len, hidden_dim)
    prompt_hs = np.stack([
        outputs.hidden_states[0][layer_idx].squeeze(0).float().cpu().numpy()
        for layer_idx in range(num_layers)
    ])

    if num_steps == 0:
        # Model generated nothing — return empty
        hidden_dim = outputs.hidden_states[0][0].shape[-1]
        return {
            "hidden_states": np.zeros((num_layers, 0, hidden_dim)),
            "prompt_hidden_states": prompt_hs,
            "generated_text": "",
            "prompt_len": prompt_len,
        }

    # Each step t (1-indexed) has one token per layer
    stitched = []
    for layer_idx in range(num_layers):
        layer_steps = []
        for step in range(1, len(outputs.hidden_states)):
            # (1, 1, hidden_dim) -> (hidden_dim,)
            h = outputs.hidden_states[step][layer_idx].squeeze(0).squeeze(0)
            layer_steps.append(h.float().cpu().numpy())
        stitched.append(np.stack(layer_steps))  # (num_generated, hidden_dim)

    hidden_states = np.stack(stitched)  # (layers, num_generated, hidden_dim)

    generated_tokens = outputs.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()

    return {
        "hidden_states": hidden_states,
        "prompt_hidden_states": prompt_hs,
        "generated_text": generated_text,
        "prompt_len": prompt_len,
    }


def extract_teacher_forced(model, tokenizer, prompt, reference, max_length=128):
    """Single forward pass on prompt + ground truth response.

    Returns dict:
        hidden_states: np.ndarray (layers, seq_len, hidden_dim)
            Full sequence including prompt and response tokens.
        prompt_len: int
        response_len: int
    """
    # Tokenize prompt and response separately to know the boundary
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = len(prompt_ids)

    # Full sequence: prompt + response
    full_text = prompt + reference
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden_dim)
    hidden_states = np.stack(
        [layer_h.squeeze(0).float().cpu().numpy() for layer_h in outputs.hidden_states]
    )  # (layers, seq_len, hidden_dim)

    return {
        "hidden_states": hidden_states,
        "prompt_len": prompt_len,
        "response_len": seq_len - prompt_len,
    }


def check_correctness(generated_text, messages, dataset_file):
    """Check if generated text is correct for this example."""
    dataset_name = dataset_file.split("/")[-1]
    return evaluate_sample(generated_text, messages, dataset_name)
