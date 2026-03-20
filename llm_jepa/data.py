"""Dataset loading, tokenization, and label masking.

Supports three data modes:
- regular: standard SFT (input_ids, labels, attention_mask)
- jepa: adds separately-tokenized user and assistant views for 3-view forward
- stp: adds user_start_end and assistant_start_end indices for span extraction
"""

import logging

import torch
from datasets import load_dataset

from llm_jepa.models import get_adapter

logger = logging.getLogger("llm_jepa")


def _find_start_end(content, tokenizer, input_ids, attention_mask, model_name=""):
    """Find the start and end index of content tokens within input_ids.

    Searches backwards (last occurrence) to handle repeated substrings.
    Returns (start - 1, end - 1) where the range [start, end) contains the content.
    """
    tokens = tokenizer.encode(content, add_special_tokens=False)
    # Handle OpenELM leading newline quirk
    if "apple/OpenELM" in model_name and len(tokens) >= 2:
        if tokens[0] == 29871 and tokens[1] == 13:
            tokens = tokens[2:]

    decoded_content = [tokenizer.decode(t) for t in tokens]
    decoded_input = [tokenizer.decode(t) for t in input_ids]

    for i in range(len(input_ids) - len(tokens), -1, -1):
        if attention_mask[i] == 1 and decoded_input[i:i + len(tokens)] == decoded_content:
            assert i > 0
            return i - 1, i + len(tokens) - 1

    return None


def load_and_prepare_dataset(
    data_file: str,
    tokenizer,
    model_name: str,
    max_length: int = 512,
    strategy: str = "regular",
    predictors: int = 0,
    train_all: bool = False,
    plain: bool = False,
    front_pred: bool = False,
    reverse_pred: bool = False,
    plain_jepa: bool = False,
    same_predictor: bool = False,
):
    """Load JSONL dataset and tokenize for training.

    Args:
        data_file: Path to JSONL file with 'messages' field.
        tokenizer: HuggingFace tokenizer.
        model_name: Model name (for adapter lookup).
        max_length: Max sequence length.
        strategy: "regular", "jepa", or "stp".
        predictors: Number of predictor tokens to append.
        train_all: If True, compute loss on all tokens (not just assistant).
        plain: Skip chat template formatting.
        front_pred: Place predictor tokens at front of user message.
        reverse_pred: Swap user/assistant for JEPA prediction direction.
        plain_jepa: Skip chat template for JEPA views only.
        same_predictor: Use the same predictor token ID for all positions.
    """
    adapter = get_adapter(model_name)
    dataset = load_dataset("json", data_files=data_file)["train"]

    if torch.cuda.is_available() and torch.cuda.current_device() == 0:
        logger.info(f"Loaded {len(dataset)} examples from {data_file}")

    # Determine if we need separate user/assistant tokenizations
    needs_views = strategy == "jepa"
    needs_spans = strategy == "stp"

    def tokenize_conversations(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        # JEPA views
        user_input_ids_list = []
        user_labels_list = []
        user_attention_mask_list = []
        assistant_input_ids_list = []
        assistant_labels_list = []
        assistant_attention_mask_list = []

        # STP spans
        user_start_end_list = []
        assistant_start_end_list = []

        for msg_idx, messages in enumerate(examples["messages"]):
            # --- Full conversation tokenization ---
            full_messages = adapter.get_messages(messages)
            if plain:
                if train_all:
                    formatted = messages[1]["content"] + "<|eot_id|>"
                else:
                    formatted = messages[1]["content"] + "<|perception|>" + messages[2]["content"] + "<|eot_id|>"
            else:
                formatted = tokenizer.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False,
                )

            tokenized = tokenizer(
                formatted, truncation=True, max_length=max_length,
                padding="max_length", return_tensors=None,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            if train_all:
                labels = [
                    input_ids[i] if attention_mask[i] == 1 else -100
                    for i in range(len(input_ids))
                ]
            else:
                labels = _create_masked_labels(messages, tokenizer, input_ids, attention_mask)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

            # --- JEPA: separate user and assistant views ---
            if needs_views:
                # User view
                if data_file.startswith("hellaswag"):
                    user_messages = examples["text"][msg_idx]
                else:
                    if reverse_pred:
                        user_messages = adapter.get_assistant_messages(messages)
                    else:
                        user_messages = adapter.get_user_messages(messages)

                # Add predictor tokens
                to_add = predictors
                while to_add > 0:
                    pred_id = 1 if same_predictor else to_add
                    token = f"<|predictor_{pred_id}|>"
                    if front_pred:
                        user_messages[0]["content"] = token + user_messages[0]["content"]
                    else:
                        user_messages[0]["content"] += token
                    to_add -= 1

                if plain or plain_jepa:
                    formatted_user = user_messages[0]["content"]
                else:
                    formatted_user = tokenizer.apply_chat_template(
                        user_messages, tokenize=False, add_generation_prompt=False,
                    )
                tok_user = tokenizer(
                    formatted_user, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors=None,
                )
                user_input_ids_list.append(tok_user["input_ids"])
                user_labels_list.append([-100] * len(tok_user["input_ids"]))
                user_attention_mask_list.append(tok_user["attention_mask"])

                # Assistant view
                if data_file.startswith("hellaswag"):
                    assistant_messages = examples["code"][msg_idx]
                else:
                    if reverse_pred:
                        assistant_messages = adapter.get_user_messages(messages)
                    else:
                        assistant_messages = adapter.get_assistant_messages(messages)

                if plain or plain_jepa:
                    formatted_asst = assistant_messages[0]["content"]
                else:
                    formatted_asst = tokenizer.apply_chat_template(
                        assistant_messages, tokenize=False, add_generation_prompt=False,
                    )
                tok_asst = tokenizer(
                    formatted_asst, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors=None,
                )
                assistant_input_ids_list.append(tok_asst["input_ids"])
                assistant_labels_list.append([-100] * len(tok_asst["input_ids"]))
                assistant_attention_mask_list.append(tok_asst["attention_mask"])

            # --- STP: find span boundaries ---
            if needs_spans:
                # User span
                if data_file.startswith("hellaswag"):
                    content = examples["text"][msg_idx][0]["content"] + "\n"
                elif "allenai/OLMo" in model_name:
                    content = messages[1]["content"] + "\n"
                else:
                    content = messages[1]["content"]
                user_span = _find_start_end(
                    content, tokenizer, input_ids, attention_mask, model_name,
                )

                # Assistant span
                if data_file.startswith("hellaswag"):
                    asst_content = " " + examples["code"][msg_idx][0]["content"] + "\n"
                else:
                    asst_content = messages[2]["content"]
                asst_span = _find_start_end(
                    asst_content, tokenizer, input_ids, attention_mask, model_name,
                )

                if user_span is None or asst_span is None:
                    # Content was truncated — drop this example
                    input_ids_list.pop()
                    labels_list.pop()
                    attention_mask_list.pop()
                    continue

                user_start_end_list.append(list(user_span))
                assistant_start_end_list.append(list(asst_span))

        result = {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }
        if needs_views:
            result.update({
                "input_ids_user": user_input_ids_list,
                "labels_user": user_labels_list,
                "attention_mask_user": user_attention_mask_list,
                "input_ids_assistant": assistant_input_ids_list,
                "labels_assistant": assistant_labels_list,
                "attention_mask_assistant": assistant_attention_mask_list,
            })
        if needs_spans:
            result.update({
                "user_start_end": user_start_end_list,
                "assistant_start_end": assistant_start_end_list,
            })
        return result

    original_len = len(dataset)
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
        remove_columns=dataset.column_names,
    )
    if needs_spans and len(tokenized_dataset) < original_len:
        dropped = original_len - len(tokenized_dataset)
        logger.warning(
            f"Dropped {dropped}/{original_len} examples "
            f"({dropped/original_len:.1%}) due to truncation at max_length={max_length}"
        )
    return tokenized_dataset


def _create_masked_labels(messages, tokenizer, input_ids, attention_mask):
    """Create labels with input tokens masked (-100), only assistant tokens unmasked."""
    labels = [-100] * len(input_ids)

    for msg in messages:
        if msg["role"] == "assistant":
            assistant_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
            decoded_assistant = [tokenizer.decode(t) for t in assistant_tokens]
            decoded_input = [tokenizer.decode(t) for t in input_ids]

            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                if attention_mask[i] == 1 and decoded_input[i:i + len(assistant_tokens)] == decoded_assistant:
                    for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                        if attention_mask[j] == 1:
                            labels[j] = input_ids[j]
                    break

    return labels
