"""Model and tokenizer setup with adapter registry.

Model-specific quirks (tokenizer source, chat templates, attention impl)
are handled by adapter objects. Any HF model with a chat_template works
out of the box via DefaultAdapter. Known models get explicit adapters.
"""

import copy
import logging
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("llm_jepa")

SPECIAL_TOKENS = [
    "<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>",
    "<|predictor_4|>", "<|predictor_5|>", "<|predictor_6|>",
    "<|predictor_7|>", "<|predictor_8|>", "<|predictor_9|>",
    "<|predictor_10|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "<|perception|>",
]

OPENELM_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Handles model-specific tokenizer setup and message formatting."""

    def setup_tokenizer(self, tokenizer, model_name: str):
        """Apply any model-specific tokenizer modifications."""
        pass

    def get_messages(self, messages: list[dict]) -> list[dict]:
        """Format full conversation messages for this model."""
        return messages

    def get_user_messages(self, messages: list[dict]) -> list[dict]:
        """Extract user-only messages."""
        return copy.deepcopy(messages)[1:2]

    def get_assistant_messages(self, messages: list[dict]) -> list[dict]:
        """Extract assistant-only messages."""
        return messages[2:3]

    def get_attn_implementation(self) -> str | None:
        """Return attention implementation override, or None for default."""
        return None

    def find_content_adjustment(self, content: str, messages: list[dict]) -> str:
        """Adjust content string for find_start_end. Default: no adjustment."""
        return content

    @abstractmethod
    def name(self) -> str:
        ...


class DefaultAdapter(ModelAdapter):
    """Works with any HF model that has a chat_template."""

    def name(self) -> str:
        return "default"


class GemmaAdapter(ModelAdapter):
    """Gemma merges system+user and swaps assistant role to 'user'."""

    def name(self) -> str:
        return "gemma"

    def get_messages(self, messages):
        full = copy.deepcopy(messages)[1:3]
        full[0]["content"] = messages[0]["content"] + "\n\n" + full[0]["content"]
        return full

    def get_assistant_messages(self, messages):
        assistant = copy.deepcopy(messages)[2:3]
        assistant[0]["role"] = "user"
        return assistant

    def get_attn_implementation(self):
        return "eager"


class OpenELMAdapter(ModelAdapter):
    """OpenELM uses Llama-2 tokenizer with a custom chat template."""

    def name(self) -> str:
        return "openelm"

    def setup_tokenizer(self, tokenizer, model_name):
        # OpenELM doesn't ship its own tokenizer
        pass

    def load_tokenizer(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.chat_template = OPENELM_CHAT_TEMPLATE
        return tokenizer

    def get_attn_implementation(self):
        return None


class PhiAdapter(ModelAdapter):
    """Phi models need a BOS token added."""

    def name(self) -> str:
        return "phi"

    def setup_tokenizer(self, tokenizer, model_name):
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        if torch.cuda.is_available() and torch.cuda.current_device() == 0:
            logger.info("Added <|startoftext|> BOS token for Phi")


class OLMoAdapter(ModelAdapter):
    """OLMo needs newline adjustment for span finding."""

    def name(self) -> str:
        return "olmo"

    def find_content_adjustment(self, content, messages):
        return content + "\n"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, ModelAdapter] = {
    "google/gemma": GemmaAdapter(),
    "apple/OpenELM": OpenELMAdapter(),
    "microsoft/phi": PhiAdapter(),
    "allenai/OLMo": OLMoAdapter(),
}


def get_adapter(model_name: str) -> ModelAdapter:
    """Find the adapter for a model name, or return DefaultAdapter."""
    for prefix, adapter in ADAPTER_REGISTRY.items():
        if prefix in model_name:
            return adapter
    return DefaultAdapter()


# ---------------------------------------------------------------------------
# LinearPredictor (used by STP with --linear_predictor)
# ---------------------------------------------------------------------------

class LinearPredictor(nn.Module):
    def __init__(self, dx: int, dy: int | None = None, bias: bool = False):
        super().__init__()
        if dy is None:
            dy = dx
        self.M = nn.Linear(dx, dy, bias=bias)
        nn.init.xavier_uniform_(self.M.weight, gain=1.0)
        if bias:
            nn.init.zeros_(self.M.bias)
        self.dx = dx
        self.dy = dy

    def forward(self, x):
        assert x.dim() == 2 and x.shape[1] == self.dx
        return self.M(x)


# ---------------------------------------------------------------------------
# Model + tokenizer setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_rank: int = 16,
    pretrain: bool = False,
    seed: int | None = None,
    linear_predictor: bool = False,
    load_lp: bool = False,
    add_mask_token: bool = False,
    tokenizer_name: str | None = None,
):
    """Load model and tokenizer, apply adapter quirks, optional LoRA.

    Args:
        model_name: HuggingFace model name or local path to config directory.
        use_lora: Whether to apply LoRA.
        lora_rank: LoRA rank (alpha = rank * 2).
        pretrain: Initialize from config (random weights) instead of pretrained.
        seed: Random seed for weight init when pretrain=True.
        linear_predictor: Attach a LinearPredictor module to the model.
        load_lp: Load linear predictor weights from safetensors.
        add_mask_token: Add a <|mask|> special token (needed for STP random_span_mask).
        tokenizer_name: HuggingFace tokenizer name (if different from model_name).
    """
    adapter = get_adapter(model_name)

    # --- Tokenizer ---
    tokenizer_source = tokenizer_name or model_name
    if isinstance(adapter, OpenELMAdapter) and tokenizer_name is None:
        tokenizer = adapter.load_tokenizer(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.chat_template is None:
            logger.warning(f"{tokenizer_source} has no chat_template. Generation may not work correctly.")

    adapter.setup_tokenizer(tokenizer, model_name)

    # Add special tokens
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if torch.cuda.is_available() and torch.cuda.current_device() == 0:
            logger.info(f"Added {len(new_tokens)} special tokens")

    if add_mask_token and tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if torch.cuda.is_available() and torch.cuda.current_device() == 0:
            logger.info(f"Added <|mask|> token: {tokenizer.mask_token_id}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Device map ---
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            device_map = "auto"

    # --- Model loading ---
    attn_impl = adapter.get_attn_implementation()
    extra_kwargs = {}
    if attn_impl is not None:
        extra_kwargs["attn_implementation"] = attn_impl

    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            device = torch.device(f"cuda:{rank}")
            model.to(device)
            for p in model.parameters():
                torch.distributed.broadcast(p.data, src=0)
            for b in model.buffers():
                torch.distributed.broadcast(b.data, src=0)
    elif load_lp:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        d = getattr(model.config, "hidden_size", None) or model.config.model_dim
        model.linear_predictor = LinearPredictor(d)
        state_dict = load_file(os.path.join(model_name, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
            **extra_kwargs,
        )

    # Resize embeddings for new tokens
    if new_tokens or (add_mask_token and tokenizer.mask_token is not None):
        model.resize_token_embeddings(len(tokenizer))

    # --- LoRA ---
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if torch.cuda.is_available() and torch.cuda.current_device() == 0:
            model.print_trainable_parameters()

    # --- Linear predictor ---
    if linear_predictor and not load_lp:
        from llm_jepa.utils import set_seeds as _set_seeds
        if seed is not None:
            _set_seeds(seed)
        d = getattr(model.config, "hidden_size", None) or model.config.model_dim
        model.linear_predictor = LinearPredictor(d)

    return model, tokenizer
