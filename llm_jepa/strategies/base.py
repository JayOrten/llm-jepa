"""Base RepresentationTrainer — shared forward/loss skeleton.

Subclasses override prepare_inputs() and extract_embeddings() to define
how auxiliary embedding pairs (emb_a, emb_b) are computed. The base class
owns compute_loss() which combines the LM loss with the auxiliary loss.
"""

import logging

import torch
from transformers import Trainer

from llm_jepa.losses import get_loss_fn
from llm_jepa.utils import last_token_index

logger = logging.getLogger("llm_jepa")


class RepresentationTrainer(Trainer):
    """Base class for JEPA/STP training strategies.

    Subclasses must implement:
        prepare_inputs(inputs) -> dict of model kwargs
        extract_embeddings(model, inputs, hidden_states) -> (emb_a, emb_b, weights)
    """

    def __init__(self, *args, **kwargs):
        self.lbd = kwargs.pop("lbd", 0.1)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.last_token = kwargs.pop("last_token", -1)
        self.loss_type = kwargs.pop("loss_type", "cosine")
        self.lbd_warmup = kwargs.pop("lbd_warmup", False)
        self.min_lbd = kwargs.pop("min_lbd", 0.0)
        self.linear_predictor = kwargs.pop("linear_predictor", False)
        super().__init__(*args, **kwargs)
        self.loss_fn = get_loss_fn(self.loss_type)

    def get_lbd(self) -> float:
        """Lambda with optional linear warmup."""
        if not self.lbd_warmup:
            return self.lbd
        progress = self.state.global_step / max(self.state.max_steps - 1, 1)
        return self.min_lbd + (self.lbd - self.min_lbd) * progress

    def _last_token_index(self, input_ids, attention_mask) -> torch.Tensor:
        """Find last non-pad token index with self.last_token offset."""
        return last_token_index(input_ids, attention_mask, offset=self.last_token)

    def unwrap(self, model):
        """Unwrap DDP/FSDP wrapper."""
        return getattr(model, "module", model)

    def prepare_inputs(self, inputs: dict) -> dict:
        """Build model forward kwargs from batch. Override in subclass."""
        raise NotImplementedError

    def extract_embeddings(self, model, inputs, hidden_states):
        """Extract (emb_a, emb_b, weights) from hidden states. Override in subclass.

        Returns:
            (emb_a, emb_b, weights) where emb_a and emb_b are (batch, hidden_dim)
            and weights is (batch,) or None. Return (None, None, None) to skip aux loss.
        """
        raise NotImplementedError

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute LM loss + lambda * aux_loss."""
        # Subclass builds the model inputs
        llm_inputs = self.prepare_inputs(inputs)

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)

        lm_loss = outputs.loss

        # Extract embeddings and compute aux loss
        emb_a, emb_b, weights = self.extract_embeddings(model, inputs, outputs)

        if emb_a is not None:
            if self.linear_predictor:
                emb_a = self.unwrap(model).linear_predictor(emb_a)

            if self.loss_type == "cosine" and weights is not None:
                aux_loss = self.loss_fn(emb_a, emb_b, weights)
            else:
                aux_loss = self.loss_fn(emb_a, emb_b)
        else:
            aux_loss = 0.0

        total_loss = self.gamma * lm_loss + self.get_lbd() * aux_loss

        # Log component losses for callbacks
        if hasattr(self, "state") and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self._last_lm_loss = lm_loss.item() if torch.is_tensor(lm_loss) else lm_loss
            self._last_aux_loss = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject component losses into the log dict."""
        if hasattr(self, "_last_lm_loss"):
            logs["lm_loss"] = self._last_lm_loss
        if hasattr(self, "_last_aux_loss"):
            logs["aux_loss"] = self._last_aux_loss
        super().log(logs, *args, **kwargs)
