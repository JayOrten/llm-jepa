"""JEPA strategy: 3-view forward, last-token embedding comparison.

Concatenates [full, user, assistant] as a single batch, runs one forward pass,
then extracts last-token embeddings from the user and assistant portions to
compute auxiliary loss.
"""

import torch

from llm_jepa.strategies.base import RepresentationTrainer
from llm_jepa.utils import last_token_index


class JEPATrainer(RepresentationTrainer):
    """JEPA: predict assistant representation from user representation."""

    def __init__(self, *args, **kwargs):
        self.additive_mask = kwargs.pop("additive_mask", False)
        self.jepa_ratio = kwargs.pop("jepa_ratio", -1.0)
        self.avg_encoding = kwargs.pop("avg_encoding", False)
        super().__init__(*args, **kwargs)

    def _build_additive_mask(self, k: int):
        """Build causal additive attention mask of size (k, k)."""
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask

    def prepare_inputs(self, inputs):
        if self.additive_mask:
            return self._prepare_additive(inputs)

        # Standard 3-view: concatenate [full, user, assistant] along batch dim
        return {
            "input_ids": torch.cat([
                inputs["input_ids"],
                inputs["input_ids_user"],
                inputs["input_ids_assistant"],
            ], dim=0),
            "labels": torch.cat([
                inputs["labels"],
                inputs["labels_user"],
                inputs["labels_assistant"],
            ], dim=0),
            "attention_mask": torch.cat([
                inputs["attention_mask"],
                inputs["attention_mask_user"],
                inputs["attention_mask_assistant"],
            ], dim=0),
        }

    def _prepare_additive(self, inputs):
        """Pack user+assistant into one sequence with block-diagonal causal mask."""
        if self.jepa_ratio > 0.0 and torch.rand(1).item() > self.jepa_ratio:
            self._skip_jepa = True
            return {
                "input_ids": inputs["input_ids"],
                "labels": inputs["labels"],
                "attention_mask": inputs["attention_mask"],
            }

        self._skip_jepa = False
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device

        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(device)

        lt_full = self._last_token_index(inputs["input_ids"], inputs["attention_mask"])
        lt_user = last_token_index(inputs["input_ids_user"], inputs["attention_mask_user"], offset=self.last_token)
        lt_asst = last_token_index(inputs["input_ids_assistant"], inputs["attention_mask_assistant"], offset=self.last_token)

        for i in range(batch_size):
            length = lt_full[i] + 1
            lu = lt_user[i] + 1
            la = lt_asst[i] + 1

            # Pack assistant tokens after user tokens
            inputs["input_ids_user"][i, lu:lu + la] = inputs["input_ids_assistant"][i, :la]
            inputs["labels_user"][i, lu:lu + la] = inputs["labels_assistant"][i, :la]

            # Build masks
            mask[i, :, :length, :length] = self._build_additive_mask(length)
            mask[i + batch_size, :, :lu, :lu] = self._build_additive_mask(lu)
            mask[i + batch_size, :, lu:lu + la, lu:lu + la] = self._build_additive_mask(la)

        self._lt_user = lt_user
        self._lt_assistant = lt_asst + lt_user + 1

        return {
            "input_ids": torch.cat([inputs["input_ids"], inputs["input_ids_user"]], dim=0),
            "labels": torch.cat([inputs["labels"], inputs["labels_user"]], dim=0),
            "attention_mask": mask,
        }

    def extract_embeddings(self, model, inputs, outputs):
        hidden_states = outputs.hidden_states[-1]

        if self.additive_mask:
            if getattr(self, "_skip_jepa", False):
                return None, None, None
            batch_size = hidden_states.shape[0] // 2
            user_hs = hidden_states[batch_size:]
            asst_hs = user_hs  # same sequence, different positions
            idx_user = self._lt_user
            idx_asst = self._lt_assistant
        else:
            batch_size = hidden_states.shape[0] // 3
            user_hs = hidden_states[batch_size:batch_size * 2]
            asst_hs = hidden_states[batch_size * 2:]
            idx_user = self._last_token_index(inputs["input_ids_user"], inputs["attention_mask_user"])
            idx_asst = self._last_token_index(inputs["input_ids_assistant"], inputs["attention_mask_assistant"])

        first_dim = idx_user.shape[0]

        if self.avg_encoding:
            emb_user = torch.zeros((first_dim, user_hs.shape[-1]), device=user_hs.device)
            emb_asst = torch.zeros((first_dim, asst_hs.shape[-1]), device=asst_hs.device)
            for i in range(first_dim):
                emb_user[i] = user_hs[i, :idx_user[i] + 1].mean(dim=0)
                emb_asst[i] = asst_hs[i, :idx_asst[i] + 1].mean(dim=0)
        else:
            emb_user = user_hs[range(first_dim), idx_user, :]
            emb_asst = asst_hs[range(first_dim), idx_asst, :]

        return emb_user, emb_asst, None
