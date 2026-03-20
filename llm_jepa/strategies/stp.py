"""STP (Semantic Tube Prediction) strategy.

Operates on span-level representations extracted from a single forward pass.
Supports multiple modes: e2e, mean, random_span, curvature, random_span_mask.
"""

import logging
import random

import torch
import torch.nn.functional as F
from transformers import Trainer

from llm_jepa.losses import get_loss_fn
from llm_jepa.strategies.base import RepresentationTrainer
from llm_jepa.utils import last_token_index

logger = logging.getLogger("llm_jepa")


class STPTrainer(RepresentationTrainer):
    """Semantic Tube Prediction: span subtraction embeddings."""

    def __init__(self, *args, **kwargs):
        # STP-specific params
        self.linear_mode = kwargs.pop("linear_mode", "random_span")
        self.span_max_length = kwargs.pop("span_max_length", -1)
        self.span_times = kwargs.pop("span_times", 1)
        self.span_layer = kwargs.pop("span_layer", -1)
        self.span_zero = kwargs.pop("span_zero", False)
        self.span_e2e = kwargs.pop("span_e2e", False)
        self.span_all = kwargs.pop("span_all", False)
        self.span_draw_both = kwargs.pop("span_draw_both", False)
        self.span_uniform = kwargs.pop("span_uniform", False)
        self.length_adjustment = kwargs.pop("length_adjustment", None)
        self.curvature_sign = kwargs.pop("curvature_sign", False)

        # Mask mode
        self.random_span_mask = kwargs.pop("random_span_mask", False)
        self.random_span_mask_recover = kwargs.pop("random_span_mask_recover", False)

        # JEPA-compat params that STP also supports
        self.additive_mask = kwargs.pop("additive_mask", False)
        self.jepa_ratio = kwargs.pop("jepa_ratio", -1.0)
        self.avg_encoding = kwargs.pop("avg_encoding", False)

        if self.random_span_mask:
            assert self.span_times == 1, "span_times must be 1 when random_span_mask is set"

        super().__init__(*args, **kwargs)

        rank = getattr(self.args, "process_index", 0)
        self._g = torch.Generator(device=self.args.device)
        self._g.manual_seed(self.args.seed + rank * 3)
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_token_id = tokenizer.mask_token_id

    # ------------------------------------------------------------------
    # Span sampling
    # ------------------------------------------------------------------

    def _get_s_t(self, full_length: int):
        """Sample a random span [s, t) within [0, full_length)."""
        if self.span_draw_both:
            while True:
                p1 = torch.randint(0, full_length, (1,), generator=self._g, device=self._g.device)
                p2 = torch.randint(1, full_length + 1, (1,), generator=self._g, device=self._g.device)
                s = torch.min(p1, p2)
                t = torch.max(p1, p2)
                if self.span_zero:
                    s = torch.tensor(0, device=self._g.device)
                if self.span_max_length >= 0 and t - s > self.span_max_length:
                    t = s + self.span_max_length
                if s < t and t - s < full_length:
                    break
        elif self.span_uniform:
            total = full_length * (full_length + 1) // 2
            while True:
                r = torch.randint(total, (1,), generator=self._g, device=self._g.device, dtype=torch.long)
                two_n_plus_1 = 2 * full_length + 1
                s = torch.floor((two_n_plus_1 - torch.sqrt((two_n_plus_1**2 - 8 * r.float()))) / 2).long()
                prev = s * (2 * full_length - s + 1) // 2
                t = s + 1 + (r - prev)
                if t - s < full_length:
                    break
        else:
            if self.span_zero:
                s = torch.tensor(0, device=self._g.device)
            else:
                s = torch.randint(0, full_length, (), generator=self._g, device=self._g.device)
            max_t = full_length + 1
            if self.span_max_length >= 0:
                max_t = min(max_t, s + 1 + self.span_max_length)
            while True:
                t = torch.randint(s + 1, max_t, (), generator=self._g, device=self._g.device)
                if t - s < full_length:
                    break
        return s, t

    # ------------------------------------------------------------------
    # Embedding extraction helpers
    # ------------------------------------------------------------------

    def _get_span_embeddings(self, hidden_states, user_se, asst_se, s_off, t_off):
        """Extract before+after and patch embeddings via endpoint subtraction.

        The key insight of STP: representation of a span is approximated as
        Enc(end) - Enc(start), and the "rest" = Enc(full) - Enc(span).
        """
        us = user_se[0] + 1
        ue = user_se[1] + 1
        ast = asst_se[0] + 1
        ae = asst_se[1] + 1

        # Map offsets to absolute positions
        if s_off + us < ue:
            ps = us + s_off
        else:
            ps = ast + s_off - (ue - us)
        if t_off + us < ue:
            pt = us + t_off
        else:
            pt = ast + t_off - (ue - us)

        h = hidden_states
        us_emb = h[us - 1]
        ue_emb = h[ue - 1]
        ast_emb = h[ast - 1]
        ae_emb = h[ae - 1]
        ps_emb = h[ps - 1]
        pt_emb = h[pt - 1]

        if ps >= ast:
            before = ue_emb - us_emb + ps_emb - ast_emb
            patch = pt_emb - ps_emb
            after = ae_emb - pt_emb
        elif pt <= ue:
            before = ps_emb - us_emb
            patch = pt_emb - ps_emb
            after = ue_emb - pt_emb + ae_emb - ast_emb
        else:
            before = ps_emb - us_emb
            patch = ue_emb - ps_emb + pt_emb - ast_emb
            after = ae_emb - pt_emb

        return before, patch, after

    def _get_weights(self, full_length: int, patch_length: int) -> float:
        rest = full_length - patch_length
        if self.length_adjustment is None:
            return 1.0
        elif self.length_adjustment == "cosine_like":
            return 2.0 * rest * patch_length / (rest * rest + patch_length * patch_length)
        elif self.length_adjustment == "jaccard_like":
            return 1.0 - abs(rest - patch_length) / (rest + patch_length)
        raise ValueError(f"Unknown length_adjustment: {self.length_adjustment}")

    def _get_curvature(self, hidden_states, start, end_exclusive):
        length = end_exclusive - start
        if length > 1:
            curvature = 0.0
            for i in range(start + 1, end_exclusive):
                prev = hidden_states[i - 1] - hidden_states[i - 2]
                curr = hidden_states[i] - hidden_states[i - 1]
                dot = torch.dot(prev, curr)
                norms = torch.norm(prev) * torch.norm(curr)
                cosine = torch.clamp(dot / norms, -1.0, 1.0)
                angle = torch.acos(cosine)
                curvature += angle if self.curvature_sign else torch.abs(angle)
            return curvature, length - 1
        return 0.0, 0

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    def prepare_inputs(self, inputs):
        """STP uses a single forward pass on the full sequence."""
        if self.random_span_mask:
            self._prepare_mask_views(inputs)

        if self.linear_mode is not None and not self.random_span_mask:
            # For e2e/mean/random_span/curvature: just pass the full sequence
            self._user_start_end = inputs["user_start_end"]
            self._assistant_start_end = inputs["assistant_start_end"]
            return {
                "input_ids": inputs["input_ids"],
                "labels": inputs["labels"],
                "attention_mask": inputs["attention_mask"],
            }

        # For random_span_mask or JEPA-style 3-view (fallback)
        if "input_ids_user" in inputs:
            return {
                "input_ids": torch.cat([inputs["input_ids"], inputs["input_ids_user"], inputs["input_ids_assistant"]], dim=0),
                "labels": torch.cat([inputs["labels"], inputs["labels_user"], inputs["labels_assistant"]], dim=0),
                "attention_mask": torch.cat([inputs["attention_mask"], inputs["attention_mask_user"], inputs["attention_mask_assistant"]], dim=0),
            }
        return {
            "input_ids": inputs["input_ids"],
            "labels": inputs["labels"],
            "attention_mask": inputs["attention_mask"],
        }

    def _prepare_mask_views(self, inputs):
        """Build masked user/assistant views for random_span_mask mode."""
        bs = inputs["input_ids"].shape[0]
        inputs["input_ids_user"] = torch.zeros_like(inputs["input_ids"]) + self.eos_token_id
        inputs["labels_user"] = torch.zeros_like(inputs["labels"])
        inputs["attention_mask_user"] = torch.zeros_like(inputs["attention_mask"])
        inputs["input_ids_assistant"] = torch.zeros_like(inputs["input_ids"]) + self.eos_token_id
        inputs["labels_assistant"] = torch.zeros_like(inputs["labels"])
        inputs["attention_mask_assistant"] = torch.zeros_like(inputs["attention_mask"])

        for i in range(bs):
            user_start = inputs["user_start_end"][i, 0] + 1
            user_end = inputs["user_start_end"][i, 1] + 1
            assistant_start = inputs["assistant_start_end"][i, 0] + 1
            assistant_end = inputs["assistant_start_end"][i, 1] + 1

            if self.span_e2e:
                assistant_start = user_end
            if self.span_all:
                user_start = 0

            user_length = user_end - user_start
            assistant_length = assistant_end - assistant_start
            full_length = user_length + assistant_length

            s, t = self._get_s_t(full_length)

            # Map offsets to absolute positions
            if s + user_start < user_end:
                ps = user_start + s
            else:
                ps = assistant_start + s - user_length
            if t + user_start < user_end:
                pt = user_start + t
            else:
                pt = assistant_start + t - user_length

            assert pt > ps

            # Build masked view (user) and target view (assistant)
            total_len = assistant_end - user_start

            if ps >= assistant_start:
                inputs["input_ids_user"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                inputs["input_ids_user"][i, ps - user_start:pt - user_start] = self.mask_token_id
                inputs["labels_user"][i, :] = -100
                inputs["attention_mask_user"][i, :total_len] = 1
                if self.random_span_mask_recover:
                    inputs["input_ids_assistant"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :total_len] = 1
                else:
                    span_len = pt - ps
                    inputs["input_ids_assistant"][i, :span_len] = inputs["input_ids"][i, ps:pt]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :span_len] = 1
            elif pt <= user_end:
                inputs["input_ids_user"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                inputs["input_ids_user"][i, ps - user_start:pt - user_start] = self.mask_token_id
                inputs["labels_user"][i, :] = -100
                inputs["attention_mask_user"][i, :total_len] = 1
                if self.random_span_mask_recover:
                    inputs["input_ids_assistant"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :total_len] = 1
                else:
                    span_len = pt - ps
                    inputs["input_ids_assistant"][i, :span_len] = inputs["input_ids"][i, ps:pt]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :span_len] = 1
            else:
                # Span crosses user/assistant boundary
                inputs["input_ids_user"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                inputs["input_ids_user"][i, ps - user_start:user_end - user_start] = self.mask_token_id
                inputs["input_ids_user"][i, assistant_start - user_start:pt - user_start] = self.mask_token_id
                inputs["labels_user"][i, :] = -100
                inputs["attention_mask_user"][i, :total_len] = 1
                if self.random_span_mask_recover:
                    inputs["input_ids_assistant"][i, :total_len] = inputs["input_ids"][i, user_start:assistant_end]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :total_len] = 1
                else:
                    part1_len = user_end - ps
                    part2_len = pt - assistant_start
                    inputs["input_ids_assistant"][i, :part1_len] = inputs["input_ids"][i, ps:user_end]
                    inputs["input_ids_assistant"][i, part1_len:part1_len + part2_len] = inputs["input_ids"][i, assistant_start:pt]
                    inputs["labels_assistant"][i, :] = -100
                    inputs["attention_mask_assistant"][i, :part1_len + part2_len] = 1

    def extract_embeddings(self, model, inputs, outputs):
        hidden_states = outputs.hidden_states[self.span_layer]

        if self.linear_mode == "e2e":
            return self._extract_e2e(hidden_states)
        elif self.linear_mode == "mean":
            return self._extract_mean(hidden_states)
        elif self.linear_mode == "random_span":
            return self._extract_random_span(hidden_states)
        elif self.linear_mode == "curvature":
            return self._extract_curvature(hidden_states, outputs)
        elif self.random_span_mask:
            return self._extract_mask_mode(hidden_states, inputs, outputs)
        else:
            # Fallback: JEPA-style last-token comparison from 3-view forward
            return self._extract_jepa_fallback(hidden_states, inputs)

    # ------------------------------------------------------------------
    # Mode-specific extractors
    # ------------------------------------------------------------------

    def _extract_e2e(self, hidden_states):
        """End-to-end: Enc(end) - Enc(start) for user and assistant."""
        bs = hidden_states.shape[0]
        se_u = self._user_start_end
        se_a = self._assistant_start_end
        user_emb = hidden_states[range(bs), se_u[:, 1]] - hidden_states[range(bs), se_u[:, 0]]
        asst_emb = hidden_states[range(bs), se_a[:, 1]] - hidden_states[range(bs), se_a[:, 0]]
        return user_emb, asst_emb, None

    def _extract_mean(self, hidden_states):
        """Mean pooling over user and assistant spans."""
        bs = hidden_states.shape[0]
        se_u = self._user_start_end
        se_a = self._assistant_start_end
        user_emb = torch.zeros((bs, hidden_states.shape[-1]), device=hidden_states.device)
        asst_emb = torch.zeros((bs, hidden_states.shape[-1]), device=hidden_states.device)
        for i in range(bs):
            user_emb[i] = hidden_states[i, se_u[i, 0] + 1:se_u[i, 1] + 1].mean(dim=0)
            asst_emb[i] = hidden_states[i, se_a[i, 0] + 1:se_a[i, 1] + 1].mean(dim=0)
        return user_emb, asst_emb, None

    def _extract_random_span(self, hidden_states):
        """Random span: carve out a patch, compare patch vs rest."""
        bs = hidden_states.shape[0]
        total = bs * self.span_times
        hd = hidden_states.shape[-1]
        device = hidden_states.device

        user_emb = torch.zeros((total, hd), device=device)
        asst_emb = torch.zeros((total, hd), device=device)
        weights = torch.ones(total, device=device)

        se_u = self._user_start_end
        se_a = self._assistant_start_end

        for j in range(total):
            i = j // self.span_times
            if self.span_e2e:
                se_a[i, 0] = se_u[i, 1]
            if self.span_all:
                se_u[i, 0] = 0

            user_len = se_u[i, 1] - se_u[i, 0]
            asst_len = se_a[i, 1] - se_a[i, 0]
            full_len = user_len + asst_len

            s, t = self._get_s_t(full_len)
            before, patch, after = self._get_span_embeddings(
                hidden_states[i], se_u[i], se_a[i], s, t,
            )
            user_emb[j] = before + after
            asst_emb[j] = patch
            weights[j] = self._get_weights(full_len, t - s)

        return user_emb, asst_emb, weights

    def _extract_curvature(self, hidden_states, outputs):
        """Curvature mode: minimize angular curvature of hidden state trajectory."""
        bs = hidden_states.shape[0]
        se_u = self._user_start_end
        se_a = self._assistant_start_end

        curvature = torch.zeros(bs, device=hidden_states.device)
        for i in range(bs):
            uc, un = self._get_curvature(hidden_states[i], se_u[i, 0] + 1, se_u[i, 1] + 1)
            ac, an = self._get_curvature(hidden_states[i], se_a[i, 0] + 1, se_a[i, 1] + 1)
            if un + an > 0:
                curvature[i] = (uc + ac) / (un + an)

        # Override compute_loss behavior: curvature IS the aux loss
        # We return None to skip normal loss computation, but store
        # curvature for compute_loss to pick up
        self._curvature_loss = torch.mean(curvature)
        return None, None, None

    def _extract_mask_mode(self, hidden_states, inputs, outputs):
        """Random span mask mode: compare masked vs unmasked views."""
        # In mask mode, we did a 3-view forward
        batch_size = hidden_states.shape[0] // 3
        user_hs = hidden_states[batch_size:batch_size * 2]
        asst_hs = hidden_states[batch_size * 2:]
        idx_user = last_token_index(inputs["input_ids_user"], inputs["attention_mask_user"], offset=-1)
        idx_asst = last_token_index(inputs["input_ids_assistant"], inputs["attention_mask_assistant"], offset=-1)
        first_dim = idx_user.shape[0]
        emb_user = user_hs[range(first_dim), idx_user, :]
        emb_asst = asst_hs[range(first_dim), idx_asst, :]
        return emb_user, emb_asst, None

    def _extract_jepa_fallback(self, hidden_states, inputs):
        """Fallback for when STP operates in JEPA-like 3-view mode."""
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to handle curvature mode specially."""
        self._curvature_loss = None

        # Standard path
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # If curvature mode set aux loss directly
        if self._curvature_loss is not None:
            if return_outputs:
                lm_loss = result[1].loss
                total = self.gamma * lm_loss + self.get_lbd() * self._curvature_loss
                return (total, result[1])
            else:
                # result is already just lm_loss * gamma + 0
                # We need to add curvature
                return result + self.get_lbd() * self._curvature_loss

        return result
