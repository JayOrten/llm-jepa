"""Utility functions: seeding, logging setup, last-token indexing."""

import logging
import random

import numpy as np
import torch


logger = logging.getLogger("llm_jepa")


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO"):
    """Configure the llm_jepa logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def last_token_index(input_ids: torch.Tensor, attention_mask: torch.Tensor, offset: int = -1) -> torch.Tensor:
    """Find the index of the last non-padding token in each sequence, then apply offset.

    Args:
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        offset: offset from end of non-pad region. -1 = last token, -2 = second-to-last, etc.

    Returns:
        (batch,) tensor of indices.
    """
    indices = []
    for i in range(input_ids.shape[0]):
        length = 0
        started = False
        for j in range(input_ids.shape[1]):
            if attention_mask[i, j] != 0:
                started = True
                length = j + 1
            elif started:
                break
        indices.append(length + offset)
    return torch.tensor(indices, device=input_ids.device)


def is_rank_zero() -> bool:
    """Check if we're on CUDA device 0 (or no CUDA)."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.current_device() == 0
