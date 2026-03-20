"""Loss functions for representation learning.

All functions take (a, b) embedding tensors of shape (batch, hidden_dim)
and return a scalar loss. No dependency on Trainer or config.
"""

import torch
import torch.nn.functional as F


def cosine_loss(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """1 - weighted mean cosine similarity."""
    sim = F.cosine_similarity(a, b, dim=-1)
    if weights is not None:
        return 1.0 - torch.sum(sim * weights) / torch.sum(weights)
    return 1.0 - torch.mean(sim)


def l2_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm of (a - b)."""
    return torch.linalg.norm(a - b, ord=2, dim=-1).mean()


def mse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((a - b) ** 2)


def infonce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss."""
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    logits = torch.mm(a_norm, b_norm.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


LOSS_REGISTRY = {
    "cosine": cosine_loss,
    "l2": l2_loss,
    "mse": mse_loss,
    "infonce": infonce_loss,
}


def get_loss_fn(name: str):
    """Return a loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {', '.join(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]
