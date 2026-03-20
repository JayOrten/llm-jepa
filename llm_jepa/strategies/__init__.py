"""Training strategies for LLM-JEPA."""

from llm_jepa.strategies.base import RepresentationTrainer
from llm_jepa.strategies.jepa import JEPATrainer
from llm_jepa.strategies.stp import STPTrainer

STRATEGY_REGISTRY = {
    "jepa": JEPATrainer,
    "stp": STPTrainer,
}


def get_trainer_class(strategy_name: str):
    """Return the trainer class for the given strategy name.

    'regular' returns the HuggingFace Trainer directly (no aux loss).
    """
    if strategy_name == "regular":
        from transformers import Trainer
        return Trainer
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: regular, {', '.join(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy_name]
