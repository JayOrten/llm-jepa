"""Configuration via Dynaconf.

Settings are loaded from TOML files with CLI overrides via --set key=value.
"""

import argparse
import sys

from dynaconf import Dynaconf, Validator


def load_settings(argv: list[str] | None = None) -> Dynaconf:
    """Parse CLI args and return a Dynaconf settings object.

    Usage:
        python train.py --config configs/experiments/foo.toml
        python train.py --config configs/experiments/foo.toml --set strategy.name=stp --set strategy.lambda_=0.02
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, nargs="+", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", default=False)
    args, _ = parser.parse_known_args(argv)

    # Always load default.toml first, then any user-specified configs
    settings_files = ["configs/default.toml"] + args.config

    settings = Dynaconf(
        settings_files=settings_files,
        envvar_prefix="LLM_JEPA",
        merge_enabled=True,
    )

    # Apply --set overrides
    for override in args.overrides:
        key, _, value = override.partition("=")
        # Auto-convert types
        value = _auto_cast(value)
        settings.set(key, value)

    settings.set("dry_run", args.dry_run)

    # Validate
    settings.validators.register(
        Validator("model.name", must_exist=True),
        Validator("strategy.name", is_in=("regular", "jepa", "stp")),
        Validator("training.batch_size", gte=1),
        Validator("training.num_epochs", gte=1),
    )
    settings.validators.validate()

    return settings


def _auto_cast(value: str):
    """Try to cast string to int, float, or bool."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
