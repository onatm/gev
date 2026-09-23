"""Backend-dispatched bounded training profile entry point."""

from __future__ import annotations

from pathlib import Path

from ..configuration.resolved import resolve_experiment_config


def profile_train(config, warmup_steps: int, measure_steps: int, output: Path, *,
                  data_root: str = "data") -> dict:
    resolved = resolve_experiment_config(config)
    return resolved.profile_train(warmup_steps, measure_steps, output,
                                  data_root=data_root)
