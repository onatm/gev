"""Stable TOML I/O for typed study-child configurations."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

from ..configuration.config import ExperimentConfig


def _toml_key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value, ensure_ascii=False)


def _toml_literal(value: Any) -> str:
    if value is None:
        raise ValueError("TOML has no null value")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML floats must be finite")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def serialize_toml(value: dict[str, Any]) -> str:
    """Serialize nested tables in stable insertion order; omit optional nulls."""
    lines: list[str] = []

    def table(entries: dict[str, Any], path: tuple[str, ...]) -> None:
        scalars = [(key, item) for key, item in entries.items()
                   if not isinstance(item, dict) and item is not None]
        children = [(key, item) for key, item in entries.items() if isinstance(item, dict)]
        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, item in scalars:
            lines.append(f"{_toml_key(key)} = {_toml_literal(item)}")
        for key, item in children:
            table(item, (*path, key))

    table(value, ())
    return "\n".join(lines) + "\n"


def write_seed_config(config: ExperimentConfig, seed: int, directory: Path, *,
                      save_every: int | None = None) -> Path:
    """Write a typed child config with the requested seed and snapshot policy."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    training = dataclasses.replace(config.training, seed=seed)
    if save_every is not None and training.save_every is None:
        training = dataclasses.replace(training, save_every=save_every)
    config = dataclasses.replace(config, training=training)
    value = {
        "experiment": {"id": config.experiment_id},
        "protocol": dataclasses.asdict(config.protocol),
        "model": dataclasses.asdict(config.model),
        "backend": dataclasses.asdict(config.backend),
        "training": dataclasses.asdict(config.training),
        "runtime": dataclasses.asdict(config.runtime),
    }
    path = Path(directory) / f"seed-{seed}.toml"
    path.write_text(serialize_toml(value), encoding="utf-8")
    return path
