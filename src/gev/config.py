"""Strict, frozen experiment configuration."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    """A configuration is malformed or unsafe to run."""


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    expected_model_type: str
    marker_ids: dict[str, int] | None = None
    marker_artifact: str | None = None


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    learning_rate: float
    dtype: str
    context_length: int
    state_cap: int = 384
    branch_cap: int = 1024
    packed_cap: int = 2048
    logical_batch: int = 8
    microbatch: int = 1
    weight_decay: float = 0.01
    head_learning_rate: float | None = None
    max_steps: int | None = None
    p_none: float = 0.1
    p_none_distract: float = 0.12
    p_distract: float = 0.15
    p_none_pair: float = 0.25
    save_every: int | None = None


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    device: str
    mps_fallback: bool
    output_root: str
    attn_implementation: str = "eager"
    gradient_checkpointing: bool = False
    empty_cache: bool = False
    execution_mode: str = "rows"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig


def _only(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")


def _required(table: dict[str, Any], keys: set[str], where: str) -> None:
    missing = keys - set(table)
    if missing:
        raise ConfigError(f"{where}: missing key(s): {', '.join(sorted(missing))}")


def load_config(path: str | pathlib.Path) -> ExperimentConfig:
    path = pathlib.Path(path)
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    _only(raw, {"experiment", "model", "training", "runtime"}, "root")
    for section, keys in {
        "experiment": ({"id"}, {"id"}),
        "model": ({"name", "revision", "expected_model_type", "marker_ids", "marker_artifact"}, {"name", "revision", "expected_model_type"}),
        "training": ({"seed", "epochs", "learning_rate", "dtype", "context_length", "state_cap", "branch_cap", "packed_cap", "logical_batch", "microbatch", "weight_decay", "head_learning_rate", "max_steps", "save_every", "p_none", "p_none_distract", "p_distract", "p_none_pair"}, {"seed", "epochs", "learning_rate", "dtype", "context_length"}),
        "runtime": ({"device", "mps_fallback", "output_root", "attn_implementation", "gradient_checkpointing", "empty_cache", "execution_mode"}, {"device", "mps_fallback", "output_root"}),
    }.items():
        if isinstance(keys, tuple):
            allowed, required = keys
        else:
            allowed = required = keys
        value = raw.get(section)
        if not isinstance(value, dict):
            raise ConfigError(f"{section}: expected a TOML table")
        _only(value, allowed, section)
        _required(value, required, section)
    exp, model, training, runtime = raw["experiment"], raw["model"], raw["training"], raw["runtime"]
    if not isinstance(exp["id"], str) or not exp["id"].strip():
        raise ConfigError("experiment.id must be non-empty")
    if not isinstance(model["name"], str) or not model["name"]:
        raise ConfigError("model.name must be non-empty")
    if not isinstance(model["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ConfigError("model.revision must be a 40-character lowercase hexadecimal commit")
    if model["expected_model_type"] != "gemma3_text":
        raise ConfigError("model.expected_model_type must be gemma3_text")
    marker_ids = model.get("marker_ids")
    if marker_ids is not None:
        roles = {"state", "question", "option_start", "option_end", "decide"}
        if not isinstance(marker_ids, dict) or set(marker_ids) != roles or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in marker_ids.values()
        ):
            raise ConfigError("model.marker_ids must map the five semantic roles to non-negative integers")
    if isinstance(training["seed"], bool) or not isinstance(training["seed"], int) or training["seed"] < 0:
        raise ConfigError("training.seed must be non-negative")
    if isinstance(training["epochs"], bool) or not isinstance(training["epochs"], int) or not 1 <= training["epochs"] <= 100:
        raise ConfigError("training.epochs must be in [1, 100]")
    if isinstance(training["learning_rate"], bool) or not isinstance(training["learning_rate"], (int, float)) or not math.isfinite(training["learning_rate"]) or not 0 < training["learning_rate"] <= 1:
        raise ConfigError("training.learning_rate must be finite and in (0, 1]")
    if training["dtype"] not in {"fp32", "bf16"}:
        raise ConfigError("training.dtype must be fp32 or bf16")
    if isinstance(training["context_length"], bool) or not isinstance(training["context_length"], int) or not 1 <= training["context_length"] <= 32768:
        raise ConfigError("training.context_length must be in [1, 32768]")
    for name, low, high in (("state_cap", 1, 32768), ("branch_cap", 1, 32768), ("packed_cap", 1, 32768)):
        value = training.get(name, {"state_cap": 384, "branch_cap": 1024, "packed_cap": 2048}[name])
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ConfigError(f"training.{name} must be in [{low}, {high}]")
    for name in ("logical_batch", "microbatch"):
        value = training.get(name, 8 if name == "logical_batch" else 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"training.{name} must be a positive integer")
    if training.get("microbatch", 1) > training.get("logical_batch", 8):
        raise ConfigError("training.microbatch must not exceed logical_batch")
    for name in ("weight_decay", "p_none", "p_none_distract", "p_distract", "p_none_pair"):
        value = training.get(name, {"weight_decay": .01, "p_none": .1, "p_none_distract": .12, "p_distract": .15, "p_none_pair": .25}[name])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name.startswith("p_") and value > 1):
            raise ConfigError(f"training.{name} is invalid")
    if sum(training.get(k, d) for k, d in (("p_none", .1), ("p_none_distract", .12), ("p_distract", .15))) > 1:
        raise ConfigError("training augmentation probabilities sum to more than one")
    if training.get("head_learning_rate") is not None and (isinstance(training["head_learning_rate"], bool) or not isinstance(training["head_learning_rate"], (int, float)) or not math.isfinite(training["head_learning_rate"]) or training["head_learning_rate"] <= 0):
        raise ConfigError("training.head_learning_rate must be positive")
    if training.get("max_steps") is not None and (isinstance(training["max_steps"], bool) or not isinstance(training["max_steps"], int) or training["max_steps"] < 1):
        raise ConfigError("training.max_steps must be positive")
    if training.get("save_every") is not None and (isinstance(training["save_every"], bool) or not isinstance(training["save_every"], int) or training["save_every"] < 1):
        raise ConfigError("training.save_every must be positive")
    if training.get("state_cap", 384) > training["context_length"]:
        raise ConfigError("training.state_cap must not exceed context_length")
    if training["dtype"] == "bf16" and runtime["device"] == "cpu":
        raise ConfigError("training.dtype=bf16 requires an MPS or CUDA-capable runtime")
    if runtime["device"] not in {"auto", "cpu", "mps"}:
        raise ConfigError("runtime.device must be auto, cpu, or mps")
    if not isinstance(runtime["mps_fallback"], bool):
        raise ConfigError("runtime.mps_fallback must be boolean")
    if not isinstance(runtime["output_root"], str) or not runtime["output_root"]:
        raise ConfigError("runtime.output_root must be non-empty")
    if runtime.get("attn_implementation", "eager") not in {"eager", "sdpa"}:
        raise ConfigError("runtime.attn_implementation must be eager or sdpa")
    for name in ("gradient_checkpointing", "empty_cache"):
        if not isinstance(runtime.get(name, False), bool):
            raise ConfigError(f"runtime.{name} must be boolean")
    if runtime.get("execution_mode", "rows") not in {"rows", "packed"}:
        raise ConfigError("runtime.execution_mode must be rows or packed")
    return ExperimentConfig(
        exp["id"], ModelConfig(**model), TrainingConfig(**{**training, **{k: training.get(k, d) for k, d in (("state_cap", 384), ("branch_cap", 1024), ("packed_cap", 2048), ("logical_batch", 8), ("microbatch", 1), ("weight_decay", .01), ("head_learning_rate", None), ("max_steps", None), ("save_every", None), ("p_none", .1), ("p_none_distract", .12), ("p_distract", .15), ("p_none_pair", .25) )}}), RuntimeConfig(**{**runtime, "attn_implementation": runtime.get("attn_implementation", "eager"), "gradient_checkpointing": runtime.get("gradient_checkpointing", False), "empty_cache": runtime.get("empty_cache", False), "execution_mode": runtime.get("execution_mode", "rows")})
    )
