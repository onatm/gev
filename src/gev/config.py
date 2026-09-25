"""Experiment configuration: one TOML file with ``[model]`` and ``[training]`` tables."""

from __future__ import annotations

import dataclasses
import re
import tomllib
from pathlib import Path
from typing import Any

FAMILIES = ("gemma3", "gemma4")
BACKENDS = ("torch", "mlx")
DTYPES = ("bf16", "fp32")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ConfigError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    family: str
    markers: tuple[str, ...] = ("<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>")
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    pointer_width: int = 256


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    epochs: int = 2
    learning_rate: float = 1e-4
    head_learning_rate: float | None = None
    weight_decay: float = 0.01
    logical_batch: int = 8
    microbatch: int = 1
    max_steps: int | None = None
    save_every: int | None = None
    state_cap: int = 384
    branch_cap: int = 2048
    p_none: float = 0.1
    p_none_distract: float = 0.12
    p_distract: float = 0.15
    p_none_pair: float = 0.25


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    backend: str = "torch"
    device: str = "auto"
    dtype: str = "bf16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        model, training = self.model, self.training
        checks = [
            (model.family in FAMILIES, f"model.family must be one of {FAMILIES}"),
            (re.fullmatch(r"[0-9a-f]{40}", model.revision) is not None,
             "model.revision must be a 40-character commit SHA"),
            (len(model.markers) == 5 and len(set(model.markers)) == 5, "model.markers must be 5 distinct tokens"),
            (self.backend in BACKENDS, f"backend must be one of {BACKENDS}"),
            (self.dtype in DTYPES, f"dtype must be one of {DTYPES}"),
            (self.backend != "mlx" or model.family == "gemma4", "the MLX backend supports gemma4 only"),
            (self.attn_implementation in ("eager", "sdpa", "flash_attention_2"),
             "attn_implementation must be eager, sdpa, or flash_attention_2"),
            (training.epochs >= 1 and training.logical_batch >= 1, "epochs and logical_batch must be positive"),
            (1 <= training.microbatch <= training.logical_batch, "microbatch must be in [1, logical_batch]"),
            (training.learning_rate > 0, "learning_rate must be positive"),
            (training.state_cap <= training.branch_cap, "state_cap must not exceed branch_cap"),
            (min(training.p_none, training.p_none_distract, training.p_distract, training.p_none_pair) >= 0
             and training.p_none + training.p_none_distract + training.p_distract <= 1,
             "augmentation probabilities must be nonnegative and the choice ones sum to at most 1"),
        ]
        for ok, message in checks:
            if not ok:
                raise ConfigError(message)

    @property
    def head_learning_rate(self) -> float:
        return self.training.head_learning_rate or self.training.learning_rate

    def replace(self, **changes: Any) -> "Config":
        training = {k: changes.pop(k) for k in list(changes) if k in TrainingConfig.__dataclass_fields__}
        if training:
            changes["training"] = dataclasses.replace(self.training, **training)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, table: Any, where: str):
    if not isinstance(table, dict):
        raise ConfigError(f"{where} must be a table")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(table) - set(fields)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    values = {key: tuple(value) if isinstance(value, list) else value for key, value in table.items()}
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def from_dict(raw: dict) -> Config:
    raw = dict(raw)
    model = _build(ModelConfig, raw.pop("model", None), "model")
    training = _build(TrainingConfig, raw.pop("training", {}), "training")
    return _build(Config, {**raw, "model": model, "training": training}, "config")


def load(path: str | Path) -> Config:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return from_dict(raw)
