"""Validated model/backend selection and scientific-versus-operational identity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from ..models.registry import DEFAULT_MODEL_REGISTRY, ModelRegistry, ResolvedModel
from ..models.policy import (MODEL_POLICIES, auto_device_requires_probe,
                             validate_model_policy)
from ..training.schedule import TrainingSchedule

SUPPORTED_PROTOCOLS = frozenset({("kev-decision-v7", 1)})


def scientific_recipe(config: ExperimentConfig) -> dict[str, Any]:
    """Return only protocol/model/training choices that define the experiment."""
    training = dataclasses.asdict(config.training)
    for operational in ("seed", "max_steps", "save_every"):
        training.pop(operational, None)
    model = {
        "family": config.model.family,
        "name": config.model.name,
        "revision": config.model.revision,
        "expected_model_type": config.model.expected_model_type,
        "marker_ids": config.model.marker_ids,
    }
    model_policy = MODEL_POLICIES.get(config.model.family)
    if model_policy is not None and model_policy.source_weights_dtype is not None:
        model.update(source_weights_dtype=model_policy.source_weights_dtype,
                     compute_dtype=config.training.dtype)
    return {
        "protocol": dataclasses.asdict(config.protocol),
        "model": model,
        "training": training,
        "execution_mode": config.runtime.execution_mode,
        "representation_version": 1,
    }


def _recipe_digest(recipe: dict[str, Any]) -> str:
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    config: ExperimentConfig
    model: ResolvedModel
    recipe: dict[str, Any]
    recipe_sha256: str

    def select_device(self, requested: str | None = None) -> str:
        """Resolve ``auto`` only to a device the selected backend supports."""
        selected = self.model.select_device(
            self.config.runtime.device if requested is None else requested)
        validate_model_policy(self.config, self.model, effective_device=selected)
        return selected

    def validate_runtime_available(self) -> None:
        """Fail explicit unavailable accelerator requests before downloading weights."""
        self.select_device()

    def validate_policy_before_data_access(self) -> None:
        """Probe auto only when a supported policy combination depends on device."""
        if auto_device_requires_probe(self.config):
            self.select_device()

    def seed_rng(self, seed: int | None = None) -> None:
        """Seed backend-owned global RNGs before tokenizer/model construction."""
        self.model.seed_rng(self.config.training.seed if seed is None else seed)

    def create_model(self, *, temperature: float = 1.0, backbone: object | None = None,
                     attn_implementation: str | None = None,
                     gradient_checkpointing: bool | None = None,
                     seed: int | None = None) -> object:
        """Construct through the selected backend after validating the pair."""
        self.validate_runtime_available()
        arguments = dict(
            model_name=self.config.model.name,
            revision=self.config.model.revision,
            temperature=temperature,
            backbone=backbone,
            attn_implementation=attn_implementation or self.config.runtime.attn_implementation,
            gradient_checkpointing=(self.config.runtime.gradient_checkpointing
                                    if gradient_checkpointing is None else gradient_checkpointing),
            seed=seed,
        )
        if self.config.model.family == "gemma4_e2b_text":
            arguments["compute_dtype"] = self.config.training.dtype
        return self.model.create_model(**arguments)

    def load_tokenizer(self):
        return self.model.load_tokenizer(self.config.model.name, self.config.model.revision)

    def load_markers(self, tokenizer, artifact_path: str | Path | None = None):
        path = artifact_path or self.config.model.marker_artifact
        return self.model.load_markers(path, tokenizer)

    def encode_record(self, tokenizer, record: dict, markers, *,
                      state_cap: int | None = None, branch_cap: int | None = None,
                      packed_cap: int | None = None) -> dict:
        training = self.config.training
        return self.model.encode_record(
            tokenizer, record, markers,
            state_cap=training.state_cap if state_cap is None else state_cap,
            branch_cap=training.branch_cap if branch_cap is None else branch_cap,
            packed_cap=training.packed_cap if packed_cap is None else packed_cap)

    def create_predictor(self, model: object, tokenizer: object, markers: object, *,
                         temperature: float = 1.0, execution_mode: str = "rows") -> object:
        return self.model.create_predictor(
            model, tokenizer, markers,
            state_cap=self.config.training.state_cap,
            branch_cap=self.config.training.branch_cap,
            packed_cap=self.config.training.packed_cap,
            temperature=temperature, execution_mode=execution_mode)

    def load_checkpoint(self, directory, *, device="cpu", tokenizer=None,
                        expected_marker_map=None, backbone_loader=None,
                        attn_implementation=None):
        arguments = dict(
            config=self.config, device=device, tokenizer=tokenizer,
            expected_marker_map=expected_marker_map,
            backbone_loader=backbone_loader,
            attn_implementation=attn_implementation)
        if self.config.model.family == "gemma4_e2b_text":
            arguments["compute_dtype"] = self.config.training.dtype
        return self.model.load_checkpoint(directory, **arguments)

    def save_checkpoint(self, model: object, directory, metadata: dict, tokenizer=None):
        return self.model.save_checkpoint(model, directory, metadata, tokenizer)

    def checkpoint_fingerprint(self, directory) -> str:
        return self.model.checkpoint_fingerprint(directory)

    def trainable_fingerprint(self, model: object) -> str:
        return self.model.trainable_fingerprint(model)

    def train(self, model: object, schedule: TrainingSchedule,
              output: str | Path, **kwargs) -> dict:
        return self.model.train(model, schedule, self.config, output, **kwargs)

    def profile_train(self, warmup_steps: int, measure_steps: int, output: str | Path,
                      *, data_root: str | Path = "data") -> dict:
        return self.model.profile_train(self.config, warmup_steps, measure_steps,
                                        output, data_root=data_root)

    def compare_precision_profiles(self, model, encodings, record_ids, *,
                                   device: str, include_bf16: bool = True):
        return self.model.compare_precision_profiles(
            model, encodings, record_ids, device=device, include_bf16=include_bf16)

    def measure_execution(self, model, encodings, *, records: int, warmup: int = 2) -> dict:
        return self.model.measure_execution(model, encodings, records=records, warmup=warmup)

    def provenance(self, *, output_path: str | None = None) -> dict[str, Any]:
        """Fully resolved configuration, including non-scientific run controls."""
        config = dataclasses.asdict(self.config)
        operational_controls = {
            "seed": self.config.training.seed,
            "max_steps": self.config.training.max_steps,
            "save_every": self.config.training.save_every,
            "device": self.config.runtime.device,
            "attention_implementation": self.config.runtime.attn_implementation,
            "gradient_checkpointing": self.config.runtime.gradient_checkpointing,
            "empty_cache": self.config.runtime.empty_cache,
            "output_root": self.config.runtime.output_root,
        }
        if output_path is not None:
            operational_controls["output_path"] = output_path
        return {
            "study_id": self.config.experiment_id,
            "model_output_id": self.model.output_model_id,
            "protocol": dataclasses.asdict(self.config.protocol),
            "model_family": self.model.family.family_id,
            "family_runtime": type(self.model.family_runtime).__name__,
            "backend": self.model.backend.backend_id,
            "scientific_recipe": self.recipe,
            "scientific_recipe_sha256": self.recipe_sha256,
            "resolved_config": config,
            "operational_controls": operational_controls,
        }


def resolve_experiment_config(config: ExperimentConfig, *,
                              registry: ModelRegistry | None = None) -> ResolvedExperimentConfig:
    """Validate family/backend/protocol/runtime capability contracts, no downloads."""
    protocol_identity = (config.protocol.id, config.protocol.version)
    if protocol_identity not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported experiment protocol: {protocol_identity!r}")
    resolved_model = (registry or DEFAULT_MODEL_REGISTRY).resolve(
        config.model.family, config.backend.id)
    if config.model.expected_model_type != resolved_model.family.architecture:
        raise ValueError(
            f"model expected type {config.model.expected_model_type!r} does not match "
            f"family architecture {resolved_model.family.architecture!r}"
        )
    if (config.model.marker_ids is not None
            and set(config.model.marker_ids) != set(resolved_model.family.marker_roles)):
        raise ValueError("model marker ID roles do not match the selected model family")
    validate_model_policy(config, resolved_model)
    recipe = scientific_recipe(config)
    return ResolvedExperimentConfig(config, resolved_model, recipe, _recipe_digest(recipe))
