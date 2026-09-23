"""Single-run training stages used by ``gev train`` and study children."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..configuration.config import ExperimentConfig, load_config
from ..data.access import file_digest, load_verified_split, split_path, validate_training_rows
from ..data.continuation import (build_continuation, load_prepared,
                                 validate_init_metadata)
from ..configuration.resolved import ResolvedExperimentConfig, resolve_experiment_config
from ..backends.torch.environment import configure_runtime
from ..training.schedule import TrainingSchedule


def _config(path: str | Path, *, max_steps: int | None, device: str | None) -> tuple[ExperimentConfig, ResolvedExperimentConfig]:
    config = load_config(path)
    if device is not None:
        config = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, device=device))
    if max_steps is not None:
        if isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("--max-steps must be a positive integer")
        config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=max_steps))
    return config, resolve_experiment_config(config)


def _checkpoint_metadata(*, config: ExperimentConfig, resolved: ResolvedExperimentConfig,
                         markers, source_hash: str,
                         manifest_hash: str, metrics: dict, output: str | Path,
                         extra: dict | None = None) -> dict:
    return {
        **resolved.provenance(output_path=str(output)),
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "base_model_type": config.model.expected_model_type,
        "marker_ids": markers.ids,
        "marker_strings": markers.strings,
        "marker_bos": markers.bos,
        "tokenizer_revision": markers.tokenizer_revision,
        "tokenizer_sha256": getattr(markers, "tokenizer_sha256", None),
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "head_width": resolved.model.family.pointer_width,
        "lora": {"r": resolved.model.family.lora_rank,
                 "alpha": resolved.model.family.lora_alpha,
                 "dropout": resolved.model.family.lora_dropout,
                 "targets": list(resolved.model.family.lora_targets)},
        "dtype": config.training.dtype,
        "weights_dtype": "fp32",
        "state_cap": config.training.state_cap,
        "branch_cap": config.training.branch_cap,
        "packed_cap": config.training.packed_cap,
        "representation_version": 1,
        "device": metrics["device"],
        "training": metrics,
        "config": dataclasses.asdict(config),
        **(extra or {}),
    }


def train_stage(config_path: str | Path, *, data_root: str | Path, output: str | Path,
                max_steps: int | None = None, device: str | None = None,
                resume: str | Path | None = None) -> dict:
    """Train from verified decision-v7 train data and save its inference checkpoint."""
    output = Path(output)
    resume_path = Path(resume) if resume is not None else None
    if resume_path is not None and (output / "checkpoint").exists():
        raise FileExistsError("--resume requires a fresh --out directory; refusing to overwrite its existing inference checkpoint")
    config, resolved = _config(config_path, max_steps=max_steps, device=device)
    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()
    rows, _manifest, manifest_hash = load_verified_split(
        data_root, "decision-v7", "train", training=True)
    validate_training_rows(rows, "decision-v7")

    resolved.seed_rng()
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer)
    model = resolved.create_model(temperature=1.0)
    schedule = TrainingSchedule(rows, config, tokenizer, markers,
                                encoder=resolved.encode_record)
    source_path = split_path(data_root, "decision-v7", "train")
    source_hash = file_digest(source_path)
    resume_input = None
    if resume_path is not None:
        output.mkdir(parents=True, exist_ok=True)
        resume_input = {"snapshot": str(resume_path.resolve()),
                        "snapshot_sha256": file_digest(resume_path)}
        (output / "resume_input.json").write_text(json.dumps(resume_input, indent=2) + "\n")
    manifest = {"suite": "decision-v7", "split": "train", "source_sha256": source_hash,
                "manifest_sha256": manifest_hash}
    metrics = resolved.train(model, schedule, output,
                             source_hash=source_hash, manifest=manifest, resume=resume_path)
    metadata = _checkpoint_metadata(config=config, resolved=resolved,
                                    markers=markers,
                                    source_hash=source_hash, manifest_hash=manifest_hash,
                                    metrics=metrics, output=output,
                                    extra={"resume_input": resume_input})
    resolved.save_checkpoint(model, output / "checkpoint", metadata, tokenizer)
    return metrics


def warm_start_stage(config_path: str | Path, *, init_from: str | Path,
                     data_root: str | Path, output: str | Path,
                     max_steps: int | None = None, device: str | None = None) -> dict:
    """Warm-start from a validated v7 checkpoint with a fresh optimizer."""
    output = Path(output)
    config, resolved = _config(config_path, max_steps=max_steps, device=device)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    init_run = Path(init_from)
    init_checkpoint = init_run / "checkpoint" if (init_run / "checkpoint").exists() else init_run
    init_meta = validate_init_metadata(init_checkpoint, config)
    prepared_path = output.with_name(output.name + "-data")
    if not prepared_path.exists():
        build_continuation(data_root, out=prepared_path, seed=config.training.seed)

    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()
    resolved.seed_rng()
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer)
    model, _loaded_meta = resolved.load_checkpoint(
        init_checkpoint, device="cpu", tokenizer=tokenizer, expected_marker_map=markers)
    rows, prepared = load_prepared(prepared_path, data_root=data_root, seed=config.training.seed)
    schedule = TrainingSchedule(rows, config, tokenizer, markers,
                                encoder=resolved.encode_record)
    initial_weights_fingerprint = resolved.trainable_fingerprint(model)
    source_hash = prepared["combined_sha256"]
    manifest = {"suite": "continuation-night2", "split": "train",
                "source_sha256": source_hash,
                "manifest_sha256": prepared["night2_sha256"],
                "replay_ids_sha256": prepared["replay_ids_sha256"]}
    metrics = resolved.train(model, schedule, output,
                             source_hash=source_hash, manifest=manifest)
    extra = {"continuation": {
        "init_checkpoint_fingerprint": resolved.checkpoint_fingerprint(init_checkpoint),
        "initial_weights_fingerprint": initial_weights_fingerprint,
        "initializer_kind": init_meta["initializer_kind"],
        "initializer_config": init_meta.get("config") if init_meta["initializer_kind"] == "full-v7" else None,
        "fresh_optimizer": True,
        "optimizer_initial_step": 0,
        "prepared": prepared,
        "diagnostic_smoke_init": init_meta["initializer_kind"] != "full-v7",
    }}
    metadata = _checkpoint_metadata(config=config, resolved=resolved,
                                    markers=markers,
                                    source_hash=source_hash,
                                    manifest_hash=prepared["night2_sha256"],
                                    metrics=metrics, output=output, extra=extra)
    resolved.save_checkpoint(model, output / "checkpoint", metadata, tokenizer)
    return {**metrics, "init_checkpoint_fingerprint": resolved.checkpoint_fingerprint(init_checkpoint)}
