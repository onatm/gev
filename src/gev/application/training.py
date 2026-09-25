"""Single-run training stages used by ``gev train`` and study children."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

from ..configuration.config import ExperimentConfig, load_config
from ..data.access import file_digest, load_verified_split, split_path, validate_training_rows
from ..data.continuation import (build_continuation, load_prepared,
                                 validate_init_metadata)
from ..configuration.resolved import ResolvedExperimentConfig, resolve_experiment_config
from ..backends.torch.environment import configure_runtime
from ..training.schedule import TrainingSchedule


def _configure_runtime(config, resolved) -> None:
    if config.backend.id == "torch":
        configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    elif config.backend.id == "mlx":
        from ..backends.mlx.qualification import require_tf32_disabled_before_mlx_import

        require_tf32_disabled_before_mlx_import()


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
        "compute_dtype": config.training.dtype,
        "weights_dtype": metrics.get("weights_dtype", "fp32"),
        "source_weights_dtype": metrics.get("source_weights_dtype"),
        "state_cap": config.training.state_cap,
        "branch_cap": config.training.branch_cap,
        "packed_cap": config.training.packed_cap,
        "representation_version": 1,
        "device": metrics["device"],
        "training": metrics,
        "parity_qualification": metrics.get("parity_qualification"),
        "base_qualification": metrics.get("base_qualification"),
        "qualification": metrics.get("qualification"),
        "config": dataclasses.asdict(config),
        **(extra or {}),
    }


def _row_isolation_check(model, encodings: list[dict], *, backend: str,
                         compute_dtype: str) -> tuple[bool, float]:
    """Compare joint row outputs to isolated execution without scoring accuracy."""
    import numpy as np

    if not encodings:
        return False, math.inf
    selected = list(encodings[:2])
    if len(selected) == 1:
        selected.append(selected[0])
    model.eval()
    if backend == "torch":
        import torch

        with torch.inference_mode():
            singles = [model.forward_one(encoding) for encoding in selected]
            together = model.forward_rows_batch(selected)
    else:
        singles = [model.forward_one(encoding) for encoding in selected]
        together = model.forward_rows_batch(selected)
    if len(together) != len(singles):
        return False, math.inf
    maximum = 0.0
    for one, batch in zip(singles, together, strict=True):
        if len(one) != len(batch):
            return False, math.inf
        for expected, actual in zip(one, batch, strict=True):
            if tuple(expected.shape) != tuple(actual.shape):
                return False, math.inf
            if backend == "torch":
                import torch

                expected_values, actual_values = expected.detach().float(), actual.detach().float()
                if (not torch.isfinite(expected_values).all()
                        or not torch.isfinite(actual_values).all()):
                    return False, math.inf
                delta = float((expected_values - actual_values).abs().max().cpu())
            else:
                import mlx.core as mx

                mx.eval(expected, actual)
                expected_values = np.asarray(expected.astype(mx.float32))
                actual_values = np.asarray(actual.astype(mx.float32))
                if not np.isfinite(expected_values).all() or not np.isfinite(actual_values).all():
                    return False, math.inf
                delta = float(np.max(np.abs(expected_values - actual_values)))
            maximum = max(maximum, delta)
    if backend == "mlx":
        tolerance = 0.0
    else:
        import torch

        compute_type = torch.bfloat16 if compute_dtype == "bf16" else torch.float32
        tolerance = (4 if compute_dtype == "bf16" else 32) * torch.finfo(compute_type).eps
    return maximum <= tolerance, maximum


def _gemma4_checkpoint_receipt(*, config, resolved, model, metrics, checks,
                               development_report, development_rows,
                               development_manifest_sha256,
                               development_report_sha256) -> dict:
    from ..evaluation.development import selected_ids_sha256
    from ..models.qualification import (qualification_code_sha256,
                                        qualification_policy_sha256,
                                        seal_qualification_receipt,
                                        validate_qualification_receipt)
    from ..models.policy import GEMMA4_POLICY

    required = GEMMA4_POLICY.required_qualification_checks(
        config.backend.id, config.training.dtype)
    selected_ids = sorted(row["_meta"]["id"] for row in development_rows)
    coverage = development_report.get("coverage", {})
    mechanisms = development_report.get("mechanism_checks", {})
    checks["development_coverage"] = (
        coverage.get("requested_records") == len(selected_ids)
        and coverage.get("evaluated_records") == len(selected_ids)
        and coverage.get("requested_questions") == coverage.get("evaluated_questions")
        and coverage.get("evaluated_questions", 0) > 0
        and coverage.get("rejected_records") == 0
        and coverage.get("truncated_records") == 0)
    mechanism_failures = mechanisms.get("failures", 0) if isinstance(mechanisms, dict) else None
    checks["development_mechanisms"] = (
        isinstance(mechanisms, dict) and mechanisms.get("passed") is True
        and not isinstance(mechanism_failures, bool) and mechanism_failures == 0)
    passed = all(checks.get(name) is True for name in required)
    payload = {
        "format": "gev.trained-checkpoint-qualification", "version": 1,
        "status": "passed" if passed else "failed",
        "model_output_id": resolved.model.output_model_id,
        "family": resolved.model.family.family_id,
        "backend": resolved.model.backend.backend_id,
        "base": {"name": config.model.name, "revision": config.model.revision,
                 "source_weights_dtype": "bf16", "compute_dtype": config.training.dtype},
        "policy_sha256": qualification_policy_sha256(
            config.model.family, config.backend.id, config.training.dtype),
        "code_sha256": qualification_code_sha256(
            config.backend.id, Path(__file__).resolve().parents[3]),
        "checks": checks,
        "training_complete": bool(metrics.get("complete")),
        "training_steps": int(metrics.get("logical_steps", 0)),
        "development": {
            "suite": "decision-v7", "split": "development",
            "manifest_sha256": development_manifest_sha256,
            "report_sha256": development_report_sha256,
            "selected_ids": selected_ids,
            "selected_ids_sha256": selected_ids_sha256(
                [{"_meta": {"id": identifier}} for identifier in selected_ids]),
            "selected_record_count": len(selected_ids), "coverage": coverage,
            "mechanism_checks": mechanisms,
            "scores": {"clean": development_report.get("clean", {}),
                       "per_source": development_report.get("per_source", {})},
        },
        "trainable_parameters_sha256": resolved.trainable_fingerprint(model),
        "checkpoint_ready": False, "checkpoint_tensor_sha256": None,
    }
    receipt = seal_qualification_receipt(payload)
    if passed:
        validate_qualification_receipt(
            receipt, config=config, backend=config.backend.id,
            expected_code_sha256=receipt["code_sha256"], require_checkpoint_ready=False)
    return receipt


def train_stage(config_path: str | Path, *, data_root: str | Path, output: str | Path,
                max_steps: int | None = None, device: str | None = None,
                resume: str | Path | None = None) -> dict:
    """Train from verified decision-v7 train data and save its inference checkpoint."""
    output = Path(output)
    resume_path = Path(resume) if resume is not None else None
    if resume_path is not None and (output / "checkpoint").exists():
        raise FileExistsError("--resume requires a fresh --out directory; refusing to overwrite its existing inference checkpoint")
    config, resolved = _config(config_path, max_steps=max_steps, device=device)
    _configure_runtime(config, resolved)
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
        if resume_path.is_dir():
            digest = hashlib.sha256()
            for item in sorted(path for path in resume_path.rglob("*") if path.is_file()):
                digest.update(item.relative_to(resume_path).as_posix().encode())
                digest.update(file_digest(item).encode())
            snapshot_hash = digest.hexdigest()
        else:
            snapshot_hash = file_digest(resume_path)
        resume_input = {"snapshot": str(resume_path.resolve()),
                        "snapshot_sha256": snapshot_hash}
        (output / "resume_input.json").write_text(json.dumps(resume_input, indent=2) + "\n")
    manifest = {"suite": "decision-v7", "split": "train", "source_sha256": source_hash,
                "manifest_sha256": manifest_hash}
    metrics = resolved.train(model, schedule, output,
                             source_hash=source_hash, manifest=manifest, resume=resume_path)
    qualification_error = None
    if config.model.family == "gemma4_e2b_text":
        from ..domain.materialize import materialize
        from ..evaluation.benchmark import evaluate_records
        from ..evaluation.development import select_development_rows
        from ..evaluation.metrics import grouped_metrics
        from ..models.qualification import validate_qualification_receipt

        output.mkdir(parents=True, exist_ok=True)
        if config.backend.id == "mlx":
            metrics["base_qualification"] = model.provenance["base_qualification"]
        else:
            metrics["base_qualification"] = model.source_provenance
        dev_rows, _dev_manifest, dev_manifest_hash = load_verified_split(
            data_root, "decision-v7", "development")
        selected_dev = [row for row in select_development_rows(dev_rows)
                        if not row.get("_meta", {}).get("pair_id")]
        if not selected_dev:
            raise RuntimeError("Gemma 4 development selection has no complete independent rows")
        development_encodings = [resolved.encode_record(tokenizer, materialize(row), markers)
                                 for row in selected_dev]
        model.eval()
        predictor = resolved.create_predictor(model, tokenizer, markers,
                                              temperature=1.0, execution_mode="rows")
        development_report, development_rows = evaluate_records(
            selected_dev, predictor, output / "development", 1.0)
        development_report["per_source"] = grouped_metrics(development_rows, "source")
        development_report["provenance"] = {
            "model_output_id": resolved.model.output_model_id,
            "suite": "decision-v7", "split": "development",
            "manifest_sha256": dev_manifest_hash,
            "selected_record_count": len(selected_dev),
            "execution_mode": "rows", "temperature": 1.0,
        }
        development_report_path = output / "development" / "report.json"
        development_report_path.write_text(
            json.dumps(development_report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        development_report_sha256 = file_digest(development_report_path)
        isolation_passed, isolation_delta = _row_isolation_check(
            model, development_encodings, backend=config.backend.id,
            compute_dtype=config.training.dtype)
        if config.backend.id == "mlx":
            from ..backends.mlx.qualification import collect_mlx_training_checks

            checks = collect_mlx_training_checks(model, metrics, development_encodings)
        elif config.backend.id == "torch":
            from ..backends.torch.qualification import (
                causal_attention_masks_valid, collect_torch_training_checks)

            checks = collect_torch_training_checks(model, metrics)
            checks["causal_attention_masks"] = causal_attention_masks_valid(
                model, development_encodings, compute_dtype=config.training.dtype)
        else:
            raise ValueError(f"Gemma 4 training backend is not implemented: {config.backend.id}")
        checks["row_isolation"] = isolation_passed
        metrics["qualification_checks"] = checks
        metrics["row_isolation_max_logit_delta"] = isolation_delta
        qualification = _gemma4_checkpoint_receipt(
            config=config, resolved=resolved, model=model, metrics=metrics, checks=checks,
            development_report=development_report, development_rows=selected_dev,
            development_manifest_sha256=dev_manifest_hash,
            development_report_sha256=development_report_sha256)
        metrics["qualification"] = qualification
        (output / "qualification.json").write_text(
            json.dumps(qualification, indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8")
        (output / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if qualification["status"] != "passed":
            qualification_error = (
                "Gemma 4 training qualification checks failed; "
                "diagnostic artifacts were retained and no checkpoint was saved")
    if qualification_error is not None:
        raise RuntimeError(qualification_error)
    metadata = _checkpoint_metadata(config=config, resolved=resolved,
                                    markers=markers,
                                    source_hash=source_hash, manifest_hash=manifest_hash,
                                    metrics=metrics, output=output,
                                    extra={"resume_input": resume_input,
                                           **({"development_report": development_report}
                                              if config.model.family == "gemma4_e2b_text"
                                              else {})})
    resolved.save_checkpoint(model, output / "checkpoint", metadata, tokenizer)
    if config.model.family == "gemma4_e2b_text":
        from ..artifacts.checkpoint_identity import read_checkpoint_manifest

        final_receipt = read_checkpoint_manifest(output / "checkpoint")["qualification"]
        metrics["qualification"] = final_receipt
        (output / "qualification.json").write_text(
            json.dumps(final_receipt, indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8")
        (output / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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

    _configure_runtime(config, resolved)
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
