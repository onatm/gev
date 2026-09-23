"""Evaluation stages shared by direct commands and study orchestration."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from ..configuration.config import config_from_mapping, load_config
from ..data.access import file_digest, load_verified_split, split_path
from ..evaluation.benchmark import evaluate_records
from ..artifacts.checkpoint_identity import checkpoint_fingerprint, read_checkpoint_manifest
from ..configuration.resolved import resolve_experiment_config
from ..backends.torch.environment import configure_runtime


def config_for_run(config_path: str | Path | None, run: str | Path):
    """Use an explicit config override or the resolved config embedded in the run."""
    if config_path is not None:
        return load_config(config_path)
    run_path = Path(run)
    checkpoint = run_path / "checkpoint" if (run_path / "checkpoint").exists() else run_path
    manifest = read_checkpoint_manifest(checkpoint)
    return config_from_mapping(manifest["training"]["resolved_config"])


def evaluation_provenance(*, suite: str, split: str, suite_sha256: str,
                          source_sha256: str, checkpoint_meta: dict,
                          checkpoint_hash: str, execution_mode: str,
                          trained_execution_mode: str) -> dict:
    """Keep evaluation identity separate from the checkpoint train identity."""
    return {
        "suite": suite, "split": split, "suite_sha256": suite_sha256,
        "manifest_sha256": suite_sha256, "source_sha256": source_sha256,
        "training_source_sha256": checkpoint_meta["lineage"].get("source_sha256"),
        "training_manifest_sha256": checkpoint_meta["lineage"].get("manifest_sha256"),
        "training_complete": checkpoint_meta["training"].get("metrics", {}).get("complete"),
        "checkpoint_fingerprint": checkpoint_hash,
        "execution_mode": execution_mode,
        "trained_execution_mode": trained_execution_mode,
    }


def evaluate_stage(run: str | Path, *, suite: str, split: str, data_root: str | Path,
                   output: str | Path, config_path: str | Path | None = None,
                   temperature: float | None = None, device: str | None = None,
                   execution: str | None = None) -> dict:
    config = config_for_run(config_path, run)
    if device is not None:
        config = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, device=device))
    resolved = resolve_experiment_config(config)
    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()

    rows, _manifest, manifest_hash = load_verified_split(data_root, suite, split)
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer)
    runtime_device = device or config.runtime.device
    if runtime_device == "auto":
        runtime_device = resolved.select_device()
    run_path = Path(run)
    checkpoint = run_path / "checkpoint" if (run_path / "checkpoint").exists() else run_path
    metadata = read_checkpoint_manifest(checkpoint)
    if (metadata["execution"].get("state_cap") != config.training.state_cap
            or metadata["execution"].get("branch_cap") != config.training.branch_cap
            or metadata["execution"].get("packed_cap") != config.training.packed_cap):
        raise ValueError("evaluation config caps do not match checkpoint contract")
    model, checkpoint_meta = resolved.load_checkpoint(
        checkpoint, device=runtime_device, tokenizer=tokenizer, expected_marker_map=markers)
    selected_temperature = checkpoint_meta["calibration"].get("temperature", 1.0) if temperature is None else temperature
    model.head.temperature = selected_temperature
    mode = execution or "rows"
    predictor = resolved.create_predictor(model, tokenizer, markers,
                                          temperature=selected_temperature,
                                          execution_mode=mode)
    report, _ = evaluate_records(rows, predictor, output, 1.0)

    digest = checkpoint_fingerprint(checkpoint)
    report["provenance"] = evaluation_provenance(
        suite=suite, split=split, suite_sha256=manifest_hash,
        source_sha256=file_digest(split_path(data_root, suite, split)),
        checkpoint_meta=checkpoint_meta, checkpoint_hash=digest,
        execution_mode=mode,
        trained_execution_mode=checkpoint_meta["execution"].get("execution_mode", "rows"))
    report["provenance"].update({
        "study_id": config.experiment_id,
        "protocol": dataclasses.asdict(config.protocol),
        "model_family": resolved.model.family.family_id,
        "backend": resolved.model.backend.backend_id,
        "scientific_recipe_sha256": resolved.recipe_sha256,
        "resolved_config": dataclasses.asdict(config),
        "output_path": str(output),
        "temperature": selected_temperature,
    })
    report["provenance"]["model_fingerprint"] = resolved.checkpoint_fingerprint(checkpoint)
    report_path = Path(output) / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def locked_evaluation_stage(run: str | Path, *, selection: str | Path,
                            suites: tuple[str, ...], data_root: str | Path,
                            output: str | Path, ledger: str | Path,
                            config_path: str | Path | None = None,
                            device: str | None = None) -> dict:
    """Reserve all suites first; the callback is the only test split loader."""
    from ..evaluation.locked import run_locked

    config = config_for_run(config_path, run)
    if device is not None:
        config = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, device=device))
    resolved = resolve_experiment_config(config)
    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()

    def load_test(suite: str):
        from ..data.suites import _fetch_locked_test_file, load_manifest

        tokenizer = resolved.load_tokenizer()
        markers = resolved.load_markers(tokenizer)
        run_path = Path(run)
        checkpoint = run_path / "checkpoint" if (run_path / "checkpoint").exists() else run_path
        runtime_device = device or config.runtime.device
        if runtime_device == "auto":
            runtime_device = resolved.select_device()
        model, _ = resolved.load_checkpoint(checkpoint, device=runtime_device,
                                            tokenizer=tokenizer, expected_marker_map=markers)
        test_path = Path(data_root) / suite / "test.jsonl"
        if not test_path.exists():
            _fetch_locked_test_file(suite, test_path, load_manifest(suite))
        rows, _manifest, _manifest_hash = load_verified_split(
            data_root, suite, "test", allow_test=True, _locked_test=True)
        predictor = resolved.create_predictor(model, tokenizer, markers)
        return rows, predictor

    return run_locked(selection=selection, suites=suites, data_root=data_root,
                      output=output, ledger=ledger, load_test=load_test,
                      code_sha="gev-locked", run=run)
