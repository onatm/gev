"""Diagnostic and configuration-validation command handlers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .common import load_cli_config


def handle_diagnose(args) -> int:
    action = args.diagnose_action
    if action == "config":
        from ..configuration.resolved import resolve_experiment_config
        config = load_cli_config(args.config)
        resolved = resolve_experiment_config(config)
        print(json.dumps({"status": "valid", "experiment_id": config.experiment_id,
                          "protocol": dataclasses.asdict(config.protocol),
                          "model_family": resolved.model.family.family_id,
                          "backend": resolved.model.backend.backend_id,
                          "scientific_recipe_sha256": resolved.recipe_sha256}, indent=2))
        return 0
    if action == "environment":
        from ..diagnostics.environment import doctor
        result = doctor()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if action == "model":
        if args.tiny or args.probe:
            if args.tiny and args.config:
                raise SystemExit("--tiny and --config are mutually exclusive")
            if args.probe and args.config:
                configured = load_cli_config(args.config)
                if configured.backend.id == "mlx":
                    result = _probe_mlx(configured, args.device, data=args.data, output=args.out)
                    print(json.dumps(result, indent=2, sort_keys=True))
                    return 0 if result["status"] == "passed" else 2
            from ..backends.torch.gemma3 import check_model
            config_path = None if args.tiny else (args.config or "configs/gemma3-1b-v7.toml")
            result = check_model(tiny=args.tiny, config_path=config_path, device=args.device)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] in {"passed", "diagnostic"} else 2
        from ..diagnostics.inspect_model import inspect_model
        config = load_cli_config(args.config or "configs/gemma3-1b-v7.toml")
        result = inspect_model(config, args.output_root or config.runtime.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "verified" else 2
    if action == "train":
        from ..diagnostics.profiling import profile_train
        result = profile_train(load_cli_config(args.config), args.warmup_steps,
                               args.measure_steps, Path(args.out), data_root=args.data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "passed" else 2
    if action == "precision":
        from ..diagnostics.precision import check_precision
        result = check_precision(load_cli_config(args.config), Path(args.out), run=args.run,
                                 records=args.records, data_root=args.data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "passed" else 2
    if action == "execution":
        return _execution(args)
    raise ValueError(f"unknown diagnose action: {action}")


def _probe_mlx(config, device, *, data="data", output=None):
    """Run the single pinned FP32 compute qualification on development data."""
    import dataclasses
    from ..backends.mlx.qualification import (
        require_tf32_disabled_before_mlx_import, run_base_qualification)

    try:
        # This must precede resolve.validate_runtime_available(), which imports MLX.
        require_tf32_disabled_before_mlx_import()
        if device is not None:
            config = dataclasses.replace(
                config, runtime=dataclasses.replace(config.runtime, device=device))
        from ..configuration.resolved import resolve_experiment_config

        resolved = resolve_experiment_config(config)
        resolved.validate_runtime_available()
        resolved.seed_rng()
        tokenizer = resolved.load_tokenizer()
        markers = resolved.load_markers(tokenizer)
        result = run_base_qualification(
            model_name=config.model.name, revision=config.model.revision,
            data_root=data, resolved=resolved, tokenizer=tokenizer, markers=markers,
            repo_root=Path(__file__).resolve().parents[3])
        if output is not None:
            destination = Path(output)
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite qualification receipt: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True,
                                            allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(destination)
        return result
    except Exception as exc:
        return {"status": "failed", "protocol": "mlx-gemma4-fp32-qualification",
                "model": config.model.name, "revision": config.model.revision,
                "error": f"{type(exc).__name__}: {exc}"}


def _execution(args) -> int:
    config = load_cli_config(args.config)
    if args.device is not None:
        config = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, device=args.device))
    from ..configuration.resolved import resolve_experiment_config
    resolved = resolve_experiment_config(config)
    from ..models.specs import BackendCapability
    resolved.model.require_capability(BackendCapability.EXECUTION_DIAGNOSTICS)
    from ..backends.torch.environment import configure_runtime
    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()

    from ..data.access import load_verified_split
    from ..diagnostics.execution import select_representative_records, write_measurement
    from ..domain.materialize import materialize

    rows, _manifest, _manifest_hash = load_verified_split(args.data, "decision-v7", "development")
    rows = select_representative_records(rows, args.records)
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer)
    device = args.device or config.runtime.device
    if device == "auto":
        device = resolved.select_device()
    run_path = Path(args.run)
    checkpoint = run_path / "checkpoint" if (run_path / "checkpoint").exists() else run_path
    model, _ = resolved.load_checkpoint(checkpoint, device=device, tokenizer=tokenizer,
                                        expected_marker_map=markers)
    model.head.temperature = 1.0
    encodings = [resolved.encode_record(tokenizer, materialize(row), markers)
                 for row in rows[:args.records]]
    result = resolved.measure_execution(model, encodings, records=len(encodings))
    result["selection"] = {
        "sources": {source: sum(row.get("_meta", {}).get("source") == source for row in rows)
                    for source in sorted({row.get("_meta", {}).get("source") for row in rows})},
        "question_count": sum(len(row.get("questions", {})) for row in rows),
    }
    write_measurement(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 2
