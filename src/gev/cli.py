"""Command line entry point for Gev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import tempfile
import random
import dataclasses
import hashlib
import os

from .config import ConfigError, load_config
from .inspect_model import inspect_model
from .runtime import configure_runtime, doctor
from .data.suites import SuiteError, audit, fetch_file, fetch_manifest, load_manifest, load_split, manifest_digest, smoke, token_length_audit, verify
from .tokenization import MarkerMap, encode
from .materialize import materialize
from .evaluation.benchmark import evaluate_records

def _split_path(root: str, suite: str, split: str) -> Path:
    direct = Path(root) / f"{split}.jsonl"
    return direct if direct.exists() else Path(root) / suite / f"{split}.jsonl"

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_provenance(*, suite: str, split: str, suite_sha256: str,
                           source_sha256: str, checkpoint_meta: dict,
                           checkpoint_fingerprint: str, execution_mode: str,
                           trained_execution_mode: str) -> dict:
    """Keep evaluation identity separate from the checkpoint's train identity."""
    return {
        "suite": suite, "split": split, "suite_sha256": suite_sha256,
        "manifest_sha256": suite_sha256, "source_sha256": source_sha256,
        "training_source_sha256": checkpoint_meta.get("source_sha256"),
        "training_manifest_sha256": checkpoint_meta.get("manifest_sha256"),
        "training_complete": checkpoint_meta.get("training", {}).get("complete"),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "execution_mode": execution_mode,
        "trained_execution_mode": trained_execution_mode,
    }


def _load_cli_split(root: str, suite: str, split: str, *, training: bool = False, allow_test: bool = False,
                    _locked_test: bool = False):
    """Resolve only an official manifest or an explicitly derived smoke child."""
    if allow_test and not _locked_test:
        raise SuiteError("test loading is restricted to the locked evaluation path")
    base = Path(root)
    child = base / "manifest.json"
    if not child.exists():
        candidate = base / suite / "manifest.json"
        child = candidate if candidate.exists() else None
    if child is not None:
        manifest = json.loads(child.read_text(encoding="utf-8"))
        if not manifest.get("smoke_only"):
            raise SuiteError("local suite manifest is not an approved smoke child")
        if manifest.get("parent", {}).get("suite") != suite:
            raise SuiteError("smoke child parent suite mismatch")
        if training and split != "train":
            raise SuiteError("smoke child training is restricted to child train.jsonl")
        path = child.parent / f"{split}.jsonl"
        return load_split(path, suite, split, manifest, allow_test=allow_test), manifest, _digest(child)
    manifest = load_manifest(suite)
    path = base / suite / f"{split}.jsonl"
    return load_split(path, suite, split, manifest, allow_test=allow_test), manifest, manifest_digest(suite)

def _validate_training_rows(rows: list[dict], suite: str) -> None:
    if not rows: raise ValueError("training data is empty")
    manifest = load_manifest("decision-v7") if suite == "decision-v7" else None
    trainable = set((manifest or {}).get("trainable_sources", ()))
    forbidden = set((manifest or {}).get("eval_only_sources", ())) | set((manifest or {}).get("heldout_sources", ()))
    for row in rows:
        meta = row.get("_meta", {})
        if not meta.get("id") or not meta.get("source") or not meta.get("group_id"):
            raise ValueError("custom training data requires _meta.id, _meta.source, and _meta.group_id")
        if meta.get("variant", "clean") not in {"clean", "none_present", "none_absent"}:
            raise ValueError(f"unsupported training variant: {meta.get('variant')}")
        if meta["source"] in forbidden or (trainable and meta["source"] not in trainable):
            raise ValueError(f"training data contains non-trainable or held-out source: {meta['source']}")
    if suite == "transfer-v4":
        raise ValueError("transfer-v4 is development-only and cannot be used for training")


def _config(path: str):
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gev")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="run actual CPU/MPS runtime diagnostics")
    inspect = commands.add_parser("inspect-model", help="inspect pinned config/tokenizer, never model weights")
    inspect.add_argument("--config", default="configs/gemma3-1b-v7.toml")
    inspect.add_argument("--output-root", default=None)
    check_model_cmd = commands.add_parser("check-model", help="run the actual Gemma row model diagnostic")
    check_model_cmd.add_argument("--tiny", action="store_true", help="use an explicitly random tiny Gemma diagnostic")
    check_model_cmd.add_argument("--config", default=None, help="config TOML for the real pinned model")
    check_model_cmd.add_argument("--device", choices=("cpu", "mps"), default=None)
    train_cmd = commands.add_parser("train", help="train adapters and pointer head on a labelled split")
    train_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml")
    train_cmd.add_argument("--suite", default="decision-v7", choices=("decision-v7", "transfer-v4"))
    train_cmd.add_argument("--split", default="train", choices=("train",))
    train_cmd.add_argument("--data", default="data")
    train_cmd.add_argument("--out", required=True)
    train_cmd.add_argument("--max-steps", type=int, default=None)
    train_cmd.add_argument("--device", choices=("cpu", "mps"), default=None)
    train_cmd.add_argument("--resume", default=None, help="resume an atomic logical-boundary snapshot")
    continuation_cmd = commands.add_parser("continue-training", help="warm-start the pinned Night 2 continuation")
    continuation_cmd.add_argument("--config", default="configs/gemma3-1b-night2.toml")
    continuation_cmd.add_argument("--init-from", required=True); continuation_cmd.add_argument("--data", default="data")
    continuation_cmd.add_argument("--out", required=True); continuation_cmd.add_argument("--max-steps", type=int, default=None)
    continuation_cmd.add_argument("--device", choices=("cpu", "mps"), default=None); continuation_cmd.add_argument("--dry-run", action="store_true"); continuation_cmd.add_argument("--audit", action="store_true")
    profile_cmd = commands.add_parser("profile-train", help="profile a bounded real training window")
    profile_cmd.add_argument("--config", default="configs/gemma3-1b-mps-fast.toml"); profile_cmd.add_argument("--data", default="data")
    profile_cmd.add_argument("--warmup-steps", type=int, default=2); profile_cmd.add_argument("--measure-steps", type=int, default=4); profile_cmd.add_argument("--out", required=True)
    precision_cmd = commands.add_parser("check-precision", help="run bounded precision agreement checks")
    precision_cmd.add_argument("--run", required=True, help="trained checkpoint run or checkpoint directory")
    precision_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml")
    precision_cmd.add_argument("--records", type=int, default=16)
    precision_cmd.add_argument("--data", default="data")
    precision_cmd.add_argument("--out", required=True)
    parity_cmd = commands.add_parser("check-execution", help="measure rows versus packed development execution")
    parity_cmd.add_argument("--run", required=True); parity_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml")
    parity_cmd.add_argument("--records", type=int, default=8); parity_cmd.add_argument("--device", choices=("cpu", "mps"), default=None)
    parity_cmd.add_argument("--data", default="data"); parity_cmd.add_argument("--out", required=True)
    eval_cmd = commands.add_parser("eval", help="evaluate a checkpoint on a non-test split")
    eval_cmd.add_argument("--run", required=True); eval_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml")
    eval_cmd.add_argument("--suite", default="decision-v7", choices=("decision-v7", "transfer-v4")); eval_cmd.add_argument("--split", default="development", choices=("development", "calibration"))
    eval_cmd.add_argument("--data", default="data"); eval_cmd.add_argument("--out", required=True); eval_cmd.add_argument("--temperature", type=float, default=None); eval_cmd.add_argument("--device", choices=("cpu", "mps"), default=None); eval_cmd.add_argument("--execution", choices=("rows", "packed"), default=None)
    locked_cmd = commands.add_parser("eval-locked", help="explicit, once-only frozen test evaluation")
    locked_cmd.add_argument("--run", required=True); locked_cmd.add_argument("--selection", required=True)
    locked_cmd.add_argument("--suites", default="decision-v7,transfer-v4"); locked_cmd.add_argument("--data", default="data")
    locked_cmd.add_argument("--out", required=True); locked_cmd.add_argument("--ledger", default="runs/locked-ledger.jsonl")
    locked_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml"); locked_cmd.add_argument("--device", choices=("cpu", "mps"), default=None)
    register_cmd = commands.add_parser("register-candidate", help="pre-register a verified production candidate")
    register_cmd.add_argument("--run", required=True); register_cmd.add_argument("--study", required=True); register_cmd.add_argument("--out", required=True)
    experiment_cmd = commands.add_parser("experiment", help="plan or run the multi-seed study")
    experiment_cmd.add_argument("--config", default="configs/gemma3-1b-v7.toml"); experiment_cmd.add_argument("--seeds", default="0,1,2")
    experiment_cmd.add_argument("--data", default="data"); experiment_cmd.add_argument("--out", default="runs/study"); experiment_cmd.add_argument("--dry-run", action="store_true")
    experiment_cmd.add_argument("--max-steps", type=int, default=None); experiment_cmd.add_argument("--smoke", action="store_true"); experiment_cmd.add_argument("--existing-run", default=None)
    calibrate_cmd = commands.add_parser("calibrate", help="fit a protocol-approved temperature from saved rows")
    calibrate_cmd.add_argument("--run", required=True); calibrate_cmd.add_argument("--rows", default=None); calibrate_cmd.add_argument("--protocol", choices=("kev-screening", "kev-release"), default="kev-screening")
    calibrate_cmd.add_argument("--out", required=True); calibrate_cmd.add_argument("--update-checkpoint", action="store_true")
    compare_cmd = commands.add_parser("compare", help="paired-bootstrap two saved evaluations")
    compare_cmd.add_argument("--candidate", required=True); compare_cmd.add_argument("--reference", required=True); compare_cmd.add_argument("--aggregation", choices=("micro", "macro"), default="micro")
    compare_cmd.add_argument("--samples", type=int, default=1000); compare_cmd.add_argument("--seed", type=int, default=0); compare_cmd.add_argument("--out", required=True); compare_cmd.add_argument("--calibrated", action="store_true"); compare_cmd.add_argument("--allow-test-compare", action="store_true")
    validate = commands.add_parser("validate-config", help="validate a frozen TOML experiment")
    validate.add_argument("config")
    data = commands.add_parser("data", help="verify and fetch frozen public suites")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch")
    fetch.add_argument("suite", choices=("decision-v7", "transfer-v4"))
    fetch.add_argument("split", choices=("train", "calibration", "development", "test"))
    fetch.add_argument("--allow-test", action="store_true")
    fetch.add_argument("--output", default="data")
    continuation_fetch = data_commands.add_parser("continuation-fetch")
    continuation_fetch.add_argument("--data-root", default="data")
    continuation_prepare = data_commands.add_parser("continuation-prepare")
    continuation_prepare.add_argument("--data-root", default="data"); continuation_prepare.add_argument("--output", required=True)
    for command in ("verify", "audit"):
        check = data_commands.add_parser(command)
        check.add_argument("suite", choices=("decision-v7", "transfer-v4"))
        check.add_argument("split", choices=("train", "calibration", "development", "test"))
        check.add_argument("path")
        check.add_argument("--allow-test", action="store_true")
        if command == "audit":
            check.add_argument("--config", default=None)
            check.add_argument("--markers", default=None)
            check.add_argument("--augment", action="store_true")
            check.add_argument("--seeds", default="0,1,2", help="training seeds for token audit")
            check.add_argument("--data-root", default="data")
            check.add_argument("--output", default="runs/reference/token-length-audit.json")
    smoke_cmd = data_commands.add_parser("smoke", help="write a group-preserving structural child suite")
    smoke_cmd.add_argument("--out", default="data/smoke")
    smoke_cmd.add_argument("--train-records", type=int, default=128)
    smoke_cmd.add_argument("--dev-records", type=int, default=64)
    smoke_cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "experiment":
        from .experiment import run
        seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
        run(args.config, seeds=seeds, data=args.data, out=args.out, dry_run=args.dry_run,
            max_steps=args.max_steps, smoke=args.smoke, existing_run=args.existing_run)
        return 0
    if args.command == "calibrate":
        from .evaluation.calibration import calibrate
        print(json.dumps(calibrate(args.run, rows=args.rows, protocol=args.protocol, out=args.out,
                                   update_checkpoint=args.update_checkpoint), indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        from .evaluation.compare import compare
        print(json.dumps(compare(args.candidate, args.reference, aggregation=args.aggregation,
                                  samples=args.samples, seed=args.seed, out=args.out,
                                   calibrated=args.calibrated, allow_test_compare=args.allow_test_compare), indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        result = doctor()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    elif args.command == "validate-config":
        value = _config(args.config)
        print(json.dumps({"status": "valid", "experiment_id": value.experiment_id}, indent=2))
    elif args.command == "data":
        try:
            if args.data_command == "smoke":
                result = smoke(Path("data"), args.train_records, args.dev_records, Path(args.out), args.seed)
                print(json.dumps(result, indent=2, sort_keys=True)); return 0
            if args.data_command == "continuation-fetch":
                from .data.continuation import fetch_night2
                print(json.dumps(fetch_night2(args.data_root), indent=2, sort_keys=True)); return 0
            if args.data_command == "continuation-prepare":
                from .data.continuation import build_continuation
                print(json.dumps(build_continuation(args.data_root, out=args.output), indent=2, sort_keys=True)); return 0
            root = Path("references/suites") / args.suite
            if args.data_command == "audit" and args.config and args.markers:
                result = token_length_audit(Path(args.data_root), args.config, Path(args.markers),
                                             augment_train=args.augment,
                                             seeds=tuple(int(s) for s in args.seeds.split(",")),
                                             output=Path(args.output))
                print(json.dumps(result, indent=2, sort_keys=True)); return 0
            with tempfile.TemporaryDirectory() as directory:
                manifest = fetch_manifest(args.suite, Path(directory) / "manifest.json")
                if args.data_command == "fetch":
                    result = fetch_file(args.split, args.suite, Path(args.output) / args.suite / f"{args.split}.jsonl", manifest, allow_test=args.allow_test)
                elif args.data_command == "verify":
                    result = verify(Path(args.path), manifest, args.split, allow_test=args.allow_test, suite=args.suite)
                else:
                    result = audit(Path(args.path), manifest, args.split, allow_test=args.allow_test, suite=args.suite)
            print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result.get("valid", True) else 1
        except (SuiteError, OSError, ValueError) as exc:
            raise SystemExit(f"data error: {exc}") from exc
    elif args.command == "check-model":
        from .models.gemma import check_model
        result = check_model(tiny=args.tiny, config_path=args.config, device=args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"passed", "diagnostic"} else 2
    elif args.command == "train":
        if args.resume and (Path(args.out) / "checkpoint").exists():
            raise SystemExit("--resume requires a fresh --out directory; refusing to overwrite its existing inference checkpoint")
        value = _config(args.config)
        if args.device is not None:
            value = dataclasses.replace(value, runtime=dataclasses.replace(value.runtime, device=args.device))
        configure_runtime(value.runtime.device, value.runtime.mps_fallback)
        from .models.gemma import GemmaRowModel, load_real_backbone
        from .training.loop import train
        from .checkpoint import save_checkpoint
        if args.max_steps is not None and (isinstance(args.max_steps, bool) or args.max_steps < 1):
            raise SystemExit("--max-steps must be a positive integer")
        if args.max_steps is not None:
            value = dataclasses.replace(value, training=dataclasses.replace(value.training, max_steps=args.max_steps))
        rows, manifest, manifest_hash = _load_cli_split(args.data, args.suite, args.split, training=True)
        _validate_training_rows(rows, args.suite)
        import torch
        torch.manual_seed(value.training.seed); random.seed(value.training.seed)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(value.model.name, revision=value.model.revision)
        marker_path = Path(value.model.marker_artifact or "runs/reference/model-marker-map.json")
        markers = MarkerMap.load(marker_path, tokenizer)
        model = GemmaRowModel(load_real_backbone(value.model.name, value.model.revision), temperature=1.0)
        source_hash = _digest(Path(args.data) / ("train.jsonl" if (Path(args.data) / "manifest.json").exists() else f"{args.suite}/{args.split}.jsonl"))
        resume_input = None
        if args.resume:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            resume_input = {"snapshot": str(Path(args.resume).resolve()), "snapshot_sha256": _digest(Path(args.resume))}
            (Path(args.out) / "resume_input.json").write_text(json.dumps(resume_input, indent=2) + "\n")
        metrics = train(model, rows, tokenizer, markers, value, args.out, source_hash=source_hash, manifest={"suite": args.suite, "split": args.split, "source_sha256": source_hash, "manifest_sha256": manifest_hash}, resume=args.resume)
        marker_artifact = json.loads(marker_path.read_text(encoding="utf-8"))
        save_checkpoint(model, Path(args.out) / "checkpoint", {"model_name": value.model.name, "model_revision": value.model.revision, "base_model_type": value.model.expected_model_type, "marker_ids": markers.ids, "marker_strings": markers.strings, "marker_bos": markers.bos, "tokenizer_revision": markers.tokenizer_revision, "tokenizer_sha256": marker_artifact.get("tokenizer_sha256"), "source_sha256": source_hash, "manifest_sha256": manifest_hash, "head_width": model.head.query.out_features, "lora": {"r": 16, "alpha": 32, "dropout": .05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}, "dtype": value.training.dtype, "weights_dtype": "fp32", "state_cap": value.training.state_cap, "branch_cap": value.training.branch_cap, "packed_cap": value.training.packed_cap, "representation_version": 1, "device": metrics["device"], "training": metrics, "config": dataclasses.asdict(value), "resume_input": resume_input}, tokenizer)
        print(json.dumps(metrics, indent=2, sort_keys=True)); return 0
    elif args.command == "continue-training":
        from .data.continuation import build_continuation, load_prepared, validate_init_metadata, audit_prepared_tokens
        from .checkpoint import load_checkpoint, save_checkpoint, checkpoint_fingerprint
        value = _config(args.config)
        if args.device is not None:
            value = dataclasses.replace(value, runtime=dataclasses.replace(value.runtime, device=args.device))
        if args.max_steps is not None and (isinstance(args.max_steps, bool) or args.max_steps < 1):
            raise SystemExit("--max-steps must be a positive integer")
        if args.max_steps is not None:
            value = dataclasses.replace(value, training=dataclasses.replace(value.training, max_steps=args.max_steps))
        if Path(args.out).exists():
            raise SystemExit(f"refusing to overwrite run: {args.out}")
        init_checkpoint = Path(args.init_from) / "checkpoint" if (Path(args.init_from) / "checkpoint").exists() else Path(args.init_from)
        init_meta = validate_init_metadata(init_checkpoint, value)
        prepared_path = Path(args.out).with_name(Path(args.out).name + "-data")
        if prepared_path.exists():
            _prepared_rows, plan = load_prepared(prepared_path, data_root=args.data, seed=value.training.seed)
        else:
            plan = build_continuation(args.data, out=prepared_path, seed=value.training.seed)
        if args.dry_run:
            result = {"status": "dry-run", "plan": plan,
                              "init_from": str(Path(args.init_from).resolve()),
                              "init_metadata": {"model_name": init_meta["model_name"], "model_revision": init_meta["model_revision"],
                               "adapter_sha256": init_meta["adapter_sha256"], "pointer_sha256": init_meta["pointer_sha256"],
                               "initializer_kind": init_meta["initializer_kind"],
                               "initializer_fingerprint": init_meta["initializer_fingerprint"]},
                              "model_load": "not performed"}
            if getattr(args, "audit", False):
                result["token_audit"] = audit_prepared_tokens(prepared_path, value,
                    value.model.marker_artifact or "runs/reference/model-marker-map.json")
            print(json.dumps(result, indent=2, sort_keys=True)); return 0
        configure_runtime(value.runtime.device, value.runtime.mps_fallback)
        import torch
        torch.manual_seed(value.training.seed); random.seed(value.training.seed)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(value.model.name, revision=value.model.revision)
        marker_path = Path(value.model.marker_artifact or "runs/reference/model-marker-map.json")
        markers = MarkerMap.load(marker_path, tokenizer)
        checkpoint = init_checkpoint
        initializer_kind = init_meta["initializer_kind"]
        initializer_config = init_meta.get("config") if initializer_kind == "full-v7" else None
        model, _loaded_meta = load_checkpoint(checkpoint, config=value, device="cpu", tokenizer=tokenizer, expected_marker_map=markers)
        rows, prepared = load_prepared(prepared_path, data_root=args.data, seed=value.training.seed)
        initial_weights_fingerprint = hashlib.sha256(b"".join(
            tensor.detach().cpu().contiguous().numpy().tobytes()
            for name, tensor in sorted(model.state_dict().items())
            if "lora_" in name or name.startswith("head.")
        )).hexdigest()
        from .training.loop import train
        metrics = train(model, rows, tokenizer, markers, value, args.out,
                        source_hash=prepared["combined_sha256"], manifest={"suite": "continuation-night2", "split": "train",
                        "source_sha256": prepared["combined_sha256"], "manifest_sha256": prepared["night2_sha256"],
                        "replay_ids_sha256": prepared["replay_ids_sha256"]})
        marker_artifact = json.loads(marker_path.read_text(encoding="utf-8"))
        save_checkpoint(model, Path(args.out) / "checkpoint", {"model_name": value.model.name,
            "model_revision": value.model.revision, "base_model_type": value.model.expected_model_type,
            "marker_ids": markers.ids, "marker_strings": markers.strings, "marker_bos": markers.bos,
            "tokenizer_revision": markers.tokenizer_revision, "tokenizer_sha256": marker_artifact.get("tokenizer_sha256"),
            "source_sha256": prepared["combined_sha256"], "manifest_sha256": prepared["night2_sha256"],
            "head_width": model.head.query.out_features, "lora": {"r": 16, "alpha": 32, "dropout": .05,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            "dtype": value.training.dtype, "weights_dtype": "fp32", "state_cap": value.training.state_cap,
            "branch_cap": value.training.branch_cap, "packed_cap": value.training.packed_cap,
            "representation_version": 1, "device": metrics["device"], "training": metrics,
             "config": dataclasses.asdict(value), "continuation": {"init_checkpoint_fingerprint": checkpoint_fingerprint(checkpoint),
             "initial_weights_fingerprint": initial_weights_fingerprint,
             "initializer_kind": initializer_kind,
             "initializer_config": initializer_config,
             "fresh_optimizer": True, "optimizer_initial_step": 0,
             "prepared": prepared, "diagnostic_smoke_init": initializer_kind != "full-v7"}}, tokenizer)
        print(json.dumps({**metrics, "init_checkpoint_fingerprint": checkpoint_fingerprint(checkpoint)}, indent=2, sort_keys=True)); return 0
    elif args.command == "profile-train":
        from .profiling import profile_train
        result = profile_train(_config(args.config), args.warmup_steps, args.measure_steps, Path(args.out), data_root=args.data)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result.get("status") == "passed" else 2
    elif args.command == "check-precision":
        from .precision import check_precision
        result = check_precision(_config(args.config), Path(args.out), run=args.run, records=args.records, data_root=args.data)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result.get("status") == "passed" else 2
    elif args.command == "check-execution":
        value = _config(args.config)
        if args.device is not None:
            value = dataclasses.replace(value, runtime=dataclasses.replace(value.runtime, device=args.device))
        configure_runtime(value.runtime.device, value.runtime.mps_fallback)
        import torch
        from transformers import AutoTokenizer
        from .checkpoint import load_checkpoint
        rows, _manifest, _hash = _load_cli_split(args.data, "decision-v7", "development")
        from .execution import select_representative_records
        rows = select_representative_records(rows, args.records)
        tokenizer = AutoTokenizer.from_pretrained(value.model.name, revision=value.model.revision)
        markers = MarkerMap.load(value.model.marker_artifact or "runs/reference/model-marker-map.json", tokenizer)
        device = args.device or value.runtime.device
        if device == "auto": device = "mps" if torch.backends.mps.is_available() else "cpu"
        checkpoint = Path(args.run) / "checkpoint" if (Path(args.run) / "checkpoint").exists() else Path(args.run)
        model, _ = load_checkpoint(checkpoint, config=value, device=device, tokenizer=tokenizer, expected_marker_map=markers)
        model.head.temperature = 1.0
        encoded = [encode(tokenizer, materialize(row), markers, state_cap=value.training.state_cap,
                          branch_cap=value.training.branch_cap, packed_cap=value.training.packed_cap)
                   for row in rows[:args.records]]
        from .execution import measure, write_measurement
        result = measure(model, encoded, records=len(encoded))
        result["selection"] = {"sources": {source: sum(row.get("_meta", {}).get("source") == source for row in rows)
                                            for source in sorted({row.get("_meta", {}).get("source") for row in rows})},
                               "question_count": sum(len(row.get("questions", {})) for row in rows)}
        write_measurement(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result.get("status") == "passed" else 2
    elif args.command == "eval":
        value = _config(args.config)
        if args.device is not None:
            value = dataclasses.replace(value, runtime=dataclasses.replace(value.runtime, device=args.device))
        configure_runtime(value.runtime.device, value.runtime.mps_fallback)
        from .checkpoint import load_checkpoint
        rows, manifest, manifest_hash = _load_cli_split(args.data, args.suite, args.split)
        import torch
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(value.model.name, revision=value.model.revision)
        markers = MarkerMap.load(value.model.marker_artifact or "runs/reference/model-marker-map.json", tokenizer)
        device = args.device or value.runtime.device
        if device == "auto": device = "mps" if torch.backends.mps.is_available() else "cpu"
        checkpoint = Path(args.run) / "checkpoint" if (Path(args.run) / "checkpoint").exists() else Path(args.run)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("state_cap") != value.training.state_cap or metadata.get("branch_cap") != value.training.branch_cap or metadata.get("packed_cap") != value.training.packed_cap:
            raise SystemExit("evaluation config caps do not match checkpoint contract")
        model, checkpoint_meta = load_checkpoint(checkpoint, config=value, device=device, tokenizer=tokenizer, expected_marker_map=markers)
        temperature = checkpoint_meta.get("temperature", 1.0) if args.temperature is None else args.temperature
        model.head.temperature = temperature
        from .evaluation.predictors import LocalPredictor
        predictor = LocalPredictor(model, tokenizer, markers, state_cap=value.training.state_cap,
                                   branch_cap=value.training.branch_cap, packed_cap=value.training.packed_cap,
                                   temperature=temperature, execution_mode=args.execution or "rows")
        report, _ = evaluate_records(rows, predictor, args.out, 1.0)
        source_path = _split_path(args.data, args.suite, args.split)
        checkpoint_fingerprint = hashlib.sha256()
        for name in ("metadata.json", "adapter_model.safetensors", "pointer.safetensors"):
            checkpoint_fingerprint.update((checkpoint / name).read_bytes())
        report["provenance"] = _evaluation_provenance(
            suite=args.suite, split=args.split, suite_sha256=manifest_hash,
            source_sha256=_digest(source_path), checkpoint_meta=checkpoint_meta,
            checkpoint_fingerprint=checkpoint_fingerprint.hexdigest(),
            execution_mode=args.execution or "rows",
            trained_execution_mode=checkpoint_meta.get("execution_mode", "rows"))
        from .checkpoint import checkpoint_fingerprint as stable_checkpoint_fingerprint
        report["provenance"]["model_fingerprint"] = stable_checkpoint_fingerprint(checkpoint)
        report_path = Path(args.out) / "report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
        print(json.dumps(report, indent=2)); return 0
    elif args.command == "eval-locked":
        from .evaluation.locked import run_locked
        from .checkpoint import load_checkpoint, checkpoint_fingerprint
        value = _config(args.config)
        if args.device is not None:
            value = dataclasses.replace(value, runtime=dataclasses.replace(value.runtime, device=args.device))
        suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
        def load_test(suite):
            import torch
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(value.model.name, revision=value.model.revision)
            markers = MarkerMap.load(value.model.marker_artifact or "runs/reference/model-marker-map.json", tokenizer)
            checkpoint = Path(args.run) / "checkpoint" if (Path(args.run) / "checkpoint").exists() else Path(args.run)
            model, _ = load_checkpoint(checkpoint, config=value, device=args.device or ("mps" if torch.backends.mps.is_available() else "cpu"), tokenizer=tokenizer, expected_marker_map=markers)
            from .evaluation.predictors import LocalPredictor
            rows, _manifest, _hash = _load_cli_split(args.data, suite, "test", allow_test=True, _locked_test=True)
            return rows, LocalPredictor(model, tokenizer, markers, state_cap=value.training.state_cap,
                                        branch_cap=value.training.branch_cap, packed_cap=value.training.packed_cap)
        result = run_locked(selection=args.selection, suites=suites, data_root=args.data, output=args.out,
                            ledger=args.ledger, load_test=load_test, code_sha="gev-locked", run=args.run)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    elif args.command == "register-candidate":
        from .evaluation.locked import register_candidate
        result = register_candidate(run=args.run, study=args.study, out=args.out)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    else:
        value = _config(args.config)
        result = inspect_model(value, args.output_root or value.runtime.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result.get("status") != "verified":
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
