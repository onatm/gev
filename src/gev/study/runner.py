"""Multi-seed study orchestration.

The planner is intentionally independent of model loading.  The runner uses
fresh ``gev train``/``gev evaluate`` child processes so one GPU and one model are
owned by one trial at a time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ..configuration.config import load_config
from ..data.access import load_verified_split
from ..data.suites import REQUIRED_FILE_HASHES
from ..configuration.resolved import resolve_experiment_config
from ..models.policy import development_only_family
from .config import write_seed_config
from .process import run_child
from ..training.batching import variant_count_for_request

KEV_SHA = "08ab0b87d27cb5577a3b371ad7ed4e4686b0502b"
REFERENCE_IDENTITIES = {
    "decision_v7_manifest": "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2",
    "kev_current_adapter": "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8",
    "kev_v7_base_adapter": "c917edefdfd72b3e9ba71455584700acc70595f6",
    "qwen_base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
}


def _sha_bytes(value) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str | None:
    return _sha_bytes(path.read_bytes()) if path.exists() else None


def _requests(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verified(root: Path, suite: str, split: str) -> tuple[list[dict], dict]:
    """Verify bytes, counts, and the parent identity before planning."""
    rows, manifest, _manifest_hash = load_verified_split(root, suite, split)
    return rows, manifest


def plan(config_path: str | Path, *, seeds=(0, 1, 2), data="data", smoke=False) -> dict:
    if not seeds:
        raise ValueError("at least one study seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(tuple(seeds)):
        raise ValueError("seeds must be unique")
    config = load_config(config_path)
    resolved = resolve_experiment_config(config)
    resolved.validate_policy_before_data_access()
    root = Path(data)
    development_only = development_only_family(resolved.model.family.family_id)
    train = None
    verified_splits = {}
    try:
        is_child = (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()).get("smoke_only")
        splits = ((("decision-v7", "train"), ("decision-v7", "development"))
                  if is_child or development_only else
                  (("decision-v7", "train"), ("decision-v7", "calibration"),
                   ("decision-v7", "development"), ("transfer-v4", "development")))
        for suite, split in splits:
            verified_splits[f"{suite}:{split}"] = _verified(root, suite, split)
        train = verified_splits["decision-v7:train"][0]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot plan: verified {data} suites are required ({exc})") from exc
    manifest = verified_splits["decision-v7:train"][1]
    actual = []
    for seed in seeds:
        if train is None:
            variants = None
        else:
            variants = sum(variant_count_for_request(r, seed=seed, epoch=e,
                         p_none=config.training.p_none, p_none_distract=config.training.p_none_distract,
                         p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair)
                           for e in range(config.training.epochs) for r in train)
        actual.append({"seed": seed, "records": len(train) * config.training.epochs if train else manifest["files"]["train.jsonl"]["records"] * config.training.epochs,
                       "variants": variants, "source_exposures": manifest["files"]["train.jsonl"]["records"] * config.training.epochs,
                       "steps": config.training.epochs * (((len(train) if train else manifest["files"]["train.jsonl"]["records"]) + config.training.logical_batch - 1) // config.training.logical_batch)})
    recipe = resolved.recipe
    recipe_hash = resolved.recipe_sha256
    allowed_splits = {"train", "calibration", "development"}
    if root.joinpath("manifest.json").exists() and manifest.get("smoke_only"):
        files = {"decision-v7": {split: {"expected_sha256": info["sha256"], "present_sha256": info["sha256"], "records": info["records"]}
                                  for split, info in manifest["files"].items()}}
    elif development_only:
        files = {"decision-v7": {
            split: {"expected_sha256": REQUIRED_FILE_HASHES["decision-v7"][split],
                    "present_sha256": _file_sha(root / "decision-v7" / f"{split}.jsonl")}
            for split in ("train", "development")}}
    else:
        files = {suite: {split: {"expected_sha256": digest, "present_sha256": _file_sha((root / suite / f"{split}.jsonl"))}
                        for split, digest in values.items() if split in allowed_splits}
             for suite, values in REQUIRED_FILE_HASHES.items() for _ in [0]}
    return {"status": "planned", "config": str(config_path), "config_sha256": recipe_hash,
            "scientific_recipe": recipe, "resolved_config": dataclasses.asdict(config),
            "study_id": config.experiment_id, "protocol": dataclasses.asdict(config.protocol),
            "model_family": resolved.model.family.family_id,
            "model_output_id": resolved.model.output_model_id,
            "backend": resolved.model.backend.backend_id,
            "kev_sha": KEV_SHA, "reference_identities": REFERENCE_IDENTITIES,
            "model": {"name": config.model.name, "revision": config.model.revision,
                      "family": resolved.model.family.family_id, "backend": resolved.model.backend.backend_id,
                      "output_model_id": resolved.model.output_model_id,
                      "dtype": config.training.dtype,
                      "device": config.runtime.device, "attention": config.runtime.attn_implementation},
            "data": files, "seeds": list(seeds), "trials": actual,
            "recipe": {"epochs": config.training.epochs, "logical_batch": config.training.logical_batch,
                       "microbatch": config.training.microbatch, "p_none_pair": config.training.p_none_pair,
                       "smoke": smoke},
            "selection_rule": ("completed decision-v7 development macro NLL" if development_only
                               else "completed transfer-clean micro accuracy; lower transfer Brier; dev macro NLL"),
            "no_test_loading": True, "smoke_child": bool(root.joinpath("manifest.json").exists() and manifest.get("smoke_only")),
            "runner": "sequential isolated gev train/eval child processes"}


def _report_summary(report: dict) -> dict:
    """Keep study stdout/result.json compact; the complete report stays on disk."""
    summary = {key: report[key] for key in
               ("coverage", "mechanism_checks", "provenance") if key in report}
    if "clean" in report:
        summary["clean"] = {key: report["clean"][key] for key in
                             ("acc", "nll", "brier", "ece") if key in report["clean"]}
    if "tasks" in report:
        summary["tasks"] = {name: {"nll": value["nll"]}
                             for name, value in report["tasks"].items()
                             if "nll" in value}
    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(config_path: str | Path, *, seeds=(0, 1, 2), data="data", out="runs/study",
        dry_run=False, max_steps=None, smoke=False, existing_run=None) -> dict:
    result = plan(config_path, seeds=seeds, data=data, smoke=smoke)
    result["operational_controls"] = {
        "output_path": str(out), "max_steps": max_steps,
        "smoke": smoke, "existing_run": existing_run,
    }
    if dry_run:
        print(json.dumps(result, indent=2, sort_keys=True)); return result
    root = Path(out)
    if root.exists(): raise FileExistsError(f"refusing to overwrite experiment: {root}")
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    trials = []
    config_path = Path(config_path).resolve()
    data = str(Path(data).resolve())
    existing_run = str(Path(existing_run).resolve()) if existing_run else None
    configs = root / "configs"
    configs.mkdir()
    started_at = _utc_now()
    study_started = time.monotonic()
    current = {"state": "running", "seed": None, "seed_index": None, "seed_total": len(seeds),
               "stage": "starting", "step": None, "total_steps": None, "latest_loss": None,
               "started_at": started_at, "updated_at": started_at, "elapsed_seconds": 0.0,
               "recent_stage": None, "log_path": None}
    planned_steps = {trial["seed"]: trial["steps"] for trial in result["trials"]}
    base_config = load_config(config_path)
    development_only = development_only_family(base_config.model.family)
    training_config = base_config.training
    result.update(status="running", trials=[])

    def save_status(**changes):
        current.update(changes)
        current["updated_at"] = _utc_now()
        _atomic_json(root / "status.json", current)

    def save_result():
        result["trials"] = list(trials)
        _atomic_json(root / "result.json", result)

    save_status()
    save_result()
    interrupted = False
    for index, seed in enumerate(seeds, 1):
        seed_log_dir = root / "logs" / f"seed-{seed}"
        seed_log_dir.mkdir(parents=True)
        trial = root / f"seed-{seed}"
        full_study = (max_steps is None and not smoke and not existing_run
                      and training_config.max_steps is None and not result.get("smoke_child"))
        cfg = write_seed_config(base_config, seed, configs,
                                save_every=100 if full_study else None)
        legacy = existing_run is not None
        stage_started = time.monotonic()
        active = {"stage": None, "log_path": None, "latest": {}, "stage_total": None}

        def execute(stage, command):
            log_path = seed_log_dir / f"{stage}.log"
            progress_label = f"seed {seed} ({index}/{len(seeds)}) {stage}"
            active.update(stage=stage, log_path=str(log_path), latest={}, stage_total=None)
            print(f"[{progress_label}] starting; log {log_path}", flush=True)
            initial_progress = {}
            if stage == "train":
                actual_steps = max_steps if max_steps is not None else training_config.max_steps
                if actual_steps is None:
                    actual_steps = planned_steps.get(seed)
                initial_progress = {"step": 0, "total_steps": actual_steps,
                                    "latest_loss": None,
                                    "full_training_steps": planned_steps.get(seed)}
            save_status(state="running", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                        elapsed_seconds=time.monotonic() - study_started, log_path=str(log_path),
                        stage=stage, recent_stage={"stage": stage, "outcome": "running",
                        "log_path": str(log_path)}, **initial_progress)

            def update(progress, elapsed, heartbeat):
                active["latest"] = dict(progress)
                if stage == "train":
                    step = progress.get("step")
                    total_steps = progress.get("total_steps")
                    latest_loss = progress.get("latest_loss")
                    progress_fields = {
                        "step": current.get("step") if step is None else step,
                        "total_steps": (current.get("total_steps") if total_steps is None
                                        else total_steps),
                        "latest_loss": (current.get("latest_loss") if latest_loss is None
                                        else latest_loss),
                    }
                else:
                    progress_fields = {}
                save_status(state="running", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                            elapsed_seconds=time.monotonic() - study_started, stage_elapsed_seconds=elapsed,
                            log_path=str(log_path), recent_stage={"stage": stage, "outcome": "running",
                            "log_path": str(log_path)}, stage=stage, **progress_fields)

            try:
                child = run_child(command, log_path=log_path, on_update=update,
                                  progress_label=progress_label, stage=stage)
            except KeyboardInterrupt:
                save_status(state="interrupted", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                            stage=stage, elapsed_seconds=time.monotonic() - study_started,
                            stage_elapsed_seconds=time.monotonic() - stage_started,
                            log_path=str(log_path), recent_stage={"stage": stage, "outcome": "interrupted",
                            "log_path": str(log_path)})
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                stage_result = {"stage": stage, "outcome": "failed", "exit_code": None,
                                "error": error, "log_path": str(log_path)}
                save_status(state="running", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                            stage=stage, elapsed_seconds=time.monotonic() - study_started,
                            log_path=str(log_path), recent_stage=stage_result)
                print(f"[{progress_label}] failed: {error}; log {log_path}", flush=True)
                return stage_result
            error = None
            if child["returncode"]:
                lines = [line.strip() for line in child["output_excerpt"].splitlines() if line.strip()]
                error = lines[-1] if lines else f"child exited with status {child['returncode']}"
                print(f"[{progress_label}] failed (exit {child['returncode']}): {error}; log {log_path}", flush=True)
            else:
                print(f"[{progress_label}] finished in {child['elapsed_seconds']:.1f}s", flush=True)
            stage_result = {"stage": stage, "outcome": "failed" if error else "completed",
                            "exit_code": child["returncode"], "log_path": str(log_path)}
            if error:
                excerpt = child["output_excerpt"][-4000:]
                stage_result.update(error=error, output_excerpt=excerpt, stderr_excerpt=excerpt)
            final_progress = ({"step": child.get("step"), "total_steps": child.get("total_steps"),
                               "latest_loss": child.get("latest_loss")} if stage == "train" else {})
            save_status(state="running", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                        elapsed_seconds=time.monotonic() - study_started,
                        stage_elapsed_seconds=child["elapsed_seconds"],
                        log_path=str(log_path), recent_stage=stage_result, stage=stage,
                        **final_progress)
            return stage_result

        try:
            if existing_run:
                command = [sys.executable, "-m", "gev", "evaluate", existing_run,
                           "--suite", "decision-v7", "--split", "development",
                           "--data", data, "--temperature", "1", "--out", str(trial / "development")]
                stage_result = execute("development", command)
                stages = [stage_result]
            else:
                train_cmd = [sys.executable, "-m", "gev", "train", str(cfg),
                             "--data", data,
                             "--out", str(trial)]
                if max_steps is not None: train_cmd += ["--max-steps", str(max_steps)]
                stages = [execute("train", train_cmd)]
                if stages[-1]["outcome"] == "completed":
                    evaluations = (("development", "decision-v7", "development"),) if development_only else (
                        ("calibration", "decision-v7", "calibration"),
                        ("development", "decision-v7", "development"),
                        ("transfer", "transfer-v4", "development"))
                    for name, suite, split in evaluations:
                        eval_cmd = [sys.executable, "-m", "gev", "evaluate", str(trial),
                                    "--suite", suite, "--split", split,
                                    "--data", data, "--temperature", "1", "--out", str(trial / name)]
                        stage_result = execute(name, eval_cmd)
                        stages.append(stage_result)
                        if stage_result["outcome"] == "failed":
                            break
                    if (not development_only
                            and all(stage["outcome"] == "completed" for stage in stages)):
                        calibration_cmd = [sys.executable, "-m", "gev", "calibrate", "--run",
                                           str(trial / "calibration"), "--protocol", "kev-screening",
                                           "--out", str(trial / "calibration" / "calibration.json")]
                        stages.append(execute("temperature", calibration_cmd))
            failed_stage = next((stage for stage in stages if stage["outcome"] == "failed"), None)
            if failed_stage:
                trial_result = {"seed": seed, "completed": False, "eligible": False,
                                "legacy_existing_run": legacy, "status": "failed",
                                "stage": failed_stage["stage"], "exit_code": failed_stage["exit_code"],
                                "error": failed_stage["error"], "log_path": failed_stage["log_path"],
                                "path": str(trial)}
                if "output_excerpt" in failed_stage:
                    trial_result["output_excerpt"] = failed_stage["output_excerpt"]
                    trial_result["stderr_excerpt"] = failed_stage["stderr_excerpt"]
                trials.append(trial_result)
                _atomic_json(seed_log_dir / "result.json", trial_result)
                if trial.exists(): _atomic_json(trial / "result.json", trial_result)
                save_result()
                continue
            training_metrics = {}
            metrics_path = trial / "training_metrics.json"
            if metrics_path.exists(): training_metrics = json.loads(metrics_path.read_text())
            report_names = ("development",) if development_only else ("calibration", "development", "transfer")
            report_values = [json.loads((trial / name / "report.json").read_text()) for name in
                             report_names
                             if (trial / name / "report.json").exists()]
            reports_complete = len(report_values) == len(report_names) and all(
                t.get("coverage", {}).get("rejected_records", 1) == 0 and
                t.get("coverage", {}).get("truncated_records", 1) == 0 and
                t.get("coverage", {}).get("evaluated_records") == t.get("coverage", {}).get("requested_records") and
                t.get("coverage", {}).get("evaluated_questions") == t.get("coverage", {}).get("requested_questions") and
                t.get("mechanism_checks", {}).get("passed", False) for t in report_values)
            complete = bool(training_metrics.get("complete")) and not smoke and reports_complete
            trial_result = {"seed": seed, "completed": complete,
                            "eligible": complete and not legacy and not result.get("smoke_child"),
                            "status": "completed" if complete else "diagnostic",
                            "legacy_existing_run": legacy, "path": str(trial)}
            if not legacy:
                reports = {}
                for name in report_names:
                    report_path = trial / name / "report.json"
                    if report_path.exists(): reports[name] = _report_summary(json.loads(report_path.read_text()))
                trial_result["reports"] = reports
            trials.append(trial_result)
            _atomic_json(seed_log_dir / "result.json", trial_result)
            if trial.exists(): _atomic_json(trial / "result.json", trial_result)
            save_result()
        except KeyboardInterrupt:
            interrupted = True
            trial_result = {"seed": seed, "completed": False, "eligible": False,
                            "status": "interrupted", "stage": current.get("stage"),
                            "log_path": current.get("log_path"), "path": str(trial)}
            trials.append(trial_result)
            _atomic_json(seed_log_dir / "result.json", trial_result)
            if trial.exists(): _atomic_json(trial / "result.json", trial_result)
            save_status(state="interrupted", recent_stage=current.get("recent_stage"))
            save_result()
            break
        except Exception as exc:
            stage = current.get("stage") or "orchestration"
            error = f"{type(exc).__name__}: {exc}"
            log_path = current.get("log_path")
            failure = {"stage": stage, "outcome": "failed", "exit_code": None,
                       "error": error, "log_path": log_path}
            save_status(state="running", seed=seed, seed_index=index - 1, seed_total=len(seeds),
                        stage=stage, elapsed_seconds=time.monotonic() - study_started,
                        log_path=log_path, recent_stage=failure)
            trial_result = {"seed": seed, "completed": False, "eligible": False,
                            "status": "failed", "stage": stage, "exit_code": None,
                            "error": error, "log_path": log_path, "path": str(trial)}
            trials.append(trial_result)
            _atomic_json(seed_log_dir / "result.json", trial_result)
            if trial.exists(): _atomic_json(trial / "result.json", trial_result)
            save_result()

    if interrupted:
        result["status"] = "interrupted"
        result["promotion"] = {"selected_seed": None, "rule": result["selection_rule"], "eligible_seeds": []}
        save_status(state="interrupted")
        save_result()
        print(f"Experiment interrupted; status: {root / 'status.json'}", flush=True)
        raise SystemExit(130)

    if any(t.get("status") == "failed" for t in trials):
        result["status"] = "failed"
    else:
        result["status"] = "completed" if all(t.get("completed") for t in trials) else "diagnostic"
    eligible = [t for t in trials if t.get("eligible") and
                (t.get("reports", {}).get("development", {}).get("clean") if development_only
                 else t.get("reports", {}).get("transfer", {}).get("clean"))]
    def dev_macro_nll(trial):
        tasks = trial["reports"].get("development", {}).get("tasks", {})
        values = [value["nll"] for name, value in tasks.items() if not name.startswith("unknowable_")]
        return statistics.fmean(values) if values else float("inf")
    if development_only:
        eligible.sort(key=dev_macro_nll)
    else:
        eligible.sort(key=lambda t: (-t["reports"]["transfer"]["clean"]["acc"],
                                    t["reports"]["transfer"]["clean"]["brier"], dev_macro_nll(t)))
    result["promotion"] = {"selected_seed": eligible[0]["seed"] if eligible else None,
                            "rule": result["selection_rule"], "eligible_seeds": [t["seed"] for t in eligible]}
    result["aggregate"] = {}
    for name in (("development",) if development_only else ("development", "transfer")):
        values = [t["reports"][name]["clean"] for t in eligible
                  if t.get("reports", {}).get(name, {}).get("clean")]
        if values:
            result["aggregate"][name] = {metric: {"mean": statistics.fmean(v[metric] for v in values),
                "std": statistics.stdev(v[metric] for v in values) if len(values) > 1 else 0.0}
                for metric in ("acc", "nll", "brier", "ece")}
    save_result()
    save_status(state="failed" if result["status"] == "failed" else "completed",
                study_status=result["status"], stage="finished",
                elapsed_seconds=time.monotonic() - study_started,
                recent_stage=current.get("recent_stage"))
    terminal_result = {key: result[key] for key in ("status", "promotion", "aggregate")}
    terminal_result["trials"] = [
        {key: trial[key] for key in
         ("seed", "status", "eligible", "stage", "exit_code", "error", "log_path")
         if key in trial}
        for trial in trials
    ]
    terminal_result["result_path"] = str(root / "result.json")
    terminal_result["status_path"] = str(root / "status.json")
    print(json.dumps(terminal_result, indent=2, sort_keys=True))
    if result["status"] == "failed": raise SystemExit(1)
    return result
