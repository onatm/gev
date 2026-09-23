"""Fail-closed, once-only protocol for the frozen test suites.

This module is intentionally boring: all checks which do not require test
examples happen before the reservation, and the reservation and its ledger
updates use one lock.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .benchmark import prediction_rows, summarize
from ..data.suites import KNOWN_SUITES, load_manifest, manifest_digest

V7_TRAIN_SHA256 = "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad"
V7_MANIFEST_SHA256 = "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
NIGHT2_RECORDS = 3425
NIGHT2_STEPS = 429
FULL_V7_SOURCE_RECORDS = 12576
FULL_V7_PROCESSED_RECORDS = 25152
FULL_V7_STEPS = 3144


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(model_fingerprint: str, suite_manifest: str) -> str:
    value = {"model_fingerprint": model_fingerprint, "suite_manifest": suite_manifest}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value, *, lines: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if lines:
        text = "".join(json.dumps(item, allow_nan=False, sort_keys=True) + "\n" for item in value)
    else:
        text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint(path: str | Path) -> Path:
    root = Path(path)
    return root / "checkpoint" if (root / "checkpoint").is_dir() else root


def _actual_checkpoint_fingerprint(checkpoint: Path) -> tuple[str, dict]:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    for name, field in (("adapter_model.safetensors", "adapter_sha256"),
                        ("pointer.safetensors", "pointer_sha256")):
        file = checkpoint / name
        if not file.exists() or metadata.get(field) != _sha(file):
            raise ValueError(f"checkpoint {name} hash does not match metadata")
    from ..checkpoint import checkpoint_fingerprint
    return checkpoint_fingerprint(checkpoint), metadata


def _training_lineage(metadata: dict) -> str:
    """Return the accepted full-training lineage, never trusting ``complete`` alone."""
    training = metadata.get("training", {})
    config = metadata.get("config", {})
    recipe = config.get("training", {})
    full_recipe = (config.get("experiment_id") == "gemma3-1b-v7" and
                   recipe.get("epochs") == 2 and recipe.get("logical_batch") == 8 and
                   recipe.get("context_length") == 384 and recipe.get("p_none_pair", .25) == .25 and
                   recipe.get("seed") in {0, 1, 2})
    full_v7 = (metadata.get("source_sha256") == V7_TRAIN_SHA256 and
               metadata.get("manifest_sha256") == V7_MANIFEST_SHA256 and
                training.get("complete") is True and training.get("logical_steps") == FULL_V7_STEPS and
                training.get("processed_records") == FULL_V7_PROCESSED_RECORDS and
                 training.get("source_count") == FULL_V7_SOURCE_RECORDS and full_recipe and
                 not metadata.get("diagnostic_smoke_init") and not metadata.get("smoke_only") and
                 not metadata.get("continuation", {}).get("diagnostic_smoke_init"))
    continuation = metadata.get("continuation", {})
    prepared = continuation.get("prepared", {})
    full_night2 = (config.get("experiment_id") == "gemma3-1b-night2" and
                    training.get("complete") is True and prepared.get("records") == NIGHT2_RECORDS and
                    prepared.get("logical_steps") == NIGHT2_STEPS and
                    prepared.get("night2_records") == 1425 and prepared.get("replay_records") == 2000 and
                    prepared.get("source_hashes", {}).get("decision-v7/train.jsonl") == V7_TRAIN_SHA256 and
                    prepared.get("source_hashes", {}).get("night2/dates_unknowable.jsonl") and
                    prepared.get("replay_count") == 2000 and
                    continuation.get("init_checkpoint_fingerprint") and
                    continuation.get("initializer_kind") == "full-v7" and
                    continuation.get("diagnostic_smoke_init") is False and
                    not metadata.get("diagnostic_smoke_init") and not metadata.get("smoke_only"))
    if full_v7:
        return "full-v7"
    if full_night2:
        return "full-night2"
    raise ValueError("checkpoint lacks a verified full v7 or full night2 training lineage")


def _temperature(selection: dict, checkpoint_meta: dict | None) -> float:
    value = selection.get("temperature", 1.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("selection temperature must be finite and positive")
    stored = (checkpoint_meta or {}).get("temperature", 1.0)
    if not isinstance(stored, (int, float)) or not math.isfinite(stored) or stored <= 0 or float(stored) != float(value):
        raise ValueError("selection temperature does not match checkpoint")
    fit = selection.get("temperature_fit")
    if fit is not None:
        if (not isinstance(fit, dict) or fit.get("split") != "development" or
                fit.get("suite", "decision-v7") != "decision-v7"):
            raise ValueError("locked temperature must reference a development fit")
        if fit.get("temperature") is not None and float(fit["temperature"]) != float(value):
            raise ValueError("temperature fit and selection temperature differ")
    return float(value)


def _manifest_info(suite: str) -> tuple[str, dict]:
    if suite not in KNOWN_SUITES:
        raise ValueError(f"unknown suite: {suite}")
    manifest = load_manifest(suite)
    return manifest_digest(suite), manifest


def _validate_selection(selection: dict, suites: tuple[str, ...], actual_fingerprint: str | None,
                       checkpoint_meta: dict | None) -> tuple[dict[str, str], float]:
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("suites must be a non-empty list of unique suite names")
    if not isinstance(selection.get("model_fingerprint"), str) or not selection["model_fingerprint"]:
        raise ValueError("selection requires model_fingerprint")
    if actual_fingerprint is not None and selection["model_fingerprint"] != actual_fingerprint:
        raise ValueError("selection model fingerprint does not match checkpoint weights")
    if checkpoint_meta is not None:
        continuation = checkpoint_meta.get("continuation", {})
        if (checkpoint_meta.get("diagnostic_smoke_init") or checkpoint_meta.get("smoke_only") or
                continuation.get("diagnostic_smoke_init")):
            # A continuation may carry this marker, but only its verified
            # full-night2 lineage can make it acceptable.
            if checkpoint_meta.get("continuation", {}).get("prepared", {}).get("records") != NIGHT2_RECORDS:
                raise ValueError("smoke checkpoints cannot be evaluated as locked candidates")
        _training_lineage(checkpoint_meta)
    selected_suites = selection.get("suites")
    if not isinstance(selected_suites, dict):
        raise ValueError("selection requires suite metadata")
    hashes: dict[str, str] = {}
    for suite in suites:
        manifest_hash, manifest = _manifest_info(suite)
        item = selected_suites.get(suite)
        if not isinstance(item, dict):
            raise ValueError(f"selection has no suite {suite}")
        if item.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"selection manifest hash mismatch: {suite}")
        expected = manifest["files"]["test.jsonl"]["sha256"]
        if item.get("data_sha256") != expected:
            raise ValueError(f"selection test-data hash mismatch: {suite}")
        hashes[suite] = manifest_hash
    return hashes, _temperature(selection, checkpoint_meta)


def _reserve(ledger: Path, reservations: list[dict]) -> None:
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = _entries(ledger)
        old_keys = {entry.get("key") for entry in old}
        if any(item["key"] in old_keys for item in reservations):
            raise ValueError("locked evaluation key is already reserved")
        _write_entries(ledger, reservations)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _finish(ledger: Path, entries: list[dict]) -> None:
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _write_entries(ledger, entries)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_locked(*, selection: str | Path, suites: tuple[str, ...], data_root: str | Path,
               output: str | Path, ledger: str | Path,
               load_test: Callable[[str], tuple[list[dict], Callable]],
               code_sha: str = "working-tree", run: str | Path | None = None,
               checkpoint_meta: dict | None = None,
               model_fingerprint: str | None = None) -> dict:
    """Reserve every requested suite before invoking the injected test loader."""
    del data_root  # The callback owns the deliberately late test-data access.
    selection_path = Path(selection)
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    actual = model_fingerprint
    meta = checkpoint_meta
    if run is not None:
        checkpoint = _checkpoint(run)
        actual, meta = _actual_checkpoint_fingerprint(checkpoint)
        recorded_checkpoint = selected.get("checkpoint", {}).get("path")
        if recorded_checkpoint and Path(recorded_checkpoint).resolve() != checkpoint.resolve():
            raise ValueError("selection is registered for a different checkpoint")
        recorded_metadata = selected.get("checkpoint", {}).get("metadata_sha256")
        if recorded_metadata and recorded_metadata != _sha(checkpoint / "metadata.json"):
            raise ValueError("selection checkpoint metadata does not match checkpoint")
    elif actual is None or meta is None:
        raise ValueError("locked evaluation requires an actual checkpoint or verified model metadata and fingerprint")
    study = selected.get("study")
    if not isinstance(study, dict) or not study.get("path") or not study.get("sha256"):
        raise ValueError("selection requires study provenance")
    study_path = Path(study["path"]).resolve()
    if not study_path.is_file() or _sha(study_path) != study["sha256"]:
        raise ValueError("selection study provenance is stale or missing")
    manifests, temperature = _validate_selection(selected, suites, actual, meta)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite locked evaluation: {destination}")
    fingerprint = selected["model_fingerprint"]
    ledger_path = Path(ledger)
    started = []
    for suite in suites:
        started.append({"key": _key(fingerprint, manifests[suite]), "status": "started", "started_at": _now(),
                        "model_fingerprint": fingerprint, "suite": suite, "suite_manifest": manifests[suite],
                        "selection": str(selection_path), "code_sha": code_sha,
                        "selected_weight_sha256": fingerprint, "selection_model_fingerprint": fingerprint,
                        "protocol_version": "locked-eval-v7"})
    _reserve(ledger_path, started)
    completed: list[dict] = []
    try:
        destination.mkdir(parents=True)
        for reservation in started:
            suite = reservation["suite"]
            records, predictor = load_test(suite)  # first test touch, after all reservations
            if not records:
                raise ValueError(f"{suite} test loader returned no records")
            if getattr(predictor, "temperature", 1.0) != 1.0:
                raise ValueError("locked predictor must be configured for raw temperature 1")
            requested_questions = sum(len(record.get("questions", {})) for record in records)
            rows, prediction_items = [], []
            prediction_path = destination / suite / "predictions.jsonl"
            (destination / suite).mkdir(parents=True)
            for record in records:
                prediction = predictor(record)
                served_temperature = prediction.get("inference_temperature", 1.0)
                if served_temperature != 1.0:
                    raise ValueError("locked predictor must serve raw temperature 1")
                new_rows = prediction_rows(record, prediction)
                item = {"id": record["_meta"]["id"], "prediction": prediction, "rows": new_rows}
                rows.extend(new_rows); prediction_items.append(item)
            if len(rows) != requested_questions:
                raise ValueError(f"{suite} row/question coverage mismatch")
            report = summarize(rows, temperature)
            report.update(provenance={"suite": suite, "split": "test", "suite_sha256": manifests[suite],
                                      "source_sha256": selected["suites"][suite]["data_sha256"],
                                      "locked_ledger_key": reservation["key"], "model_fingerprint": fingerprint},
                          coverage={"requested_records": len(records), "requested_questions": requested_questions,
                                     "evaluated_records": len(records), "evaluated_questions": len(rows),
                                     "rejected_records": 0, "truncated_records": 0})
            suite_out = destination / suite
            _atomic_json(prediction_path, prediction_items, lines=True)
            _atomic_json(suite_out / "rows.json", rows)
            report["rows_sha256"] = _sha(suite_out / "rows.json")
            report["provenance"]["rows_sha256"] = report["rows_sha256"]
            report["coverage"]["complete_questions"] = report["coverage"]["evaluated_questions"] == requested_questions
            _atomic_json(suite_out / "report.json", report)
            completed.append({**reservation, "status": "complete", "completed_at": _now(), "output": str(suite_out),
                              "rows_sha256": _sha(suite_out / "rows.json")})
    except Exception as exc:
        failed = [{**reservation, "status": "failed", "failed_at": _now(),
                   "error": f"{type(exc).__name__}: {exc}"} for reservation in started]
        _finish(ledger_path, failed)
        raise
    _finish(ledger_path, completed)
    return {"status": "complete", "output": str(destination), "reservations": completed}


def register_candidate(*, run: str | Path, study: str | Path, out: str | Path) -> dict:
    """Create a pre-registration record from verified checkpoint/run metadata."""
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selection: {destination}")
    checkpoint = _checkpoint(run)
    _, metadata = _actual_checkpoint_fingerprint(checkpoint)
    lineage = _training_lineage(metadata)
    study_path = Path(study).resolve(); study_value = json.loads(study_path.read_text(encoding="utf-8"))
    promotion = study_value.get("promotion")
    trials = study_value.get("trials")
    if not isinstance(promotion, dict) or not isinstance(trials, list):
        raise ValueError("study must be a study result with promotion and trials")
    selected_seed = promotion.get("selected_seed")
    selected = next((trial for trial in trials if trial.get("seed") == selected_seed), None)
    if not selected or selected.get("eligible") is not True or selected.get("completed") is not True:
        raise ValueError("study selected trial is not completed and eligible")
    if selected_seed not in {0, 1, 2}:
        raise ValueError("study selected seed must be one of 0, 1, or 2")
    selected_path = _checkpoint(selected.get("path", ""))
    if selected_path.resolve() != checkpoint.resolve():
        raise ValueError("--run is not the study's selected trial")
    for name in ("calibration", "development", "transfer"):
        report = selected.get("reports", {}).get(name)
        if not isinstance(report, dict) or report.get("coverage", {}).get("rejected_records", 1) or \
                report.get("coverage", {}).get("evaluated_records") != report.get("coverage", {}).get("requested_records") or \
                not report.get("mechanism_checks", {}).get("passed", False):
            raise ValueError(f"selected trial has incomplete {name} report")
    fingerprint, _ = _actual_checkpoint_fingerprint(checkpoint)
    training = metadata.get("training", {})
    selection = {"protocol": "locked-eval-v7", "model_fingerprint": fingerprint,
                 "temperature": float(metadata.get("temperature", 1.0)),
                 "checkpoint": {"path": str(checkpoint), "metadata_sha256": _sha(checkpoint / "metadata.json")},
                 "study": {"path": str(study_path), "sha256": _sha(study_path), "selected_seed": selected_seed},
                 "training": {"source_sha256": training.get("source_sha256") or metadata.get("source_sha256"),
                               "manifest_sha256": metadata.get("manifest_sha256"),
                               "complete": metadata.get("training", {}).get("complete"),
                               "lineage": lineage}, "suites": {}}
    for suite in ("decision-v7", "transfer-v4"):
        manifest_hash, manifest = _manifest_info(suite)
        selection["suites"][suite] = {"manifest_sha256": manifest_hash,
                                       "data_sha256": manifest["files"]["test.jsonl"]["sha256"]}
    if metadata.get("temperature_fit"):
        selection["temperature_fit"] = metadata["temperature_fit"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection
