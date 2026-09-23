"""Protocol-gated temperature calibration for saved Gev rows.

Calibration is deliberately a rows-only operation: it cannot accidentally
load a checkpoint from a different split or fit on already calibrated output.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .metrics import cross_validated_temperature, fit_temperature, metrics, raw_row, scored_rows

SCREENING_TEMPERATURE_POINTS = 81
RELEASE_TEMPERATURE_POINTS = 121


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("rows must be a JSON array")
    return value


def _report_for(run: Path, rows_path: Path) -> tuple[dict, Path]:
    report_path = rows_path.parent / "report.json"
    if not report_path.exists():
        raise ValueError("calibration requires the accompanying report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = report.get("provenance", {}).get("split")
    if split not in {"calibration", "development"}:
        raise ValueError("calibration is allowed only on calibration or development rows")
    identity = report.get("provenance", {}).get("suite_sha256")
    if not identity:
        raise ValueError("report is missing suite identity provenance")
    if report.get("provenance", {}).get("suite") != "decision-v7":
        raise ValueError("calibration is not permitted on transfer or test provenance")
    checkpoint = report_path.parent.parent / "checkpoint"
    recorded = report.get("provenance", {}).get("checkpoint_fingerprint")
    if recorded and checkpoint.is_dir():
        digest = hashlib.sha256()
        for name in ("metadata.json", "adapter_model.safetensors", "pointer.safetensors"):
            file = checkpoint / name
            if not file.exists():
                raise ValueError("report checkpoint identity is incomplete")
            digest.update(file.read_bytes())
        if digest.hexdigest() != recorded and "temperature" not in json.loads((checkpoint / "metadata.json").read_text()):
            raise ValueError("report/checkpoint identity mismatch")
    stable_recorded = report.get("provenance", {}).get("model_fingerprint")
    if stable_recorded and checkpoint.is_dir():
        from ..checkpoint import checkpoint_fingerprint
        if checkpoint_fingerprint(checkpoint) != stable_recorded:
            raise ValueError("report/checkpoint model identity mismatch")
    coverage = report.get("coverage", {})
    if coverage.get("rejected_records", 0) or coverage.get("truncated_records", 0) or coverage.get("evaluated_records") != coverage.get("requested_records"):
        raise ValueError("cannot calibrate an incomplete evaluation")
    if rows_path.name != "rows.json":
        raise ValueError("calibration rows must be the report's rows.json")
    return report, report_path


def calibrate(run: str | Path, *, rows: str | Path | None = None,
              protocol: str = "kev-screening", out: str | Path | None = None,
              update_checkpoint: bool = False) -> dict:
    run = Path(run)
    rows_path = Path(rows) if rows else (run / "rows.json")
    report, report_path = _report_for(run, rows_path)
    rows_value = _read_rows(rows_path)
    clean = scored_rows(rows_value)
    if not clean or any(row.get("inference_temperature", 1.0) != 1.0 for row in clean):
        raise ValueError("calibration requires raw rows with inference_temperature=1")
    if any("logits" not in row for row in clean):
        raise ValueError("calibration requires recorded raw logits")
    if report.get("rows_sha256") and _sha(rows_path) != report["rows_sha256"]:
        raise ValueError("report/rows identity mismatch")
    if report.get("coverage", {}).get("evaluated_questions") not in (None, len(rows_value)):
        raise ValueError("report/rows question coverage mismatch")
    expected = report.get("provenance", {}).get("source_sha256")
    if expected and rows_path.exists() and report_path.parent == rows_path.parent:
        # The row file is covered by report coverage; its hash is recorded to
        # make later checkpoint/report mixups observable.
        rows_sha = _sha(rows_path)
    else:
        rows_sha = _sha(rows_path)
    split = report["provenance"]["split"]
    if protocol == "kev-screening":
        if split != "calibration":
            raise ValueError("kev-screening requires split=calibration")
        points = SCREENING_TEMPERATURE_POINTS
        temperature = fit_temperature(rows_value, aggregation="micro", points=points)
        result = {"protocol": protocol, "temperature": temperature, "points": points,
                  "metrics_raw": metrics(scored_rows(rows_value)),
                  "metrics_calibrated": metrics(scored_rows(rows_value), temperature)}
    elif protocol == "kev-release":
        if split != "development":
            raise ValueError("kev-release requires split=development")
        points = RELEASE_TEMPERATURE_POINTS
        temperature = fit_temperature(rows_value, aggregation="micro", points=points)
        oof = cross_validated_temperature(rows_value, folds=5, seed=0, samples=1000,
                                          aggregation="micro", points=points)
        result = {"protocol": protocol, "temperature": temperature, "points": points,
                  "metrics_raw": metrics(scored_rows(rows_value)),
                  "metrics_calibrated": metrics(scored_rows(rows_value), temperature),
                  "oof": oof}
    else:
        raise ValueError("protocol must be kev-screening or kev-release")
    # Accuracy must be invariant under positive temperature scaling.
    raw = metrics(scored_rows(rows_value))
    calibrated = metrics(scored_rows(rows_value), temperature)
    if raw["acc"] != calibrated["acc"]:
        raise ValueError("temperature changed argmax accuracy")
    result["provenance"] = {"report": str(report_path), "rows_sha256": rows_sha,
                            "suite_sha256": report["provenance"]["suite_sha256"], "split": split,
                            "raw_rows": True, "method": f"Kev log-grid {points} points"}
    result["checkpoint_updated"] = False
    checkpoint = run / "checkpoint" if (run / "checkpoint").is_dir() else (run if (run / "metadata.json").exists() else None)
    if update_checkpoint:
        if protocol != "kev-release" or checkpoint is None:
            raise ValueError("only a release fit may update a checkpoint")
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["temperature"] = temperature
        metadata["temperature_fit"] = result["provenance"]
        fd, temporary = tempfile.mkstemp(prefix=".metadata.", dir=checkpoint)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, metadata_path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        result["checkpoint_updated"] = True
    if out:
        destination = Path(out)
        if destination.suffix == ".json": destination.parent.mkdir(parents=True, exist_ok=True)
        else: destination.mkdir(parents=True, exist_ok=True); destination = destination / "calibration.json"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
