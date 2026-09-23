"""Paired comparisons of complete saved-row evaluations."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import metrics, paired_bootstrap, raw_row, scored_rows
from ..data.suites import load_manifest


def _load(directory: str | Path, *, allow_test=False):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "rows.json").read_text(encoding="utf-8"))
    coverage = report.get("coverage", {})
    if coverage.get("rejected_records", 0) or coverage.get("truncated_records", 0) or coverage.get("evaluated_records") != coverage.get("requested_records"):
        raise ValueError(f"incomplete evaluation: {directory}")
    if coverage.get("evaluated_questions") is not None and coverage["evaluated_questions"] != len(rows):
        raise ValueError(f"row/question coverage mismatch: {directory}")
    provenance = dict(report.get("provenance", {}))
    for key in ("suite", "split", "suite_sha256", "source_sha256"):
        if key not in provenance and key in report: provenance[key] = report[key]
    allowed_splits = {"calibration", "development", "test"} if allow_test else {"calibration", "development"}
    if not provenance.get("suite_sha256") or provenance.get("split") not in allowed_splits:
        raise ValueError("comparison requires suite identity and a non-test split (or explicit test mode)")
    if not provenance.get("source_sha256") and provenance.get("suite"):
        info = load_manifest(provenance["suite"])["files"].get(provenance["split"] + ".jsonl")
        if info: provenance["source_sha256"] = info["sha256"]
    if not provenance.get("source_sha256"):
        raise ValueError("comparison requires source identity")
    return report, rows


def compare(candidate: str | Path, reference: str | Path, *, aggregation="micro",
            samples=1000, seed=0, out: str | Path | None = None,
            calibrated=False, allow_test_compare=False) -> dict:
    if aggregation not in {"micro", "macro"}:
        raise ValueError("aggregation must be micro or macro")
    cand_report, cand_rows = _load(candidate, allow_test=allow_test_compare); ref_report, ref_rows = _load(reference, allow_test=allow_test_compare)
    def provenance(report):
        value = dict(report.get("provenance", {}))
        for key in ("suite", "split", "suite_sha256", "source_sha256"):
            if key not in value and key in report: value[key] = report[key]
        if not value.get("source_sha256") and value.get("suite"):
            value["source_sha256"] = load_manifest(value["suite"])["files"][value["split"] + ".jsonl"]["sha256"]
        return value
    cp, rp = provenance(cand_report), provenance(ref_report)
    for key in ("suite_sha256", "split", "source_sha256"):
        if cp.get(key) != rp.get(key):
            raise ValueError(f"comparison identity mismatch: {key}")
    if allow_test_compare and cp.get("split") == "test":
        for report in (cand_report, ref_report):
            if not report.get("provenance", {}).get("locked_ledger_key"):
                raise ValueError("test comparison requires locked-ledger provenance")
    cand_clean, ref_clean = scored_rows(cand_rows), scored_rows(ref_rows)
    if any(row.get("inference_temperature", 1.0) != 1.0 for row in cand_clean + ref_clean):
        if not calibrated:
            raise ValueError("raw comparison requires inference_temperature=1 rows")
    if calibrated:
        # A calibrated comparison must be explicit and both sides must carry
        # the same recorded serving temperature.
        ct = {row.get("inference_temperature", 1.0) for row in cand_clean}
        rt = {row.get("inference_temperature", 1.0) for row in ref_clean}
        if len(ct) != 1 or len(rt) != 1 or next(iter(ct)) <= 0 or next(iter(rt)) <= 0 or next(iter(ct)) == 1.0 or next(iter(rt)) == 1.0:
            raise ValueError("calibrated comparison requires one positive non-unit temperature per side")
    else:
        cand_clean, ref_clean = [raw_row(r) for r in cand_clean], [raw_row(r) for r in ref_clean]
    result = {"candidate": metrics(cand_clean), "reference": metrics(ref_clean),
              "aggregation": aggregation, "calibrated": calibrated,
              "provenance": {"suite_sha256": cp["suite_sha256"], "split": cp["split"],
                             "source_sha256": cp.get("source_sha256"),
                             "candidate_temperature": next(iter({r.get("inference_temperature", 1.0) for r in cand_clean})),
                             "reference_temperature": next(iter({r.get("inference_temperature", 1.0) for r in ref_clean}))}, "paired": {}}
    for metric in ("acc", "nll", "brier", "ece", "coverage_at_5pct_error"):
        result["paired"][metric] = paired_bootstrap(cand_clean, ref_clean, samples=samples,
                                                      seed=seed, metric=metric, aggregation=aggregation)
    if out:
        destination = Path(out); destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
