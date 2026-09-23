"""Strict saved-row benchmark runner, adapted from Kev benchmark.py."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from ..domain.api_request import api_request
from .metrics import EPSILON, grouped_metrics, metrics, raw_row, scored_rows, unknowable_report


def _labels(q):
    if q["type"] == "choice":
        keys = list(q.get("criteria", {})); return keys, keys.index(q["label"])
    if q["type"] == "noul": return ["false", "true"], int(q["label"])
    keys = [str(i) for i in range(len(q["criteria"]))]; return keys, int(q["label"])


def validate_distribution(raw, keys):
    if set(raw) != set(keys): raise ValueError("probability keys do not match requested options")
    p = np.array([raw[k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    if total <= 0 or abs(total - 1) > max(1e-5, len(keys) * .005 + 1e-8): raise ValueError(f"invalid probability sum: {total}")
    return p / total, total


def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]): raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]; rows = []
    for qid, q in record["questions"].items():
        keys, label = _labels(q); p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta.get("group_id", meta["id"]), "question": qid, "source": meta["source"],
               "task": q.get("src"), "type": q["type"], "variant": meta.get("variant", "clean"), "keys": keys, "label": label,
               "control_id": meta.get("control_id"), "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"),
               "parent": meta.get("parent_id") or (meta["id"] if meta.get("variant", "clean") == "clean" else meta.get("group_id", meta["id"])),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        if "logits" in prediction:
            raw = prediction["logits"][qid]
            if set(raw) != set(keys) or not all(math.isfinite(float(raw[k])) for k in keys): raise ValueError("logit keys or values do not match the requested options")
            row["logits"] = [float(raw[k]) for k in keys]
            if "inference_temperature" in prediction:
                row["inference_temperature"] = prediction["inference_temperature"]
        if "raw_logits" in prediction:
            raw = prediction["raw_logits"][qid]
            if set(raw) != set(keys) or not all(math.isfinite(float(raw[k])) for k in keys):
                raise ValueError("raw logit keys or values do not match the requested options")
            row["raw_logits"] = [float(raw[k]) for k in keys]
        rows.append(row)
    return rows


def _request_hash(record):
    payload = api_request(record)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def summarize(rows, temperature=1.0, heldout_sources=()):
    clean = [r for r in rows if r["variant"] == "clean"]
    knowable = [r for r in clean if r["source"] != "unknowable"]
    tasks = grouped_metrics(clean, "task") if clean else {}
    variants = grouped_metrics(rows, "variant") if rows else {}
    lookup = {(r["id"], r["question"]): r for r in clean}; diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["parent"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.asarray(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    from .mechanism import paired_flip
    task_nll = [v["nll"] for k, v in tasks.items() if not k.startswith("unknowable_") or k.startswith("unknowable_control")]
    raw_clean = [raw_row(row) for row in knowable]
    calibrated_clean = metrics(knowable, temperature) if knowable else None
    raw_metrics = metrics(raw_clean) if raw_clean else None
    return {"objective": -float(np.mean(task_nll)) if task_nll else None, "paired_flip": paired_flip(clean),
            "unknowable": unknowable_report(clean), "clean": metrics(knowable) if knowable else None, "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in heldout_sources], "task") if any(r["source"] in heldout_sources for r in clean) else {},
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None, "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "raw_clean": raw_metrics, "calibrated_clean": calibrated_clean,
            "nll_provenance": {"raw": raw_metrics["nll"] if raw_metrics else None,
                               "calibrated": calibrated_clean["nll"] if calibrated_clean else None,
                               "raw_logits": bool(raw_clean) and all("logits" in r for r in raw_clean)},
            "metric_policy": {"version": 2, "selective_ties": "whole_confidence_groups", "coverage_at_error": "in-sample maximum over confidence thresholds",
                              "nll": "exact from logits when recorded; otherwise from floored probabilities", "nll_floor": EPSILON,
                              "renormalize_returned_probabilities": True, "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}


def evaluate_records(records, predictor, output, temperature=1.0):
    output = Path(output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    predictor_temperature = getattr(predictor, "temperature", 1.0)
    if predictor_temperature != 1.0 and temperature != 1.0:
        raise ValueError("additional benchmark temperature would be applied twice")
    output.mkdir(parents=True); rows, predictions, latencies = [], [], []
    start = time.perf_counter(); coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                                             "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                try:
                    before = time.perf_counter(); pred = predictor(record); new = prediction_rows(record, pred)
                except Exception as exc:
                    coverage["rejected_records"] += 1
                    (output / "failure.json").write_text(json.dumps({"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(exc).__name__}, indent=2) + "\n")
                    raise
                elapsed = float(pred.get("latency_ms", (time.perf_counter() - before) * 1000)); latencies.append(elapsed)
                item = {"request_sha256": _request_hash(record), "id": record["_meta"]["id"], "prediction": pred, "rows": new}
                stream.write(json.dumps(item, allow_nan=False) + "\n"); stream.flush()
                predictions.append(item); rows.extend(new); coverage["evaluated_records"] += 1; coverage["evaluated_questions"] += len(new)
    except Exception: raise
    (output / "rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    report = summarize(rows, temperature)
    from .mechanism import mechanism_checks
    report.update(coverage=coverage, latency_ms={"median": float(np.median(latencies)) if latencies else None,
                  "p95": float(np.quantile(latencies, .95)) if latencies else None},
                  calibration={"inference_temperature": getattr(predictor, "temperature", None), "additional_temperature": temperature,
                               "logits_recorded": bool(rows) and all("logits" in r for r in rows)},
                   mechanism_checks=mechanism_checks(records, predictor), elapsed_seconds=time.perf_counter() - start)
    report["rows_sha256"] = hashlib.sha256((output / "rows.json").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, rows
