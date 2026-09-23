"""Development-only rows/packed execution parity measurement."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def select_representative_records(records: list[dict], limit: int) -> list[dict]:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("records must be a positive integer")
    if not records:
        raise ValueError("development selection is empty")
    ordered = sorted(records, key=lambda row: row.get("_meta", {}).get("id", ""))
    if limit >= 8:
        quotas = {"agnews": 3, "yelp": 2, "banking77": 3}
        selected = []
        for source, quota in quotas.items():
            candidates = [row for row in ordered if row.get("_meta", {}).get("source") == source]
            candidates.sort(key=lambda row: (-len(row.get("questions", {})), row.get("_meta", {}).get("id", "")))
            if len(candidates) < quota:
                raise ValueError(f"development selection lacks {source} representatives")
            selected.extend(candidates[:quota])
        remaining = [row for row in ordered if row not in selected]
        selected.extend(remaining[:limit - len(selected)])
        return selected
    return ordered[:min(limit, len(ordered))]


def measure(model, encodings, *, records: int, warmup: int = 2) -> dict:
    if isinstance(records, bool) or records <= 0:
        raise ValueError("records must be a positive integer")
    if not encodings:
        raise ValueError("cannot measure an empty selection")
    if records > len(encodings):
        raise ValueError("records exceeds the selected encoding count")
    values = list(encodings[:records])
    device = next(model.parameters()).device

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

    result = {"records": len(values), "device": str(device), "modes": {},
              "prefix": {"miss_count": 0, "hit_count": 0, "reused_prefix_positions": 0}}
    for mode in ("rows", "packed"):
        call = model.forward_rows_batch if mode == "rows" else model.forward_packed_batch
        for value in values[:min(warmup, len(values))]:
            with torch.no_grad():
                call([value])
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            output = call(values)
        sync()
        elapsed = time.perf_counter() - start
        result["modes"][mode] = {"seconds": elapsed, "latency_ms": elapsed * 1000 / max(len(values), 1),
                                 "question_count": sum(len(row) for row in output)}
    deltas = {"packed_vs_rows": {"logit": 0.0, "probability": 0.0},
              "cached_first_vs_rows": {"logit": 0.0, "probability": 0.0},
              "cached_repeat_vs_rows": {"logit": 0.0, "probability": 0.0}}
    prefix_miss_seconds = prefix_hit_seconds = 0.0
    for encoding in values:
        with torch.no_grad():
            rows = model.forward_one(encoding)
            packed = model.forward_packed_one(encoding)
        prefix_start = time.perf_counter()
        prefix = model.prefill_prefix(tuple(encoding["ids"][:encoding["state_length"]]),
                                      tuple(encoding["pos"][:encoding["state_length"]]))
        with torch.no_grad():
            cached = model.forward_with_prefix(encoding, prefix)
        if device.type == "mps": torch.mps.synchronize()
        prefix_miss_seconds += time.perf_counter() - prefix_start
        result["prefix"]["miss_count"] += 1
        result["prefix"]["reused_prefix_positions"] += encoding["state_length"]
        hit_start = time.perf_counter()
        with torch.no_grad():
            repeated = model.forward_with_prefix(encoding, prefix)
        if device.type == "mps": torch.mps.synchronize()
        prefix_hit_seconds += time.perf_counter() - hit_start
        result["prefix"]["hit_count"] += 1
        if not (len(rows) == len(packed) == len(cached) == len(repeated)):
            raise AssertionError("rows, packed, and cached question counts differ")
        for row_logits, packed_logits, cached_logits, repeat_logits in zip(rows, packed, cached, repeated):
            if not (row_logits.shape == packed_logits.shape == cached_logits.shape == repeat_logits.shape):
                raise AssertionError("rows, packed, and cached option counts differ")
            for name, other in (("packed_vs_rows", packed_logits), ("cached_first_vs_rows", cached_logits),
                                ("cached_repeat_vs_rows", repeat_logits)):
                deltas[name]["logit"] = max(deltas[name]["logit"], float((row_logits - other).abs().max()))
                deltas[name]["probability"] = max(
                    deltas[name]["probability"],
                    float((torch.softmax(row_logits, -1) - torch.softmax(other, -1)).abs().max()))
    result["prefix"].update({"miss_seconds": prefix_miss_seconds, "hit_seconds": prefix_hit_seconds,
                              "miss_latency_ms": prefix_miss_seconds * 1000 / max(len(values), 1),
                              "hit_latency_ms": prefix_hit_seconds * 1000 / max(2 * len(values), 1)})
    threshold = 1e-3 if device.type == "mps" else 1e-4
    result["deltas"] = deltas
    result["max_logit_delta"] = max(value["logit"] for value in deltas.values())
    result["max_probability_delta"] = max(value["probability"] for value in deltas.values())
    result["threshold"] = threshold
    result["numerical_passed"] = all(value["probability"] <= threshold for value in deltas.values())
    result["status"] = "passed" if result["numerical_passed"] else "failed"
    return result


def write_measurement(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
