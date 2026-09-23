"""Measured precision qualification for the pinned, trained row model."""
from __future__ import annotations

import contextlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

from .materialize import materialize
from .tokenization import encode


THRESHOLDS = {"fp32_eager_sdpa_max_abs_probability": 1e-3, "bf16_max_abs_probability": .02}


def _set_attention(model, implementation: str) -> None:
    decoder = model.decoder
    setter = getattr(decoder, "set_attn_implementation", None)
    if setter is not None:
        setter(implementation)
    else:
        decoder.config._attn_implementation = implementation
    decoder.config.use_cache = False


def _profile(model, encodings: list[dict], ids: list[str], *, device: str,
             attention: str, dtype: str, baseline: dict[str, list[float]] | None) -> dict[str, Any]:
    try:
        _set_attention(model, attention)
        model.eval()
        autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if dtype == "bf16" else contextlib.nullcontext())
        # Keep rows isolated during the diagnostic as well as during the
        # production model's row construction.  This bounds MPS's attention
        # workspace for long representatives without changing the records.
        probabilities = {}
        with torch.no_grad():
            for start in range(0, len(encodings), 1):
                with autocast:
                    chunk_logits = model.forward_batch(encodings[start:start + 1])
                probabilities.update({
                    f"{ids[start]}:{question_index}": torch.softmax(value.float(), -1).cpu().tolist()
                    for question_index, value in enumerate(chunk_logits[0])
                })
        model.train()
        model.zero_grad(set_to_none=True)
        # One real backward probe per profile is sufficient to qualify the
        # runtime and avoids retaining 16 long decoder graphs simultaneously.
        with autocast:
            logits = model.forward_batch(encodings[:1])
            loss = sum(torch.nn.functional.cross_entropy(value.float().unsqueeze(0),
                                                         torch.tensor([encodings[0]["labels"][j]], device=value.device))
                       for j, value in enumerate(logits[0]))
        loss = loss / max(len(logits[0]), 1)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        finite_gradient = bool(gradients) and all(g is not None and bool(torch.isfinite(g).all().item()) for g in gradients)
        result: dict[str, Any] = {"attention": attention, "dtype": dtype, "status": "passed",
                                  "probabilities": probabilities, "loss": float(loss.detach().cpu()),
                                  "finite_gradient": finite_gradient, "max_abs_probability": None,
                                  "kl_divergence": None, "argmax_flips": None, "compared_questions": 0}
        if not finite_gradient or not torch.isfinite(loss).item():
            result["status"] = "failed"
        if baseline is not None:
            deltas, kls, flips = [], [], 0
            for key, values in probabilities.items():
                before = torch.tensor(baseline[key], dtype=torch.float64)
                after = torch.tensor(values, dtype=torch.float64)
                deltas.append(float((before - after).abs().max()))
                kls.append(float((before * (before.clamp_min(1e-12).log() - after.clamp_min(1e-12).log())).sum()))
                flips += int(before.argmax() != after.argmax())
            result.update(max_abs_probability=max(deltas, default=None), kl_divergence=sum(kls),
                          argmax_flips=flips, compared_questions=len(deltas))
        return result
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        return {"attention": attention, "dtype": dtype, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "max_abs_probability": None,
                "kl_divergence": None, "argmax_flips": None, "finite_gradient": None}


def compare_precision_profiles(model, encodings: list[dict], record_ids: list[str], *, device: str,
                               include_bf16: bool = True) -> list[dict[str, Any]]:
    """Run the production forward/backward comparison on one model instance."""
    profiles = [("eager", "fp32"), ("sdpa", "fp32")]
    if include_bf16:
        profiles.append(("sdpa", "bf16"))
    results: list[dict[str, Any]] = []
    baseline = None
    for attention, dtype in profiles:
        result = _profile(model, encodings, record_ids, device=device, attention=attention,
                          dtype=dtype, baseline=baseline)
        results.append(result)
        if baseline is None and result.get("status") == "passed":
            baseline = result["probabilities"]
    return results


def _development_rows(root: str, records: int) -> list[dict]:
    """Load the verified development artifact without selecting training data."""
    # Use the same manifest-aware loader as the CLI; accepting an arbitrary
    # similarly named JSONL file would make the qualification non-reproducible.
    from .cli import _load_cli_split
    rows, _manifest, _manifest_hash = _load_cli_split(root, "decision-v7", "development")
    if not rows:
        raise FileNotFoundError(f"verified development artifact not found below: {root}")
    # Stable representative selection: retain multi-question rows and the longest
    # banking row, then fill in source order.  Never randomize evaluation rows.
    banking = [row for row in rows if row.get("_meta", {}).get("source") == "banking77"]
    selected = sorted(banking, key=lambda row: len(json.dumps(row, ensure_ascii=False)), reverse=True)[:1]
    selected += [row for row in rows if row.get("_meta", {}).get("source") == "agnews" and len(row["questions"]) > 1]
    selected += [row for row in rows if len(row["questions"]) > 1 and row not in selected]
    selected += [row for row in rows if row not in selected]
    return selected[:records]


def check_precision(config, output: Path, *, run: str, records: int = 16, data_root: str = "data") -> dict:
    if records < 1:
        raise ValueError("records must be positive")
    result: dict[str, Any] = {"status": "failed", "model_revision": config.model.revision,
                              "hardware": platform.platform(), "thresholds": THRESHOLDS,
                              "runtime": {"attn": config.runtime.attn_implementation,
                                          "gradient_checkpointing": config.runtime.gradient_checkpointing,
                                          "use_cache": False, "master_dtype": "fp32"}, "comparisons": []}
    try:
        import transformers
        from .checkpoint import load_checkpoint
        from .models.gemma import load_real_backbone
        from .tokenization import MarkerMap
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
        markers = MarkerMap.load(config.model.marker_artifact or "runs/reference/model-marker-map.json", tokenizer)
        rows = _development_rows(data_root, records)
        encodings = [encode(tokenizer, materialize(row), markers, state_cap=config.training.state_cap,
                            branch_cap=config.training.branch_cap, packed_cap=config.training.packed_cap) for row in rows]
        device = "mps" if config.runtime.device == "auto" and torch.backends.mps.is_available() else config.runtime.device
        if device == "auto": device = "cpu"
        checkpoint = Path(run) / "checkpoint" if (Path(run) / "checkpoint").exists() else Path(run)
        model, metadata = load_checkpoint(
            checkpoint, config=config, device=device, tokenizer=tokenizer, expected_marker_map=markers,
            backbone_loader=lambda name, revision: load_real_backbone(
                name, revision, attn_implementation="eager", gradient_checkpointing=config.runtime.gradient_checkpointing))
        result["model"] = {"run": str(checkpoint), "metadata_revision": metadata.get("model_revision"),
                           "records": len(rows), "record_ids": [r.get("_meta", {}).get("id") for r in rows],
                           "transformers": transformers.__version__}
        comparisons = compare_precision_profiles(model, encodings,
                                                  [str(r.get("_meta", {}).get("id")) for r in rows],
                                                  device=device, include_bf16=device == "mps")
        result["comparisons"] = comparisons
        result["gradient_probe_records"] = 1
        fp = next((c for c in comparisons if c["attention"] == "sdpa" and c["dtype"] == "fp32"), None)
        bf = next((c for c in comparisons if c["dtype"] == "bf16"), None)
        valid_fp = fp and fp.get("status") == "passed" and fp.get("max_abs_probability") is not None
        valid_bf = not bf or (bf.get("status") == "passed" and bf.get("max_abs_probability") is not None)
        fp_ok = bool(valid_fp and fp["max_abs_probability"] <= THRESHOLDS["fp32_eager_sdpa_max_abs_probability"])
        bf_ok = bool(valid_bf and (not bf or bf["max_abs_probability"] <= THRESHOLDS["bf16_max_abs_probability"]))
        grads_ok = all(c.get("finite_gradient") is True for c in comparisons)
        result["qualification"] = {"fp32_eager_sdpa": fp_ok, "bf16": bf_ok, "finite_gradients": grads_ok}
        result["status"] = "passed" if fp_ok and bf_ok and grads_ok else "failed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["qualification"] = {"fp32_eager_sdpa": False, "bf16": False, "finite_gradients": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
