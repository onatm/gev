"""Measured precision qualification for the pinned, trained row model."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from ..domain.materialize import materialize


THRESHOLDS = {"fp32_eager_sdpa_max_abs_probability": 1e-3, "bf16_max_abs_probability": .02}


def _development_rows(root: str, records: int) -> list[dict]:
    """Load the verified development artifact without selecting training data."""
    # Accept only a verified official split or an approved smoke child.
    from ..data.access import load_verified_split
    rows, _manifest, _manifest_hash = load_verified_split(root, "decision-v7", "development")
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
        from ..configuration.resolved import resolve_experiment_config
        resolved = resolve_experiment_config(config)
        from ..models.specs import BackendCapability
        resolved.model.require_capability(BackendCapability.PRECISION_DIAGNOSTICS)
        resolved.validate_runtime_available()
        result["selection"] = resolved.provenance(output_path=str(output))
        import transformers
        tokenizer = resolved.load_tokenizer()
        markers = resolved.load_markers(tokenizer)
        rows = _development_rows(data_root, records)
        encodings = [resolved.encode_record(tokenizer, materialize(row), markers) for row in rows]
        device = resolved.select_device()
        checkpoint = Path(run) / "checkpoint" if (Path(run) / "checkpoint").exists() else Path(run)
        model, metadata = resolved.load_checkpoint(
            checkpoint, device=device, tokenizer=tokenizer, expected_marker_map=markers,
            attn_implementation="eager")
        result["model"] = {"run": str(checkpoint), "metadata_revision": metadata["identity"]["base"]["revision"],
                           "records": len(rows), "record_ids": [r.get("_meta", {}).get("id") for r in rows],
                           "transformers": transformers.__version__}
        comparisons = resolved.model.compare_precision_profiles(
            model, encodings, [str(r.get("_meta", {}).get("id")) for r in rows],
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
