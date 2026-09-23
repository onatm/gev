"""Bounded, real-model training-window profiling (never a complete recipe run)."""
from __future__ import annotations

import contextlib
import json
import platform
import time
from pathlib import Path

import torch

from .precision import _development_rows
from .materialize import materialize
from .tokenization import MarkerMap, encode
from .training.batching import variants_for_request
from .training.objective import logical_loss


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _memory(device: str) -> dict:
    import psutil
    values = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if device == "mps":
        peak = getattr(torch.mps, "max_memory_allocated", None)
        values.update({"allocated_bytes": torch.mps.current_allocated_memory(),
                       "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                       # PyTorch 2.8 on some Apple builds does not expose a
                       # peak counter; current allocation is the honest fallback.
                       "peak_allocated_bytes": peak() if peak is not None else torch.mps.current_allocated_memory()})
    else:
        values.update({"allocated_bytes": None, "driver_allocated_bytes": None,
                       "peak_allocated_bytes": None})
    return values


def profile_train(config, warmup_steps: int, measure_steps: int, output: Path, *, data_root: str = "data") -> dict:
    if warmup_steps < 0 or measure_steps < 1:
        raise ValueError("warmup-steps must be non-negative and measure-steps positive")
    device = "mps" if config.runtime.device == "auto" and torch.backends.mps.is_available() else config.runtime.device
    if device == "auto": device = "cpu"
    result = {"status": "failed", "model_revision": config.model.revision,
              "hardware": platform.platform(), "cpu_brand": platform.processor(),
              "runtime": {"attn": config.runtime.attn_implementation,
                          "gradient_checkpointing": config.runtime.gradient_checkpointing,
                          "empty_cache": config.runtime.empty_cache, "use_cache": False},
              "warmup_steps": warmup_steps, "measure_steps": measure_steps,
              "logical_batch": config.training.logical_batch, "microbatch": config.training.microbatch,
              "device": device}
    try:
        from transformers import AutoTokenizer
        from .models.gemma import GemmaRowModel, load_real_backbone
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
        markers = MarkerMap.load(config.model.marker_artifact or "runs/reference/model-marker-map.json", tokenizer)
        rows = _development_rows(data_root, config.training.logical_batch)
        variants = [variant for row in rows for variant in variants_for_request(
            row, seed=config.training.seed, epoch=0, tokenizer=tokenizer, markers=markers,
            caps=(config.training.state_cap, config.training.branch_cap, config.training.packed_cap),
            p_none=config.training.p_none, p_none_distract=config.training.p_none_distract,
            p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair)]
        model = GemmaRowModel(load_real_backbone(
            config.model.name, config.model.revision,
            attn_implementation=config.runtime.attn_implementation,
            gradient_checkpointing=config.runtime.gradient_checkpointing)).to(device).train()
        head = list(model.head.parameters())
        head_ids = {id(p) for p in head}
        lora = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": lora, "lr": config.training.learning_rate},
            {"params": head, "lr": config.training.head_learning_rate or config.training.learning_rate}],
            weight_decay=config.training.weight_decay)
        payload = [{"record": variant.record, "encoding": variant.encoding} for variant in variants]
        runtime_dtype = config.training.dtype

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(variants), config.training.microbatch):
                part = variants[start:start + config.training.microbatch]
                autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                            if runtime_dtype == "bf16" else contextlib.nullcontext())
                with autocast:
                    logits = model.forward_batch([v.encoding for v in part])
                    loss = logical_loss(logits, [{"record": v.record, "encoding": v.encoding} for v in part])
                    loss = loss * (len(part) / len(variants))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("non-finite profiling loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _sync(device)

        for _ in range(warmup_steps):
            step()
            if config.runtime.empty_cache and device == "mps": torch.mps.empty_cache()
        _sync(device)
        timings, memory = [], []
        for _ in range(measure_steps):
            _sync(device)
            started = time.perf_counter()
            step()
            timings.append(time.perf_counter() - started)
            memory.append(_memory(device))
            if config.runtime.empty_cache and device == "mps": torch.mps.empty_cache()
        result.update(status="passed", reason="bounded real-model measurement", timings_seconds=timings,
                      mean_seconds=sum(timings) / len(timings), memory_per_step=memory,
                      selected_ids=[v.original_source_id for v in variants],
                      selected_variant_count=len(variants), selected_question_lengths=[
                          len(v.encoding["ids"]) for v in variants],
                      physical_microbatches=(len(variants) + config.training.microbatch - 1) // config.training.microbatch,
                      model_weights_dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
                      master_dtype="fp32")
    except Exception as exc:
        result.update(reason=f"{type(exc).__name__}: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
