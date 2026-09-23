"""PyTorch eager/SDPA/BF16 numerical qualification for the Gemma backend."""

from __future__ import annotations

import contextlib
from typing import Any

import torch


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
        probabilities = {}
        with torch.no_grad():
            for start in range(len(encodings)):
                with autocast:
                    chunk_logits = model.forward_batch(encodings[start:start + 1])
                probabilities.update({
                    f"{ids[start]}:{question_index}": torch.softmax(value.float(), -1).cpu().tolist()
                    for question_index, value in enumerate(chunk_logits[0])
                })
        model.train()
        model.zero_grad(set_to_none=True)
        with autocast:
            logits = model.forward_batch(encodings[:1])
            loss = sum(torch.nn.functional.cross_entropy(
                value.float().unsqueeze(0),
                torch.tensor([encodings[0]["labels"][index]], device=value.device))
                       for index, value in enumerate(logits[0]))
        loss = loss / max(len(logits[0]), 1)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        finite_gradient = bool(gradients) and all(
            gradient is not None and bool(torch.isfinite(gradient).all().item())
            for gradient in gradients)
        result: dict[str, Any] = {
            "attention": attention, "dtype": dtype, "status": "passed",
            "probabilities": probabilities, "loss": float(loss.detach().cpu()),
            "finite_gradient": finite_gradient, "max_abs_probability": None,
            "kl_divergence": None, "argmax_flips": None, "compared_questions": 0,
        }
        if not finite_gradient or not torch.isfinite(loss).item():
            result["status"] = "failed"
        if baseline is not None:
            deltas, kls, flips = [], [], 0
            for key, values in probabilities.items():
                before = torch.tensor(baseline[key], dtype=torch.float64)
                after = torch.tensor(values, dtype=torch.float64)
                deltas.append(float((before - after).abs().max()))
                kls.append(float((before * (before.clamp_min(1e-12).log()
                                             - after.clamp_min(1e-12).log())).sum()))
                flips += int(before.argmax() != after.argmax())
            result.update(max_abs_probability=max(deltas, default=None),
                          kl_divergence=sum(kls), argmax_flips=flips,
                          compared_questions=len(deltas))
        return result
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        return {"attention": attention, "dtype": dtype, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "max_abs_probability": None,
                "kl_divergence": None, "argmax_flips": None, "finite_gradient": None}


def compare_precision_profiles(model, encodings: list[dict], record_ids: list[str], *,
                               device: str, include_bf16: bool = True) -> list[dict[str, Any]]:
    profiles = [("eager", "fp32"), ("sdpa", "fp32")]
    if include_bf16:
        profiles.append(("sdpa", "bf16"))
    results: list[dict[str, Any]] = []
    baseline = None
    for attention, dtype in profiles:
        result = _profile(model, encodings, record_ids, device=device,
                          attention=attention, dtype=dtype, baseline=baseline)
        results.append(result)
        if baseline is None and result.get("status") == "passed":
            baseline = result["probabilities"]
    return results
