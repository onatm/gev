"""Torch runtime environment configuration."""

from __future__ import annotations

import os


def configure_runtime(device: str, mps_fallback: bool) -> None:
    """Apply the accelerator policy before importing torch/transformers."""
    requested = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if requested is not None and requested.lower() not in {"0", "false", "no", "off", "1", "true", "yes", "on"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be a boolean value")
    effective = requested is not None and requested.lower() in {"1", "true", "yes", "on"}
    if not mps_fallback and effective:
        raise RuntimeError("runtime.mps_fallback=false conflicts with PYTORCH_ENABLE_MPS_FALLBACK")
    if mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
