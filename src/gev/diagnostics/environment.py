"""Machine and accelerator diagnostics. Diagnostics never hide unsupported MPS."""

from __future__ import annotations

import platform
import os
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _memory() -> str:
    try:
        import psutil
        import resource
        vm = psutil.virtual_memory()
        return (
            f"total={vm.total},available={vm.available},"
            f"process_max_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}"
        )
    except Exception as exc:  # pragma: no cover - platform-specific
        return f"memory=unavailable ({exc})"


def _torch_smoke(device: str, dtype_name: str) -> dict[str, Any]:
    import torch

    model = torch.nn.Linear(4, 3, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.ones((2, 4), device=device)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    try:
        if dtype_name == "bf16":
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                output = model(inputs)
        else:
            output = model(inputs)
        loss = output.float().square().mean()
        loss.backward()
        finite_loss = bool(torch.isfinite(loss).item())
        finite_gradient = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters())
        optimizer.step()
        updated = any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, model.parameters()))
        actual_dtype = str(output.dtype).removeprefix("torch.")
        dtype_ok = dtype_name == "fp32" and actual_dtype == "float32" or dtype_name == "bf16" and actual_dtype == "bfloat16"
        status = "passed" if finite_loss and finite_gradient and updated and dtype_ok else "failed"
        return {"status": status, "seconds": round(time.perf_counter() - started, 4), "loss": float(loss.detach().cpu()),
                "requested_dtype": dtype_name, "actual_output_dtype": actual_dtype, "finite_loss": finite_loss,
                "finite_gradient": finite_gradient, "parameter_updated": updated}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def doctor() -> dict[str, Any]:
    cpu_brand = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu_brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip() or cpu_brand
        except (OSError, subprocess.CalledProcessError):
            pass
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_brand,
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "memory": _memory(),
    }
    try:
        import torch
        result["torch"] = torch.__version__
        result["packages"] = {
            name: _package_version(name)
            for name in ("transformers", "peft", "accelerate", "huggingface-hub", "safetensors", "numpy")
        }
        result["mps"] = {"built": bool(torch.backends.mps.is_built()), "available": bool(torch.backends.mps.is_available())}
        result["cpu_fp32"] = _torch_smoke("cpu", "fp32")
        if result["mps"]["available"]:
            result["mps_fp32"] = _torch_smoke("mps", "fp32")
            result["mps_bf16"] = _torch_smoke("mps", "bf16")
        else:
            result["mps_fp32"] = {"status": "not-run", "reason": "MPS is not available"}
            result["mps_bf16"] = {"status": "not-run", "reason": "MPS is not available"}
    except Exception as exc:
        result["torch"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    statuses = [result[key].get("status") for key in ("cpu_fp32", "mps_fp32", "mps_bf16") if isinstance(result.get(key), dict)]
    result["status"] = "passed" if statuses and all(status in {"passed", "not-run"} for status in statuses) else "failed"
    return result


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"
