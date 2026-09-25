"""Content-addressed Gemma 4 trained-checkpoint qualification receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .policy import MODEL_POLICIES

RECEIPT_FORMAT = "gev.trained-checkpoint-qualification"
RECEIPT_VERSION = 1

_COMMON_CODE = (
    "src/gev/models/policy.py", "src/gev/models/qualification.py",
    "src/gev/models/specs.py", "src/gev/models/registry.py",
    "src/gev/configuration/config.py", "src/gev/configuration/resolved.py",
    "src/gev/application/training.py", "src/gev/application/evaluation.py",
    "src/gev/application/prediction.py", "src/gev/evaluation/benchmark.py",
    "src/gev/evaluation/development.py", "src/gev/domain/tokenization.py",
    "src/gev/training/batching.py", "src/gev/training/schedule.py",
)
_BACKEND_CODE = {
    "torch": ("src/gev/backends/torch/__init__.py",
              "src/gev/backends/torch/gemma4.py", "src/gev/backends/torch/gemma3.py",
              "src/gev/backends/torch/pointer.py", "src/gev/backends/torch/masks.py",
              "src/gev/backends/torch/objective.py",
              "src/gev/backends/torch/qualification.py",
              "src/gev/backends/torch/training.py", "src/gev/backends/torch/checkpoint.py"),
    "mlx": ("src/gev/backends/mlx/__init__.py", "src/gev/backends/mlx/gemma4.py",
            "src/gev/backends/mlx/pointer.py", "src/gev/backends/mlx/training.py",
            "src/gev/backends/mlx/checkpoint.py", "src/gev/backends/mlx/qualification.py"),
}


def qualification_policy_sha256(family: str, backend: str, compute_dtype: str) -> str:
    policy = MODEL_POLICIES.get(family)
    if policy is None:
        raise ValueError(f"no trained-checkpoint policy for model family {family!r}")
    value = {"family": family, "backend": backend, "compute_dtype": compute_dtype,
             "required_checks": policy.required_qualification_checks(backend, compute_dtype),
             "source_weights_dtype": policy.source_weights_dtype,
             "implemented": sorted(policy.implemented),
             "required_execution_mode": policy.required_execution_mode,
             "required_microbatch": policy.required_microbatch,
             "development_evaluation": policy.development_evaluation,
             "development_evaluation_scope": policy.development_evaluation_scope}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def qualification_code_sha256(backend: str, repo_root: str | Path) -> str:
    if backend not in _BACKEND_CODE:
        raise ValueError(f"no qualification code inventory for backend {backend!r}")
    root = Path(repo_root)
    names = (*_COMMON_CODE, *_BACKEND_CODE[backend])
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def seal_qualification_receipt(payload: dict) -> dict:
    receipt = dict(payload)
    receipt.pop("receipt_sha256", None)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return receipt


def trainable_content_sha256(backend: str, model) -> str:
    """Hash sorted trainable content with names, dtypes, and shapes included."""
    import numpy as np

    if backend == "torch":
        values = [(name, value) for name, value in model.state_dict().items()
                  if "lora_" in name or name.startswith("head.")]
        arrays = [(name, str(value.dtype), list(value.shape),
                   value.detach().cpu().contiguous().numpy().tobytes())
                  for name, value in values]
    elif backend == "mlx":
        import mlx.core as mx
        from mlx.utils import tree_flatten

        arrays = []
        for name, value in tree_flatten(model.trainable_parameters()):
            mx.eval(value)
            arrays.append((name, str(value.dtype), list(value.shape),
                           np.ascontiguousarray(np.asarray(value)).tobytes()))
    else:
        raise ValueError(f"no trainable fingerprint implementation for {backend!r}")
    digest = hashlib.sha256()
    for name, dtype, shape, data in sorted(arrays):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(dtype.encode("ascii") + b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(data)
    return digest.hexdigest()


def bind_checkpoint_tensors(receipt: dict, tensor_hashes: dict[str, str]) -> dict:
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    payload["checkpoint_ready"] = True
    payload["checkpoint_tensor_sha256"] = dict(sorted(tensor_hashes.items()))
    return seal_qualification_receipt(payload)


def validate_qualification_sidecars(directory: str | Path, receipt: dict) -> None:
    """Require checkpoint-local receipt/report copies to match the manifest receipt."""
    root = Path(directory)
    try:
        sidecar_receipt = json.loads((root / "qualification.json").read_text(encoding="utf-8"))
        report_bytes = (root / "development_report.json").read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint qualification sidecar is missing or invalid: {exc}") from exc
    if sidecar_receipt != receipt:
        raise ValueError("checkpoint sidecar qualification differs from manifest")
    if hashlib.sha256(report_bytes).hexdigest() != receipt.get("development", {}).get(
            "report_sha256"):
        raise ValueError("checkpoint development report digest mismatch")


def _value(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key)


def validate_qualification_receipt(receipt: dict, *, config, backend: str,
                                   expected_code_sha256: str | None = None,
                                   expected_checkpoint_hashes: dict[str, str] | None = None,
                                   require_checkpoint_ready: bool = True) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("Gemma 4 checkpoint qualification receipt is missing")
    digest = receipt.get("receipt_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    if digest != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("Gemma 4 checkpoint qualification receipt hash mismatch")

    model_config = _value(config, "model")
    training_config = _value(config, "training")
    family = _value(model_config, "family")
    name = _value(model_config, "name")
    revision = _value(model_config, "revision")
    compute_dtype = _value(training_config, "dtype")
    policy = MODEL_POLICIES.get(family)
    if policy is None:
        raise ValueError("checkpoint qualification receipt has no model policy")
    output_model_id = "gev-gemma4-e2b" if family == "gemma4_e2b_text" else family
    if (receipt.get("format") != RECEIPT_FORMAT
            or receipt.get("version") != RECEIPT_VERSION
            or receipt.get("status") != "passed"
            or receipt.get("model_output_id") != output_model_id
            or receipt.get("family") != family or receipt.get("backend") != backend
            or receipt.get("base") != {
                "name": name, "revision": revision,
                "source_weights_dtype": "bf16", "compute_dtype": compute_dtype}):
        raise ValueError("Gemma 4 checkpoint qualification identity mismatch")
    if receipt.get("policy_sha256") != qualification_policy_sha256(
            family, backend, compute_dtype):
        raise ValueError("Gemma 4 checkpoint qualification policy mismatch")
    if expected_code_sha256 is not None and receipt.get("code_sha256") != expected_code_sha256:
        raise ValueError("Gemma 4 checkpoint qualification code hash mismatch")
    required_checks = policy.required_qualification_checks(backend, compute_dtype)
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) is not True
                                           for name in required_checks):
        raise ValueError("Gemma 4 checkpoint qualification checks did not pass")
    if (not isinstance(receipt.get("training_complete"), bool)
            or isinstance(receipt.get("training_steps"), bool)
            or not isinstance(receipt.get("training_steps"), int)
            or receipt["training_steps"] < 1):
        raise ValueError("Gemma 4 checkpoint training identity is invalid")
    development = receipt.get("development")
    if (not isinstance(development, dict)
            or (development.get("suite"), development.get("split")) !=
            ("decision-v7", "development")):
        raise ValueError("Gemma 4 checkpoint development scope mismatch")
    ids = development.get("selected_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids)
            or development.get("selected_record_count") != len(ids)
            or development.get("selected_ids_sha256") != hashlib.sha256(
                json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False)
                .encode("utf-8")).hexdigest()):
        raise ValueError("Gemma 4 checkpoint development selection identity mismatch")
    coverage = development.get("coverage", {})
    if (not isinstance(coverage, dict)
            or coverage.get("requested_records") != len(ids)
            or coverage.get("evaluated_records") != len(ids)
            or coverage.get("requested_questions") != coverage.get("evaluated_questions")
            or coverage.get("evaluated_questions", 0) < 1
            or coverage.get("rejected_records") != 0
            or coverage.get("truncated_records") != 0):
        raise ValueError("Gemma 4 checkpoint development coverage is incomplete")
    mechanisms = development.get("mechanism_checks")
    if (not isinstance(mechanisms, dict) or mechanisms.get("passed") is not True
            or isinstance(mechanisms.get("failures", 0), bool)
            or mechanisms.get("failures", 0) != 0
            or checks.get("development_mechanisms") is not True):
        raise ValueError("Gemma 4 checkpoint development mechanism summary failed")
    manifest_hash = development.get("manifest_sha256")
    report_hash = development.get("report_sha256")
    trainable_hash = receipt.get("trainable_parameters_sha256")
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value)
           for value in (manifest_hash, report_hash, trainable_hash,
                         receipt.get("code_sha256"))):
        raise ValueError("Gemma 4 checkpoint qualification digest is invalid")
    ready = receipt.get("checkpoint_ready")
    hashes = receipt.get("checkpoint_tensor_sha256")
    if require_checkpoint_ready:
        if (ready is not True or not isinstance(hashes, dict)
                or set(hashes) != {"adapter", "pointer"}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in hashes.values())):
            raise ValueError("Gemma 4 checkpoint qualification is not checkpoint-ready")
        if (expected_checkpoint_hashes is not None
                and hashes != expected_checkpoint_hashes):
            raise ValueError("Gemma 4 checkpoint tensor hashes mismatch")
    elif ready is not False or hashes is not None:
        raise ValueError("Gemma 4 pre-checkpoint qualification has an invalid ready state")
