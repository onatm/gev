"""Family-level runtime policy, separate from backend capability declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .specs import BackendCapability, GEMMA4_E2B_REVISION


@dataclass(frozen=True)
class ModelPolicy:
    """Implemented runtime and qualification constraints for a family."""

    family_id: str
    supported_compute_dtypes: frozenset[str]
    implemented: frozenset[tuple[str, str, str]]
    source_weights_dtype: str | None = None
    required_execution_mode: str | None = None
    required_microbatch: int | None = None
    forbid_mps_fallback: bool = False
    forbid_gradient_checkpointing: bool = False
    bf16_trainable_state_dtype: str | None = None
    bf16_optimizer_state_dtype: str | None = None
    development_evaluation: tuple[str, str, str] | None = None
    development_evaluation_scope: str = "decision-v7/development"
    qualification_checks: tuple[tuple[str, str, tuple[str, ...]], ...] = ()

    def required_qualification_checks(self, backend: str, compute_dtype: str) -> tuple[str, ...]:
        for configured_backend, configured_dtype, checks in self.qualification_checks:
            if (backend, compute_dtype) == (configured_backend, configured_dtype):
                return checks
        raise ValueError(
            f"no qualification policy for {self.family_id}/{backend}/{compute_dtype}")


_TORCH_DEVICES = frozenset({"cpu", "mps", "cuda"})
GEMMA3_POLICY = ModelPolicy(
    family_id="gemma3_text",
    supported_compute_dtypes=frozenset({"fp32", "bf16"}),
    implemented=frozenset(
        ("torch", device, dtype)
        for device in _TORCH_DEVICES
        for dtype in (("fp32", "bf16") if device == "mps" else ("fp32",))
    ),
)

# Both backends implement the selected Gemma 4 precision matrix. MLX remains
# GPU-only; its precision-specific trained receipt is checked before save/load.
GEMMA4_POLICY = ModelPolicy(
    family_id="gemma4_e2b_text",
    supported_compute_dtypes=frozenset({"fp32", "bf16"}),
    implemented=(frozenset({("mlx", "gpu", "bf16"), ("mlx", "gpu", "fp32")}) | frozenset(
        ("torch", device, dtype)
        for device in _TORCH_DEVICES
        for dtype in ("fp32", "bf16")
    )),
    source_weights_dtype="bf16",
    required_execution_mode="rows",
    required_microbatch=1,
    forbid_mps_fallback=True,
    forbid_gradient_checkpointing=True,
    bf16_trainable_state_dtype="fp32",
    bf16_optimizer_state_dtype="fp32",
    development_evaluation=("decision-v7", "development", "rows"),
    development_evaluation_scope=(
        "decision-v7/development: source-balanced complete independent rows; "
        "exclude incomplete contrastive pairs"
    ),
    qualification_checks=tuple(
        (backend, compute_dtype,
         ("source_inventory_exact", "decoder_compute_dtype", "fp32_lora_and_pointer",
          "fp32_optimizer_state", "finite_loss_gradients", "nonzero_trainable_update",
          "row_isolation", "development_coverage", "development_mechanisms",
          *backend_checks))
        for backend, backend_checks in (
            ("torch", ("causal_attention_masks",)),
            ("mlx", ("boundary_gathers_valid", "decoder_masks_valid",
                     "shared_kv_mapping_exact")),
        )
        for compute_dtype in ("bf16", "fp32")
    ),
)

MODEL_POLICIES = {policy.family_id: policy for policy in (GEMMA3_POLICY, GEMMA4_POLICY)}


def _device_capability(device: str) -> BackendCapability | None:
    return {
        "cpu": BackendCapability.CPU,
        "mps": BackendCapability.MPS,
        "cuda": BackendCapability.CUDA,
        "gpu": BackendCapability.MLX_GPU,
    }.get(device)


def validate_model_policy(config, resolved_model, *, effective_device: str | None = None) -> None:
    """Reject unsupported family/backend/device/precision recipes before loading weights."""
    if config.runtime.device != "auto":
        capability = _device_capability(config.runtime.device)
        if capability is None or capability not in resolved_model.backend.capabilities:
            raise ValueError(
                f"backend {config.backend.id} does not support runtime device "
                f"{config.runtime.device!r}"
            )

    policy = MODEL_POLICIES.get(config.model.family)
    if policy is None:
        # Custom registry families retain their backend-defined behavior.
        return

    if config.training.dtype not in policy.supported_compute_dtypes:
        raise ValueError(
            f"{policy.family_id} supports compute dtype(s): "
            f"{', '.join(sorted(policy.supported_compute_dtypes))}"
        )
    if config.backend.id == "mlx":
        requested_device = "gpu" if config.runtime.device == "auto" else config.runtime.device
    else:
        requested_device = config.runtime.device

    if config.model.family == "gemma4_e2b_text":
        if (config.model.name, config.model.revision, config.model.expected_model_type) != (
            "google/gemma-4-E2B", GEMMA4_E2B_REVISION, "gemma4"
        ):
            raise ValueError("Gemma 4 E2B requires the pinned PRETRAINED base checkpoint")
        if config.model.marker_ids != dict(
            zip(resolved_model.family.marker_roles, range(6, 11), strict=True)
        ):
            raise ValueError("Gemma 4 E2B requires the verified existing marker rows 6..10")

    mlx_pairing = config.model.family == "gemma4_e2b_text" and config.backend.id == "mlx"
    if mlx_pairing and policy.forbid_mps_fallback and config.runtime.mps_fallback:
        raise ValueError(f"{policy.family_id} does not support Torch MPS fallback")
    if mlx_pairing and policy.forbid_gradient_checkpointing and config.runtime.gradient_checkpointing:
        raise ValueError(f"{policy.family_id} does not support gradient checkpointing")
    if (policy.required_execution_mode is not None
            and config.runtime.execution_mode != policy.required_execution_mode):
        raise ValueError(
            f"{policy.family_id} supports {policy.required_execution_mode} execution only"
        )
    if (mlx_pairing and policy.required_microbatch is not None
            and config.training.microbatch != policy.required_microbatch):
        raise ValueError(
            f"{policy.family_id} currently requires microbatch={policy.required_microbatch}"
        )

    # Auto selection is hardware-dependent. Validate the actual result in
    # ResolvedExperimentConfig.select_device before any model/tokenizer loading.
    if (effective_device is None and config.runtime.device == "auto"
            and config.backend.id == "torch"):
        return
    device = effective_device or requested_device
    combination = (config.backend.id, device, config.training.dtype)
    if combination not in policy.implemented:
        raise ValueError(
            f"unsupported {policy.family_id} backend/device/compute combination: "
            f"{config.backend.id}/{device}/{config.training.dtype}"
        )


def validate_development_evaluation_scope(family_id: str, suite: str, split: str,
                                         execution_mode: str) -> None:
    """Enforce a family's explicitly scoped non-held-out development evaluation."""
    policy = MODEL_POLICIES.get(family_id)
    if policy is not None and policy.development_evaluation is not None:
        expected_suite, expected_split, expected_mode = policy.development_evaluation
        if suite != expected_suite:
            raise ValueError(
                f"{family_id} evaluation is limited to {expected_suite} {expected_split}"
            )
        if split != expected_split:
            raise ValueError(f"{family_id} evaluation is {expected_split}-only")
        if execution_mode != expected_mode:
            raise ValueError(f"{family_id} evaluation supports {expected_mode}-only execution")


def development_only_family(family_id: str) -> bool:
    """Whether study selection is limited to a model's development scope."""
    policy = MODEL_POLICIES.get(family_id)
    return policy is not None and policy.development_evaluation is not None


def auto_device_requires_probe(config) -> bool:
    """Whether auto may choose an unsupported device for this family/precision."""
    if config.runtime.device != "auto":
        return False
    policy = MODEL_POLICIES.get(config.model.family)
    if policy is None:
        return False
    devices = {
        "torch": ("cuda", "mps", "cpu"),
        "mlx": ("gpu",),
    }.get(config.backend.id, ())
    return any((config.backend.id, device, config.training.dtype) not in policy.implemented
               for device in devices)
