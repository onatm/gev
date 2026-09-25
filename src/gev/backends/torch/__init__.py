"""Current PyTorch model-construction backend (CPU, MPS, and CUDA)."""

from __future__ import annotations

import random
import hashlib
from typing import Any

from ...models.specs import BackendCapability, ModelFamilySpec


class TorchBackend:
    backend_id = "torch"
    capabilities = frozenset({
        BackendCapability.AUTOGRAD,
        BackendCapability.CAUSAL_LM,
        BackendCapability.LORA_ADAPTERS,
        BackendCapability.CPU,
        BackendCapability.MPS,
        BackendCapability.CUDA,
        BackendCapability.TRAIN_PROFILING,
        BackendCapability.PRECISION_DIAGNOSTICS,
        BackendCapability.EXECUTION_DIAGNOSTICS,
    })
    families = frozenset({"gemma3_text", "gemma4_e2b_text"})
    architectures = frozenset({"gemma3_text", "gemma4"})

    def seed_rng(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        random.seed(seed)

    def create_model(self, family: ModelFamilySpec, *, model_name: str, revision: str,
                     temperature: float = 1.0, backbone: object | None = None,
                     attn_implementation: str = "eager",
                     gradient_checkpointing: bool = False,
                     seed: int | None = None,
                     compute_dtype: str | None = None) -> Any:
        """Construct a family model using this backend's tensor/autograd stack."""
        if family.family_id not in self.families:
            raise ValueError(f"torch backend does not implement model family: {family.family_id}")
        if family.architecture not in self.architectures:
            raise ValueError(f"torch backend does not implement architecture: {family.architecture}")
        if seed is not None:
            self.seed_rng(seed)
        if family.family_id == "gemma4_e2b_text":
            if (model_name, revision) != (
                "google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
            ):
                raise ValueError("Torch Gemma 4 requires the pinned google/gemma-4-E2B revision")
            from .gemma4 import Gemma4RowModel, load_gemma4_backbone

            compute_dtype = "fp32" if compute_dtype is None else compute_dtype
            loaded = (load_gemma4_backbone(
                model_name, revision, compute_dtype=compute_dtype,
                attn_implementation=attn_implementation,
                gradient_checkpointing=False)
                if backbone is None else backbone)
            return Gemma4RowModel(
                loaded, compute_dtype=compute_dtype, temperature=temperature,
                use_peft=not loaded.__class__.__module__.startswith("peft."),
                gradient_checkpointing=gradient_checkpointing)

        from .gemma3 import GemmaRowModel, load_real_backbone

        if backbone is None:
            backbone = load_real_backbone(
                model_name, revision, attn_implementation=attn_implementation,
                gradient_checkpointing=gradient_checkpointing)
        return GemmaRowModel(backbone, temperature=temperature,
                             use_peft=not backbone.__class__.__module__.startswith("peft."))

    def select_device(self, requested: str) -> str:
        import torch

        if requested == "auto":
            if torch.cuda.is_available() and BackendCapability.CUDA in self.capabilities:
                return "cuda"
            if torch.backends.mps.is_available() and BackendCapability.MPS in self.capabilities:
                return "mps"
            return "cpu"
        if requested == "cpu" and BackendCapability.CPU in self.capabilities:
            return "cpu"
        if requested == "mps" and BackendCapability.MPS in self.capabilities:
            if not torch.backends.mps.is_available():
                raise RuntimeError("runtime.device=mps was requested, but MPS is unavailable; refusing fallback")
            return "mps"
        if requested == "cuda" and BackendCapability.CUDA in self.capabilities:
            if not torch.cuda.is_available():
                raise RuntimeError("runtime.device=cuda was requested, but CUDA is unavailable; refusing fallback")
            return "cuda"
        raise ValueError(f"torch backend does not support runtime device {requested!r}")

    def save_checkpoint(self, model: object, directory, metadata: dict, tokenizer=None):
        from .checkpoint import save_checkpoint

        return save_checkpoint(model, directory, metadata, tokenizer)

    def load_checkpoint(self, directory, *, config, device="cpu", tokenizer=None,
                        expected_marker_map=None, backbone_loader=None,
                        attn_implementation=None, compute_dtype=None):
        from .checkpoint import load_checkpoint

        return load_checkpoint(directory, config=config, device=device, tokenizer=tokenizer,
                                expected_marker_map=expected_marker_map,
                                backbone_loader=backbone_loader,
                                attn_implementation=attn_implementation,
                                compute_dtype=compute_dtype)

    def checkpoint_fingerprint(self, directory):
        from .checkpoint import checkpoint_fingerprint

        return checkpoint_fingerprint(directory)

    def trainable_fingerprint(self, model: object) -> str:
        if getattr(getattr(model, "family", None), "family_id", None) != "gemma4_e2b_text":
            payload = b"".join(
                tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(model.state_dict().items())
                if "lora_" in name or name.startswith("head.")
            )
            return hashlib.sha256(payload).hexdigest()
        from ...models.qualification import trainable_content_sha256

        return trainable_content_sha256("torch", model)

    def create_predictor(self, model: object, tokenizer: object, markers: object, *,
                         state_cap: int, branch_cap: int, packed_cap: int,
                         temperature: float = 1.0, execution_mode: str = "rows",
                         encoder=None) -> object:
        from .predictor import TorchPredictor

        return TorchPredictor(model, tokenizer, markers, state_cap=state_cap,
                              branch_cap=branch_cap, packed_cap=packed_cap,
                              temperature=temperature, execution_mode=execution_mode,
                              encoder=encoder)

    def train(self, model: object, schedule, config, output, **kwargs):
        from .training import train

        return train(model, schedule, config, output, **kwargs)

    def profile_train(self, config, warmup_steps: int, measure_steps: int,
                      output, *, data_root: str = "data") -> dict:
        from .profiling import profile_train

        return profile_train(config, warmup_steps, measure_steps, output,
                             data_root=data_root)

    def compare_precision_profiles(self, model, encodings, record_ids, *,
                                   device: str, include_bf16: bool = True):
        from .precision import compare_precision_profiles

        return compare_precision_profiles(model, encodings, record_ids,
                                          device=device, include_bf16=include_bf16)

    def measure_execution(self, model, encodings, *, records: int, warmup: int = 2) -> dict:
        from .execution import measure

        return measure(model, encodings, records=records, warmup=warmup)
