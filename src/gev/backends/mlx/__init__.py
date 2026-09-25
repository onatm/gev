"""Native MLX backend for pinned text-only Gemma 4."""

from __future__ import annotations

import random

from ...models.specs import BackendCapability, ModelFamilySpec


def _require_tf32_mode() -> None:
    from .qualification import require_tf32_disabled_before_mlx_import

    require_tf32_disabled_before_mlx_import()


class MlxBackend:
    backend_id = "mlx"
    capabilities = frozenset({
        BackendCapability.CAUSAL_LM,
        BackendCapability.LORA_ADAPTERS,
        BackendCapability.AUTOGRAD,
        BackendCapability.MLX_GPU,
        BackendCapability.BF16_SOURCE_WEIGHTS,
    })
    families = frozenset({"gemma4_e2b_text"})
    architectures = frozenset({"gemma4"})

    def seed_rng(self, seed: int) -> None:
        _require_tf32_mode()
        import mlx.core as mx

        mx.random.seed(seed)
        random.seed(seed)

    def select_device(self, requested: str) -> str:
        if requested not in {"auto", "gpu"}:
            raise ValueError(f"MLX backend does not support runtime device {requested!r}; use gpu")
        _require_tf32_mode()
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 4 MLX requires the optional mlx extra on Apple Silicon (macOS arm64)"
            ) from exc

        if not mx.metal.is_available():
            raise RuntimeError("MLX GPU was requested, but Metal is unavailable; refusing CPU fallback")
        return "gpu"

    def create_model(self, family: ModelFamilySpec, *, model_name: str, revision: str,
                     temperature: float = 1.0, backbone: object | None = None,
                     attn_implementation: str = "eager", gradient_checkpointing: bool = False,
                     compute_dtype: str = "bf16",
                     seed: int | None = None):
        if family.family_id not in self.families or family.architecture not in self.architectures:
            raise ValueError(f"MLX backend does not implement family {family.family_id}")
        if model_name != "google/gemma-4-E2B" or revision != "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f":
            raise ValueError("MLX Gemma 4 requires the pinned google/gemma-4-E2B PRETRAINED revision")
        if compute_dtype not in {"bf16", "fp32"}:
            raise ValueError("Gemma 4 MLX compute dtype must be bf16 or fp32")
        self.select_device("gpu")
        if seed is not None:
            self.seed_rng(seed)
        from .gemma4 import Gemma4RowModel, load_gemma4_backbone

        loaded = (load_gemma4_backbone(model_name, revision, compute_dtype=compute_dtype)
                  if backbone is None else backbone)
        if (not isinstance(loaded, dict) or "model" not in loaded
                or loaded.get("source_weights_dtype") != "bf16"
                or loaded.get("compute_dtype") != compute_dtype):
            raise ValueError("MLX Gemma 4 loader source/compute dtype does not match selection")
        return Gemma4RowModel(loaded, family=family, temperature=temperature)

    def create_predictor(self, model, tokenizer, markers, **kwargs):
        _require_tf32_mode()
        from .predictor import MlxPredictor

        return MlxPredictor(model, tokenizer, markers, **kwargs)

    def save_checkpoint(self, model, directory, metadata, tokenizer=None):
        _require_tf32_mode()
        from .checkpoint import save_checkpoint

        return save_checkpoint(model, directory, metadata, tokenizer)

    def load_checkpoint(self, directory, *, config, device="gpu", tokenizer=None,
                        expected_marker_map=None, backbone_loader=None, attn_implementation=None,
                        compute_dtype="bf16"):
        _require_tf32_mode()
        from .checkpoint import load_checkpoint

        return load_checkpoint(directory, config=config, device=device, tokenizer=tokenizer,
                               expected_marker_map=expected_marker_map,
                               backbone_loader=backbone_loader,
                               compute_dtype=compute_dtype)

    def checkpoint_fingerprint(self, directory):
        from ...artifacts.checkpoint_identity import checkpoint_fingerprint

        return checkpoint_fingerprint(directory)

    def trainable_fingerprint(self, model) -> str:
        _require_tf32_mode()
        from ...models.qualification import trainable_content_sha256

        return trainable_content_sha256("mlx", model)

    def train(self, model, schedule, config, output, **kwargs):
        _require_tf32_mode()
        from .training import train

        return train(model, schedule, config, output, **kwargs)
