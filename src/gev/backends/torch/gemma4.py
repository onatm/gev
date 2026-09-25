"""Text-only Torch Gemma 4 loading and independent-row fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from ...models.specs import GEMMA4_E2B, GEMMA4_E2B_REVISION
from .gemma3 import GemmaRowModel

_EXPECTED_TEXT_TENSORS = 600
_EXPECTED_TEXT_SERIALIZED_PARAMETERS = 4_647_449_891
_EXPECTED_TEXT_PARAMETERS = 4_628_569_344


def _snapshot(model_name: str, revision: str) -> Path:
    from ...infrastructure.network import use_system_ssl

    use_system_ssl()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(model_name, revision=revision)
    if info.sha != revision:
        raise ValueError("Hugging Face resolved a different Gemma 4 revision")
    siblings = {item.rfilename for item in info.siblings}
    index_names = sorted(name for name in siblings if name.endswith(".safetensors.index.json"))
    if index_names:
        index_name = index_names[0]
        path = Path(snapshot_download(
            model_name, revision=revision, allow_patterns=["config.json", index_name]))
        index = json.loads((path / index_name).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Gemma 4 safetensors index has no weight map")
        text_entries = {key: filename for key, filename in weight_map.items()
                        if key.startswith("model.language_model.")}
        if len(weight_map) != 2011 or len(text_entries) != _EXPECTED_TEXT_TENSORS:
            raise ValueError("pinned Gemma 4 safetensors index inventory mismatch")
        text_shards = sorted(set(text_entries.values()))
        if not text_shards or not set(text_shards) <= siblings:
            raise ValueError("Gemma 4 index has no complete text-decoder shard inventory")
        patterns = ["config.json", index_name, *text_shards]
        path = Path(snapshot_download(
            model_name, revision=revision, allow_patterns=patterns))
    else:
        weight_files = sorted(name for name in siblings if name.endswith(".safetensors"))
        if not weight_files:
            raise ValueError("pinned Gemma 4 snapshot has no safetensors weights")
        patterns = ["config.json", *weight_files]
        path = Path(snapshot_download(
            model_name, revision=revision, allow_patterns=patterns))
    if path.name != revision:
        raise ValueError("downloaded Gemma 4 snapshot does not match the pinned revision")
    return path


def _shared_kv_extra_names(*, layers: int, shared_layers: int) -> set[str]:
    first_shared = layers - shared_layers
    return {
        f"layers.{index}.self_attn.{module}.weight"
        for index in range(first_shared, layers)
        for module in ("k_proj", "v_proj", "k_norm")
    }


def _load_gemma4_snapshot(root: str | Path, model_name: str, revision: str, *,
                          compute_dtype: str, attn_implementation: str,
                          gradient_checkpointing: bool,
                          enforce_pinned_inventory: bool) -> nn.Module:
    """Private local-snapshot seam; only production entry enforces pinned sizes."""
    if (model_name, revision) != ("google/gemma-4-E2B", GEMMA4_E2B_REVISION):
        raise ValueError("Torch Gemma 4 requires the pinned google/gemma-4-E2B revision")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(compute_dtype)
    if dtype is None:
        raise ValueError("Gemma 4 compute dtype must be fp32 or bf16")

    root = Path(root)
    from transformers import AutoConfig, Gemma4TextModel

    config = AutoConfig.from_pretrained(root)
    text_config = getattr(config, "text_config", None)
    if (getattr(config, "model_type", None) != "gemma4"
            or getattr(text_config, "model_type", None) != "gemma4_text"):
        raise ValueError("Torch Gemma 4 requires nested Gemma4TextConfig")
    if (enforce_pinned_inventory
            and (int(text_config.hidden_size) != 1536
                 or int(text_config.num_hidden_layers) != 35)):
        raise ValueError("pinned Gemma 4 text config dimensions mismatch")
    text_config.use_cache = False
    text_config._attn_implementation = attn_implementation
    model = Gemma4TextModel(text_config).to(device="cpu", dtype=dtype)
    expected = model.state_dict()
    seen: set[str] = set()
    shapes: dict[str, list[int]] = {}
    text_parameters = 0
    from safetensors import safe_open

    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("pinned Gemma 4 snapshot has no safetensors weights")
    with torch.no_grad():
        for filename in weight_files:
            with safe_open(filename, framework="pt", device="cpu") as safe:
                for key in safe.keys():
                    if safe.get_slice(key).get_dtype() != "BF16":
                        raise ValueError(f"Gemma 4 published weight {key} is not BF16")
                    if not key.startswith("model.language_model."):
                        continue
                    name = key.removeprefix("model.language_model.")
                    if name in seen:
                        raise ValueError(f"duplicate Gemma 4 text tensor: {name}")
                    seen.add(name)
                    shape = list(safe.get_slice(key).get_shape())
                    shapes[name] = shape
                    text_parameters += math.prod(shape)
                    if name in expected:
                        if tuple(expected[name].shape) != tuple(shape):
                            raise ValueError(f"Torch Gemma 4 source shape mismatch for {name}")
                        expected[name].copy_(safe.get_tensor(key).to(dtype=dtype))

    layers = int(text_config.num_hidden_layers)
    allowed_extras = _shared_kv_extra_names(
        layers=layers, shared_layers=int(text_config.num_kv_shared_layers))
    missing = set(expected) - seen
    extras = seen - set(expected)
    if missing or extras != allowed_extras:
        raise ValueError(
            "Torch Gemma 4 tensor inventory mismatch "
            f"(missing={len(missing)}, unexpected={len(extras)})"
        )
    effective_parameters = sum(parameter.numel() for parameter in model.parameters())
    if enforce_pinned_inventory and (
            len(shapes) != _EXPECTED_TEXT_TENSORS
            or text_parameters != _EXPECTED_TEXT_SERIALIZED_PARAMETERS
            or effective_parameters != _EXPECTED_TEXT_PARAMETERS):
        raise ValueError(
            "pinned Gemma 4 text inventory mismatch "
            f"(tensors={len(shapes)}, serialized={text_parameters}, "
            f"effective={effective_parameters})"
        )
    if any(parameter.dtype != dtype for parameter in model.state_dict().values()):
        raise ValueError(f"Torch Gemma 4 decoder was not materialized in {compute_dtype}")
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model._gev_source = {
        "model": model_name,
        "revision": revision,
        "source_weights_dtype": "bf16",
        "compute_dtype": compute_dtype,
        "source_text_tensor_count": len(shapes),
        "source_text_parameters": text_parameters,
        "effective_text_parameters": effective_parameters,
        "source_name_shape_sha256": hashlib.sha256(json.dumps(
            shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return model


def load_gemma4_backbone(model_name: str, revision: str, *, compute_dtype: str = "fp32",
                         attn_implementation: str = "eager",
                         gradient_checkpointing: bool = False) -> nn.Module:
    """Stream pinned BF16 decoder shards into a text-only Gemma4TextModel."""
    if (model_name, revision) != ("google/gemma-4-E2B", GEMMA4_E2B_REVISION):
        raise ValueError("Torch Gemma 4 requires the pinned google/gemma-4-E2B revision")
    return _load_gemma4_snapshot(
        _snapshot(model_name, revision), model_name, revision,
        compute_dtype=compute_dtype, attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing, enforce_pinned_inventory=True)


class Gemma4RowModel(GemmaRowModel):
    """Gemma 4 text rows with BF16 decoder or FP32 decoder and FP32 pointer."""

    backend_id = "torch"

    def __init__(self, backbone: nn.Module, *, compute_dtype: str = "fp32",
                 temperature: float = 1.0, use_peft: bool = True,
                 gradient_checkpointing: bool = False) -> None:
        if compute_dtype not in {"fp32", "bf16"}:
            raise ValueError("Gemma 4 compute dtype must be fp32 or bf16")
        if getattr(backbone.config, "model_type", None) != "gemma4_text":
            raise ValueError("Torch Gemma 4 requires a Gemma4TextModel backbone")
        source_provenance = getattr(backbone, "_gev_source", None)
        backbone.to(dtype=torch.bfloat16 if compute_dtype == "bf16" else torch.float32)
        super().__init__(backbone, temperature=temperature, use_peft=use_peft,
                         gradient_checkpointing=gradient_checkpointing,
                         family=GEMMA4_E2B)
        # PEFT's default adapter autocast promotes LoRA parameters to FP32;
        # restore it explicitly when loading an already wrapped fixture/model.
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        self.compute_dtype = compute_dtype
        self.native_bf16_compute = compute_dtype == "bf16"
        self.source_provenance = (dict(source_provenance)
                                  if isinstance(source_provenance, dict) else None)
        if any(parameter.dtype != torch.float32 for parameter in self.head.parameters()):
            raise ValueError("Gemma 4 pointer head must remain FP32")

    def forward_batch(self, encodings: list[dict], *, execution_mode: str = "rows"):
        if execution_mode != "rows":
            raise ValueError("Torch Gemma 4 supports rows execution only")
        return self.forward_rows_batch(encodings)

    def forward_packed_batch(self, encodings: list[dict]):
        raise ValueError("Torch Gemma 4 supports rows execution only")


def _tiny_config(*, hidden_size: int = 32, layers: int = 2):
    from transformers import Gemma4TextConfig

    if hidden_size not in {16, 32} or layers not in {2, 4}:
        raise ValueError("tiny Gemma 4 requires hidden_size 16/32 and 2/4 layers")
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=hidden_size // 8,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention"] * (layers - 1) + ["full_attention"],
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=0,
        use_cache=False,
    )


def build_tiny_gemma4_model(*, compute_dtype: str = "fp32", hidden_size: int = 32,
                            layers: int = 2, temperature: float = 1.0) -> Gemma4RowModel:
    """Random text-only fixture for tests; never a substitute for the pinned base."""
    from transformers import Gemma4TextModel

    torch.manual_seed(0)
    return Gemma4RowModel(Gemma4TextModel(_tiny_config(hidden_size=hidden_size, layers=layers)),
                          compute_dtype=compute_dtype, temperature=temperature)
