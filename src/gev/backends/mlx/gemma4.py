"""Strict native-weight loading and independent-row Gemma 4 inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ...domain.tokenization import rows_of
from ...models.specs import GEMMA4_E2B, GEMMA4_E2B_REVISION
from .pointer import MlxPointerHead

_EXPECTED_TENSORS = 2011
_EXPECTED_TEXT_TENSORS = 600
_EXPECTED_TEXT_SERIALIZED_PARAMETERS = 4_647_449_891
_EXPECTED_TEXT_PARAMETERS = 4_628_569_344
_BOUNDARY_IDS = (6, 7, 8, 9, 10, 239673, 239674, 239675, 239676, 262143)
_TARGETS = frozenset(GEMMA4_E2B.lora_targets)


def _snapshot(model_name: str, revision: str) -> Path:
    from ...infrastructure.network import use_system_ssl

    use_system_ssl()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(model_name, revision=revision)
    if info.sha != revision:
        raise ValueError("Hugging Face resolved a different Gemma 4 revision")
    path = Path(snapshot_download(model_name, revision=revision))
    if path.name != revision:
        raise ValueError("downloaded Gemma 4 snapshot does not match the pinned revision")
    return path


def _validate_source_inventory(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    if config.get("model_type") != "gemma4" or text.get("model_type") != "gemma4_text":
        raise ValueError("checkpoint is not top-level gemma4 with a gemma4_text decoder")
    if any(value for table in (config, text) for key in ("quantization_config", "quantization")
           if (value := table.get(key)) is not None):
        raise ValueError("quantized Gemma 4 checkpoints are not accepted")
    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("pinned Gemma 4 snapshot has no safetensors weights")
    from safetensors import safe_open

    all_keys: set[str] = set()
    text_keys: dict[str, list[int]] = {}
    total_parameters = 0
    for filename in weight_files:
        with safe_open(filename, framework="np") as safe:
            for key in safe.keys():
                if key in all_keys:
                    raise ValueError(f"duplicate Gemma 4 weight tensor: {key}")
                all_keys.add(key)
                shape = list(safe.get_slice(key).get_shape())
                total_parameters += math.prod(shape)
                if safe.get_slice(key).get_dtype() != "BF16":
                    raise ValueError(f"Gemma 4 published weight {key} is not BF16")
                if key.startswith("model.language_model."):
                    text_keys[key.removeprefix("model.language_model.")] = shape
    text_serialized = sum(math.prod(shape) for shape in text_keys.values())
    if (len(all_keys) != _EXPECTED_TENSORS or len(text_keys) != _EXPECTED_TEXT_TENSORS
            or text_serialized != _EXPECTED_TEXT_SERIALIZED_PARAMETERS):
        raise ValueError("Gemma 4 safetensors name/shape inventory differs from the qualified pinned base")
    return {"config": config, "text_shapes": text_keys, "tensor_count": len(all_keys),
            "text_tensor_count": len(text_keys), "text_serialized_parameters": text_serialized,
            "serialized_parameters": total_parameters}


def _shared_kv_extra_names(*, layers: int, shared_layers: int) -> set[str]:
    first_shared = layers - shared_layers
    return {f"language_model.model.layers.{index}.self_attn.{module}.weight"
            for index in range(first_shared, layers)
            for module in ("k_proj", "v_proj", "k_norm")}


def _install_lora(text_model) -> dict:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    keys: set[str] = set()
    target_counts = {target: 0 for target in GEMMA4_E2B.lora_targets}
    for layer in text_model.layers:
        for name, module in layer.named_modules():
            target = name.rsplit(".", 1)[-1]
            if target in _TARGETS and isinstance(module, nn.Linear):
                keys.add(name)
                target_counts[target] += 1
    absent = sorted(target for target, count in target_counts.items() if count == 0)
    if absent:
        raise ValueError("Gemma 4 MLX decoder is missing configured LoRA target linears: "
                         + ", ".join(absent))
    linear_to_lora_layers(text_model, 35, {
        "rank": GEMMA4_E2B.lora_rank,
        "scale": GEMMA4_E2B.lora_alpha / GEMMA4_E2B.lora_rank,
        "dropout": GEMMA4_E2B.lora_dropout,
        "keys": keys,
    })
    expected = sum(target_counts.values())
    # MLX names LoRA factors according to its tuner implementation. Count actual
    # wrapped layers using the concrete type rather than accepting an empty set.
    from mlx_lm.tuner.lora import LoRALinear

    observed = sum(isinstance(module, LoRALinear) for _, module in text_model.named_modules())
    if observed != expected:
        raise ValueError(f"Gemma 4 LoRA target count mismatch: expected {expected}, got {observed}")
    return {"target_counts": target_counts, "layers": expected, "rank": GEMMA4_E2B.lora_rank,
            "alpha": GEMMA4_E2B.lora_alpha, "dropout": GEMMA4_E2B.lora_dropout,
            "targets": sorted(_TARGETS)}


def _verify_marker_embeddings(decoder) -> dict:
    import numpy as np

    result = {}
    for name, weight in (("main", decoder.embed_tokens.weight),
                         ("per_layer", decoder.embed_tokens_per_layer.weight)):
        mx.eval(weight)
        rows = np.asarray(weight[mx.array(_BOUNDARY_IDS, dtype=mx.int32)]
                          .astype(mx.float32), dtype=np.float32)
        marker_rows, high_boundary_rows = rows[:5], rows[5:]
        if (not np.isfinite(rows).all()
                or np.any(np.linalg.norm(rows, axis=-1) == 0)
                or np.any(np.linalg.norm(high_boundary_rows, axis=-1) == 0)):
            raise ValueError(f"Gemma 4 {name} marker/boundary embeddings are non-finite or zero")
        pairwise = [float(np.linalg.norm(marker_rows[i] - marker_rows[j]))
                    for i in range(5) for j in range(i + 1, 5)]
        if min(pairwise) == 0:
            raise ValueError(f"Gemma 4 {name} marker embeddings alias")
        result[name] = {"finite": True, "nonzero": True,
                        "boundary_ids": list(_BOUNDARY_IDS),
                        "boundary_sha256": hashlib.sha256(
                            np.ascontiguousarray(rows).tobytes()).hexdigest(),
                        "high_boundary_nonzero": True,
                        "pairwise_distance_min": min(pairwise),
                        "pairwise_distance_max": max(pairwise)}
    return result


class _TorchCompatibleRMSNorm(nn.Module):
    """Match Transformers Gemma4RMSNorm's explicit float32/pow operation order."""

    def __init__(self, source):
        super().__init__()
        self.eps = source.eps
        self.with_scale = hasattr(source, "weight")
        if self.with_scale:
            self.weight = source.weight

    def __call__(self, value):
        value_float = value.astype(mx.float32)
        mean_square = mx.mean(mx.square(value_float), axis=-1, keepdims=True) + self.eps
        normalized = value_float * mx.power(mean_square, -0.5)
        if self.with_scale:
            normalized = normalized * self.weight.astype(mx.float32)
        return normalized.astype(value.dtype)


class _TorchCompatibleLinear(nn.Module):
    """Use float32 accumulation with output cast to the selected compute dtype."""

    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.has_bias = "bias" in source
        if self.has_bias:
            self.bias = source.bias

    def __call__(self, value):
        output_shape = (*value.shape[:-1], self.weight.shape[0])
        inputs = value.reshape(-1, value.shape[-1]).astype(mx.float32)
        output = mx.matmul(inputs, self.weight.astype(mx.float32).T)
        if self.has_bias:
            output = output + self.bias.astype(mx.float32)
        return output.reshape(output_shape).astype(value.dtype)


def _install_torch_compatible_rmsnorms(decoder) -> int:
    replacements = []
    for name, module in decoder.named_modules():
        if isinstance(module, nn.RMSNorm) or type(module).__name__ == "RMSNormNoScale":
            replacements.append((name, _TorchCompatibleRMSNorm(module)))
    if not replacements:
        raise ValueError("Gemma 4 decoder has no RMSNorm modules to adapt")
    decoder.update_modules(tree_unflatten(replacements), strict=True)
    return len(replacements)


def _install_torch_compatible_ple_projection(decoder) -> None:
    source = decoder.per_layer_model_projection
    decoder.update_modules({
        "per_layer_model_projection": _TorchCompatibleLinear(source),
    }, strict=True)


def load_gemma4_backbone(model_name: str, revision: str, *, compute_dtype: str = "bf16"):
    """Load the exact pinned BF16 source in native BF16 or optional FP32 diagnostics."""
    if compute_dtype not in {"bf16", "fp32"}:
        raise ValueError("Gemma 4 source weights are BF16; compute dtype must be bf16 or fp32")
    path = _snapshot(model_name, revision)
    inventory = _validate_source_inventory(path)
    from mlx_lm.utils import _get_classes, load_config
    from mlx.utils import tree_flatten

    config = load_config(path)
    if config.get("model_type") != "gemma4":
        raise ValueError("MLX-LM did not load the pinned gemma4 architecture")
    model_class, args_class = _get_classes(config)
    base = model_class(args_class.from_dict(config))
    weights = {}
    for filename in sorted(path.glob("model*.safetensors")):
        weights.update(mx.load(str(filename)))
    sanitized = base.sanitize(weights) if hasattr(base, "sanitize") else weights
    expected_shapes = {name: tuple(value.shape) for name, value in tree_flatten(base.parameters())}
    expected_names = set(expected_shapes)
    extra_names = set(sanitized) - expected_names
    missing_names = expected_names - set(sanitized)
    allowed_extras = _shared_kv_extra_names(
        layers=int(config["text_config"]["num_hidden_layers"]),
        shared_layers=int(config["text_config"]["num_kv_shared_layers"]))
    if extra_names != allowed_extras or missing_names:
        raise ValueError("strict MLX Gemma 4 name inventory mismatch "
                         f"(missing={len(missing_names)}, unexpected={len(extra_names)})")
    # The 60 per-layer k/v projection and k-norm copies for shared-KV layers
    # are redundant in MLX's shared-cache decoder and were verified above.
    # All remaining base weights are loaded with MLX strict=True shape checking.
    for name in allowed_extras:
        sanitized.pop(name)
    if set(sanitized) != set(expected_shapes):
        raise ValueError("strict MLX Gemma 4 tensor names changed during shared-KV filtering")
    names = sorted(sanitized)
    for start in range(0, len(names), 32):
        tensors = []
        for name in names[start:start + 32]:
            value = sanitized.pop(name)
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(f"strict MLX Gemma 4 shape mismatch for {name}")
            value = value.astype(mx.bfloat16 if compute_dtype == "bf16" else mx.float32)
            tensors.append((name, value))
        base.update(tree_unflatten(tensors), strict=False)
    del sanitized, weights, expected_shapes
    mx.eval(base.parameters())
    if not hasattr(base, "language_model") or not hasattr(base.language_model, "model"):
        raise ValueError("MLX Gemma 4 model has no text-only decoder path")
    decoder = base.language_model.model
    if decoder.config.num_hidden_layers != 35 or decoder.config.hidden_size != 1536:
        raise ValueError("loaded Gemma 4 text decoder configuration mismatch")
    loaded = tree_flatten(decoder.parameters())
    layer_scalars = [(name, value) for name, value in loaded if name.endswith("layer_scalar")]
    loaded_parameters = sum(value.size for name, value in loaded if not name.endswith("layer_scalar"))
    if loaded_parameters != _EXPECTED_TEXT_PARAMETERS or len(layer_scalars) != 35:
        raise ValueError("loaded Gemma 4 text decoder parameter inventory mismatch: "
                         f"expected {_EXPECTED_TEXT_PARAMETERS} weights and 35 layer scalars, "
                         f"got {loaded_parameters} weights and {len(layer_scalars)} scalars")
    expected_dtype = mx.bfloat16 if compute_dtype == "bf16" else mx.float32
    for name, value in loaded:
        if value.dtype != expected_dtype:
            raise ValueError(f"loaded base tensor {name} is not {compute_dtype}")
    rmsnorm_count = _install_torch_compatible_rmsnorms(decoder)
    _install_torch_compatible_ple_projection(decoder)
    marker_embeddings = _verify_marker_embeddings(decoder)
    decoder.freeze()
    lora = _install_lora(decoder)
    inventory_sha256 = hashlib.sha256(json.dumps(
        inventory["text_shapes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    base_qualification = {
        "revision": revision, "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
        "inventory": {"tensor_count": inventory["tensor_count"],
                      "text_tensor_count": inventory["text_tensor_count"],
                      "text_serialized_parameters": inventory["text_serialized_parameters"],
                      "effective_text_parameters": _EXPECTED_TEXT_PARAMETERS,
                      "name_shape_sha256": inventory_sha256},
        "lora": lora, "marker_embeddings": marker_embeddings,
        "rmsnorm_policy": "torch-fp32-pow-v1", "rmsnorm_count": rmsnorm_count,
        "ple_projection_policy": "torch-fp32-accumulate-bf16-output-v1" if compute_dtype == "bf16"
                                  else "torch-fp32-accumulate-fp32-output-v1",
    }
    return {"model": base, "path": str(path), "revision": revision,
            "inventory": {key: value for key, value in inventory.items() if key != "config"},
            "lora": lora, "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
            "marker_embeddings": marker_embeddings,
            "rmsnorm_policy": "torch-fp32-pow-v1", "rmsnorm_count": rmsnorm_count,
            "ple_projection_policy": "torch-fp32-accumulate-bf16-output-v1" if compute_dtype == "bf16"
                                      else "torch-fp32-accumulate-fp32-output-v1",
            "base_qualification": base_qualification}


class Gemma4RowModel(nn.Module):
    """One independent state-plus-question decoder call per decision row."""

    backend_id: str = "mlx"

    def __init__(self, loaded: dict, *, family=GEMMA4_E2B, temperature: float = 1.0):
        super().__init__()
        self.base = loaded["model"]
        self.head = MlxPointerHead(1536, family.pointer_width, temperature)
        self.family = family
        self.provenance = {key: value for key, value in loaded.items() if key != "model"}
        self.provenance["model_output_id"] = family.output_model_id
        self.execution_mode = "rows"

    @property
    def decoder(self):
        return self.base.language_model.model

    def forward_one(self, encoded: dict) -> list:
        """Decode questions as isolated state-plus-question rows in input order."""
        return self._forward_one_rows(encoded)

    def _forward_one_rows(self, encoded: dict) -> list:
        state_ids, _state_positions, questions = rows_of(encoded)
        output = []
        for question in questions:
            ids = mx.array([state_ids + question["ids"]], dtype=mx.int32)
            hidden = self.decoder(ids)
            decide = len(state_ids) + question["decide"]
            options = [len(state_ids) + index for index in question["opts"]]
            output.append(self.head(hidden[0, decide], hidden[0, options]))
        return output

    def forward_rows_batch(self, encodings: list[dict]) -> list[list]:
        return [self.forward_one(encoded) for encoded in encodings]

    def forward_batch(self, encodings: list[dict], *, execution_mode: str = "rows") -> list[list]:
        if execution_mode != "rows":
            raise ValueError("Gemma 4 MLX supports rows execution only")
        return self.forward_rows_batch(encodings)

    def probs(self, encodings: dict | list[dict]):
        logits = self.forward_one(encodings) if isinstance(encodings, dict) else self.forward_rows_batch(encodings)
        if isinstance(encodings, dict):
            return [self.head.probabilities(value) for value in logits]
        return [[self.head.probabilities(value) for value in record] for record in logits]
