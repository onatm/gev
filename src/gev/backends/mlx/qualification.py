"""Single precision-keyed engineering qualification protocol for Gemma 4 MLX."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ...data.suites import REQUIRED_FILE_HASHES
from ...domain.tokenization import rows_of


def load_torch_cpu_oracle(snapshot: str | Path, *, compute_dtype: str = "fp32"):
    """Load pinned BF16 source tensors into a frozen FP32 Torch CPU oracle."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4TextModel

    if compute_dtype != "fp32":
        raise ValueError("the active Gemma 4 qualification supports FP32 compute only")
    root = Path(snapshot)
    config = AutoConfig.from_pretrained(root)
    if (getattr(config, "model_type", None) != "gemma4"
            or getattr(config.text_config, "model_type", None) != "gemma4_text"):
        raise ValueError("Torch oracle requires nested Gemma4TextConfig")
    model = Gemma4TextModel(config.text_config).to(dtype=torch.float32, device="cpu").eval()
    expected = model.state_dict()
    seen, source_shapes = set(), {}
    source_tensor_count = source_text_parameters = 0
    for filename in sorted(root.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as safe:
            for key in safe.keys():
                source_tensor_count += 1
                if not key.startswith("model.language_model."):
                    continue
                name = key.removeprefix("model.language_model.")
                seen.add(name)
                shape = tuple(safe.get_slice(key).get_shape())
                source_shapes[name] = list(shape)
                source_text_parameters += math.prod(shape)
                if name in expected and tuple(expected[name].shape) != shape:
                    raise ValueError(f"Torch oracle shape mismatch for {name}")
                if safe.get_slice(key).get_dtype() != "BF16":
                    raise ValueError(f"Torch oracle source weight {name} is not published BF16")
                if name in expected:
                    with torch.no_grad():
                        expected[name].copy_(safe.get_tensor(key).to(dtype=torch.float32))
    missing = set(expected) - seen
    extras = seen - set(expected)
    first_shared = config.text_config.num_hidden_layers - config.text_config.num_kv_shared_layers
    allowed_extras = {f"layers.{index}.self_attn.{module}.weight"
                      for index in range(first_shared, config.text_config.num_hidden_layers)
                      for module in ("k_proj", "v_proj", "k_norm")}
    if missing or extras != allowed_extras:
        raise ValueError(f"Torch oracle tensor inventory mismatch (missing={len(missing)}, extras={len(extras)})")
    if any(value.dtype != torch.float32 for value in model.state_dict().values()):
        raise ValueError("Torch oracle weights are not FP32 compute")
    model._gev_qualification = {
        "source_tensor_count": source_tensor_count,
        "source_text_tensor_count": len(source_shapes),
        "source_text_parameters": source_text_parameters,
        "source_name_shape_sha256": hashlib.sha256(json.dumps(
            source_shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "effective_text_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_revision": BASE_REVISION,
        "source_dtype": "bf16",
        "compute_dtype": "fp32",
    }
    return model


def _compare_attention_masks(torch_model, ids: list[int]) -> dict:
    import numpy as np
    import mlx.core as mx
    import torch
    from mlx_lm.models.base import create_attention_mask
    from transformers.masking_utils import (create_causal_mask,
                                            create_sliding_window_causal_mask)

    length = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long)
    positions = torch.arange(length, dtype=torch.long).unsqueeze(0)
    config = getattr(torch_model, "config", torch_model)
    with torch.inference_mode():
        embeddings = torch.zeros((1, length, config.hidden_size), dtype=torch.float32)
        mask_arguments = {"config": config, "inputs_embeds": embeddings,
                          "attention_mask": torch.ones_like(input_ids),
                          "past_key_values": None, "position_ids": positions,
                          "allow_is_causal_skip": False}
        torch_masks = {
            "full_attention": create_causal_mask(**mask_arguments),
            "sliding_attention": create_sliding_window_causal_mask(**mask_arguments),
        }
    mlx_hidden = mx.zeros((1, length, config.hidden_size), dtype=mx.float32)
    result = {}
    for kind, window in (("full_attention", None), ("sliding_attention", 512)):
        torch_mask = torch_masks[kind]
        if torch_mask is None or not isinstance(torch_mask, torch.Tensor):
            raise ValueError(f"Torch {kind} mask was not materialized")
        torch_allowed = (torch_mask if torch_mask.dtype == torch.bool else torch_mask == 0)
        torch_allowed = torch_allowed.squeeze().cpu().numpy()
        mlx_mask = create_attention_mask(mlx_hidden, window_size=window, return_array=True)
        mx.eval(mlx_mask)
        mlx_allowed = np.asarray(mlx_mask, dtype=np.bool_)
        result[kind] = {"match": bool(torch_allowed.shape == mlx_allowed.shape
                                      and np.array_equal(torch_allowed, mlx_allowed)),
                         "shape": list(mlx_allowed.shape),
                         "allowed_positions": int(mlx_allowed.sum())}
    return result

QUALIFICATION_PROTOCOL = "mlx-gemma4-qualification"
BASE_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
DEVELOPMENT_SHA256 = REQUIRED_FILE_HASHES["decision-v7"]["development"]
DATASET_MANIFEST_SHA256 = "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
SELECTION_ALGORITHM = "source-balanced-shortest-longest-canonical-json-v1"
SELECTED_IDS_SHA256 = "b7d1717e7ff124cf881b5a261e9279491839fea5fb8dfbcb4c7da6ab19b15123"
POINTER_HEAD_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
SELECTED_LAYER_INDICES = (0, 14, 34)
BOUNDARY_IDS = (6, 7, 8, 9, 10, 239673, 239674, 239675, 239676, 262143)


@dataclass(frozen=True)
class PrecisionGatePolicy:
    compute_dtype: str
    gate_kind: str
    max_abs_probability: float | None = None
    require_argmax_agreement: bool = True


FP32_DIAGNOSTIC_GATE = PrecisionGatePolicy(
    compute_dtype="fp32", gate_kind="independent_implementation_diagnostic",
    require_argmax_agreement=False)
@dataclass(frozen=True)
class QualificationPolicy:
    protocol: str = QUALIFICATION_PROTOCOL
    protocol_version: int = 1
    family: str = "gemma4_e2b_text"
    backend: str = "mlx"
    base_model: str = "google/gemma-4-E2B"
    base_revision: str = BASE_REVISION
    source_weights_dtype: str = "bf16"
    dataset_suite: str = "decision-v7"
    dataset_split: str = "development"
    dataset_split_sha256: str = DEVELOPMENT_SHA256
    dataset_manifest_sha256: str = DATASET_MANIFEST_SHA256
    selection_algorithm: str = SELECTION_ALGORITHM
    selected_ids_sha256: str = SELECTED_IDS_SHA256
    rows_per_source: int = 2
    require_multi_question_row: bool = True
    boundary_ids: tuple[int, ...] = BOUNDARY_IDS
    pointer_head_seeds: tuple[int, ...] = POINTER_HEAD_SEEDS
    selected_layer_indices: tuple[int, ...] = SELECTED_LAYER_INDICES
    precision_gates: tuple[PrecisionGatePolicy, ...] = (FP32_DIAGNOSTIC_GATE,)
    mlx_tf32_disabled: bool = True
    exact_weight_shapes: bool = True
    exact_marker_and_boundary_gathers: bool = True
    exact_attention_masks: bool = True
    exact_shared_kv_mapping: bool = True
    expected_trainable_lora_matrices: int = 205
    pointer_width: int = 256
    expected_shared_kv_sources: tuple[tuple[str, int], ...] = (
        ("full_attention", 14), ("sliding_attention", 13))

    def gate_for(self, compute_dtype: str) -> PrecisionGatePolicy:
        for gate in self.precision_gates:
            if gate.compute_dtype == compute_dtype:
                return gate
        raise ValueError(f"no Gemma 4 qualification gate for compute dtype {compute_dtype}")

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


POLICY = QualificationPolicy()
POLICY_SHA256 = POLICY.sha256()


def require_tf32_disabled_before_mlx_import() -> None:
    """Freeze MLX matmul mode before MLX initializes a Metal context."""
    imported = "mlx.core" in sys.modules
    configured = os.environ.get("MLX_ENABLE_TF32")
    if configured is not None and configured != "0":
        raise RuntimeError("Gemma 4 MLX requires MLX_ENABLE_TF32=0 before importing mlx.core")
    if imported and configured != "0":
        raise RuntimeError(
            "Gemma 4 MLX requires MLX_ENABLE_TF32=0 before importing mlx.core; "
            "restart in a fresh process with that environment"
        )
    os.environ["MLX_ENABLE_TF32"] = "0"


def select_development_rows(rows: list[dict]) -> list[dict]:
    from ...evaluation.development import select_development_rows as select

    return select(rows, require_multi_question_row=POLICY.require_multi_question_row)


def selected_ids_sha256(rows: list[dict]) -> str:
    from ...evaluation.development import selected_ids_sha256 as digest

    return digest(rows)


def validate_development_selection(rows: list[dict]) -> list[dict]:
    selected = select_development_rows(rows)
    actual = selected_ids_sha256(selected)
    if actual != POLICY.selected_ids_sha256:
        raise ValueError(f"development selection digest mismatch: {actual}")
    return selected


def qualification_environment() -> dict:
    require_tf32_disabled_before_mlx_import()
    import mlx.core as mx

    try:
        metal_device = mx.device_info()
    except (AttributeError, RuntimeError):
        metal_device = None
    metal_device = json.loads(json.dumps(metal_device, default=str))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "mlx_device": str(mx.default_device()),
        "metal_device": metal_device,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
    }


def qualification_code_sha256(repo_root: str | Path) -> str:
    root = Path(repo_root)
    source_paths = (
        "src/gev/backends/mlx/gemma4.py",
        "src/gev/backends/mlx/__init__.py",
        "src/gev/backends/mlx/pointer.py",
        "src/gev/backends/mlx/qualification.py",
        "src/gev/backends/mlx/training.py",
        "src/gev/backends/mlx/checkpoint.py",
        "src/gev/domain/tokenization.py",
        "src/gev/training/batching.py",
        "src/gev/training/schedule.py",
        "src/gev/training/policy.py",
        "src/gev/models/specs.py",
        "src/gev/models/policy.py",
        "src/gev/models/registry.py",
        "src/gev/models/families.py",
        "models.lock.json",
        "src/gev/configuration/config.py",
        "src/gev/configuration/resolved.py",
        "src/gev/application/training.py",
        "src/gev/application/evaluation.py",
        "src/gev/application/prediction.py",
        "src/gev/commands/diagnose.py",
        "src/gev/evaluation/development.py",
    )
    digest = hashlib.sha256()
    for name in source_paths:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def seal_receipt(payload: dict) -> dict:
    if payload.get("protocol") != QUALIFICATION_PROTOCOL or payload.get("policy_sha256") != POLICY_SHA256:
        raise ValueError("cannot seal a receipt for an unknown Gemma 4 qualification policy")
    receipt = dict(payload)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return receipt


def validate_base_diagnostic_receipt(receipt: dict, *, expected_hardware: dict | None = None,
                                     expected_code_sha256: str | None = None,
                                     expected_base_revision: str = BASE_REVISION) -> None:
    if (not isinstance(receipt, dict) or receipt.get("protocol") != QUALIFICATION_PROTOCOL
            or receipt.get("policy_sha256") != POLICY_SHA256):
        raise ValueError("Gemma 4 qualification receipt policy identity mismatch")
    digest = receipt.get("receipt_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    if digest != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("Gemma 4 qualification receipt hash mismatch")
    if expected_hardware is not None and receipt.get("hardware") != expected_hardware:
        raise ValueError("Gemma 4 qualification receipt hardware mismatch")
    base = receipt.get("base", {})
    if (base.get("name") != POLICY.base_model or base.get("revision") != expected_base_revision
            or base.get("family") != POLICY.family or base.get("backend") != POLICY.backend):
        raise ValueError("Gemma 4 qualification receipt has a stale source revision")
    if base.get("source_weights_dtype") != POLICY.source_weights_dtype:
        raise ValueError("Gemma 4 qualification receipt source/compute precision mismatch")
    if base.get("compute_dtype") != "fp32":
        raise ValueError("Gemma 4 optional base diagnostic must use FP32 compute")
    if (expected_code_sha256 is not None
            and receipt.get("code_sha256") != expected_code_sha256):
        raise ValueError("Gemma 4 qualification receipt code hash is stale")
    stage = receipt.get("stages", {}).get("fp32_structure", {})
    boundaries = stage.get("boundary_gathers_exact", {})
    inventory = stage.get("mlx_source_inventory", {})
    if (receipt.get("status") != "base_structure_passed"
            or receipt.get("inference_qualified") is not False
            or stage.get("status") != "passed"
            or stage.get("source_inventory_exact") is not True
            or inventory.get("tensor_count") != 2011
            or inventory.get("text_tensor_count") != 600
            or inventory.get("text_serialized_parameters") != 4_647_449_891
            or inventory.get("effective_text_parameters") != 4_628_569_344
            or any(value is not True for value in boundaries.values())
            or set(boundaries) != {"main", "ple"}
            or stage.get("decoder_contract_exact") is not True
            or stage.get("attention_masks_exact") is not True
            or stage.get("selected_layer_outputs", {}).get("match") is not True
            or stage.get("probability_outputs", {}).get("finite") is not True):
        raise ValueError("Gemma 4 optional FP32 base diagnostic has not passed")


def _pointer_seed(seed: int) -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(0x474556 + seed)
    return {
        "query_weight": rng.normal(0, 0.015, (256, 1536)).astype(np.float32),
        "query_bias": rng.normal(0, 0.01, (256,)).astype(np.float32),
        "key_weight": rng.normal(0, 0.015, (256, 1536)).astype(np.float32),
        "key_bias": rng.normal(0, 0.01, (256,)).astype(np.float32),
    }


def collect_outputs(model, encodings: list[dict], *, backend: str,
                    head_seeds: tuple[int, ...] = POINTER_HEAD_SEEDS) -> dict:
    """Collect final hidden vectors and fixed-seed Gev pointer outputs."""
    import numpy as np

    hidden_rows: dict[str, dict] = {}
    for record_index, encoded in enumerate(encodings):
        state, _state_positions, questions = rows_of(encoded)
        record_id = (encoded.get("metadata") or {}).get("id", f"record-{record_index}")
        for question_index, question in enumerate(questions):
            ids = state + question["ids"]
            if backend == "mlx":
                import mlx.core as mx

                output = model.decoder(mx.array([ids], dtype=mx.int32))[0]
                mx.eval(output)
                as_numpy = lambda value: np.asarray(value.astype(mx.float32))
            elif backend == "torch":
                import torch

                input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
                with torch.inference_mode():
                    output = model(input_ids=input_ids,
                                   attention_mask=torch.ones_like(input_ids),
                                   use_cache=False).last_hidden_state[0]
                as_numpy = lambda value: value.detach().float().cpu().numpy()
            else:
                raise ValueError(f"unknown qualification output backend: {backend}")
            decision = len(state) + question["decide"]
            options = [len(state) + position for position in question["opts"]]
            key = f"{record_id}::q{question_index}"
            hidden_rows[key] = {"decision": as_numpy(output[decision]),
                                "options": as_numpy(output[options])}

    outputs = {"hidden": hidden_rows, "heads": {}}
    if backend == "mlx":
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        original_head = dict(tree_flatten(model.head.parameters()))
        try:
            for seed in head_seeds:
                parameters = _pointer_seed(seed)
                model.head.query.weight = mx.array(parameters["query_weight"])
                model.head.query.bias = mx.array(parameters["query_bias"])
                model.head.key.weight = mx.array(parameters["key_weight"])
                model.head.key.bias = mx.array(parameters["key_bias"])
                per_head = {}
                for key, hidden in hidden_rows.items():
                    logits = model.head(mx.array(hidden["decision"]), mx.array(hidden["options"]))
                    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
                    mx.eval(logits, probabilities)
                    per_head[key] = {"logits": np.asarray(logits),
                                     "probabilities": np.asarray(probabilities)}
                outputs["heads"][str(seed)] = per_head
        finally:
            model.head.update(tree_unflatten(list(original_head.items())))
    else:
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            for seed in head_seeds:
                parameters = _pointer_seed(seed)
                query_weight = torch.tensor(parameters["query_weight"])
                query_bias = torch.tensor(parameters["query_bias"])
                key_weight = torch.tensor(parameters["key_weight"])
                key_bias = torch.tensor(parameters["key_bias"])
                per_head = {}
                for key, hidden in hidden_rows.items():
                    decision = torch.tensor(hidden["decision"], dtype=torch.float32)
                    options = torch.tensor(hidden["options"], dtype=torch.float32)
                    query = F.linear(decision, query_weight, query_bias)
                    keys = F.linear(options, key_weight, key_bias)
                    logits = keys @ query / (256 ** 0.5)
                    probabilities = torch.softmax(logits, dim=-1)
                    per_head[key] = {"logits": logits.numpy(),
                                     "probabilities": probabilities.numpy()}
                outputs["heads"][str(seed)] = per_head
    return outputs


def compare_outputs(reference: dict, candidate: dict) -> dict:
    import numpy as np

    reference_hidden, candidate_hidden = reference["hidden"], candidate["hidden"]
    if set(reference_hidden) != set(candidate_hidden):
        return {"status": "failed", "reason": "question key inventory mismatch"}
    hidden_errors, hidden_cosines = [], []
    for key in sorted(reference_hidden):
        for field in ("decision", "options"):
            left = np.asarray(reference_hidden[key][field], dtype=np.float32)
            right = np.asarray(candidate_hidden[key][field], dtype=np.float32)
            if left.shape != right.shape:
                return {"status": "failed", "reason": f"hidden shape mismatch at {key}:{field}"}
            hidden_errors.append(float(np.max(np.abs(left - right))))
            a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            hidden_cosines.append(float(np.dot(a, b) / denominator) if denominator else 1.0)
    if set(reference["heads"]) != set(candidate["heads"]):
        return {"status": "failed", "reason": "pointer-head seed inventory mismatch"}
    probability_errors, logit_errors, argmax_flips, compared = [], [], 0, 0
    per_question = {}
    for seed in sorted(reference["heads"]):
        ref_head, candidate_head = reference["heads"][seed], candidate["heads"][seed]
        if set(ref_head) != set(candidate_head):
            return {"status": "failed", "reason": f"question inventory mismatch for head seed {seed}"}
        for key in sorted(ref_head):
            ref_logits = np.asarray(ref_head[key]["logits"], dtype=np.float32)
            actual_logits = np.asarray(candidate_head[key]["logits"], dtype=np.float32)
            ref_probs = np.asarray(ref_head[key]["probabilities"], dtype=np.float32)
            actual_probs = np.asarray(candidate_head[key]["probabilities"], dtype=np.float32)
            if (ref_logits.shape != actual_logits.shape
                    or ref_probs.shape != actual_probs.shape):
                return {"status": "failed", "reason": f"output shape mismatch at {seed}:{key}"}
            logit_errors.append(float(np.max(np.abs(ref_logits - actual_logits))))
            probability_errors.append(float(np.max(np.abs(ref_probs - actual_probs))))
            argmax_flips += int(ref_probs.argmax() != actual_probs.argmax())
            compared += 1
            per_question[f"{seed}:{key}"] = {
                "reference_logits": ref_logits.tolist(),
                "candidate_logits": actual_logits.tolist(),
                "reference_probabilities": ref_probs.tolist(),
                "candidate_probabilities": actual_probs.tolist(),
                "probability_max_abs": probability_errors[-1],
                "argmax_match": bool(ref_probs.argmax() == actual_probs.argmax()),
            }
    finite = (all(np.isfinite(array).all() for side in (reference["heads"], candidate["heads"])
                  for outputs in side.values() for item in outputs.values() for array in item.values())
              and all(np.isfinite(array).all()
                      for side in (reference_hidden, candidate_hidden)
                      for item in side.values() for array in item.values()))
    return {
        "status": "measured",
        "compared_questions": compared,
        "pointer_head_seeds": sorted(int(seed) for seed in reference["heads"]
                                      if seed.isdigit()),
        "trained_head": "trained" in reference["heads"],
        "hidden_max_abs": max(hidden_errors, default=0.0),
        "hidden_min_cosine": min(hidden_cosines, default=1.0),
        "pointer_logits_max_abs": max(logit_errors, default=0.0),
        "probabilities_max_abs": max(probability_errors, default=0.0),
        "argmax_flips": argmax_flips,
        "finite": bool(finite),
        "per_question": per_question,
    }


def collect_mlx_training_checks(model, training_metrics: dict,
                                development_encodings: list[dict]) -> dict[str, bool]:
    """Collect native MLX decoder, training, attention and shared-KV evidence."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    provenance = model.provenance
    base = provenance.get("base_qualification", {})
    inventory = base.get("inventory", {})
    compute_dtype = provenance.get("compute_dtype")
    expected_dtype = mx.bfloat16 if compute_dtype == "bf16" else mx.float32
    source_inventory_exact = (
        provenance.get("source_weights_dtype") == "bf16"
        and base.get("source_weights_dtype") == "bf16"
        and base.get("compute_dtype") == compute_dtype
        and base.get("revision") == BASE_REVISION
        and inventory.get("tensor_count") == 2011
        and inventory.get("text_tensor_count") == 600
        and inventory.get("text_serialized_parameters") == 4_647_449_891
        and inventory.get("effective_text_parameters") == 4_628_569_344
        and isinstance(inventory.get("name_shape_sha256"), str)
        and len(inventory["name_shape_sha256"]) == 64)
    frozen = [(name, value) for name, value in tree_flatten(model.decoder.parameters())
              if ".lora_" not in name]
    decoder_compute_dtype = bool(frozen) and all(value.dtype == expected_dtype
                                                  for _, value in frozen)
    parameters = dict(tree_flatten(model.trainable_parameters()))
    lora = [value for name, value in parameters.items() if ".lora_" in name]
    pointer = [value for name, value in parameters.items() if name.startswith("head.")]
    trainable_precision = (bool(lora) and len(pointer) == 4
                           and all(value.dtype == mx.float32 for value in (*lora, *pointer)))
    engineering = training_metrics.get("engineering_checks", {})
    boundary = _boundary_summary(_base_boundary_values(model, backend="mlx"))
    boundaries_valid = all(boundary.get(f"{name}_high_boundary_nonzero") is True
                            for name in ("main", "ple"))
    return {
        "source_inventory_exact": source_inventory_exact,
        "decoder_compute_dtype": decoder_compute_dtype,
        "fp32_lora_and_pointer": trainable_precision,
        "fp32_optimizer_state": engineering.get("optimizer_master_state_fp32") is True,
        "finite_loss_gradients": (engineering.get("finite_loss_gradients") is True
                                   and engineering.get("finite_optimizer_state") is True),
        "nonzero_trainable_update": (
            engineering.get("nonzero_adapter_and_head_gradients") is True
            and engineering.get("nonzero_trainable_update") is True),
        "boundary_gathers_valid": boundaries_valid,
        "decoder_masks_valid": _mlx_attention_masks_exact(model, development_encodings),
        "shared_kv_mapping_exact": (
            _mlx_shared_source_layers(model) == dict(POLICY.expected_shared_kv_sources)),
    }


def _memory_snapshot() -> dict:
    import psutil
    import mlx.core as mx

    return {"process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
            "mlx_active_bytes": int(mx.get_active_memory())}


def _release_mlx_model(*objects) -> None:
    import gc
    import mlx.core as mx

    del objects
    gc.collect()
    mx.clear_cache()


def _source_inventory_matches(torch_info: dict, mlx_info: dict) -> bool:
    return (torch_info.get("source_tensor_count") == mlx_info.get("tensor_count")
            and torch_info.get("source_text_tensor_count") == mlx_info.get("text_tensor_count")
            and torch_info.get("source_text_parameters") == mlx_info.get("text_serialized_parameters")
            and torch_info.get("source_name_shape_sha256") == mlx_info.get("name_shape_sha256")
            and torch_info.get("effective_text_parameters") == mlx_info.get("effective_text_parameters")
            and torch_info.get("source_revision") == BASE_REVISION)


def _attention_masks_exact(comparisons: dict) -> bool:
    return (set(comparisons) == {"short", "long"}
            and all(value.get("full_attention", {}).get("match") is True
                    and value.get("sliding_attention", {}).get("match") is True
                    for value in comparisons.values()))


def _mlx_attention_masks_exact(model, encodings: list[dict]) -> bool:
    """Validate MLX causal/sliding masks from architecture invariants, no Torch oracle."""
    import numpy as np
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    lengths = [len(state) + len(question["ids"])
               for encoded in encodings
               for state, _positions, questions in [rows_of(encoded)]
               for question in questions]
    if not lengths:
        return False
    for length in {min(lengths), max(lengths)}:
        hidden = mx.zeros((1, length, model.decoder.config.hidden_size), dtype=mx.bfloat16)
        positions = np.arange(length)
        causal = positions[:, None] >= positions[None, :]
        for window, expected in ((None, causal),
                                 (512, causal & ((positions[:, None] - positions[None, :]) < 512))):
            mask = create_attention_mask(hidden, window_size=window, return_array=True)
            mx.eval(mask)
            if not np.array_equal(np.asarray(mask, dtype=np.bool_).squeeze(), expected):
                return False
    return True


def _mlx_shared_source_layers(model) -> dict[str, int]:
    config = model.decoder.config
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    return {kind: max(index for index in range(first_shared)
                      if config.layer_types[index] == kind)
            for kind in sorted(set(config.layer_types))}


def run_base_qualification(*, model_name: str, revision: str, data_root: str | Path,
                           resolved, tokenizer, markers, repo_root: str | Path) -> dict:
    """Run the sole pinned FP32 structural qualification.

    Large Torch and MLX references are loaded sequentially, never all resident
    together. Only bounded decision outputs are retained between stages.
    """
    require_tf32_disabled_before_mlx_import()
    import gc
    import numpy as np

    from ...data.access import file_digest, load_verified_split, split_path
    from ...domain.materialize import materialize
    from .gemma4 import _snapshot, load_gemma4_backbone, Gemma4RowModel
    from ...data.suites import manifest_digest

    if model_name != POLICY.base_model or revision != POLICY.base_revision:
        raise ValueError("Gemma 4 qualification requires the exact pinned PRETRAINED E2B base")
    if (resolved.model.family.family_id != POLICY.family
            or resolved.model.backend.backend_id != POLICY.backend):
        raise ValueError("Gemma 4 qualification model family/backend mismatch")
    rows, manifest, loaded_manifest_sha = load_verified_split(
        data_root, POLICY.dataset_suite, POLICY.dataset_split)
    data_path = split_path(data_root, POLICY.dataset_suite, POLICY.dataset_split)
    data_sha = file_digest(data_path)
    expected_manifest_sha = manifest_digest(POLICY.dataset_suite)
    if (data_sha != POLICY.dataset_split_sha256
            or expected_manifest_sha != POLICY.dataset_manifest_sha256
            or loaded_manifest_sha != expected_manifest_sha):
        raise ValueError("Gemma 4 qualification requires the pinned verified decision-v7 development split")
    selected = validate_development_selection(rows)
    if selected_ids_sha256(selected) != SELECTED_IDS_SHA256:
        raise ValueError("Gemma 4 qualification development row selection hash mismatch")
    selected_ids = [row["_meta"]["id"] for row in selected]
    encodings = [resolved.encode_record(tokenizer, materialize(row), markers)
                 for row in selected]
    lengths = [{"id": row["_meta"]["id"], "tokens": len(encoded["ids"]),
                "questions": len(encoded["decide_idx"])}
               for row, encoded in zip(selected, encodings, strict=True)]
    short_encoding = min(encodings, key=lambda encoded: max(
        (len(rows_of(encoded)[0]) + len(question["ids"])
         for question in rows_of(encoded)[2]), default=0))
    long_encoding = max(encodings, key=lambda encoded: max(
        (len(rows_of(encoded)[0]) + len(question["ids"])
         for question in rows_of(encoded)[2]), default=0))
    short_state, _, short_questions = rows_of(short_encoding)
    long_state, _, long_questions = rows_of(long_encoding)
    short_ids = short_state + short_questions[0]["ids"]
    long_ids = long_state + max(long_questions, key=lambda row: len(row["ids"]))["ids"]

    environment = qualification_environment()
    snapshot = _snapshot(model_name, revision)
    memory = {"before_torch_fp32": _memory_snapshot()}

    torch_model = load_torch_cpu_oracle(snapshot, compute_dtype="fp32")
    torch_outputs = collect_outputs(torch_model, encodings, backend="torch")
    torch_boundary = _base_boundary_values(torch_model, backend="torch")
    torch_layers = _capture_torch_selected_layers(torch_model, short_ids)
    torch_info = dict(torch_model._gev_qualification)
    torch_contract = _torch_decoder_contract(torch_model)
    torch_masks = {"short": _compare_attention_masks(torch_model, short_ids),
                   "long": _compare_attention_masks(torch_model, long_ids)}
    del torch_model
    gc.collect()
    memory["after_torch_release"] = _memory_snapshot()

    from ...models.specs import GEMMA4_E2B

    mlx_loaded_fp32 = load_gemma4_backbone(model_name, revision, compute_dtype="fp32")
    mlx_model_fp32 = Gemma4RowModel(mlx_loaded_fp32, family=GEMMA4_E2B)
    mlx_model_fp32.eval()
    mlx_outputs_fp32 = collect_outputs(mlx_model_fp32, encodings, backend="mlx")
    mlx_boundary_fp32 = _base_boundary_values(mlx_model_fp32, backend="mlx")
    mlx_layers_fp32 = _capture_mlx_selected_layers(mlx_model_fp32, short_ids)
    mlx_contract = _mlx_decoder_contract(mlx_model_fp32)
    fp32_outputs = compare_outputs(torch_outputs, mlx_outputs_fp32)
    fp32_layers = _compare_selected_layers(torch_layers, mlx_layers_fp32)
    source_inventory_exact = _source_inventory_matches(
        torch_info, mlx_loaded_fp32["base_qualification"]["inventory"])
    decoder_contract_exact = torch_contract == mlx_contract
    boundary_exact = _exact_boundary_match(torch_boundary, mlx_boundary_fp32)
    masks_exact = _attention_masks_exact(torch_masks)
    fp32_pass = (source_inventory_exact and decoder_contract_exact
                 and _all_boundary_tables_exact(boundary_exact)
                 and masks_exact and fp32_layers["match"]
                 and fp32_outputs.get("finite") is True)
    fp32_stage = {
        "status": "passed" if fp32_pass else "failed",
        "probability_comparison_role": "diagnostic_only_no_acceptance_threshold",
        "probability_outputs": fp32_outputs,
        "selected_layer_outputs": fp32_layers,
        "source_inventory_exact": source_inventory_exact,
        "torch_source_inventory": torch_info,
        "mlx_source_inventory": mlx_loaded_fp32["base_qualification"]["inventory"],
        "decoder_contract_exact": decoder_contract_exact,
        "decoder_contract": torch_contract,
        "boundary_gathers_exact": boundary_exact,
        "torch_boundary": _boundary_summary(torch_boundary),
        "mlx_fp32_boundary": _boundary_summary(mlx_boundary_fp32),
        "attention_masks_exact": masks_exact,
        "torch_attention_masks": torch_masks,
        "attention_masks": torch_masks,
    }
    memory["after_mlx_fp32"] = _memory_snapshot()
    del mlx_model_fp32, mlx_loaded_fp32, mlx_outputs_fp32, mlx_layers_fp32
    _release_mlx_model()
    gc.collect()
    memory["after_fp32_release"] = _memory_snapshot()
    stages = {"fp32_structure": fp32_stage}
    status = "base_structure_passed" if fp32_stage["status"] == "passed" else "failed"
    return seal_receipt({
        "protocol": QUALIFICATION_PROTOCOL,
        "policy_sha256": POLICY_SHA256,
        "status": status,
        "inference_qualified": False,
        "base": {"name": model_name, "revision": revision,
                 "family": POLICY.family, "backend": POLICY.backend,
                 "source_weights_dtype": POLICY.source_weights_dtype,
                  "compute_dtype": "fp32"},
        "dataset": {"suite": POLICY.dataset_suite, "split": POLICY.dataset_split,
                    "file_sha256": data_sha, "manifest_sha256": loaded_manifest_sha},
        "selection": {"algorithm": POLICY.selection_algorithm,
                      "ids": selected_ids,
                      "ids_sha256": selected_ids_sha256([{"_meta": {"id": value}}
                                                           for value in selected_ids]),
                      "rows": lengths,
                      "row_count": len(selected),
                      "sources": sorted({row["_meta"]["source"] for row in selected})},
        "tokenizer": {"revision": markers.tokenizer_revision,
                      "sha256": getattr(markers, "tokenizer_sha256", None),
                      "marker_ids": markers.ids, "marker_strings": markers.strings},
        "hardware": environment,
        "code_sha256": qualification_code_sha256(repo_root),
        "head_seeds": list(POLICY.pointer_head_seeds),
        "boundary_ids": list(BOUNDARY_IDS),
        "precision_gates": {gate.compute_dtype: asdict(gate)
                             for gate in POLICY.precision_gates},
        "stages": stages,
        "memory": memory,
    })


def _base_boundary_values(model, *, backend: str) -> dict:
    import numpy as np

    if backend == "mlx":
        import mlx.core as mx

        decoder = model.decoder
        ids = mx.array(BOUNDARY_IDS, dtype=mx.int32)
        main = decoder.embed_tokens.weight[ids]
        ple = decoder.embed_tokens_per_layer.weight[ids]
        mx.eval(main, ple)
        main_values = np.asarray(main.astype(mx.float32))
        ple_values = np.asarray(ple.astype(mx.float32))
    elif backend == "torch":
        import torch

        ids = torch.tensor(BOUNDARY_IDS, dtype=torch.long)
        with torch.inference_mode():
            main_values = model.embed_tokens.weight[ids].float().cpu().numpy()
            ple_values = model.embed_tokens_per_layer.weight[ids].float().cpu().numpy()
    else:
        raise ValueError(f"unknown boundary backend: {backend}")
    return {"main": main_values, "ple": ple_values}


def _boundary_summary(values: dict) -> dict:
    import numpy as np

    marker_main, marker_ple = values["main"][:5], values["ple"][:5]
    def checksum(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {
        "ids": list(BOUNDARY_IDS),
        "main_shape": list(values["main"].shape),
        "ple_shape": list(values["ple"].shape),
        "main_finite": bool(np.isfinite(values["main"]).all()),
        "ple_finite": bool(np.isfinite(values["ple"]).all()),
        "main_markers_nonzero_distinct": bool(
            np.all(np.linalg.norm(marker_main, axis=-1) > 0)
            and len({row.tobytes() for row in marker_main}) == 5),
        "ple_markers_nonzero_distinct": bool(
            np.all(np.linalg.norm(marker_ple, axis=-1) > 0)
            and len({row.tobytes() for row in marker_ple}) == 5),
        "main_high_boundary_nonzero": bool(
            np.all(np.linalg.norm(values["main"][5:], axis=-1) > 0)),
        "ple_high_boundary_nonzero": bool(
            np.all(np.linalg.norm(values["ple"][5:], axis=-1) > 0)),
        "main_sha256": checksum(values["main"]),
        "ple_sha256": checksum(values["ple"]),
    }


def _exact_boundary_match(reference: dict, candidate: dict) -> dict:
    import numpy as np

    return {name: bool(np.array_equal(reference[name], candidate[name]))
            for name in ("main", "ple")}


def _all_boundary_tables_exact(matches: dict) -> bool:
    return all(matches.get(name) is True for name in ("main", "ple"))


def _torch_shared_sources(model) -> dict:
    shared_from = {}
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        if not attention.is_kv_shared_layer and attention.store_full_length_kv:
            shared_from[attention.layer_type] = index
    return shared_from


def _mlx_shared_sources(model) -> dict:
    decoder = model.decoder
    first_shared = decoder.config.num_hidden_layers - decoder.config.num_kv_shared_layers
    shared_from = {}
    for index in range(first_shared, decoder.config.num_hidden_layers):
        kind = decoder.layers[index].layer_type
        shared_from[kind] = decoder.previous_kvs[index]
    return shared_from


def _torch_decoder_contract(model) -> dict:
    config = model.config
    return {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "layer_types": list(config.layer_types),
        "rope_parameters": config.rope_parameters,
        "sliding_window": config.sliding_window,
        "num_kv_shared_layers": config.num_kv_shared_layers,
        "hidden_size_per_layer_input": config.hidden_size_per_layer_input,
        "vocab_size_per_layer_input": config.vocab_size_per_layer_input,
        "attention": [{"head_dim": config.per_layer_config[index].head_dim,
                       "kv_heads": config.per_layer_config[index].num_key_value_heads}
                      for index in range(config.num_hidden_layers)],
        "mlp_intermediate_by_layer": [layer.mlp.intermediate_size
                                       for layer in model.layers],
        "shared_kv_sources": _torch_shared_sources(model),
    }


def _mlx_decoder_contract(model) -> dict:
    decoder = model.decoder
    config = decoder.config
    return {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "layer_types": list(config.layer_types),
        "rope_parameters": config.rope_parameters,
        "sliding_window": config.sliding_window,
        "num_kv_shared_layers": config.num_kv_shared_layers,
        "hidden_size_per_layer_input": config.hidden_size_per_layer_input,
        "vocab_size_per_layer_input": config.vocab_size_per_layer_input,
        "attention": [{"head_dim": layer.self_attn.head_dim,
                       "kv_heads": layer.self_attn.n_kv_heads}
                      for layer in decoder.layers],
        "mlp_intermediate_by_layer": [
            getattr(layer.mlp.gate_proj, "linear", layer.mlp.gate_proj).weight.shape[0]
            for layer in decoder.layers],
        "shared_kv_sources": _mlx_shared_sources(model),
    }


def _capture_torch_selected_layers(model, ids: list[int]) -> dict[int, object]:
    import torch

    captured, hooks = {}, []
    for index in POLICY.selected_layer_indices:
        hooks.append(model.layers[index].register_forward_hook(
            lambda _module, _args, output, layer=index:
                captured.__setitem__(layer, output.detach().float().cpu().numpy())))
    input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    for hook in hooks:
        hook.remove()
    return captured


def _capture_mlx_selected_layers(model, ids: list[int]) -> dict[int, object]:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models import gemma4_text

    captured = {}
    original = gemma4_text.DecoderLayer.__call__

    def capture(layer, *args, **kwargs):
        output = original(layer, *args, **kwargs)
        if layer.layer_idx in POLICY.selected_layer_indices:
            value = output[0] if isinstance(output, tuple) else output
            mx.eval(value)
            captured[layer.layer_idx] = np.asarray(value.astype(mx.float32))
        return output

    gemma4_text.DecoderLayer.__call__ = capture
    try:
        mx.eval(model.decoder(mx.array([ids], dtype=mx.int32)))
    finally:
        gemma4_text.DecoderLayer.__call__ = original
    return captured


def _compare_selected_layers(reference: dict, candidate: dict) -> dict:
    import numpy as np

    if set(reference) != set(candidate):
        return {"match": False, "reason": "selected-layer inventory mismatch"}
    outputs = {}
    for index in POLICY.selected_layer_indices:
        left, right = reference[index], candidate[index]
        if left.shape != right.shape:
            outputs[str(index)] = {"match": False, "reason": "shape mismatch",
                                   "reference_shape": list(left.shape),
                                   "candidate_shape": list(right.shape)}
        else:
            outputs[str(index)] = {"match": True,
                                   "max_abs": float(np.max(np.abs(left - right))),
                                   "finite": bool(np.isfinite(left).all() and np.isfinite(right).all())}
    return {"match": all(item["match"] and item.get("finite", True)
                         for item in outputs.values()),
            "selected_layers": outputs}
