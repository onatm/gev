"""Strict, backend-tagged adapter/head-only MLX inference checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from safetensors.mlx import load_file, save_file

from ...artifacts.checkpoint_identity import (MANIFEST_FILENAME, MANIFEST_FORMAT,
                                               MANIFEST_VERSION, read_checkpoint_manifest)
from ...models.specs import GEMMA4_E2B, GEMMA4_E2B_REVISION
from ...models.qualification import (bind_checkpoint_tensors,
                                     qualification_code_sha256,
                                     trainable_content_sha256,
                                     validate_qualification_receipt,
                                     validate_qualification_sidecars)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(model, directory, metadata: dict, tokenizer=None) -> Path:
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    required = {"model_name", "model_revision", "base_model_type", "model_family", "backend",
                "protocol", "scientific_recipe", "scientific_recipe_sha256", "marker_ids",
                "marker_strings", "head_width", "lora", "dtype", "compute_dtype",
                "weights_dtype", "source_weights_dtype",
                "state_cap", "branch_cap", "packed_cap", "representation_version",
                "qualification", "base_qualification"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata missing: {', '.join(missing)}")
    compute_dtype = metadata["compute_dtype"]
    if (metadata["backend"] != "mlx" or metadata["model_family"] != GEMMA4_E2B.family_id
            or compute_dtype not in {"bf16", "fp32"}
            or metadata["dtype"] != compute_dtype
            or metadata["weights_dtype"] != compute_dtype
            or metadata["source_weights_dtype"] != "bf16"
            or metadata["base_qualification"].get("revision") != GEMMA4_E2B_REVISION
            or metadata["base_qualification"].get("source_weights_dtype") != "bf16"
            or metadata["base_qualification"].get("compute_dtype") != compute_dtype):
        raise ValueError("MLX checkpoint requires the pinned BF16 source and selected compute identity")
    qualification = metadata["qualification"]
    expected_code = qualification_code_sha256("mlx", Path(__file__).resolve().parents[4])
    validate_qualification_receipt(
        qualification, config=metadata["config"], backend="mlx",
        expected_code_sha256=expected_code, require_checkpoint_ready=False)
    trainable_sha = trainable_content_sha256("mlx", model)
    if qualification.get("trainable_parameters_sha256") != trainable_sha:
        raise ValueError("qualification receipt does not bind the checkpoint trainable tensors")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        parameters = dict(tree_flatten(model.trainable_parameters()))
        adapter = {name: value for name, value in parameters.items() if not name.startswith("head.")}
        pointer = {name.removeprefix("head."): value for name, value in parameters.items()
                   if name.startswith("head.")}
        if not adapter or set(pointer) != set(dict(tree_flatten(model.head.parameters()))):
            raise ValueError("MLX checkpoint has incomplete adapter or pointer tensors")
        if any(value.dtype != mx.float32 for value in (*adapter.values(), *pointer.values())):
            raise ValueError("MLX LoRA masters and pointer tensors must remain FP32")
        save_file(adapter, str(temporary / "adapter_model.safetensors"))
        save_file(pointer, str(temporary / "pointer.safetensors"))
        if tokenizer is not None:
            tokenizer.save_pretrained(temporary)
        adapter_shapes = {key: list(value.shape) for key, value in adapter.items()}
        pointer_shapes = {key: list(value.shape) for key, value in pointer.items()}
        adapter_dtypes = {key: str(value.dtype) for key, value in adapter.items()}
        pointer_dtypes = {key: str(value.dtype) for key, value in pointer.items()}
        tensor_hashes = {
            "adapter": _sha(temporary / "adapter_model.safetensors"),
            "pointer": _sha(temporary / "pointer.safetensors"),
        }
        qualified_checkpoint = bind_checkpoint_tensors(qualification, tensor_hashes)
        development_report = metadata.get("development_report")
        if not isinstance(development_report, dict):
            raise ValueError("Gemma 4 checkpoint metadata lacks its development report")
        report_bytes = (json.dumps(development_report, indent=2, allow_nan=False) + "\n").encode()
        if hashlib.sha256(report_bytes).hexdigest() != qualified_checkpoint[
                "development"]["report_sha256"]:
            raise ValueError("Gemma 4 checkpoint development report digest mismatch")
        (temporary / "development_report.json").write_bytes(report_bytes)
        (temporary / "qualification.json").write_text(
            json.dumps(qualified_checkpoint, indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8")
        receipt_binding = {"receipt_sha256": qualified_checkpoint["receipt_sha256"],
                           "checkpoint_tensor_sha256": tensor_hashes}
        provenance = metadata
        resolved = provenance.get("resolved_config", {})
        execution = {key: provenance[key] for key in
                      ("dtype", "compute_dtype", "weights_dtype", "source_weights_dtype", "device",
                       "state_cap", "branch_cap", "packed_cap",
                      "representation_version") if key in provenance}
        execution["execution_mode"] = "rows"
        execution["storage_backend"] = "mlx.safetensors"
        execution["runtime_controls"] = resolved.get("operational_controls", {})
        lineage = {key: provenance[key] for key in
                   ("source_sha256", "manifest_sha256", "resume_input", "smoke_only")
                   if key in provenance}
        lineage["qualification"] = qualified_checkpoint
        manifest = {
            "format": MANIFEST_FORMAT, "version": MANIFEST_VERSION,
            "identity": {
                "family": provenance["model_family"],
                "model_output_id": provenance["model_output_id"], "backend": "mlx",
                "base": {"name": provenance["model_name"], "revision": provenance["model_revision"],
                         "type": provenance["base_model_type"]},
                "tokenizer": {"revision": provenance.get("tokenizer_revision", provenance["model_revision"]),
                              "sha256": provenance.get("tokenizer_sha256")},
                "markers": {"ids": provenance["marker_ids"], "strings": provenance["marker_strings"],
                            "bos": provenance.get("marker_bos", False)},
                "protocol": provenance["protocol"],
                "recipe": {"scientific_recipe": provenance["scientific_recipe"],
                           "sha256": provenance["scientific_recipe_sha256"]},
                "model_contract": {"head_width": provenance["head_width"], "lora": provenance["lora"],
                                   "representation_version": provenance["representation_version"],
                                     "storage_backend": "mlx.safetensors",
                                     "weights_dtype": compute_dtype,
                                     "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
                                     "base_qualification": provenance["base_qualification"],
                                     "qualification": receipt_binding},
            },
            "lineage": lineage,
            "training": {"metrics": provenance.get("training", {}),
                         "config": provenance.get("config", {}),
                         "resolved_config": resolved.get("resolved_config", {}),
                         "study_id": resolved.get("study_id")},
            "execution": execution,
            "tensors": {
                "adapter": {"filename": "adapter_model.safetensors", "sha256": tensor_hashes["adapter"], "shapes": adapter_shapes, "dtypes": adapter_dtypes},
                "pointer": {"filename": "pointer.safetensors", "sha256": tensor_hashes["pointer"], "shapes": pointer_shapes, "dtypes": pointer_dtypes},
            },
             "qualification": qualified_checkpoint,
            "calibration": {"temperature": 1.0},
        }
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_checkpoint(directory, *, config, device="gpu", tokenizer=None,
                    expected_marker_map=None, backbone_loader=None,
                    compute_dtype="bf16"):
    if device != "gpu":
        raise ValueError("Gemma 4 MLX checkpoints load on the MLX GPU only")
    if compute_dtype not in {"bf16", "fp32"} or compute_dtype != config.training.dtype:
        raise ValueError("Gemma 4 MLX checkpoint compute dtype does not match its config")
    from ...configuration.resolved import resolve_experiment_config
    resolved = resolve_experiment_config(config)
    resolved.validate_runtime_available()
    directory = Path(directory)
    manifest = read_checkpoint_manifest(directory)
    identity = manifest["identity"]
    base, marker_identity, contract = (identity[key] for key in ("base", "markers", "model_contract"))
    if identity["backend"] != "mlx":
        raise ValueError("checkpoint backend mismatch")
    if identity["family"] != GEMMA4_E2B.family_id:
        raise ValueError("checkpoint model family mismatch")
    if identity.get("model_output_id") != "gev-gemma4-e2b":
        raise ValueError("checkpoint output model ID mismatch")
    if base.get("name") != config.model.name:
        raise ValueError("checkpoint model name mismatch")
    if base.get("revision") != config.model.revision:
        raise ValueError("checkpoint model revision mismatch")
    if base.get("type") != "gemma4":
        raise ValueError("checkpoint base model type mismatch")
    if identity["protocol"] != dataclasses.asdict(config.protocol):
        raise ValueError("checkpoint protocol mismatch")
    if identity["recipe"].get("sha256") != resolved.recipe_sha256:
        raise ValueError("checkpoint scientific recipe mismatch")
    if (contract.get("storage_backend") != "mlx.safetensors"
            or contract.get("weights_dtype") != compute_dtype
            or contract.get("source_weights_dtype") != "bf16"
            or contract.get("compute_dtype") != compute_dtype):
        raise ValueError("checkpoint MLX tensor storage/dtype contract mismatch")
    if contract.get("representation_version") != 1 or contract.get("head_width") != GEMMA4_E2B.pointer_width:
        raise ValueError("checkpoint model contract mismatch")
    expected_lora = {"r": GEMMA4_E2B.lora_rank, "alpha": GEMMA4_E2B.lora_alpha,
                     "dropout": GEMMA4_E2B.lora_dropout, "targets": list(GEMMA4_E2B.lora_targets)}
    if contract.get("lora") != expected_lora:
        raise ValueError("checkpoint LoRA contract mismatch")
    if marker_identity.get("bos") is not False:
        raise ValueError("checkpoint marker BOS contract mismatch")
    if expected_marker_map is not None and (
            marker_identity.get("ids") != expected_marker_map.ids
            or marker_identity.get("strings") != expected_marker_map.strings
            or expected_marker_map.bos is not False):
        raise ValueError("checkpoint marker identity mismatch")
    if marker_identity.get("ids") != config.model.marker_ids:
        raise ValueError("checkpoint marker ID contract mismatch")
    expected_marker_strings = dict(zip(
        GEMMA4_E2B.marker_roles,
        (f"<unused{index}>" for index in range(5)), strict=True))
    if marker_identity.get("strings") != expected_marker_strings:
        raise ValueError("checkpoint marker strings mismatch")
    if identity["tokenizer"].get("revision") != config.model.revision:
        raise ValueError("checkpoint tokenizer revision mismatch")
    expected_tokenizer_sha = (expected_marker_map.tokenizer_sha256
                              if expected_marker_map is not None else None)
    if (expected_tokenizer_sha is not None
            and identity["tokenizer"].get("sha256") != expected_tokenizer_sha):
        raise ValueError("checkpoint tokenizer digest mismatch")
    if (manifest["execution"].get("execution_mode") != "rows"
            or manifest["execution"].get("dtype") != compute_dtype
            or manifest["execution"].get("compute_dtype") != compute_dtype
            or manifest["execution"].get("weights_dtype") != compute_dtype
            or manifest["execution"].get("source_weights_dtype") != "bf16"
            or manifest["execution"].get("device") != "gpu"):
        raise ValueError("checkpoint execution mode mismatch")
    saved_config = manifest["training"].get("config", {})
    if (saved_config.get("backend", {}).get("id") != "mlx"
            or saved_config.get("model", {}).get("family") != GEMMA4_E2B.family_id
            or saved_config.get("model", {}).get("name") != config.model.name
            or saved_config.get("model", {}).get("revision") != config.model.revision):
        raise ValueError("checkpoint resolved config identity mismatch")
    qualification = manifest.get("qualification")
    tensor_hashes = {name: manifest["tensors"][name]["sha256"]
                     for name in ("adapter", "pointer")}
    expected_code = qualification_code_sha256(
        "mlx", Path(__file__).resolve().parents[4])
    validate_qualification_receipt(
        qualification, config=config, backend="mlx",
        expected_code_sha256=expected_code,
        expected_checkpoint_hashes=tensor_hashes)
    validate_qualification_sidecars(directory, qualification)
    contract_receipt = contract.get("qualification", {})
    if (contract_receipt.get("receipt_sha256") != qualification.get("receipt_sha256")
            or contract_receipt.get("checkpoint_tensor_sha256") != tensor_hashes):
        raise ValueError("checkpoint qualification identity binding mismatch")

    adapter_descriptor, pointer_descriptor = (manifest["tensors"][key] for key in ("adapter", "pointer"))
    adapter = load_file(str(directory / adapter_descriptor["filename"]))
    pointer = load_file(str(directory / pointer_descriptor["filename"]))
    if adapter_descriptor["shapes"] != {key: list(value.shape) for key, value in adapter.items()}:
        raise ValueError("MLX adapter tensor names/shapes mismatch")
    if pointer_descriptor["shapes"] != {key: list(value.shape) for key, value in pointer.items()}:
        raise ValueError("MLX pointer tensor names/shapes mismatch")
    if adapter_descriptor.get("dtypes") != {key: str(value.dtype) for key, value in adapter.items()}:
        raise ValueError("MLX adapter tensor dtype mismatch")
    if pointer_descriptor.get("dtypes") != {key: str(value.dtype) for key, value in pointer.items()}:
        raise ValueError("MLX pointer tensor dtype mismatch")
    fp32_dtype = str(mx.float32)
    if (any(value != fp32_dtype for value in adapter_descriptor["dtypes"].values())
            or any(value != fp32_dtype for value in pointer_descriptor["dtypes"].values())):
        raise ValueError("MLX LoRA masters and pointer checkpoint tensors must be FP32")
    from .gemma4 import Gemma4RowModel, load_gemma4_backbone

    loaded = (backbone_loader(config.model.name, config.model.revision,
                              compute_dtype=compute_dtype) if backbone_loader is not None
              else load_gemma4_backbone(config.model.name, config.model.revision,
                                        compute_dtype=compute_dtype))
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise ValueError("MLX Gemma 4 loader did not return strict base provenance")
    if (loaded.get("source_weights_dtype") != "bf16"
            or loaded.get("compute_dtype") != compute_dtype):
        raise ValueError("loaded MLX base source/compute precision mismatch")
    if contract.get("base_qualification") != loaded.get("base_qualification"):
        raise ValueError("checkpoint strict base inventory qualification mismatch")
    model = Gemma4RowModel(loaded, family=resolved.model.family,
                           temperature=manifest["calibration"].get("temperature", 1.0))
    current = dict(tree_flatten(model.trainable_parameters()))
    expected_adapter = {name for name in current if not name.startswith("head.")}
    expected_pointer = {name.removeprefix("head.") for name in current if name.startswith("head.")}
    if set(adapter) != expected_adapter or set(pointer) != expected_pointer:
        raise ValueError("strict MLX adapter/head tensor inventory mismatch")
    if any(adapter[name].dtype != current[name].dtype or adapter[name].shape != current[name].shape
           for name in adapter):
        raise ValueError("strict MLX adapter parameter dtype/shape mismatch")
    if any(pointer[name].dtype != current[f"head.{name}"].dtype
           or pointer[name].shape != current[f"head.{name}"].shape for name in pointer):
        raise ValueError("strict MLX pointer parameter dtype/shape mismatch")
    model.update(tree_unflatten(
        [(name, value) for name, value in adapter.items()]
        + [(f"head.{name}", value) for name, value in pointer.items()]))
    mx.eval(model.parameters())
    loaded_trainable_sha = trainable_content_sha256("mlx", model)
    if loaded_trainable_sha != qualification.get("trainable_parameters_sha256"):
        raise ValueError("loaded MLX adapter/head differs from its qualification receipt")
    model.eval()
    return model, manifest
