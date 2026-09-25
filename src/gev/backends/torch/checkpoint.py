"""Atomic adapter-only checkpoints."""
from __future__ import annotations
import dataclasses, hashlib, json, os, shutil, tempfile, warnings
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from ...artifacts.checkpoint_identity import (MANIFEST_FILENAME, MANIFEST_FORMAT,
                                    MANIFEST_VERSION, checkpoint_fingerprint,
                                    read_checkpoint_manifest)
from ...configuration.resolved import resolve_experiment_config
from ...models.qualification import (bind_checkpoint_tensors,
                                     qualification_code_sha256,
                                     trainable_content_sha256,
                                     validate_qualification_receipt,
                                     validate_qualification_sidecars)

FORMAT_VERSION = MANIFEST_VERSION

def _canonical(name: str) -> str:
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_checkpoint(model, directory: str | Path, metadata: dict, tokenizer=None) -> Path:
    target = Path(directory)
    if target.exists(): raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    parent = target.parent; parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        required = {"model_name", "model_revision", "marker_ids", "marker_strings", "head_width", "lora", "dtype", "state_cap", "branch_cap", "packed_cap", "representation_version", "base_model_type", "model_family", "backend", "protocol", "scientific_recipe", "scientific_recipe_sha256"}
        missing = sorted(required - set(metadata))
        if missing: raise ValueError(f"checkpoint metadata missing: {', '.join(missing)}")
        gemma4 = metadata["model_family"] == "gemma4_e2b_text"
        if gemma4:
            validate_qualification_receipt(
                metadata.get("qualification"), config=metadata["config"], backend="torch",
                expected_code_sha256=qualification_code_sha256(
                    "torch", Path(__file__).resolve().parents[4]),
                require_checkpoint_ready=False)
            if metadata["qualification"].get("trainable_parameters_sha256") != \
                    trainable_content_sha256("torch", model):
                raise ValueError("qualification receipt does not bind checkpoint trainable tensors")
        model.backbone.save_pretrained(temp, safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(temp / "pointer.safetensors"))
        if tokenizer is not None: tokenizer.save_pretrained(temp)
        adapter = temp / "adapter_model.safetensors"
        if not adapter.exists(): raise ValueError("PEFT did not write adapter_model.safetensors")
        adapter_shapes = {k: list(v.shape) for k, v in load_file(str(adapter)).items()}
        pointer_shapes = {k: list(v.shape) for k, v in model.head.state_dict().items()}
        qualification = None
        if gemma4:
            tensor_hashes = {"adapter": _sha(adapter),
                             "pointer": _sha(temp / "pointer.safetensors")}
            qualification = bind_checkpoint_tensors(
                metadata["qualification"], tensor_hashes)
            development_report = metadata.get("development_report")
            if not isinstance(development_report, dict):
                raise ValueError("Gemma 4 checkpoint metadata lacks its development report")
            report_bytes = (json.dumps(development_report, indent=2, allow_nan=False)
                            + "\n").encode()
            if hashlib.sha256(report_bytes).hexdigest() != qualification[
                    "development"]["report_sha256"]:
                raise ValueError("Gemma 4 checkpoint development report digest mismatch")
            (temp / "development_report.json").write_bytes(report_bytes)
            (temp / "qualification.json").write_text(
                json.dumps(qualification, indent=2, sort_keys=True,
                           allow_nan=False) + "\n", encoding="utf-8")
        provenance = metadata
        resolved = provenance.get("resolved_config", {})
        execution = {key: provenance[key] for key in
                     ("dtype", "compute_dtype", "weights_dtype", "source_weights_dtype", "device",
                      "state_cap", "branch_cap", "packed_cap", "representation_version")
                     if key in provenance}
        execution["execution_mode"] = provenance.get("execution_mode", provenance.get("training", {}).get("execution_mode", "rows"))
        execution["runtime_controls"] = resolved.get("operational_controls", {})
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "identity": {
                "family": provenance["model_family"],
                "model_output_id": provenance.get("model_output_id", provenance["model_family"]),
                "backend": provenance["backend"],
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
                                   "representation_version": provenance["representation_version"]},
            },
            "lineage": {key: provenance[key] for key in
                        ("source_sha256", "manifest_sha256", "continuation", "resume_input", "smoke_only",
                         "diagnostic_smoke_init") if key in provenance},
            "training": {"metrics": provenance.get("training", {}),
                         "config": provenance.get("config", {}),
                         "resolved_config": resolved.get("resolved_config", {}),
                         "training_args": provenance.get("training_args", {}),
                         "study_id": resolved.get("study_id")},
            "execution": execution,
            "tensors": {
                "adapter": {"filename": "adapter_model.safetensors", "sha256": _sha(adapter), "shapes": adapter_shapes},
                "pointer": {"filename": "pointer.safetensors", "sha256": _sha(temp / "pointer.safetensors"), "shapes": pointer_shapes},
            },
            "calibration": {"temperature": 1.0},
        }
        if gemma4:
            manifest["qualification"] = qualification
        (temp / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise
    return target

def load_checkpoint(directory: str | Path, *, config, device="cpu", tokenizer=None,
                    expected_marker_map=None, backbone_loader=None,
                    attn_implementation: str | None = None, compute_dtype=None):
    resolved = resolve_experiment_config(config)
    device = resolved.select_device(device)
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported checkpoint device: {device}")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("checkpoint device mps was requested, but MPS is unavailable; refusing fallback")
    directory = Path(directory); manifest = read_checkpoint_manifest(directory)
    identity = manifest["identity"]
    base, markers, contract = identity["base"], identity["markers"], identity["model_contract"]
    if base.get("name") != config.model.name: raise ValueError("checkpoint model name mismatch")
    if base.get("revision") != config.model.revision: raise ValueError("checkpoint model revision mismatch")
    if base.get("type") != config.model.expected_model_type: raise ValueError("checkpoint base model type mismatch")
    if identity["family"] != resolved.model.family.family_id:
        raise ValueError("checkpoint model family mismatch")
    if identity.get("model_output_id", resolved.model.output_model_id) != resolved.model.output_model_id:
        raise ValueError("checkpoint output model ID mismatch")
    if identity["backend"] != resolved.model.backend.backend_id:
        raise ValueError("checkpoint backend mismatch")
    gemma4 = config.model.family == "gemma4_e2b_text"
    qualification = None
    if gemma4:
        tensor_hashes = {name: manifest["tensors"][name]["sha256"]
                         for name in ("adapter", "pointer")}
        qualification = manifest.get("qualification")
        validate_qualification_receipt(
            qualification, config=config, backend="torch",
            expected_code_sha256=qualification_code_sha256(
                "torch", Path(__file__).resolve().parents[4]),
            expected_checkpoint_hashes=tensor_hashes)
        validate_qualification_sidecars(directory, qualification)
    if identity["protocol"] != dataclasses.asdict(config.protocol):
        raise ValueError("checkpoint protocol mismatch")
    if identity["recipe"].get("sha256") != resolved.recipe_sha256:
        raise ValueError("checkpoint scientific recipe mismatch")
    if contract.get("representation_version") != 1: raise ValueError("unknown checkpoint representation")
    if markers.get("bos", False) is not False: raise ValueError("checkpoint marker BOS contract mismatch")
    if identity["tokenizer"].get("revision") != config.model.revision: raise ValueError("checkpoint tokenizer revision mismatch")
    if contract.get("head_width") != resolved.model.family.pointer_width: raise ValueError("checkpoint pointer head width mismatch")
    expected_lora = {"r": resolved.model.family.lora_rank,
                     "alpha": resolved.model.family.lora_alpha,
                     "dropout": resolved.model.family.lora_dropout,
                     "targets": list(resolved.model.family.lora_targets)}
    if contract.get("lora") != expected_lora: raise ValueError("checkpoint LoRA contract mismatch")
    if manifest["execution"].get("device") is not None and manifest["execution"]["device"] not in {"cpu", "mps", "cuda"}: raise ValueError("checkpoint device identity is invalid")
    selected_compute_dtype = (
        config.training.dtype if compute_dtype is None else compute_dtype)
    if selected_compute_dtype != config.training.dtype:
        raise ValueError("checkpoint compute dtype does not match the resolved config")
    expected_weights_dtype = (
        selected_compute_dtype if config.model.family == "gemma4_e2b_text" else "fp32")
    execution = manifest["execution"]
    if execution.get("dtype", config.training.dtype) != config.training.dtype:
        raise ValueError("checkpoint compute dtype mismatch")
    if execution.get("weights_dtype", expected_weights_dtype) != expected_weights_dtype:
        raise ValueError("checkpoint decoder weights dtype mismatch")
    if config.model.family == "gemma4_e2b_text":
        if execution.get("compute_dtype") != selected_compute_dtype:
            raise ValueError("checkpoint compute dtype mismatch")
        if execution.get("weights_dtype") != expected_weights_dtype:
            raise ValueError("checkpoint decoder weights dtype mismatch")
        if execution.get("source_weights_dtype") != "bf16":
            raise ValueError("checkpoint source weights dtype mismatch")
    if expected_marker_map is not None and (markers.get("ids") != expected_marker_map.ids or markers.get("strings") != expected_marker_map.strings): raise ValueError("checkpoint marker identity mismatch")
    if expected_marker_map is not None and markers.get("bos") != expected_marker_map.bos: raise ValueError("checkpoint marker BOS contract mismatch")
    if expected_marker_map is not None and identity["tokenizer"].get("revision") != expected_marker_map.tokenizer_revision: raise ValueError("checkpoint tokenizer revision mismatch")
    expected_tokenizer_sha = (expected_marker_map.tokenizer_sha256
                              if expected_marker_map is not None else None)
    if (expected_tokenizer_sha is not None
            and identity["tokenizer"].get("sha256") != expected_tokenizer_sha):
        raise ValueError("checkpoint tokenizer digest mismatch")
    # Validate the backend-owned tensor descriptors before constructing or
    # downloading the frozen base model.
    adapter_descriptor = manifest["tensors"]["adapter"]
    pointer_descriptor = manifest["tensors"]["pointer"]
    adapter_preflight = load_file(str(directory / adapter_descriptor["filename"]))
    pointer_weights = load_file(str(directory / pointer_descriptor["filename"]))
    if adapter_descriptor["shapes"] != {k: list(v.shape) for k, v in adapter_preflight.items()}:
        raise ValueError("adapter tensor names/shapes mismatch")
    if pointer_descriptor["shapes"] != {k: list(v.shape) for k, v in pointer_weights.items()}:
        raise ValueError("pointer tensor names/shapes mismatch")
    if any(tensor.dtype != torch.float32 for tensor in adapter_preflight.values()):
        raise ValueError("adapter tensor dtype mismatch")
    if any(tensor.dtype != torch.float32 for tensor in pointer_weights.values()):
        raise ValueError("pointer tensor dtype mismatch")
    if backbone_loader is None:
        backbone = None
    elif config.model.family == "gemma4_e2b_text":
        backbone = backbone_loader(
            config.model.name, config.model.revision,
            compute_dtype=selected_compute_dtype)
    else:
        backbone = backbone_loader(config.model.name, config.model.revision)
    # Test/in-process loaders may return a raw Gemma module extracted from a
    # previously constructed fixture.  Transformers can retain PEFT's marker
    # attribute on that raw module; remove only that stale marker so it is not
    # mistaken for an already-wrapped model.
    stale_peft_marker = (backbone is not None
                         and not backbone.__class__.__module__.startswith("peft.")
                         and hasattr(backbone, "peft_config"))
    if stale_peft_marker:
        delattr(backbone, "peft_config")
    with warnings.catch_warnings():
        # Some in-process raw fixtures retain PEFT's marker through Module's
        # attribute forwarding.  They are still loaded as a fresh adapter
        # wrapper; production raw loaders do not emit this warning.
        if stale_peft_marker:
            warnings.filterwarnings("ignore", message="You are trying to modify a model with PEFT for a second time")
        model = resolved.create_model(backbone=backbone,
                                      attn_implementation=attn_implementation)
    from peft import set_peft_model_state_dict
    weights = adapter_preflight
    expected = {_canonical(k): list(v.shape) for k, v in model.backbone.state_dict().items() if "lora_" in k}
    actual = {k: list(v.shape) for k, v in weights.items()}
    if expected != {_canonical(k): v for k, v in actual.items()} or manifest["tensors"]["adapter"]["shapes"] != actual: raise ValueError("adapter tensor names/shapes mismatch")
    if any(parameter.dtype != torch.float32
           for name, parameter in model.backbone.named_parameters() if "lora_" in name):
        raise ValueError("constructed LoRA master parameters are not FP32")
    result = set_peft_model_state_dict(model.backbone, weights)
    missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
    unexpected = [k for k in getattr(result, "unexpected_keys", []) if "lora_" in k]
    if missing or unexpected: raise ValueError("strict PEFT adapter load mismatch")
    head_weights = pointer_weights
    if manifest["tensors"]["pointer"]["shapes"] != {k: list(v.shape) for k, v in head_weights.items()}: raise ValueError("pointer tensor names/shapes mismatch")
    model.head.load_state_dict(head_weights, strict=True)
    model.to(device).eval()
    if gemma4 and trainable_content_sha256("torch", model) != \
            qualification.get("trainable_parameters_sha256"):
        raise ValueError("loaded Torch adapter/head differs from its qualification receipt")
    return model, manifest
