"""Atomic adapter-only checkpoints."""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile, warnings
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .models.gemma import GemmaRowModel, load_real_backbone

FORMAT_VERSION = 1

def _canonical(name: str) -> str:
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checkpoint_fingerprint(directory: str | Path) -> str:
    """Stable model identity, excluding mutable calibration metadata."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    identity = {key: meta.get(key) for key in ("model_name", "model_revision", "marker_ids", "marker_strings", "marker_bos", "adapter_sha256", "pointer_sha256")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def save_checkpoint(model, directory: str | Path, metadata: dict, tokenizer=None) -> Path:
    target = Path(directory)
    if target.exists(): raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    parent = target.parent; parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        meta = {"format_version": FORMAT_VERSION, "marker_bos": False,
                "tokenizer_revision": metadata.get("model_revision"), **metadata}
        meta.setdefault("execution_mode", meta.get("training", {}).get("execution_mode", "rows"))
        required = {"model_name", "model_revision", "marker_ids", "marker_strings", "head_width", "lora", "dtype", "state_cap", "branch_cap", "packed_cap", "representation_version", "base_model_type"}
        missing = sorted(required - set(meta))
        if missing: raise ValueError(f"checkpoint metadata missing: {', '.join(missing)}")
        (temp / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
        model.backbone.save_pretrained(temp, safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(temp / "pointer.safetensors"))
        if tokenizer is not None: tokenizer.save_pretrained(temp)
        adapter = temp / "adapter_model.safetensors"
        if not adapter.exists(): raise ValueError("PEFT did not write adapter_model.safetensors")
        meta["adapter_sha256"] = _sha(adapter)
        meta["pointer_sha256"] = _sha(temp / "pointer.safetensors")
        meta["adapter_tensors"] = {k: list(v.shape) for k, v in load_file(str(adapter)).items()}
        meta["pointer_tensors"] = {k: list(v.shape) for k, v in model.head.state_dict().items()}
        (temp / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise
    return target

def load_checkpoint(directory: str | Path, *, config, device="cpu", tokenizer=None, expected_marker_map=None, backbone_loader=None):
    directory = Path(directory); meta = json.loads((directory / "metadata.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION: raise ValueError("unsupported checkpoint format")
    if meta.get("model_name") != config.model.name: raise ValueError("checkpoint model name mismatch")
    if meta.get("model_revision") != config.model.revision: raise ValueError("checkpoint model revision mismatch")
    if meta.get("base_model_type") != config.model.expected_model_type: raise ValueError("checkpoint base model type mismatch")
    if meta.get("representation_version") != 1: raise ValueError("unknown checkpoint representation")
    if meta.get("marker_bos", False) is not False: raise ValueError("checkpoint marker BOS contract mismatch")
    if meta.get("tokenizer_revision") != config.model.revision: raise ValueError("checkpoint tokenizer revision mismatch")
    if meta.get("head_width") != 256: raise ValueError("checkpoint pointer head width mismatch")
    expected_lora = {"r": 16, "alpha": 32, "dropout": .05,
                     "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    if meta.get("lora") != expected_lora: raise ValueError("checkpoint LoRA contract mismatch")
    if meta.get("device") is not None and meta["device"] not in {"cpu", "mps"}: raise ValueError("checkpoint device identity is invalid")
    if expected_marker_map is not None and (meta.get("marker_ids") != expected_marker_map.ids or meta.get("marker_strings") != expected_marker_map.strings): raise ValueError("checkpoint marker identity mismatch")
    if expected_marker_map is not None and meta.get("marker_bos") != expected_marker_map.bos: raise ValueError("checkpoint marker BOS contract mismatch")
    if expected_marker_map is not None and meta.get("tokenizer_revision") != expected_marker_map.tokenizer_revision: raise ValueError("checkpoint tokenizer revision mismatch")
    if not (directory / "adapter_model.safetensors").exists() or _sha(directory / "adapter_model.safetensors") != meta.get("adapter_sha256"): raise ValueError("adapter hash mismatch")
    if not (directory / "pointer.safetensors").exists() or _sha(directory / "pointer.safetensors") != meta.get("pointer_sha256"): raise ValueError("pointer hash mismatch")
    backbone = (backbone_loader or load_real_backbone)(config.model.name, config.model.revision)
    # Test/in-process loaders may return a raw Gemma module extracted from a
    # previously constructed fixture.  Transformers can retain PEFT's marker
    # attribute on that raw module; remove only that stale marker so it is not
    # mistaken for an already-wrapped model.
    stale_peft_marker = not backbone.__class__.__module__.startswith("peft.") and hasattr(backbone, "peft_config")
    if stale_peft_marker:
        delattr(backbone, "peft_config")
    with warnings.catch_warnings():
        # Some in-process raw fixtures retain PEFT's marker through Module's
        # attribute forwarding.  They are still loaded as a fresh adapter
        # wrapper; production raw loaders do not emit this warning.
        if stale_peft_marker:
            warnings.filterwarnings("ignore", message="You are trying to modify a model with PEFT for a second time")
        model = GemmaRowModel(backbone, use_peft=not backbone.__class__.__module__.startswith("peft."))
    from peft import set_peft_model_state_dict, load_peft_weights
    weights = load_peft_weights(str(directory), device="cpu")
    expected = {_canonical(k): list(v.shape) for k, v in model.backbone.state_dict().items() if "lora_" in k}
    actual = {k: list(v.shape) for k, v in weights.items()}
    if expected != {_canonical(k): v for k, v in actual.items()} or meta.get("adapter_tensors") != actual: raise ValueError("adapter tensor names/shapes mismatch")
    result = set_peft_model_state_dict(model.backbone, weights)
    missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
    unexpected = [k for k in getattr(result, "unexpected_keys", []) if "lora_" in k]
    if missing or unexpected: raise ValueError("strict PEFT adapter load mismatch")
    head_weights = load_file(str(directory / "pointer.safetensors"))
    if meta.get("pointer_tensors") != {k: list(v.shape) for k, v in head_weights.items()}: raise ValueError("pointer tensor names/shapes mismatch")
    model.head.load_state_dict(head_weights, strict=True)
    model.to(device).eval()
    return model, meta
