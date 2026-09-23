"""Validated, backend-neutral inference checkpoint manifest access."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"
MANIFEST_FORMAT = "gev.inference-checkpoint"
MANIFEST_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and validate the complete inference-artifact contract and tensor bytes."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    version = manifest.get("version") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("format") != MANIFEST_FORMAT
            or isinstance(version, bool) or not isinstance(version, int)
            or version != MANIFEST_VERSION):
        raise ValueError("unsupported inference checkpoint manifest")
    for section in ("identity", "lineage", "training", "execution", "tensors", "calibration"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"checkpoint manifest {section} must be an object")
    identity = manifest["identity"]
    for field in ("family", "backend", "base", "tokenizer", "markers", "protocol", "recipe", "model_contract"):
        if field not in identity:
            raise ValueError(f"checkpoint manifest identity missing {field}")
    base, tokenizer, markers, protocol, recipe = (identity[key] for key in
        ("base", "tokenizer", "markers", "protocol", "recipe"))
    if not isinstance(identity["family"], str) or not identity["family"]:
        raise ValueError("checkpoint manifest family is invalid")
    if not isinstance(identity["backend"], str) or not identity["backend"]:
        raise ValueError("checkpoint manifest backend is invalid")
    if (not isinstance(base, dict)
            or any(not isinstance(base.get(k), str) or not base[k] for k in ("name", "revision", "type"))
            or not re.fullmatch(r"[0-9a-f]{40}", str(base.get("revision", "")))):
        raise ValueError("checkpoint manifest pinned base identity is invalid")
    if (not isinstance(tokenizer, dict) or tokenizer.get("revision") != base["revision"]):
        raise ValueError("checkpoint manifest tokenizer identity is invalid")
    if (tokenizer.get("sha256") is not None
            and (not isinstance(tokenizer["sha256"], str)
                 or not _SHA256.fullmatch(tokenizer["sha256"]))):
        raise ValueError("checkpoint manifest tokenizer digest is invalid")
    if (not isinstance(markers, dict) or not isinstance(markers.get("ids"), dict)
            or not isinstance(markers.get("strings"), dict)
            or set(markers["ids"]) != set(markers["strings"])
            or not markers["ids"] or markers.get("bos") is not False):
        raise ValueError("checkpoint manifest marker identity is invalid")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in markers["ids"].values())
            or len(set(markers["ids"].values())) != len(markers["ids"])
            or any(not isinstance(value, str) or not value for value in markers["strings"].values())):
        raise ValueError("checkpoint manifest marker identity is invalid")
    if (not isinstance(protocol, dict) or not isinstance(protocol.get("id"), str)
            or not protocol["id"] or isinstance(protocol.get("version"), bool)
            or not isinstance(protocol.get("version"), int) or protocol["version"] < 1):
        raise ValueError("checkpoint manifest protocol is invalid")
    if not isinstance(identity["model_contract"], dict):
        raise ValueError("checkpoint manifest model contract is invalid")
    if not isinstance(recipe, dict) or not isinstance(recipe.get("scientific_recipe"), dict) or not _SHA256.fullmatch(str(recipe.get("sha256", ""))):
        raise ValueError("checkpoint manifest recipe identity is invalid")
    recipe_hash = hashlib.sha256(json.dumps(recipe["scientific_recipe"], sort_keys=True,
                                             separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    if recipe_hash != recipe["sha256"]:
        raise ValueError("checkpoint manifest recipe hash mismatch")
    calibration = manifest["calibration"]
    temperature = calibration.get("temperature", 1.0)
    if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("checkpoint manifest calibration temperature is invalid")
    tensors = manifest["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != {"adapter", "pointer"}:
        raise ValueError("checkpoint manifest must describe adapter and pointer tensors")
    expected_files = {"adapter": "adapter_model.safetensors", "pointer": "pointer.safetensors"}
    for logical_name, descriptor in tensors.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"checkpoint {logical_name} tensor descriptor is invalid")
        filename, digest, shapes = descriptor.get("filename"), descriptor.get("sha256"), descriptor.get("shapes")
        if (filename != expected_files[logical_name] or Path(filename).name != filename
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                or not isinstance(shapes, dict) or not shapes):
            raise ValueError(f"checkpoint {logical_name} tensor descriptor is invalid")
        if any(not isinstance(name, str) or not isinstance(shape, list)
               or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
               for name, shape in shapes.items()):
            raise ValueError(f"checkpoint {logical_name} tensor shapes are invalid")
        path = directory / filename
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"checkpoint {logical_name} tensor hash mismatch")
    return manifest


def checkpoint_fingerprint(directory: str | Path) -> str:
    """Hash immutable model identity and tensor hashes, excluding mutable calibration."""
    manifest = read_checkpoint_manifest(directory)
    identity = manifest["identity"]
    stable = {
        "family": identity["family"],
        "backend": identity["backend"],
        "base": {
            "name": identity["base"]["name"],
            "type": identity["base"]["type"],
            "revision": identity["base"]["revision"],
        },
        "tokenizer": {
            "revision": identity["tokenizer"]["revision"],
            "sha256": identity["tokenizer"].get("sha256"),
        },
        "markers": identity["markers"],
        "adapter_sha256": manifest["tensors"]["adapter"]["sha256"],
        "pointer_sha256": manifest["tensors"]["pointer"]["sha256"],
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
