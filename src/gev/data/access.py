"""Verified application-level access to official and derived suite splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .suites import SuiteError, load_manifest, load_split, manifest_digest


def split_path(root: str | Path, suite: str, split: str) -> Path:
    """Resolve a split path, preferring the root-level smoke-child layout."""
    direct = Path(root) / f"{split}.jsonl"
    return direct if direct.exists() else Path(root) / suite / f"{split}.jsonl"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_split(root: str | Path, suite: str, split: str, *, training: bool = False,
                        allow_test: bool = False, _locked_test: bool = False):
    """Load an official split or explicitly-derived smoke child with provenance.

    Test rows are available only to the locked-evaluation path. Local manifests
    are accepted only when they describe an approved smoke child of ``suite``.
    """
    if allow_test and not _locked_test:
        raise SuiteError("test loading is restricted to the locked evaluation path")
    base = Path(root)
    child = base / "manifest.json"
    if not child.exists():
        candidate = base / suite / "manifest.json"
        child = candidate if candidate.exists() else None
    if child is not None:
        manifest = json.loads(child.read_text(encoding="utf-8"))
        if not manifest.get("smoke_only"):
            raise SuiteError("local suite manifest is not an approved smoke child")
        if manifest.get("parent", {}).get("suite") != suite:
            raise SuiteError("smoke child parent suite mismatch")
        if training and split != "train":
            raise SuiteError("smoke child training is restricted to child train.jsonl")
        path = child.parent / f"{split}.jsonl"
        return load_split(path, suite, split, manifest, allow_test=allow_test), manifest, file_digest(child)
    manifest = load_manifest(suite)
    path = base / suite / f"{split}.jsonl"
    return load_split(path, suite, split, manifest, allow_test=allow_test), manifest, manifest_digest(suite)


def validate_training_rows(rows: list[dict], suite: str) -> None:
    """Reject empty, held-out, or otherwise non-trainable custom training data."""
    if not rows:
        raise ValueError("training data is empty")
    manifest = load_manifest("decision-v7") if suite == "decision-v7" else None
    trainable = set((manifest or {}).get("trainable_sources", ()))
    forbidden = set((manifest or {}).get("eval_only_sources", ())) | set(
        (manifest or {}).get("heldout_sources", ()))
    for row in rows:
        meta = row.get("_meta", {})
        if not meta.get("id") or not meta.get("source") or not meta.get("group_id"):
            raise ValueError("custom training data requires _meta.id, _meta.source, and _meta.group_id")
        if meta.get("variant", "clean") not in {"clean", "none_present", "none_absent"}:
            raise ValueError(f"unsupported training variant: {meta.get('variant')}")
        if meta["source"] in forbidden or (trainable and meta["source"] not in trainable):
            raise ValueError(f"training data contains non-trainable or held-out source: {meta['source']}")
    if suite == "transfer-v4":
        raise ValueError("transfer-v4 is development-only and cannot be used for training")
