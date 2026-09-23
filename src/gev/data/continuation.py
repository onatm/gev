"""Pinned Night 2 continuation data and replay protocol.

This module deliberately does not generate night2 data.  It only downloads the
already published artifact, verifies its bytes, and derives the training child
from the verified v7 train split.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any
import urllib.request

from .suites import atomic_write, load_manifest, load_split, REQUIRED_FILE_HASHES
from ..infrastructure.pathlookup import reference_bytes

KEV_REVISION = "08ab0b87d27cb5577a3b371ad7ed4e4686b0502b"
NIGHT2_SEED = "night2-20260920"  # artifact-generation seed; not the training seed
NIGHT2_MANIFEST_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_REVISION}/evals/night2/manifest.json"
NIGHT2_MANIFEST_SHA256 = "520b5d95e877fa972ac24d5c137c6b1e602fec709b935913dab86e40c83a34d1"
NIGHT2_DATA_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_REVISION}/evals/night2/dates_unknowable.jsonl"
NIGHT2_DATA_SHA256 = "afd8502d162163605ac446439e32c7b9083302dd78a6bfdc5e30075619e98437"
NIGHT2_RECORDS = 1425
NIGHT2_DATES = 900
NIGHT2_UNKNOWABLE = 525
REPLAY_COUNT = 2000
COMBINED_COUNT = 3425
LOGICAL_STEPS = 429
DECISION_TRAIN_SHA256 = REQUIRED_FILE_HASHES["decision-v7"]["train"]
PREPARED_RECIPE = "continuation-night2-v2"
LORA_CONTRACT = {"r": 16, "alpha": 32, "dropout": .05,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # nosec B310: pinned URL
        return response.read()


def fetch_night2(data_root: str | Path = "data") -> dict[str, Any]:
    """Fetch the frozen primary artifact using atomic, strict writes."""
    root = Path(data_root) / "night2"
    manifest_bytes = _download(NIGHT2_MANIFEST_URL)
    reference_manifest = reference_bytes("night2", "manifest.json")
    if _sha(reference_manifest) != NIGHT2_MANIFEST_SHA256 or manifest_bytes != reference_manifest:
        raise ValueError("night2 manifest bytes differ from the bundled reference")
    manifest = json.loads(manifest_bytes)
    item = manifest.get("files", {}).get("dates_unknowable.jsonl", {})
    if item.get("sha256") != NIGHT2_DATA_SHA256 or item.get("records") != NIGHT2_RECORDS:
        raise ValueError("night2 manifest identity/count mismatch")
    data = _download(NIGHT2_DATA_URL)
    if _sha(data) != NIGHT2_DATA_SHA256 or sum(1 for line in data.splitlines() if line) != NIGHT2_RECORDS:
        raise ValueError("night2 raw artifact hash/count mismatch")
    atomic_write(root / "manifest.json", manifest_bytes)
    atomic_write(root / "dates_unknowable.jsonl", data)
    return {"manifest": str(root / "manifest.json"), "data": str(root / "dates_unknowable.jsonl"),
            "data_sha256": _sha(data), "records": NIGHT2_RECORDS}


def _read_frozen_night2(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if _sha(data) != NIGHT2_DATA_SHA256:
        raise ValueError("night2 data hash mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line]
    if len(rows) != NIGHT2_RECORDS:
        raise ValueError("night2 record count mismatch")
    return rows


def build_continuation(data_root: str | Path = "data", *, out: str | Path | None = None,
                       seed: int = 1, replay_count: int = REPLAY_COUNT) -> dict[str, Any]:
    """Prepare the exact 3,425-row continuation child without changing sources."""
    root = Path(data_root)
    night2 = root / "night2" / "dates_unknowable.jsonl"
    manifest_path = root / "night2" / "manifest.json"
    if not night2.exists() or not manifest_path.exists():
        raise ValueError("frozen night2 artifact is missing; run `gev data fetch night2` first")
    night2_rows = _read_frozen_night2(night2)
    night2_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if night2_manifest.get("seed") != NIGHT2_SEED:
        raise ValueError("night2 provenance seed mismatch")
    train_manifest = load_manifest("decision-v7")
    train_path = root / "decision-v7" / "train.jsonl"
    train = load_split(train_path, "decision-v7", "train", train_manifest)
    if _sha(train_path.read_bytes()) != DECISION_TRAIN_SHA256:
        raise ValueError("decision-v7 train source hash mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("continuation replay seed must be a non-negative integer")
    if isinstance(replay_count, bool) or not isinstance(replay_count, int) or replay_count < 1:
        raise ValueError("continuation replay_count must be positive")
    replay = random.Random(f"replay:{seed}").sample(train, min(replay_count, len(train)))
    rows = night2_rows + replay
    if len(rows) != len(night2_rows) + len(replay):
        raise ValueError("continuation source assembly count mismatch")
    ids = [r["_meta"]["id"] for r in rows]
    replay_ids = [r["_meta"]["id"] for r in replay]
    sources = sorted({r["_meta"]["source"] for r in rows})
    payload = b"".join((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for r in rows)
    result = {"recipe": PREPARED_RECIPE, "seed": seed, "night2_seed": NIGHT2_SEED,
              "replay_count": replay_count, "records": len(rows),
              "logical_steps": (len(rows) + 7) // 8,
              "replay_records": len(replay), "night2_records": len(night2_rows),
              "night2_sha256": NIGHT2_DATA_SHA256, "replay_ids": ids[:len(replay)],
              "replay_ids_sha256": _sha(json.dumps(replay_ids, separators=(",", ":")).encode()),
              "source_hashes": {"decision-v7/train.jsonl": DECISION_TRAIN_SHA256,
                                 "night2/dates_unknowable.jsonl": NIGHT2_DATA_SHA256},
              "combined_sha256": _sha(payload), "sources": sources,
              "provenance": {"kev_revision": KEV_REVISION, "night2_manifest": str(manifest_path),
                              "night2_seed": NIGHT2_SEED, "replay_seed": seed,
                              "generated_by": "gev.data.continuation"}}
    result["replay_ids"] = replay_ids
    if out is not None:
        destination = Path(out)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite continuation: {destination}")
        destination.mkdir(parents=True)
        atomic_write(destination / "combined.jsonl", payload)
        atomic_write(destination / "manifest.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def load_prepared(path: str | Path, *, data_root: str | Path | None = None,
                  seed: int = 1, replay_count: int = REPLAY_COUNT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    data = (path / "combined.jsonl").read_bytes()
    if manifest.get("recipe") != PREPARED_RECIPE or manifest.get("seed") != seed or manifest.get("replay_count") != replay_count:
        raise ValueError("prepared continuation recipe is stale; regenerate it at a new recipe-specific path")
    if _sha(data) != manifest.get("combined_sha256"):
        raise ValueError("prepared continuation hash mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line]
    if len(rows) != manifest.get("records"):
        raise ValueError("prepared continuation count mismatch")
    if data_root is not None:
        expected = build_continuation(data_root, seed=seed, replay_count=replay_count)
        if manifest.get("source_hashes") != expected["source_hashes"] or manifest.get("night2_sha256") != expected["night2_sha256"]:
            raise ValueError("prepared continuation source hashes are stale; regenerate it at a new recipe-specific path")
        if manifest.get("replay_ids") != expected["replay_ids"]:
            raise ValueError("prepared continuation replay identity is stale; regenerate it at a new recipe-specific path")
    return rows, manifest


def validate_init_metadata(checkpoint: str | Path, config) -> dict[str, Any]:
    """Check the warm-start identity without loading backbone or adapter bytes."""
    directory = Path(checkpoint)
    from ..artifacts.checkpoint_identity import checkpoint_fingerprint, read_checkpoint_manifest
    manifest = read_checkpoint_manifest(directory)
    identity, lineage = manifest["identity"], manifest["lineage"]
    base, markers, contract = identity["base"], identity["markers"], identity["model_contract"]
    if identity.get("family") != "gemma3_text" or identity.get("backend") != "torch":
        raise ValueError("continuation init checkpoint family/backend mismatch")
    expected_protocol = dataclasses.asdict(config.protocol) if hasattr(config, "protocol") else {
        "id": "kev-decision-v7", "version": 1}
    if identity.get("protocol") != expected_protocol:
        raise ValueError("continuation init checkpoint protocol mismatch")
    if base.get("name") != config.model.name or base.get("revision") != config.model.revision:
        raise ValueError("continuation init checkpoint model identity mismatch")
    if (base.get("type") != "gemma3_text" or identity.get("backend") != "torch"
            or contract.get("lora") != LORA_CONTRACT):
        raise ValueError("continuation init checkpoint is not the Gemma LoRA contract")
    if contract.get("head_width") != 256:
        raise ValueError("continuation init pointer head width mismatch")
    if "qwen" in str(base.get("name", "")).lower():
        raise ValueError("Qwen checkpoints are not valid Gev continuation initializers")
    for field in ("ids", "strings"):
        if not markers.get(field):
            raise ValueError(f"continuation init manifest missing marker {field}")
    if (set(markers["ids"]) != {"state", "question", "option_start", "option_end", "decide"} or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in markers["ids"].values()) or
            len(set(markers["ids"].values())) != 5):
        raise ValueError("continuation init marker identity is invalid")
    if (set(markers["strings"]) != set(markers["ids"]) or
            any(not isinstance(value, str) or not value for value in markers["strings"].values())):
        raise ValueError("continuation init marker identity is invalid")
    training = manifest["training"].get("metrics", {})
    source_config = manifest["training"].get("config", {})
    source_training = source_config.get("training", {})
    correct_v7_config = (source_config.get("experiment_id") == "gemma3-1b-v7" and
                         source_training.get("seed") in {0, 1, 2} and source_training.get("epochs") == 2 and
                         source_training.get("learning_rate") == 0.0001 and
                         source_training.get("logical_batch") == 8 and
                         source_training.get("context_length") == 384 and
                         source_training.get("p_none_pair", .25) == .25)
    full_v7 = (lineage.get("source_sha256") == DECISION_TRAIN_SHA256 and
                lineage.get("manifest_sha256") == "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2" and
                correct_v7_config and
                training.get("complete") is True and training.get("logical_steps") == 3144 and
                training.get("processed_records") == 25152 and training.get("source_count", 12576) == 12576 and
                not lineage.get("diagnostic_smoke_init") and not lineage.get("smoke_only") and
                not lineage.get("continuation", {}).get("diagnostic_smoke_init"))
    return {"initializer_kind": "full-v7" if full_v7 else "diagnostic-smoke-or-partial",
            "initializer_fingerprint": checkpoint_fingerprint(directory),
            "config": source_config}


def audit_prepared_tokens(path: str | Path, config, marker_path: str | Path) -> dict[str, Any]:
    """Audit every prepared record with the real Gemma tokenizer, never truncate."""
    from ..configuration.resolved import resolve_experiment_config
    resolved = resolve_experiment_config(config)
    from ..domain.materialize import materialize
    from ..domain.tokenization import rows_of
    from ..training.batching import variants_for_request
    rows, _ = load_prepared(path)
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer, artifact_path=marker_path)
    caps = (config.training.state_cap, config.training.branch_cap, config.training.packed_cap)
    maxima = [0, 0, 0]; overflow = []
    for row in rows:
        try:
            variants = variants_for_request(row, seed=config.training.seed, epoch=0, tokenizer=tokenizer,
                                             markers=markers, caps=(10**9, 10**9, 10**9),
                                             p_none=config.training.p_none, p_none_distract=config.training.p_none_distract,
                                             p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair,
                                             encoder=resolved.encode_record)
            for variant in variants:
                encoded = variant.encoding
                _, _, questions = rows_of(encoded)
                values = (encoded["state_length"], encoded["state_length"] + max((len(q["ids"]) for q in questions), default=0), len(encoded["ids"]))
                maxima = [max(a, b) for a, b in zip(maxima, values)]
                if any(value > cap for value, cap in zip(values, caps)):
                    overflow.append(row["_meta"]["id"])
        except Exception as exc:
            raise ValueError(f"token audit failed for {row['_meta']['id']}; fix future caps/source, do not drop row: {exc}") from exc
    if overflow:
        raise ValueError(f"token overflow in {len(overflow)} records; fix future caps/source, do not filter original data")
    return {"records": len(rows), "overflow_records": 0, "max_state": maxima[0],
            "max_branch": maxima[1], "max_packed": maxima[2], "caps": caps,
            "tokenizer_revision": config.model.revision}
