"""Strict access to the two frozen Kev suite manifests and their raw JSONL files."""

from __future__ import annotations

import json
import os
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from ..network import use_system_ssl
from ..config import load_config
from ..hashhelper import sha256
from ..materialize import materialize
from ..pathlookup import reference_bytes
from ..tokenization import MarkerMap, encode, rows_of

KEV_SHA = "08ab0b87d27cb5577a3b371ad7ed4e4686b0502b"
HF_REVISION = "a88f56db5341397299137cb68775c2ea6e3f68cb"
MANIFEST_SHA256 = "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
MANIFEST_HASHES = {"decision-v7": MANIFEST_SHA256,
                   "transfer-v4": "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1"}
MANIFEST_URLS = {
    "decision-v7": f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_SHA}/evals/v7/decision-v7/manifest.json",
    "transfer-v4": f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_SHA}/evals/v4/transfer-v4/manifest.json",
}


class SuiteError(ValueError):
    pass

KNOWN_SUITES = frozenset(("decision-v7", "transfer-v4"))
KNOWN_SPLITS = frozenset(("train", "calibration", "development", "test"))
REQUIRED_FILE_HASHES = {
    "decision-v7": {"train": "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad",
                    "calibration": "12c2029a07d24828f5f8cc48f09f572c81457de96f3d5d16180be4c090abd5ca",
                    "development": "8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c",
                    "test": "cd7d129a84232e0c7a4e92b4840a6dfe14c8bb93c5d1904b526fa8c085325d2d"},
    "transfer-v4": {"development": "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e",
                    "test": "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517"},
}


def _sha(data: bytes) -> str:
    return sha256(data)


def _file_info(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in KNOWN_SPLITS:
        raise SuiteError(f"unknown split: {split}")
    key = split if split in manifest.get("files", {}) else f"{split}.jsonl"
    try:
        return manifest["files"][key]
    except KeyError as exc:
        raise SuiteError(f"manifest has no split {split}") from exc


def _validate_manifest_identity(suite: str, manifest: dict[str, Any]) -> None:
    if suite not in KNOWN_SUITES:
        raise SuiteError(f"unknown suite: {suite}")
    for split, digest in REQUIRED_FILE_HASHES[suite].items():
        if _file_info(manifest, split).get("sha256") != digest:
            raise SuiteError(f"manifest identity mismatch for {suite}")


def _download(url: str) -> bytes:
    use_system_ssl()
    with urllib.request.urlopen(url) as response:  # nosec B310: pinned HTTPS URLs only
        return response.read()


def load_manifest(suite: str, root: Path | None = None) -> dict[str, Any]:
    """Load a bundled manifest without making network access a prerequisite."""
    if suite not in KNOWN_SUITES:
        raise SuiteError(f"unknown suite: {suite}")
    try:
        data = ((root / "references" / "suites" / suite / "manifest.json").read_bytes()
                if root is not None else reference_bytes("suites", suite, "manifest.json"))
        if root is None and _sha(data) != MANIFEST_HASHES[suite]:
            raise SuiteError(f"bundled manifest hash mismatch for {suite}")
        manifest = json.loads(data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SuiteError(f"cannot load bundled manifest {suite}: {exc}") from exc
    if not isinstance(manifest.get("files"), dict) or not manifest.get("context"):
        raise SuiteError(f"invalid bundled manifest: {suite}")
    _validate_manifest_identity(suite, manifest)
    return manifest


def manifest_digest(suite: str, root: Path | None = None) -> str:
    """Return the verified byte digest of an official suite manifest."""
    if root is None:
        load_manifest(suite)
        return MANIFEST_HASHES[suite]
    data = (root / "references" / "suites" / suite / "manifest.json").read_bytes()
    load_manifest(suite, root=root)
    return _sha(data)


def load_split(path: Path, suite: str, split: str, manifest: dict[str, Any] | None = None,
               *, allow_test: bool = False) -> list[dict[str, Any]]:
    if suite not in KNOWN_SUITES:
        raise SuiteError(f"unknown suite: {suite}")
    if split not in KNOWN_SPLITS:
        raise SuiteError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise SuiteError("test requires explicit --allow-test")
    manifest = manifest or load_manifest(suite)
    if not manifest.get("smoke_only"):
        _validate_manifest_identity(suite, manifest)
    info = _file_info(manifest, split)
    verify_jsonl(path, info["sha256"], info["records"], info["questions"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        _validate_request(row)
    return rows


def _validate_request(row: dict[str, Any]) -> None:
    if not isinstance(row, dict) or "state" not in row or not isinstance(row.get("questions"), dict) or not row["questions"]:
        raise SuiteError("record must contain state and non-empty questions")
    meta = row.get("_meta", {})
    if not meta.get("id") or not meta.get("source") or not meta.get("group_id"):
        raise SuiteError("record provenance requires id, source, and group_id")
    for question in row["questions"].values():
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind not in {"choice", "noul", "score"}:
            raise SuiteError(f"unknown question type: {kind}")
        if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise SuiteError("choice criteria must be non-empty")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise SuiteError("score criteria must be non-empty")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise SuiteError("noul criteria must be an object")
        if "label" not in question:
            raise SuiteError("question is missing label")
        from ..representation import question_keys, validate_label, validate_target
        keys = question_keys(kind, criteria or ({} if kind == "noul" else []))
        validate_label(question, keys)
        if question.get("target") is not None:
            validate_target(question["target"], keys)


def verify_bytes(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if _sha(data) != expected_sha256:
        raise SuiteError(f"hash mismatch for {path}")
    return data


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def verify_jsonl(path: Path, expected_sha256: str, expected_records: int, expected_questions: int) -> dict[str, Any]:
    data = verify_bytes(path, expected_sha256)
    if b"\r" in data or (data and not data.endswith(b"\n")):
        raise SuiteError("JSONL must retain upstream UTF-8 LF bytes")
    records = questions = 0
    for line in data.splitlines():
        if not line: continue
        value = json.loads(line.decode("utf-8")); records += 1
        questions += len(value.get("questions", {}))
    if (records, questions) != (expected_records, expected_questions):
        raise SuiteError(f"count mismatch: got {(records, questions)}")
    return {"bytes": len(data), "sha256": expected_sha256, "records": records, "questions": questions}


def verify_data(data: bytes, expected_sha256: str, expected_records: int, expected_questions: int) -> dict[str, Any]:
    if _sha(data) != expected_sha256 or b"\r" in data or (data and not data.endswith(b"\n")):
        raise SuiteError("raw JSONL integrity check failed")
    records = questions = 0
    for line in data.splitlines():
        if line:
            value = json.loads(line.decode("utf-8")); records += 1; questions += len(value.get("questions", {}))
    if (records, questions) != (expected_records, expected_questions):
        raise SuiteError(f"count mismatch: got {(records, questions)}")
    return {"bytes": len(data), "sha256": expected_sha256, "records": records, "questions": questions}


def fetch_manifest(name: str, destination: Path) -> dict[str, Any]:
    if name not in MANIFEST_URLS: raise SuiteError(f"unknown suite: {name}")
    data = _download(MANIFEST_URLS[name])
    if _sha(data) != MANIFEST_HASHES[name]:
        raise SuiteError("authoritative decision-v7 manifest hash mismatch")
    atomic_write(destination, data)
    return json.loads(data)


def fetch_file(split: str, suite: str, destination: Path, manifest: dict[str, Any], *, allow_test=False) -> dict[str, Any]:
    if suite not in KNOWN_SUITES:
        raise SuiteError(f"unknown suite: {suite}")
    if split == "test" and not allow_test: raise SuiteError("test requires explicit --allow-test")
    _validate_manifest_identity(suite, manifest)
    info = _file_info(manifest, split)
    path = f"{'v7/decision-v7' if suite == 'decision-v7' else 'v4/transfer-v4'}/{split}.jsonl"
    url = f"https://huggingface.co/datasets/jaredpalmer/kev-suites/resolve/{HF_REVISION}/{path}"
    data = _download(url)
    result = verify_data(data, info["sha256"], info["records"], info["questions"])
    atomic_write(destination, data)
    return result


def audit(path: Path, manifest: dict[str, Any], split: str, *, allow_test: bool = False, suite: str | None = None) -> dict[str, Any]:
    if split not in KNOWN_SPLITS:
        raise SuiteError(f"unknown split: {split}")
    if split == "test" and not allow_test: raise SuiteError("test requires explicit --allow-test")
    _validate_manifest_identity(suite or ("decision-v7" if "train.jsonl" in manifest.get("files", {}) and manifest["files"]["train.jsonl"].get("records", 0) else "transfer-v4"), manifest)
    info = _file_info(manifest, split)
    result = verify_jsonl(path, info["sha256"], info["records"], info["questions"])
    result["overflow_records"] = None
    result["token_audit"] = "not_run"
    result["types"] = {}
    result["sources"] = {}
    result["variants"] = {}
    result["validation_errors"] = []
    seen_ids, seen_groups = set(), set()
    protocol = manifest.get("protocol", {})
    trainable_sources = set(manifest.get("trainable_sources", protocol.get("trainable_sources", [])))
    eval_only_sources = set(manifest.get("eval_only_sources", protocol.get("eval_only_sources", [])))
    heldout_sources = set(manifest.get("heldout_sources", protocol.get("heldout_sources", [])))
    excluded_structures = set(protocol.get("excluded_structure_keys", [])) | set(
        manifest.get("heldout_structures", protocol.get("heldout_structures", []))
    )
    excluded_shapes = set(protocol.get("excluded_shape_keys", [])) | set(
        manifest.get("heldout_shapes", protocol.get("heldout_shapes", []))
    )
    allowed_sources = trainable_sources | eval_only_sources | heldout_sources
    for line in path.read_bytes().splitlines():
        if not line: continue
        row = json.loads(line)
        _validate_request(row)
        meta = row.get("_meta", {})
        record_id = meta.get("id")
        if not record_id or record_id in seen_ids:
            result["validation_errors"].append(f"missing or duplicate request id: {record_id!r}")
        seen_ids.add(record_id)
        group_id = meta.get("group_id")
        if not group_id:
            result["validation_errors"].append(f"missing group id for {record_id!r}")
        else:
            seen_groups.add(group_id)
        source = meta.get("source")
        if allowed_sources and source not in allowed_sources:
            result["validation_errors"].append(f"source not in manifest: {meta.get('source')!r}")
        if split == "train" and trainable_sources and source not in (trainable_sources - heldout_sources):
            result["validation_errors"].append(f"non-trainable source in training: {source!r}")
        structure = meta.get("structure") or meta.get("structure_key")
        if split == "train" and structure in excluded_structures:
            result["validation_errors"].append(f"held-out structure in {record_id!r}")
        shape = meta.get("shape") or meta.get("shape_key")
        if split == "train" and shape in excluded_shapes:
            result["validation_errors"].append(f"held-out shape in {record_id!r}")
        result["sources"][meta.get("source", "unknown")] = result["sources"].get(meta.get("source", "unknown"), 0) + 1
        result["variants"][meta.get("variant", "clean")] = result["variants"].get(meta.get("variant", "clean"), 0) + 1
        for q in row.get("questions", {}).values():
            result["types"][q.get("type", "unknown")] = result["types"].get(q.get("type", "unknown"), 0) + 1
            try:
                from ..representation import question_keys, validate_label
                validate_label(q, question_keys(q["type"], q.get("criteria") or ({} if q["type"] == "noul" else [])))
            except (KeyError, ValueError) as exc:
                result["validation_errors"].append(str(exc))
    result["valid"] = not result["validation_errors"]
    result["note"] = "structural/source audit only; no Gemma token claim"
    return result


def verify(path: Path, manifest: dict[str, Any], split: str, *, allow_test: bool = False, suite: str | None = None) -> dict[str, Any]:
    """Verify a downloaded split; unlike audit this is deliberately no tokenizer work."""
    if split == "test" and not allow_test: raise SuiteError("test requires explicit --allow-test")
    suite = suite or ("decision-v7" if "train.jsonl" in manifest.get("files", {}) and manifest["files"]["train.jsonl"].get("records", 0) else "transfer-v4")
    _validate_manifest_identity(suite, manifest)
    info = _file_info(manifest, split)
    return verify_jsonl(path, info["sha256"], info["records"], info["questions"])


def smoke(data_root: Path = Path("data"), train_records: int = 128, dev_records: int = 64,
          out_dir: Path | None = None, seed: int = 0) -> dict[str, Any]:
    """Build a real, group-preserving structural subset from fetched train/dev bytes."""
    train_path = data_root / "decision-v7" / "train.jsonl"
    dev_path = data_root / "decision-v7" / "development.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise SuiteError("smoke requires verified decision-v7 train and development files")
    manifest = load_manifest("decision-v7")
    if out_dir is not None and out_dir.exists():
        raise SuiteError(f"refusing to overwrite smoke output: {out_dir}")
    train_rows = load_split(train_path, "decision-v7", "train", manifest)
    dev_rows = load_split(dev_path, "decision-v7", "development", manifest)
    def select(path: Path, budget: int) -> list[dict]:
        rows = train_rows if path == train_path else dev_rows
        groups, selected = {}, []
        for row in rows:
            key = row.get("_meta", {}).get("group_id", row.get("_meta", {}).get("id"))
            groups.setdefault(key, []).append(row)
        by_source: dict[str, list[list[dict]]] = {}
        for group in groups.values():
            by_source.setdefault(group[0].get("_meta", {}).get("source", "unknown"), []).append(group)
        rng = random.Random(seed)
        sources = list(by_source)
        rng.shuffle(sources)
        for groups_for_source in by_source.values():
            rng.shuffle(groups_for_source)
        cursor = 0
        while sources and len(selected) < budget:
            source = sources[cursor % len(sources)]; cursor += 1
            if not by_source[source]:
                sources.remove(source); continue
            group = by_source[source].pop(0)
            if selected and len(selected) + len(group) > budget: continue
            selected.extend(group)
        return selected
    train, dev = select(train_path, train_records), select(dev_path, dev_records)
    out = out_dir or data_root / "smoke"
    if out.exists():
        raise SuiteError(f"refusing to overwrite smoke output: {out}")
    out.mkdir(parents=True)
    for name, rows in (("train", train), ("development", dev)):
        atomic_write(out / f"{name}.jsonl", b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in rows))
    child_files = {}
    for name, rows in (("train", train), ("development", dev)):
        data = (out / f"{name}.jsonl").read_bytes()
        child_files[f"{name}.jsonl"] = {"sha256": _sha(data), "records": len(rows),
                                         "questions": sum(len(row["questions"]) for row in rows)}
    child = {"version": 1, "smoke_only": True, "test": False,
             "parent": {"suite": "decision-v7", "revision": HF_REVISION,
                        "manifest_sha256": MANIFEST_HASHES["decision-v7"],
                        "files": {name: info["sha256"] for name, info in
                                  (("train.jsonl", manifest["files"]["train.jsonl"]),
                                   ("development.jsonl", manifest["files"]["development.jsonl"]))}},
             "files": child_files,
             "source": "derived without resampling", "seed": seed}
    atomic_write(out / "manifest.json", (json.dumps(child, indent=2, sort_keys=True) + "\n").encode())
    return {"status": "generated", "train_records": len(train), "dev_records": len(dev),
            "train_groups": len({r.get("_meta", {}).get("group_id") for r in train}),
            "dev_groups": len({r.get("_meta", {}).get("group_id") for r in dev}),
            "test_included": False,
             "metadata_sha256": _sha(json.dumps({"train": train, "dev": dev}, sort_keys=True, ensure_ascii=False).encode()),
             "output": str(out)}


def token_length_audit(data_root: Path, config_path: str, marker_path: Path,
                       *, augment_train: bool = False, seed: int = 0, epochs: int | None = None,
                       seeds: tuple[int, ...] | None = None,
                       output: Path = Path("runs/reference/token-length-audit.json")) -> dict[str, Any]:
    """Measure actual tokenizer lengths without changing or truncating source bytes."""
    from transformers import AutoTokenizer
    from ..training.batching import variants_for_request

    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
    markers = MarkerMap.load(marker_path, tokenizer)
    caps = {"state": config.training.state_cap, "branch": config.training.branch_cap,
            "packed": config.training.packed_cap}
    epochs = config.training.epochs if epochs is None else epochs
    seeds = (seed,) if seeds is None else seeds
    if augment_train and (epochs < 1 or not seeds or any(s < 0 for s in seeds)):
        raise ValueError("training audit requires positive epochs and non-negative seeds")
    partitions = (("decision-v7", "train"), ("decision-v7", "calibration"),
                  ("decision-v7", "development"), ("transfer-v4", "development"))
    manifest_cache = {suite: load_manifest(suite) for suite, _ in partitions}
    report: dict[str, Any] = {"model": config.model.name, "revision": config.model.revision,
                              "markers": markers.ids, "caps": caps, "partitions": {}}

    def measure(name: str, rows: list[dict[str, Any]], variant: str, *,
                training_seed: int | None = None, epoch: int = 0) -> None:
        values = {"records": len(rows), "variants": 0, "questions": 0, "overflow_records": 0,
                   "max_state": 0, "max_branch": 0, "max_packed": 0,
                   "maxima": {}, "top": {"state": [], "branch": [], "packed": []},
                   "overflow_by_cap": {kind: 0 for kind in caps}, "variant": variant}
        overflow_ids: set[str] = set()
        for row in rows:
            ident = row.get("_meta", {}).get("id", "unknown")
            if training_seed is None:
                encodings = [("unaugmented", encode(tokenizer, materialize(row), markers,
                              state_cap=10**9, branch_cap=10**9, packed_cap=10**9))]
            else:
                variants = variants_for_request(row, seed=training_seed, epoch=epoch,
                            tokenizer=tokenizer, markers=markers, caps=(10**9, 10**9, 10**9),
                            p_none=config.training.p_none, p_none_distract=config.training.p_none_distract,
                            p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair)
                encodings = [(v.variant, v.encoding) for v in variants]
            for kind_name, encoded in encodings:
                _, _, questions = rows_of(encoded)
                values["variants"] += 1
                values["questions"] += len(questions)
                lengths = {"state": encoded["state_length"],
                           "branch": encoded["state_length"] + max((len(q["ids"]) for q in questions), default=0),
                           "packed": len(encoded["ids"])}
                entry = {"id": ident, "variant": kind_name}
                for kind, length in lengths.items():
                    item = {**entry, "length": length}
                    if length > values[f"max_{kind}"]:
                        values[f"max_{kind}"] = length
                        values["maxima"][kind] = item
                    top = values["top"][kind]
                    top.append(item)
                    if len(top) > 10:
                        top.sort(key=lambda value: (-value["length"], value["id"], value["variant"]))
                        top.pop()
                    if length > caps[kind]:
                        overflow_ids.add(ident)
                        values["overflow_by_cap"][kind] += 1
        values["overflow_records"] = len(overflow_ids)
        for kind in values["top"]:
            values["top"][kind].sort(key=lambda item: (-item["length"], item["id"], item["variant"]))
        report["partitions"][name] = values

    for suite, split in partitions:
        path = data_root / suite / f"{split}.jsonl"
        rows = load_split(path, suite, split, manifest_cache[suite])
        measure(f"{suite}/{split}/unaugmented", rows, "unaugmented")
        if augment_train and suite == "decision-v7" and split == "train":
            for training_seed in seeds:
                for epoch in range(epochs):
                    measure(f"{suite}/{split}/seed_{training_seed}/epoch_{epoch + 1}", rows,
                            "training", training_seed=training_seed, epoch=epoch)
    atomic_write(output, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    report["output"] = str(output)
    return report
