import hashlib
import json

import pytest

from gev.data.suites import SuiteError, fetch_file, load_split, verify_data
from gev.data.access import load_verified_split, validate_training_rows


def test_raw_integrity_counts_before_publish():
    data = b'{"questions":{"q":{"type":"noul"}}}\n'
    digest = hashlib.sha256(data).hexdigest()
    assert verify_data(data, digest, 1, 1)["records"] == 1
    with pytest.raises(SuiteError): verify_data(data + b"x", digest, 1, 1)


def test_empty_partition_is_valid_bytes():
    digest = hashlib.sha256(b"").hexdigest()
    assert verify_data(b"", digest, 0, 0)["records"] == 0


def test_child_manifest_reload_and_tamper_guard(tmp_path):
    row = {"state": "x", "questions": {"q": {"type": "noul", "label": True}},
           "_meta": {"id": "source/0", "source": "source", "group_id": "g"}}
    data = (json.dumps(row) + "\n").encode()
    path = tmp_path / "train.jsonl"; path.write_bytes(data)
    manifest = {"smoke_only": True, "files": {"train.jsonl": {
        "sha256": hashlib.sha256(data).hexdigest(), "records": 1, "questions": 1}},
        "context": {"max_state": 384}}
    assert len(load_split(path, "decision-v7", "train", manifest)) == 1
    path.write_bytes(data + b"tampered")
    with pytest.raises(SuiteError): load_split(path, "decision-v7", "train", manifest)


def test_unknown_suite_and_split_are_rejected():
    with pytest.raises(SuiteError): load_split(__file__, "not-a-suite", "train")
    with pytest.raises(SuiteError): load_split(__file__, "decision-v7", "nope")


def test_public_fetch_api_rejects_test_without_network_or_destination(tmp_path, monkeypatch):
    monkeypatch.setattr("gev.data.suites._download", lambda *_: pytest.fail("test bytes must not be fetched"))
    destination = tmp_path / "test.jsonl"
    with pytest.raises(SuiteError, match="locked evaluation path"):
        fetch_file("test", "decision-v7", destination, {})
    assert not destination.exists()


def test_application_loader_accepts_only_matching_smoke_child_and_returns_manifest_digest(tmp_path):
    row = {"state": "x", "questions": {"q": {"type": "noul", "label": True}},
           "_meta": {"id": "source/0", "source": "source", "group_id": "g"}}
    data = (json.dumps(row) + "\n").encode()
    (tmp_path / "train.jsonl").write_bytes(data)
    manifest = {"smoke_only": True, "parent": {"suite": "decision-v7"},
                "files": {"train.jsonl": {"sha256": hashlib.sha256(data).hexdigest(),
                                            "records": 1, "questions": 1}},
                "context": {"max_state": 384}}
    manifest_bytes = (json.dumps(manifest) + "\n").encode()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)

    rows, loaded_manifest, digest = load_verified_split(tmp_path, "decision-v7", "train", training=True)
    assert rows == [row]
    assert loaded_manifest == manifest
    assert digest == hashlib.sha256(manifest_bytes).hexdigest()


def test_application_loader_rejects_unapproved_child_and_nonlocked_test(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"smoke_only": False}))
    with pytest.raises(SuiteError, match="approved smoke child"):
        load_verified_split(tmp_path, "decision-v7", "train")

    (tmp_path / "manifest.json").write_text(json.dumps({
        "smoke_only": True, "parent": {"suite": "transfer-v4"}}))
    with pytest.raises(SuiteError, match="parent suite mismatch"):
        load_verified_split(tmp_path, "decision-v7", "train")

    (tmp_path / "manifest.json").write_text(json.dumps({
        "smoke_only": True, "parent": {"suite": "decision-v7"}}))
    with pytest.raises(SuiteError, match="child train"):
        load_verified_split(tmp_path, "decision-v7", "development", training=True)
    with pytest.raises(SuiteError, match="locked evaluation"):
        load_verified_split(tmp_path, "decision-v7", "test", allow_test=True)


def test_training_row_validation_preserves_provenance_and_suite_guards(monkeypatch):
    row = {"_meta": {"id": "x", "source": "allowed", "group_id": "g"}}
    monkeypatch.setattr("gev.data.access.load_manifest", lambda _: {"trainable_sources": ["allowed"]})
    validate_training_rows([row], "decision-v7")
    with pytest.raises(ValueError, match="non-trainable"):
        validate_training_rows([{**row, "_meta": {**row["_meta"], "source": "heldout"}}], "decision-v7")
    with pytest.raises(ValueError, match="development-only"):
        validate_training_rows([row], "transfer-v4")
