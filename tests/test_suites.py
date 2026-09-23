import hashlib
import json

import pytest

from gev.data.suites import SuiteError, load_split, verify_data


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
