import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gev.data import continuation as c
from gev.configuration.config import load_config


def _row(identifier):
    return {"_meta": {"id": identifier, "source": "train", "group_id": identifier},
            "state": "s", "questions": {"q": {"type": "noul", "instr": "i",
            "options": ["a"], "label": 0}}}


def _prepared_fixture(tmp_path, monkeypatch, *, seed=1, replay_count=2):
    night2 = [_row("n0"), _row("n1")]
    train = [_row(f"t{i}") for i in range(5)]
    raw = b"\n".join(json.dumps(x).encode() for x in night2) + b"\n"
    train_raw = b"train"
    monkeypatch.setattr(c, "NIGHT2_RECORDS", 2)
    monkeypatch.setattr(c, "NIGHT2_DATA_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(c, "DECISION_TRAIN_SHA256", hashlib.sha256(train_raw).hexdigest())
    monkeypatch.setattr(c, "load_manifest", lambda suite: {"seed": c.NIGHT2_SEED,
        "files": {"train.jsonl": {"sha256": c.DECISION_TRAIN_SHA256}}} if suite == "decision-v7" else {})
    monkeypatch.setattr(c, "load_split", lambda *args, **kwargs: train)
    root = tmp_path / "data"
    (root / "night2").mkdir(parents=True)
    (root / "night2" / "dates_unknowable.jsonl").write_bytes(raw)
    (root / "night2" / "manifest.json").write_text(json.dumps({"seed": c.NIGHT2_SEED}))
    # build_continuation hashes the local replay file, while load_split is mocked.
    (root / "decision-v7").mkdir()
    (root / "decision-v7" / "train.jsonl").write_bytes(train_raw)
    out = tmp_path / "prepared"
    plan = c.build_continuation(root, out=out, seed=seed, replay_count=replay_count)
    return root, out, plan, train


def test_replay_uses_optimizer_seed_and_night2_first(tmp_path, monkeypatch):
    _, _, plan, train = _prepared_fixture(tmp_path, monkeypatch)
    expected = [x["_meta"]["id"] for x in __import__("random").Random("replay:1").sample(train, 2)]
    assert plan["replay_ids"] == expected
    assert plan["replay_ids"] == [x["_meta"]["id"] for x in c.load_prepared(tmp_path / "prepared", replay_count=2)[0][2:]]
    assert [x["_meta"]["id"] for x in c.load_prepared(tmp_path / "prepared", replay_count=2)[0][:2]] == ["n0", "n1"]


def test_generator_seed_does_not_change_replay_ids(tmp_path, monkeypatch):
    _, _, one, _ = _prepared_fixture(tmp_path, monkeypatch, seed=1)
    assert one["night2_seed"] == c.NIGHT2_SEED
    assert one["seed"] == 1
    assert one["replay_ids"]


def test_prepared_recipe_and_sources_are_revalidated(tmp_path, monkeypatch):
    root, out, _, _ = _prepared_fixture(tmp_path, monkeypatch)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["recipe"] = "old-invalid-recipe"
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale"):
        c.load_prepared(out, data_root=root)


def test_prepared_bytes_are_revalidated(tmp_path, monkeypatch):
    _, out, _, _ = _prepared_fixture(tmp_path, monkeypatch)
    (out / "combined.jsonl").write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="hash"):
        c.load_prepared(out, replay_count=2)


def test_changed_shared_pool_is_rejected(tmp_path, monkeypatch):
    root, out, _, _ = _prepared_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(c, "DECISION_TRAIN_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="source hash mismatch"):
        c.load_prepared(out, data_root=root, replay_count=2)


def _metadata(tmp_path, **updates):
    adapter = tmp_path / "adapter_model.safetensors"
    pointer = tmp_path / "pointer.safetensors"
    adapter.write_bytes(b"adapter")
    pointer.write_bytes(b"pointer")
    value = {"format": "gev.inference-checkpoint", "version": 1,
             "identity": {"family": "gemma3_text", "backend": "torch",
                 "base": {"name": "google/gemma-3-1b-pt", "revision": "f" * 40, "type": "gemma3_text"},
                 "tokenizer": {"revision": "f" * 40},
                 "markers": {"ids": {"state": 1, "question": 2, "option_start": 3, "option_end": 4, "decide": 5},
                             "strings": {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "bos": False},
                 "protocol": {"id": "kev-decision-v7", "version": 1},
                 "recipe": {"scientific_recipe": {}, "sha256": hashlib.sha256(b"{}").hexdigest()},
                 "model_contract": {"lora": c.LORA_CONTRACT, "head_width": 256, "representation_version": 1}},
             "lineage": {}, "training": {"metrics": {}, "config": {}}, "execution": {},
             "tensors": {"adapter": {"filename": "adapter_model.safetensors", "sha256": hashlib.sha256(b"adapter").hexdigest(), "shapes": {"x": [1]}},
                         "pointer": {"filename": "pointer.safetensors", "sha256": hashlib.sha256(b"pointer").hexdigest(), "shapes": {"x": [1]}}},
             "calibration": {"temperature": 1.0}}
    lineage = value["lineage"]
    for key in ("source_sha256", "manifest_sha256"):
        if key in updates:
            lineage[key] = updates.pop(key)
    value["training"]["metrics"] = updates.pop("training", {})
    value["training"]["config"] = updates.pop("config", {})
    if "lora" in updates:
        value["identity"]["model_contract"]["lora"] = updates.pop("lora")
    if "marker_ids" in updates:
        value["identity"]["markers"]["ids"] = updates.pop("marker_ids")
    value.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    return value


def _config():
    return SimpleNamespace(model=SimpleNamespace(name="google/gemma-3-1b-pt", revision="f" * 40))


def test_init_metadata_rejects_wrong_rank_and_head_hash(tmp_path):
    _metadata(tmp_path, lora={"r": 8})
    with pytest.raises(ValueError, match="LoRA"):
        c.validate_init_metadata(tmp_path, _config())
    value = _metadata(tmp_path)
    value["tensors"]["adapter"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        c.validate_init_metadata(tmp_path, _config())


def test_init_metadata_classifies_complete_v7_and_fingerprints_weights(tmp_path):
    value = _metadata(tmp_path, source_sha256=c.DECISION_TRAIN_SHA256,
        manifest_sha256="a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2",
        training={"complete": True, "logical_steps": 3144, "processed_records": 25152},
        config={"experiment_id": "gemma3-1b-v7", "training": {"seed": 2, "epochs": 2,
            "learning_rate": 0.0001, "logical_batch": 8, "context_length": 384}})
    result = c.validate_init_metadata(tmp_path, _config())
    assert result["initializer_kind"] == "full-v7"
    assert result["initializer_fingerprint"]


def test_init_metadata_rejects_marker_contract(tmp_path):
    _metadata(tmp_path, marker_ids={"state": 1})
    with pytest.raises(ValueError, match="marker"):
        c.validate_init_metadata(tmp_path, _config())


def test_audit_counts_state_in_branch_limit(monkeypatch, tmp_path):
    rows = [_row("r")]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = (json.dumps(rows[0]) + "\n").encode()
    (prepared / "combined.jsonl").write_bytes(data)
    (prepared / "manifest.json").write_text(json.dumps({"recipe": c.PREPARED_RECIPE, "seed": 1,
        "replay_count": 2000, "combined_sha256": hashlib.sha256(data).hexdigest(), "records": 1}))
    monkeypatch.setattr(c, "load_prepared", lambda path: (rows, {}))
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_tokenizer",
                        lambda self, *args: object())
    monkeypatch.setattr("gev.domain.tokenization.MarkerMap.load", lambda *args: object())
    monkeypatch.setattr("gev.domain.tokenization.rows_of", lambda encoded: ([], [], [{"ids": [0] * 7}]))
    monkeypatch.setattr("gev.training.batching.variants_for_request", lambda *args, **kwargs: [SimpleNamespace(
        encoding={"state_length": 4, "ids": [0] * 10})])
    config = load_config("configs/smoke.toml")
    config = replace(config, model=replace(config.model, name="x", revision="r"),
                     training=replace(config.training, seed=1, p_none=.1,
                                      p_none_distract=.1, p_distract=.1,
                                      p_none_pair=.1, state_cap=10,
                                      branch_cap=10, packed_cap=20))
    with pytest.raises(ValueError, match="overflow"):
        c.audit_prepared_tokens(prepared, config, "markers.json")
