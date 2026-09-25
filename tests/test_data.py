import json
import random

import pytest

from gev import data
from gev.records import materialize, public_request, render
from tests.conftest import request
from tests.kev_reference import augment as reference_augment
from tests.kev_reference import render as reference_render


def choice_request():
    return {"state": "evidence", "questions": {
        "q": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}, "label": "a"},
        "soft": {"type": "choice", "criteria": {"x": "X", "y": "Y"}, "label": "x", "target": {"x": .25, "y": .75}},
    }}


def test_augmentation_matches_pinned_kev_for_many_seeds():
    for seed in range(40):
        assert data.augment(choice_request(), random.Random(seed)) == \
            reference_augment(choice_request(), random.Random(seed), .1, .12, .15)


def test_render_matches_pinned_kev():
    value = {"a": [1, {"b": None}], "c": {"d": True}}
    assert render(value) == reference_render(value)


def test_none_pair_keeps_state_and_removes_the_answer():
    present, absent = data.none_pair(choice_request(), random.Random(2))
    assert present["state"] == absent["state"] and list(present["questions"]) == ["q"]
    assert len(present["questions"]["q"]["criteria"]) == 4
    assert absent["questions"]["q"]["label"] in absent["questions"]["q"]["criteria"]
    assert "a" not in absent["questions"]["q"]["criteria"]


def test_training_variants_are_deterministic_per_seed_epoch_and_record():
    kwargs = dict(p_none=.1, p_none_distract=.12, p_distract=.15, p_none_pair=1.0)
    first = data.training_variants(request(3), seed=1, epoch=0, **kwargs)
    assert first == data.training_variants(request(3), seed=1, epoch=0, **kwargs)
    assert len(first) == 3  # augmented clean + contrastive pair


def test_epoch_order_replays_cumulative_shuffles():
    rows = [request(i) for i in range(20)]
    rng, order = random.Random(7), list(rows)
    for epoch in range(3):
        rng.shuffle(order)
        assert data.epoch_order(rows, 7, epoch) == order


def test_materialize_and_public_request():
    record = materialize(request(1))
    assert [q["keys"] for q in record["questions"]] == [["a", "b", "c"], ["false", "true"]]
    assert record["questions"][0]["label"] == 1 and record["questions"][1]["label"] == 0
    public = public_request(request(1))
    assert "label" not in public["questions"]["pick"] and "_meta" not in public
    with pytest.raises(ValueError):
        materialize({**request(1), "questions": {"q": {"type": "choice", "criteria": {"a": 1}, "label": "z"}}})


def test_load_split_verifies_bytes(sample_root):
    rows, digest = data.load_split(sample_root, "decision-v7", "train")
    assert len(rows) == 16 and len(digest) == 64
    (sample_root / "train.jsonl").write_text(json.dumps(request(0)) + "\n")
    with pytest.raises(data.DataError, match="hash mismatch"):
        data.load_split(sample_root, "decision-v7", "train")
    with pytest.raises(data.DataError, match="sample of decision-v7"):
        data.load_split(sample_root, "transfer-v4", "development")


def test_packaged_manifests_are_pinned():
    for suite in data.SUITES:
        assert data.manifest(suite)["files"]


def test_sample_is_group_preserving(tmp_path, monkeypatch):
    from tests.conftest import write_sample

    source = write_sample(tmp_path / "source", train=30, development=12)
    root = tmp_path / "root" / "decision-v7"  # sample() reads the normal <suite>/<split>.jsonl layout
    root.mkdir(parents=True)
    for split in ("train", "development"):
        (root / f"{split}.jsonl").write_bytes((source / f"{split}.jsonl").read_bytes())
    monkeypatch.setattr(data, "manifest", lambda suite: json.loads((source / "manifest.json").read_text()))
    result = data.sample(tmp_path / "root", tmp_path / "smoke", train_records=10, dev_records=4)
    assert result["files"]["train.jsonl"]["records"] == 10
    rows, _ = data.load_split(tmp_path / "smoke", "decision-v7", "train")
    assert len({row["_meta"]["source"] for row in rows}) == 2
