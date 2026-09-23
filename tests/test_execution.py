import pytest
import torch

from gev.execution import measure, select_representative_records
from gev.models.gemma import build_tiny_model


def _encoding():
    return {"ids": [2, 3, 10, 4, 10, 5, 20], "seg": [0, 0, 1, 1, 1, 1, 1],
            "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, 0, 0, 1, 1, -2],
            "decide_idx": [6], "opt_idx": [[3, 5]], "labels": [0], "state_length": 2}


def _records():
    values = []
    for source, count, questions in (("agnews", 3, 3), ("yelp", 2, 2), ("banking77", 3, 1)):
        for index in range(count):
            values.append({"_meta": {"source": source, "id": f"{source}/{index}"},
                           "questions": {str(q): {} for q in range(questions)}})
    return values


def test_selection_is_source_stratified_and_multQuestion():
    selected = select_representative_records(_records(), 8)
    assert len(selected) == 8
    assert sum(len(row["questions"]) for row in selected) == 16
    assert {row["_meta"]["source"] for row in selected} == {"agnews", "yelp", "banking77"}


def test_measure_rejects_empty_or_nonpositive_selection():
    model = build_tiny_model().eval()
    with pytest.raises(ValueError):
        measure(model, [], records=1)
    with pytest.raises(ValueError):
        measure(model, [_encoding()], records=0)


def test_bad_packed_delta_fails_even_when_cached_path_is_correct():
    model = build_tiny_model().eval()
    original = model.forward_packed_one
    def bad_packed(value):
        output = original(value)
        output[0][0] += 1
        return output
    model.forward_packed_one = bad_packed
    result = measure(model, [_encoding()], records=1, warmup=0)
    assert result["deltas"]["cached_first_vs_rows"]["probability"] <= result["threshold"]
    assert result["status"] == "failed"


def test_bad_second_cache_call_fails_repeat_gate():
    model = build_tiny_model().eval()
    original = model.forward_with_prefix
    calls = {"count": 0}

    def changed_second(encoding, prefix):
        calls["count"] += 1
        value = original(encoding, prefix)
        if calls["count"] == 2:
            value[0][0] += 1
        return value

    model.forward_with_prefix = changed_second
    result = measure(model, [_encoding()], records=1, warmup=0)
    assert result["deltas"]["cached_first_vs_rows"]["probability"] <= result["threshold"]
    assert result["status"] == "failed"
