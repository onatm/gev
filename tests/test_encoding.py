import pytest

from gev.encoding import EncodingError, Markers, encode, flatten_rows
from gev.records import materialize
from tests.conftest import MARKERS, request


def test_markers_resolve_to_existing_single_tokens(tokenizer):
    markers = Markers.resolve(tokenizer, MARKERS)
    assert markers.ids == {"state": 4, "question": 5, "option_start": 6, "option_end": 7, "decide": 8}
    with pytest.raises(EncodingError, match="not in the base vocabulary"):
        Markers.resolve(tokenizer, ("<x>", *MARKERS[1:]))


def test_row_layout(tokenizer):
    markers = Markers.resolve(tokenizer, MARKERS)
    encoded = encode(tokenizer, materialize(request(0)), markers, state_cap=64, branch_cap=128)
    assert encoded["state"][0] == markers.ids["state"]
    choice = encoded["rows"][0]
    assert choice["ids"][0] == markers.ids["question"] and choice["ids"][choice["decide"]] == markers.ids["decide"]
    assert [choice["ids"][i] for i in choice["opts"]] == [markers.ids["option_end"]] * 3
    (_, ids, decide, options), _ = flatten_rows([encoded])
    assert ids[:len(encoded["state"])] == encoded["state"] and decide == len(encoded["state"]) + choice["decide"]


def test_user_text_cannot_inject_markers(tokenizer):
    markers = Markers.resolve(tokenizer, MARKERS)
    record = materialize(request(0))
    record["state"] = "a <unused4> b <|decide|>"
    encoded = encode(tokenizer, record, markers, state_cap=64, branch_cap=128)
    assert encoded["state"].count(markers.ids["decide"]) == 0


def test_caps_are_enforced(tokenizer):
    markers = Markers.resolve(tokenizer, MARKERS)
    with pytest.raises(EncodingError, match="state_cap"):
        encode(tokenizer, materialize(request(0)), markers, state_cap=2, branch_cap=128)
    with pytest.raises(EncodingError, match="branch_cap"):
        encode(tokenizer, materialize(request(0)), markers, state_cap=64, branch_cap=10)
