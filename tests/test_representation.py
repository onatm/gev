from gev.representation import option_text, question_keys, render, to_record


def test_nested_render_and_option_forms():
    assert render({"a": ["x", {"b": "y"}]}) == "a:\n  - x\n  - b: y"
    assert option_text("k", {"detail": "v"}) == "k: detail: v"
    assert question_keys("noul", {}) == ["false", "true"]


def test_to_record_translates_keys_and_keeps_metadata_out():
    request = {"state": {"source_id": "not text", "body": "hello"}, "questions": {
        "yn": {"type": "noul", "instructions": "ok?", "label": True},
        "pick": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": None}, "label": "b"},
        "score": {"type": "score", "instructions": "rate", "criteria": ["low", "high"], "label": 1},
    }}
    record, meta = to_record(request)
    assert record["questions"][0]["options"] == ["no", "yes"]
    assert record["questions"][1]["label"] == "b"
    assert meta[2]["keys"] == ["0", "1"]
    assert "source_id" in record["state"] and "label" not in record["state"]
