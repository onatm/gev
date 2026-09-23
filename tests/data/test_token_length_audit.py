from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

from gev.configuration.config import load_config
from gev.data import suites
from gev.domain.materialize import materialize
from gev.domain.tokenization import encode, rows_of
from gev.training.batching import variants_for_request


class CharacterTokenizer:
    all_special_tokens = ()

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[1] * len(text))


def test_training_audit_counts_state_and_all_pair_variants_with_source_ids(monkeypatch, tmp_path):
    tokenizer = CharacterTokenizer()
    markers = SimpleNamespace(ids={"state": 2, "question": 3, "option_start": 4,
                                   "option_end": 5, "decide": 6},
                              strings={"state": "<state>", "question": "<question>",
                                       "option_start": "<option_start>", "option_end": "<option_end>",
                                       "decide": "<decide>"})
    request = {"state": "evidence", "questions": {"q": {
        "type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}, "label": "a"}},
        "_meta": {"id": "source/1", "source": "source"}}
    clean = encode(tokenizer, materialize(request), markers, state_cap=10**9,
                   branch_cap=10**9, packed_cap=10**9)
    _, _, clean_rows = rows_of(clean)
    branch_cap = clean["state_length"] + len(clean_rows[0]["ids"])
    assert len(clean_rows[0]["ids"]) < branch_cap
    config = load_config("configs/smoke.toml")
    config = replace(config, model=replace(config.model, name="local", revision="test"),
                     training=replace(config.training, state_cap=1000,
                                      branch_cap=branch_cap, packed_cap=1000,
                                      epochs=1, p_none=0, p_none_distract=0,
                                      p_distract=0, p_none_pair=1))
    monkeypatch.setattr(suites, "load_config", lambda _: config)
    monkeypatch.setattr(suites, "load_manifest", lambda _: {})
    monkeypatch.setattr(suites, "load_split", lambda path, *_: [request] if Path(path).name == "train.jsonl" else [])
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_tokenizer",
                        lambda self, *args: tokenizer)
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_markers",
                        lambda self, *args: markers)

    result = suites.token_length_audit(tmp_path, "unused", tmp_path / "markers",
                                       augment_train=True, seeds=(0,), output=tmp_path / "audit.json")
    clean_report = result["partitions"]["decision-v7/train/unaugmented"]
    assert clean_report["max_branch"] == branch_cap
    assert clean_report["overflow_records"] == 0

    variants = variants_for_request(request, seed=0, epoch=0, tokenizer=tokenizer,
                                    markers=markers, caps=(10**9, 10**9, 10**9),
                                    p_none=0, p_none_distract=0, p_distract=0, p_none_pair=1)
    expected = max(v.encoding["state_length"] + len(rows_of(v.encoding)[2][0]["ids"])
                   for v in variants)
    report = result["partitions"]["decision-v7/train/seed_0/epoch_1"]
    assert report["variants"] == 3
    assert report["max_branch"] == expected > branch_cap
    assert report["overflow_records"] == 1
    assert report["overflow_by_cap"]["branch"] > 0
    assert report["maxima"]["branch"]["id"] == "source/1"
    assert report["maxima"]["branch"]["variant"] == "none_present"
