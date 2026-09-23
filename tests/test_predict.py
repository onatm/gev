import io
import json
from types import SimpleNamespace

import torch

from gev.cli import main
from gev.application.prediction import predict_request
from gev.configuration.config import load_config
from gev.domain.tokenization import MarkerMap, encode


class TinyTokenizer:
    all_special_tokens = []

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) % 50 + 20 for char in text])


class FakeModel:
    def __init__(self):
        self.head = SimpleNamespace(temperature=1.0)
        self.seen_labels = None

    def eval(self):
        return self

    def probs(self, encoded):
        self.seen_labels = encoded["labels"]
        return [torch.softmax(torch.arange(len(options), dtype=torch.float32) / self.head.temperature, dim=0)
                for options in encoded["opt_idx"]]


def _markers():
    return MarkerMap(
        ids={"state": 1, "question": 2, "option_start": 3, "option_end": 4, "decide": 5},
        strings={"state": "<|state|>", "question": "<|question|>",
                 "option_start": "<|option_start|>", "option_end": "<|option_end|>",
                 "decide": "<|decide|>"},
        tokenizer_revision="test-revision",
    )


def test_predict_request_accepts_unlabeled_noul_choice_and_score_and_maps_options():
    request = {
        "state": "A return policy allows returns within 30 days.",
        "questions": {
            "truth-id": {"type": "noul", "instructions": "Is day 12 within the policy?",
                         "criteria": {"false": "No", "true": "Yes"}, "label": True},
            "choice-id": {"type": "choice", "instructions": "Choose an action.",
                          "criteria": {"allow_return": "Within 30 days", "deny_return": "After 30 days"}},
            "score-id": {"type": "score", "instructions": "Rate the fit.",
                         "criteria": ["outside", "inside"]},
        },
    }
    model = FakeModel()

    result = predict_request(request, model, TinyTokenizer(), _markers(), encoder=encode, state_cap=384,
                             branch_cap=1024, packed_cap=2048, temperature=2.0)

    assert model.seen_labels == [None, None, None]
    assert result["inference_temperature"] == 2.0
    assert [item["id"] for item in result["questions"]] == ["truth-id", "choice-id", "score-id"]
    assert result["questions"][0]["winner"] == "true"
    choice = result["questions"][1]
    assert list(choice["probabilities"]) == ["allow_return", "deny_return"]
    assert choice["winner"] == "deny_return"
    assert choice["probabilities"]["allow_return"] == torch.softmax(torch.tensor([0.0, 1.0]) / 2, dim=0)[0].item()
    assert result["questions"][2]["winner"] == "1"
    assert "label" not in choice


def test_predict_stage_uses_current_checkpoint_contract(tmp_path, monkeypatch):
    from gev.application import prediction

    config = load_config("configs/smoke.toml")
    checkpoint_dir = tmp_path / "run" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    request = {"state": "Policy: returns within 30 days.", "questions": {
        "day-12": {"type": "choice", "instructions": "Return on day 12?",
                   "criteria": {"allow": "Allow", "deny": "Deny"}},
    }}
    loaded = []

    def load_checkpoint(path, **kwargs):
        loaded.append((path, kwargs))
        return FakeModel(), {"format": "gev.inference-checkpoint"}

    resolved = SimpleNamespace(
        validate_runtime_available=lambda: None, select_device=lambda: "cpu",
        load_tokenizer=lambda: TinyTokenizer(), load_markers=lambda _tokenizer: _markers(),
        load_checkpoint=load_checkpoint,
        model=SimpleNamespace(family_runtime=SimpleNamespace(encode_record=encode)))
    monkeypatch.setattr(prediction, "config_for_run", lambda *_: config)
    monkeypatch.setattr(prediction, "resolve_experiment_config", lambda _: resolved)
    monkeypatch.setattr(prediction, "configure_runtime", lambda *_: None)
    monkeypatch.setattr(prediction, "read_checkpoint_manifest", lambda _: {"execution": {
        "state_cap": config.training.state_cap, "branch_cap": config.training.branch_cap,
        "packed_cap": config.training.packed_cap}})

    result = prediction.predict_stage(tmp_path / "run", request)
    assert loaded[0][0] == checkpoint_dir
    assert loaded[0][1]["device"] == "cpu"
    assert result["questions"][0]["winner"] == "deny"


def test_predict_cli_reads_stdin_and_emits_json(tmp_path, monkeypatch, capsys):
    from gev.commands import predict

    request = {"state": "Policy: returns within 30 days.", "questions": {
        "day-12": {"type": "choice", "instructions": "Return on day 12?",
                   "criteria": {"allow": "Allow", "deny": "Deny"}},
    }}
    seen = []
    def fake_predict_stage(run, value, **kwargs):
        seen.append((run, value, kwargs))
        return {"inference_temperature": kwargs["temperature"], "questions": [
            {"id": "day-12", "winner": "deny"}]}

    monkeypatch.setattr(predict, "predict_stage", fake_predict_stage)
    monkeypatch.setattr(predict.sys, "stdin", io.StringIO(json.dumps(request)))
    assert main(["predict", str(tmp_path / "run"), "--input", "-"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert seen[0][1] == request
    assert result["inference_temperature"] == 1.0
    assert result["questions"][0]["id"] == "day-12"
    assert result["questions"][0]["winner"] == "deny"


def test_predict_rejects_invalid_temperature_before_loading_checkpoint(tmp_path, monkeypatch):
    from gev.application import prediction

    monkeypatch.setattr(prediction, "config_for_run", lambda *_: (_ for _ in ()).throw(AssertionError("loaded")))
    for value in (float("nan"), 0.0, -1.0):
        try:
            prediction.predict_stage(tmp_path, {}, temperature=value)
        except ValueError as exc:
            assert "temperature" in str(exc)
        else:
            raise AssertionError("invalid temperature accepted")
