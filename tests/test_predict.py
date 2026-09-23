import io
import json
from types import SimpleNamespace

import torch

from gev.cli import main
from gev.config import load_config
from gev.evaluation.predictors import predict_request
from gev.tokenization import MarkerMap


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

    result = predict_request(request, model, TinyTokenizer(), _markers(), state_cap=384,
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


def test_predict_cli_reads_stdin_loads_checkpoint_and_emits_json(tmp_path, monkeypatch, capsys):
    from gev import checkpoint, cli

    config = load_config("configs/smoke.toml")
    checkpoint_dir = tmp_path / "run" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "metadata.json").write_text(json.dumps({
        "state_cap": config.training.state_cap,
        "branch_cap": config.training.branch_cap,
        "packed_cap": config.training.packed_cap,
    }))
    request = {"state": "Policy: returns within 30 days.", "questions": {
        "day-12": {"type": "choice", "instructions": "Return on day 12?",
                   "criteria": {"allow": "Allow", "deny": "Deny"}},
    }}
    loaded = []

    def fake_load_checkpoint(path, *, config, device, tokenizer, expected_marker_map):
        loaded.append((path, config, device, expected_marker_map))
        return FakeModel(), {}

    monkeypatch.setattr(cli, "configure_runtime", lambda *_: None)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(cli.MarkerMap, "load", lambda *_args, **_kwargs: _markers())
    monkeypatch.setattr(checkpoint, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        staticmethod(lambda *_args, **_kwargs: TinyTokenizer()))

    assert main(["predict", "--run", str(tmp_path / "run"), "--config", "configs/smoke.toml",
                 "--input", "-"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert loaded[0][0] == checkpoint_dir
    assert loaded[0][2] == "cpu"
    assert result["inference_temperature"] == 1.0
    assert result["questions"][0]["id"] == "day-12"
    assert result["questions"][0]["winner"] == "deny"
