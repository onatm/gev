import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gev.configuration.config import load_config
from gev.diagnostics.inspect_model import inspect_model


class FakeTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id = 100, 101, 102, 103

    def get_vocab(self):
        return self.vocabulary

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self.vocabulary[text]] if text in self.vocabulary else [103, 104])


def fake_config():
    return SimpleNamespace(
        model_type="gemma3_text", hidden_size=1152, num_hidden_layers=26,
        num_attention_heads=4, num_key_value_heads=1, head_dim=256,
        intermediate_size=6912, vocab_size=262144, max_position_embeddings=32768,
        sliding_window=512, layer_types=["full_attention" if i in {5, 11, 17, 23} else "sliding_attention" for i in range(26)],
        rope_parameters={"sliding_attention": {"rope_theta": 10000.0}, "full_attention": {"rope_theta": 1000000.0}},
    )


def install_fake_transformers(monkeypatch, tokenizer, config=None, error=None):
    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if error:
                raise error
            return config or fake_config()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if error:
                raise error
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=AutoConfig, AutoTokenizer=AutoTokenizer))


def test_discovers_reserved_rows_and_writes_verified_artifact(monkeypatch, tmp_path):
    vocab = {f"<unused{i}>": 200 + i for i in range(5)}
    install_fake_transformers(monkeypatch, FakeTokenizer(vocab))
    result = inspect_model(load_config("configs/smoke.toml"), str(tmp_path))
    assert result["status"] == "verified"
    artifact = json.loads((tmp_path / "reference" / "model-marker-map.json").read_text())
    assert set(artifact["markers"]) == {"state", "question", "option_start", "option_end", "decide"}
    assert artifact["bos"] is False


@pytest.mark.parametrize("vocab", [
    {"<unused0>": 200, "<unused1>": 201, "<unused2>": 202, "<unused3>": 203},
    {"<unused0>": 200, "<unused1>": 201, "<unused2>": 202, "<unused3>": 203, "<unused4>": 100},
    {"<unused0>": 200, "<unused1>": 201, "<unused2>": 202, "<unused3>": 203, "<unused4>": 204},
])
def test_bad_marker_discovery_never_writes(monkeypatch, tmp_path, vocab):
    if len(vocab) == 5 and vocab["<unused4>"] == 204:
        class EmptyTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[])
        tokenizer = EmptyTokenizer(vocab)
    else:
        tokenizer = FakeTokenizer(vocab)
    install_fake_transformers(monkeypatch, tokenizer)
    result = inspect_model(load_config("configs/smoke.toml"), str(tmp_path))
    assert result["status"] == "failed"
    assert not (tmp_path / "reference" / "model-marker-map.json").exists()


def test_auth_failure_is_honest_and_has_no_artifact(monkeypatch, tmp_path):
    install_fake_transformers(monkeypatch, FakeTokenizer({}), error=RuntimeError("401 gated repository"))
    result = inspect_model(load_config("configs/smoke.toml"), str(tmp_path))
    assert result["status"] == "blocked"
    assert "token" not in result["error"].lower()
    assert not (tmp_path / "reference" / "model-marker-map.json").exists()


def test_inspection_rejects_unresolved_backend_before_transformers_access(tmp_path):
    config = load_config("configs/smoke.toml")
    config = replace(config, backend=replace(config.backend, id="mlx"))
    result = inspect_model(config, str(tmp_path))
    assert result["status"] == "failed"
    assert "unknown model backend" in result["error"]
    assert not (tmp_path / "reference" / "model-marker-map.json").exists()
