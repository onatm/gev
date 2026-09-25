import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gev.configuration.config import load_config
from gev.diagnostics.inspect_model import inspect_model
from gev.diagnostics.inspect_model import _gemma4_architecture
from gev.models.families import Gemma4E2BTextRuntime
from gev.domain.tokenization import MarkerMap


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
    assert "does not implement model family" in result["error"]
    assert not (tmp_path / "reference" / "model-marker-map.json").exists()


def test_gemma4_architecture_requires_nested_text_config_facts():
    expected_layers = ["full_attention" if index in {4, 9, 14, 19, 24, 29, 34}
                       else "sliding_attention" for index in range(35)]
    text = SimpleNamespace(
        model_type="gemma4_text", hidden_size=1536, num_hidden_layers=35,
        num_attention_heads=8, num_key_value_heads=1, head_dim=256,
        intermediate_size=6144, global_head_dim=512, sliding_window_pattern=5,
        vocab_size=262144, max_position_embeddings=131072, sliding_window=512,
        hidden_size_per_layer_input=256, vocab_size_per_layer_input=262144,
        num_kv_shared_layers=20, layer_types=expected_layers,
        per_layer_config={0: SimpleNamespace(head_dim=256, num_key_value_heads=1),
                          4: SimpleNamespace(head_dim=512, num_key_value_heads=1)})
    config = SimpleNamespace(model_type="gemma4", text_config=text)
    actual, mismatches = _gemma4_architecture(config)
    assert actual["global_layer_indices"] == [4, 9, 14, 19, 24, 29, 34]
    assert mismatches == {}
    text.num_kv_shared_layers = 19
    assert "num_kv_shared_layers" in _gemma4_architecture(config)[1]


def test_gemma4_registers_existing_unused_rows_without_resizing_embeddings(monkeypatch):
    class AddedToken:
        def __init__(self, content, normalized):
            self.content, self.normalized = content, normalized

    class Tokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__({f"<unused{i}>": 6 + i for i in range(5)})
            self.initial_size = 262144
            self.added = None

        def __len__(self):
            return self.initial_size

        def add_tokens(self, values):
            self.added = values
            return len(values)

        def __call__(self, text, add_special_tokens=False):
            token = text if text in self.vocabulary else text.content
            return SimpleNamespace(input_ids=[self.vocabulary[token]])

    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AddedToken=AddedToken, AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    runtime = Gemma4E2BTextRuntime()
    loaded = runtime.load_tokenizer("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")
    assert loaded is tokenizer
    assert len(loaded) == 262144
    assert [item.content for item in loaded.added] == [f"<unused{i}>" for i in range(5)]
    assert all(item.normalized is False for item in loaded.added)
    artifact = {"tokenizer_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
                "bos": False, "markers": {
                    role: {"token": f"<unused{i}>", "id": 6 + i}
                    for i, role in enumerate(("state", "question", "option_start", "option_end", "decide"))}}
    marker_map = MarkerMap.from_dict(artifact, loaded, roles=tuple(artifact["markers"]))
    assert list(marker_map.ids.values()) == [6, 7, 8, 9, 10]
