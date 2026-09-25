"""Offline fixtures: a tiny word-level tokenizer, tiny random Gemma decoders, and a synthetic suite."""

import hashlib
import json

import pytest

from gev.config import Config, ModelConfig, TrainingConfig

WORDS = ["a", "b", "c", "d", "yes", "no", "is", "it", "the", "case", "pick", "one", "option", "true", "false",
         "x", "y", "z", "0", "1", "2", ":", "-", "none", "other", "of", "above"]
MARKERS = ("<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>")


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(["<pad>", "<eos>", "<bos>", "<unk>", *MARKERS, *WORDS])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
                                   bos_token="<bos>", unk_token="<unk>")


@pytest.fixture
def tokenizer():
    return make_tokenizer()


def gemma3_backbone():
    import torch
    from transformers import Gemma3TextConfig, Gemma3TextModel

    torch.manual_seed(0)
    return Gemma3TextModel(Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=256, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"], query_pre_attn_scalar=8, pad_token_id=0,
        use_cache=False))


def gemma4_text_config():
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=1, head_dim=8, global_head_dim=8, max_position_embeddings=256, sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
        vocab_size_per_layer_input=64, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
        pad_token_id=0, use_cache=False)


def gemma4_backbone():
    import torch
    from transformers import Gemma4TextModel

    torch.manual_seed(0)
    return Gemma4TextModel(gemma4_text_config())


def tiny_config(family="gemma3", backend="torch", **training) -> Config:
    name, revision = ("google/gemma-3-1b-pt", "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29") if family == "gemma3" \
        else ("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")
    defaults = dict(epochs=1, logical_batch=4, microbatch=2, learning_rate=1e-2, state_cap=64, branch_cap=128)
    return Config(name="tiny", backend=backend, device="cpu" if backend == "torch" else "auto", dtype="fp32",
                  model=ModelConfig(name=name, revision=revision, family=family, markers=MARKERS,
                                    lora_rank=4, lora_alpha=8, lora_dropout=0.0, pointer_width=16),
                  training=TrainingConfig(**{**defaults, **training}))


def request(index: int, variant: str = "clean") -> dict:
    label = ["a", "b", "c"][index % 3]
    return {"state": {"case": f"it is {label}"},
            "questions": {"pick": {"type": "choice", "instructions": "pick one option", "src": "letters",
                                   "criteria": {"a": "the a", "b": "the b", "c": "the c"}, "label": label},
                          "truth": {"type": "noul", "instructions": f"is it {label}", "src": "truth",
                                    "label": index % 2 == 0}},
            "_meta": {"id": f"r{index}", "group_id": f"g{index}", "source": ["x", "y"][index % 2],
                      "variant": variant}}


def write_sample(root, train=16, development=6):
    """A sample-layout data root (see gev.data.load_split) with synthetic decision-v7 rows."""
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, count in (("train", train), ("development", development), ("calibration", development)):
        body = "".join(json.dumps(request(i + (0 if split == "train" else 100))) + "\n" for i in range(count)).encode()
        (root / f"{split}.jsonl").write_bytes(body)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256(body).hexdigest(), "records": count,
                                   "questions": 2 * count}
    (root / "manifest.json").write_text(json.dumps({"parent": {"suite": "decision-v7"}, "files": files}))
    return root


@pytest.fixture
def sample_root(tmp_path):
    return write_sample(tmp_path / "data")
