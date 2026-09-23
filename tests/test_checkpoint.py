import json

import pytest

from gev.checkpoint import save_checkpoint, load_checkpoint
from gev.models.gemma import build_tiny_model
from gev.config import ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig


def _config(revision="0" * 40, name="tiny"):
    return ExperimentConfig("test", ModelConfig(name, revision, "gemma3_text"), TrainingConfig(0, 1, 1e-4, "fp32", 384), RuntimeConfig("cpu", False, "runs"))


def _metadata():
    return {"model_name": "tiny", "model_revision": "0" * 40, "base_model_type": "gemma3_text", "marker_ids": {"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, "marker_strings": {"state": "<unused0>", "question": "<unused1>", "option_start": "<unused2>", "option_end": "<unused3>", "decide": "<unused4>"}, "head_width": 256, "lora": {"r": 16, "alpha": 32, "dropout": .05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}, "dtype": "fp32", "state_cap": 384, "branch_cap": 1024, "packed_cap": 2048, "representation_version": 1}


def test_checkpoint_roundtrip_and_strict_head(tmp_path):
    model = build_tiny_model(layers=6, hidden_size=32)
    path = save_checkpoint(model, tmp_path / "ck", _metadata())
    fresh = build_tiny_model(layers=6, hidden_size=32)
    loaded, meta = load_checkpoint(path, config=_config(), backbone_loader=lambda *_: fresh.decoder.base_model.model)
    assert meta["format_version"] == 1
    assert set(loaded.head.state_dict()) == set(model.head.state_dict())


def test_checkpoint_rejects_wrong_revision_before_loading_base(tmp_path):
    model = build_tiny_model(layers=6, hidden_size=32)
    path = save_checkpoint(model, tmp_path / "ck", _metadata())
    with pytest.raises(ValueError, match="revision"):
        load_checkpoint(path, config=_config("1" * 40), backbone_loader=lambda *_: pytest.fail("base loaded"))


def test_checkpoint_rejects_corrupt_adapter_before_loading_base(tmp_path):
    model = build_tiny_model(layers=6, hidden_size=32)
    path = save_checkpoint(model, tmp_path / "ck", _metadata())
    adapter = path / "adapter_model.safetensors"
    adapter.write_bytes(adapter.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(path, config=_config(), backbone_loader=lambda *_: pytest.fail("base loaded"))


def test_checkpoint_metadata_is_json_and_records_hashes(tmp_path):
    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["adapter_sha256"] and metadata["pointer_sha256"]
    assert metadata["adapter_tensors"] and metadata["pointer_tensors"]
