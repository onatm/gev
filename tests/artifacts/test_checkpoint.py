import json
import hashlib

import pytest

from gev.backends.torch.checkpoint import save_checkpoint, load_checkpoint
from gev.backends.torch.gemma3 import build_tiny_model
from gev.configuration.config import ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig


def _config(revision="0" * 40, name="tiny"):
    return ExperimentConfig("test", ModelConfig(name, revision, "gemma3_text"), TrainingConfig(0, 1, 1e-4, "fp32", 384), RuntimeConfig("cpu", False, "runs"))


def _metadata():
    from dataclasses import asdict
    config = _config()
    from gev.configuration.resolved import resolve_experiment_config
    recipe = resolve_experiment_config(config).recipe
    return {"model_name": "tiny", "model_revision": "0" * 40, "base_model_type": "gemma3_text", "model_family": "gemma3_text", "backend": "torch", "protocol": asdict(config.protocol), "scientific_recipe": recipe, "scientific_recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(), "resolved_config": {"resolved_config": asdict(config), "study_id": config.experiment_id}, "marker_ids": {"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, "marker_strings": {"state": "<unused0>", "question": "<unused1>", "option_start": "<unused2>", "option_end": "<unused3>", "decide": "<unused4>"}, "head_width": 256, "lora": {"r": 16, "alpha": 32, "dropout": .05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}, "dtype": "fp32", "state_cap": 384, "branch_cap": 1024, "packed_cap": 2048, "representation_version": 1, "training": {"complete": False}}


def test_checkpoint_roundtrip_and_strict_head(tmp_path):
    model = build_tiny_model(layers=6, hidden_size=32)
    path = save_checkpoint(model, tmp_path / "ck", _metadata())
    fresh = build_tiny_model(layers=6, hidden_size=32)
    loaded, meta = load_checkpoint(path, config=_config(), backbone_loader=lambda *_: fresh.decoder.base_model.model)
    assert meta["version"] == 1
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


def test_checkpoint_manifest_describes_tensor_hashes_and_shapes(tmp_path):
    from gev.artifacts.checkpoint_identity import checkpoint_fingerprint, read_checkpoint_manifest
    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    metadata = read_checkpoint_manifest(path)
    assert metadata["format"] == "gev.inference-checkpoint"
    assert metadata["version"] == 1
    assert metadata["tensors"]["adapter"]["sha256"]
    assert metadata["tensors"]["pointer"]["sha256"]
    assert metadata["tensors"]["adapter"]["shapes"]
    assert metadata["tensors"]["pointer"]["shapes"]
    assert not (path / "metadata.json").exists()
    fingerprint = checkpoint_fingerprint(path)
    metadata["calibration"].update(temperature=1.75, fit={"suite": "decision-v7"})
    (path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert checkpoint_fingerprint(path) == fingerprint


@pytest.mark.parametrize("version", [2, True, 1.0])
def test_checkpoint_manifest_rejects_unsupported_version(tmp_path, version):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME, read_checkpoint_manifest

    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsupported inference checkpoint manifest"):
        read_checkpoint_manifest(path)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("base", "name", "another/base"),
    ("base", "type", "another_architecture"),
    ("tokenizer", "sha256", "a" * 64),
])
def test_checkpoint_fingerprint_tracks_pinned_model_and_tokenizer_identity(
        tmp_path, section, field, value):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME, checkpoint_fingerprint

    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    original = checkpoint_fingerprint(path)
    manifest["identity"][section][field] = value
    manifest_path.write_text(json.dumps(manifest))

    assert checkpoint_fingerprint(path) != original


def test_checkpoint_rejects_tokenizer_digest_mismatch_before_base_load(tmp_path):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME
    from gev.domain.tokenization import MarkerMap

    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["tokenizer"]["sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    expected_markers = MarkerMap(
        manifest["identity"]["markers"]["ids"],
        manifest["identity"]["markers"]["strings"],
        _config().model.revision,
        tokenizer_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        load_checkpoint(path, config=_config(), expected_marker_map=expected_markers,
                        backbone_loader=lambda *_: pytest.fail("base loaded"))


def test_checkpoint_manifest_rejects_malformed_tokenizer_digest(tmp_path):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME, read_checkpoint_manifest

    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["tokenizer"]["sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="tokenizer digest"):
        read_checkpoint_manifest(path)


@pytest.mark.parametrize("section,value,message", [
    ("family", "other-family", "family"),
    ("backend", "other-backend", "backend"),
])
def test_checkpoint_identity_mismatches_fail_before_base_load(tmp_path, section, value, message):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME
    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"][section] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, config=_config(), backbone_loader=lambda *_: pytest.fail("base loaded"))


def test_checkpoint_protocol_mismatch_fails_before_base_load(tmp_path):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME
    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["protocol"]["version"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol"):
        load_checkpoint(path, config=_config(), backbone_loader=lambda *_: pytest.fail("base loaded"))


def test_checkpoint_tensor_shape_descriptor_is_checked_before_base_load(tmp_path):
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME
    path = save_checkpoint(build_tiny_model(layers=6, hidden_size=32), tmp_path / "ck", _metadata())
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["tensors"]["pointer"]["shapes"]["query.weight"] = [1, 1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pointer tensor names/shapes"):
        load_checkpoint(path, config=_config(), backbone_loader=lambda *_: pytest.fail("base loaded"))
