import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import Gemma4TextModel

from gev.application.training import _checkpoint_metadata
from gev.backends.torch.checkpoint import load_checkpoint
from gev.backends.torch.gemma4 import _tiny_config
from gev.backends.torch.checkpoint import save_checkpoint
from gev.configuration.config import BackendConfig, load_config
from gev.configuration.resolved import resolve_experiment_config
from gev.domain.tokenization import MarkerMap
from gev.evaluation.development import selected_ids_sha256
from gev.models.policy import GEMMA4_POLICY
from gev.models.qualification import (qualification_code_sha256,
                                      qualification_policy_sha256,
                                      seal_qualification_receipt)


def _config(compute_dtype="fp32", backend="torch"):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    return dataclasses.replace(
        config,
        backend=BackendConfig(backend),
        training=dataclasses.replace(config.training, dtype=compute_dtype),
        runtime=dataclasses.replace(
            config.runtime, device="cpu" if backend == "torch" else "gpu",
            gradient_checkpointing=False),
    )


def _metadata(config, resolved, model):
    markers = MarkerMap(
        config.model.marker_ids,
        dict(zip(config.model.marker_ids,
                 (f"<unused{i}>" for i in range(5)), strict=True)),
        config.model.revision,
    )
    checks = {name: True for name in GEMMA4_POLICY.required_qualification_checks(
        config.backend.id, config.training.dtype)}
    report = {"coverage": {"requested_records": 1, "evaluated_records": 1,
                            "requested_questions": 1, "evaluated_questions": 1,
                            "rejected_records": 0, "truncated_records": 0},
              "clean": {"nll": 0.7}}
    report_bytes = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    receipt = seal_qualification_receipt({
        "format": "gev.trained-checkpoint-qualification", "version": 1,
        "status": "passed", "model_output_id": "gev-gemma4-e2b",
        "family": config.model.family, "backend": config.backend.id,
        "base": {"name": config.model.name, "revision": config.model.revision,
                 "source_weights_dtype": "bf16", "compute_dtype": config.training.dtype},
        "policy_sha256": qualification_policy_sha256(
            config.model.family, config.backend.id, config.training.dtype),
        "code_sha256": qualification_code_sha256(
            config.backend.id, Path(__file__).resolve().parents[2]),
        "checks": checks, "training_complete": False, "training_steps": 2,
        "development": {
            "suite": "decision-v7", "split": "development", "manifest_sha256": "c" * 64,
            "selected_ids": ["dev/one"],
            "selected_ids_sha256": selected_ids_sha256([{"_meta": {"id": "dev/one"}}]),
            "selected_record_count": 1,
            "coverage": report["coverage"],
            "mechanism_checks": {"passed": True, "failures": 0},
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "scores": {"clean": {"nll": 0.7}},
        },
        "trainable_parameters_sha256": resolved.trainable_fingerprint(model),
        "checkpoint_ready": False, "checkpoint_tensor_sha256": None,
    })
    metadata = _checkpoint_metadata(
        config=config, resolved=resolved, markers=markers,
        source_hash="a" * 64, manifest_hash="b" * 64,
        metrics={"device": "cpu", "weights_dtype": config.training.dtype,
                 "source_weights_dtype": "bf16"},
        output="test-run", extra={"development_report": report})
    metadata["qualification"] = receipt
    return metadata


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_torch_gemma4_checkpoint_roundtrip_preserves_precision_and_logits(
        tmp_path, compute_dtype):
    config = _config(compute_dtype)
    resolved = resolve_experiment_config(config)
    torch.manual_seed(17)
    base = Gemma4TextModel(_tiny_config())
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    model = resolved.create_model(backbone=base).eval()
    metadata = _metadata(config, resolved, model)
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint", metadata)
    expected = model.forward_one({
        "ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1],
        "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, -1, -1, 0, 1, -2],
        "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2,
    })[0]
    callback_dtypes = []

    def load_fixture(_name, _revision, *, compute_dtype):
        callback_dtypes.append(compute_dtype)
        fresh = Gemma4TextModel(_tiny_config())
        fresh.load_state_dict(base_state)
        return fresh

    loaded, manifest = resolved.load_checkpoint(
        checkpoint, device="cpu", backbone_loader=load_fixture)
    actual = loaded.forward_one({
        "ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1],
        "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, -1, -1, 0, 1, -2],
        "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2,
    })[0]

    assert callback_dtypes == [compute_dtype]
    assert manifest["execution"]["compute_dtype"] == compute_dtype
    assert loaded.decoder.embed_tokens.weight.dtype == (
        torch.bfloat16 if compute_dtype == "bf16" else torch.float32)
    assert torch.equal(expected, actual)
    assert all(parameter.dtype == torch.float32 for parameter in loaded.head.parameters())


def test_torch_gemma4_cross_backend_checkpoint_is_rejected_before_base_load(tmp_path):
    config = _config()
    resolved = resolve_experiment_config(config)
    model = resolved.create_model(backbone=Gemma4TextModel(_tiny_config()))
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint", _metadata(config, resolved, model))
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["backend"] = "mlx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint backend mismatch"):
        load_checkpoint(checkpoint, config=config,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded"))


def test_torch_gemma4_qualification_tampering_is_rejected_before_base_load(tmp_path):
    config = _config("bf16")
    resolved = resolve_experiment_config(config)
    model = resolved.create_model(backbone=Gemma4TextModel(_tiny_config()))
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint",
                                 _metadata(config, resolved, model))
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualification"]["checks"]["source_inventory_exact"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="qualification receipt hash mismatch"):
        load_checkpoint(checkpoint, config=config,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded"))


def test_forged_mechanism_failure_receipt_is_rejected_before_base_load(tmp_path):
    config = _config("bf16")
    resolved = resolve_experiment_config(config)
    model = resolved.create_model(backbone=Gemma4TextModel(_tiny_config()))
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint",
                                 _metadata(config, resolved, model))
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = {key: value for key, value in manifest["qualification"].items()
               if key != "receipt_sha256"}
    receipt["development"]["mechanism_checks"] = {"passed": False, "failures": 1}
    manifest["qualification"] = seal_qualification_receipt(receipt)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="development mechanism summary failed"):
        load_checkpoint(checkpoint, config=config,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded"))


@pytest.mark.parametrize("sidecar", ["qualification.json", "development_report.json"])
def test_torch_gemma4_receipt_sidecar_tampering_fails_before_base_load(
        tmp_path, sidecar):
    config = _config("fp32")
    resolved = resolve_experiment_config(config)
    model = resolved.create_model(backbone=Gemma4TextModel(_tiny_config()))
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint",
                                 _metadata(config, resolved, model))
    path = checkpoint / sidecar
    if sidecar == "qualification.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "failed"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")

    expected_error = ("sidecar qualification differs" if sidecar == "qualification.json"
                      else "development report digest mismatch")
    with pytest.raises(ValueError, match=expected_error):
        load_checkpoint(checkpoint, config=config,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded"))


@pytest.mark.parametrize(("field", "value", "message"), [
    ("compute_dtype", "bf16", "compute dtype mismatch"),
    ("source_weights_dtype", "fp32", "source weights dtype mismatch"),
])
def test_gemma4_checkpoint_precision_identity_is_checked_before_base_load(
        tmp_path, field, value, message):
    config = _config("fp32")
    resolved = resolve_experiment_config(config)
    model = resolved.create_model(backbone=Gemma4TextModel(_tiny_config()))
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint", _metadata(config, resolved, model))
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_checkpoint(checkpoint, config=config,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded"))
