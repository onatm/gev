import dataclasses
import copy
import hashlib
from pathlib import Path
import json

import pytest

from gev.configuration.config import BackendConfig, load_config
from gev.evaluation.development import selected_ids_sha256
from gev.models.qualification import (
    bind_checkpoint_tensors, qualification_code_sha256,
    qualification_policy_sha256, seal_qualification_receipt,
    validate_qualification_receipt, validate_qualification_sidecars)
from gev.models.policy import GEMMA4_POLICY


def _config(backend="torch", dtype="bf16"):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    return dataclasses.replace(
        config,
        backend=BackendConfig(backend),
        training=dataclasses.replace(config.training, dtype=dtype),
        runtime=dataclasses.replace(config.runtime,
                                    device="cpu" if backend == "torch" else "gpu"),
    )


def _receipt(config, backend="torch"):
    ids = ["dev/a", "dev/b"]
    checks = {name: True for name in
              GEMMA4_POLICY.required_qualification_checks(backend, config.training.dtype)}
    return seal_qualification_receipt({
        "format": "gev.trained-checkpoint-qualification", "version": 1,
        "status": "passed", "model_output_id": "gev-gemma4-e2b",
        "family": config.model.family, "backend": backend,
        "base": {"name": config.model.name, "revision": config.model.revision,
                 "source_weights_dtype": "bf16", "compute_dtype": config.training.dtype},
        "policy_sha256": qualification_policy_sha256(
            config.model.family, backend, config.training.dtype),
        "code_sha256": qualification_code_sha256(
            backend, Path(__file__).resolve().parents[2]),
        "checks": checks, "training_complete": False, "training_steps": 5,
        "development": {
            "suite": "decision-v7", "split": "development", "manifest_sha256": "a" * 64,
            "report_sha256": "f" * 64,
            "selected_ids": ids, "selected_ids_sha256": selected_ids_sha256(
                [{"_meta": {"id": identifier}} for identifier in ids]),
            "selected_record_count": len(ids),
            "coverage": {"requested_records": 2, "evaluated_records": 2,
                         "requested_questions": 4, "evaluated_questions": 4,
                         "rejected_records": 0, "truncated_records": 0},
            "mechanism_checks": {"passed": True, "failures": 0},
            "scores": {"clean": {"acc": 0.5, "nll": 0.7}},
        },
        "trainable_parameters_sha256": "b" * 64,
        "checkpoint_ready": False, "checkpoint_tensor_sha256": None,
    })


def test_training_receipt_seals_source_compute_policy_and_checkpoint_tensors():
    config = _config()
    receipt = _receipt(config)
    validate_qualification_receipt(
        receipt, config=config, backend="torch",
        expected_code_sha256=qualification_code_sha256(
            "torch", Path(__file__).resolve().parents[2]),
        require_checkpoint_ready=False)

    final = bind_checkpoint_tensors(receipt, {"adapter": "c" * 64, "pointer": "d" * 64})
    validate_qualification_receipt(final, config=config, backend="torch",
                                   expected_checkpoint_hashes={
                                       "adapter": "c" * 64, "pointer": "d" * 64})
    with pytest.raises(ValueError, match="tensor hashes mismatch"):
        validate_qualification_receipt(
            final, config=config, backend="torch",
            expected_checkpoint_hashes={"adapter": "0" * 64, "pointer": "d" * 64})


def test_training_receipt_fails_closed_on_backend_precision_and_check_tampering():
    config = _config()
    receipt = _receipt(config)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_qualification_receipt(receipt, config=config, backend="mlx",
                                       require_checkpoint_ready=False)

    changed = {**receipt, "receipt_sha256": "0" * 64}
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        validate_qualification_receipt(changed, config=config, backend="torch",
                                       require_checkpoint_ready=False)

    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    payload["checks"] = dict(payload["checks"], fp32_optimizer_state=False)
    failed_check = seal_qualification_receipt(payload)
    with pytest.raises(ValueError, match="checks did not pass"):
        validate_qualification_receipt(failed_check, config=config, backend="torch",
                                       require_checkpoint_ready=False)

    mechanism_failure = copy.deepcopy(receipt)
    mechanism_failure.pop("receipt_sha256")
    mechanism_failure["development"]["mechanism_checks"] = {
        "passed": False, "failures": 1}
    with pytest.raises(ValueError, match="development mechanism summary failed"):
        validate_qualification_receipt(
            seal_qualification_receipt(mechanism_failure), config=config,
            backend="torch", require_checkpoint_ready=False)

    invalid_report_digest = copy.deepcopy(receipt)
    invalid_report_digest.pop("receipt_sha256")
    invalid_report_digest["development"]["report_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="qualification digest is invalid"):
        validate_qualification_receipt(
            seal_qualification_receipt(invalid_report_digest), config=config,
            backend="torch", require_checkpoint_ready=False)


def test_checkpoint_sidecars_bind_manifest_receipt_and_report_bytes(tmp_path):
    config = _config()
    receipt = bind_checkpoint_tensors(
        _receipt(config), {"adapter": "c" * 64, "pointer": "d" * 64})
    report = {"coverage": {"evaluated_records": 2}, "clean": {"nll": 0.7}}
    report_bytes = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    receipt["development"]["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    # Re-seal because the report digest is itself part of the content-addressed receipt.
    receipt = seal_qualification_receipt(receipt)
    (tmp_path / "qualification.json").write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "development_report.json").write_bytes(report_bytes)

    validate_qualification_sidecars(tmp_path, receipt)
    (tmp_path / "development_report.json").write_bytes(report_bytes + b" ")
    with pytest.raises(ValueError, match="development report digest mismatch"):
        validate_qualification_sidecars(tmp_path, receipt)
