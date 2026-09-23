import pytest
from gev.cli import _evaluation_provenance, main
from gev.runtime import _torch_smoke


def test_validate_config(capsys):
    assert main(["validate-config", "configs/smoke.toml"]) == 0
    assert '"status": "valid"' in capsys.readouterr().out


def test_cpu_runtime_smoke_is_real():
    result = _torch_smoke("cpu", "fp32")
    assert result["status"] == "passed"


def test_resume_rejects_existing_inference_checkpoint_before_loading_model(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint"
    checkpoint.mkdir(parents=True)
    with pytest.raises(SystemExit, match="fresh --out"):
        main(["train", "--config", "configs/smoke.toml", "--resume", str(tmp_path / "snapshot.pt"), "--out", str(tmp_path / "run")])


def test_eval_provenance_keeps_ood_identity_separate_from_incomplete_training():
    provenance = _evaluation_provenance(
        suite="transfer-v4", split="development", suite_sha256="transfer-manifest",
        source_sha256="transfer-source", checkpoint_meta={
            "source_sha256": "train-source", "manifest_sha256": "train-manifest",
            "training": {"complete": False},
        }, checkpoint_fingerprint="checkpoint", execution_mode="rows",
        trained_execution_mode="rows")
    assert provenance["suite_sha256"] == "transfer-manifest"
    assert provenance["training_manifest_sha256"] == "train-manifest"
    assert provenance["training_complete"] is False
