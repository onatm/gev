import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gev.application import evaluation as stages
from gev.configuration.config import load_config


def test_evaluate_stage_uses_raw_t1_and_verified_non_test_split(monkeypatch, tmp_path):
    import hashlib
    from dataclasses import asdict
    from gev.artifacts.checkpoint_identity import MANIFEST_FILENAME
    checkpoint = tmp_path / "run" / "checkpoint"
    checkpoint.mkdir(parents=True)
    for filename in ("adapter_model.safetensors", "pointer.safetensors"):
        (checkpoint / filename).write_bytes(b"weights")
    config = load_config("configs/smoke.toml")
    digest = hashlib.sha256(b"weights").hexdigest()
    metadata = {"format": "gev.inference-checkpoint", "version": 1,
        "identity": {"family": "gemma3_text", "backend": "torch",
            "base": {"name": config.model.name, "revision": config.model.revision,
                     "type": config.model.expected_model_type},
            "tokenizer": {"revision": config.model.revision},
            "markers": {"ids": {"state": 1}, "strings": {"state": "s"}, "bos": False},
            "protocol": asdict(config.protocol),
            "recipe": {"scientific_recipe": {}, "sha256": hashlib.sha256(b"{}").hexdigest()},
            "model_contract": {"head_width": 256, "lora": {}, "representation_version": 1}},
        "lineage": {}, "training": {"metrics": {}, "config": {},
                                      "resolved_config": asdict(config)},
        "execution": {"state_cap": 384, "branch_cap": 1024, "packed_cap": 2048,
                      "execution_mode": "rows"},
        "tensors": {name: {"filename": filename, "sha256": digest, "shapes": {"x": [1]}}
                    for name, filename in (("adapter", "adapter_model.safetensors"),
                                           ("pointer", "pointer.safetensors"))},
        "calibration": {"temperature": 1.0}}
    (checkpoint / MANIFEST_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "evaluation"
    output.mkdir()
    rows = [{"_meta": {"id": "dev/1"}}]
    loader_calls = []
    monkeypatch.setattr(stages, "load_verified_split", lambda *args, **kwargs:
                        loader_calls.append((args, kwargs)) or (rows, {}, "manifest"))
    monkeypatch.setattr(stages, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(stages, "file_digest", lambda _: "source-hash")
    monkeypatch.setattr(stages, "split_path", lambda *args: tmp_path / "development.jsonl")
    monkeypatch.setattr("gev.backends.torch.TorchBackend.checkpoint_fingerprint",
                        lambda self, _: "model-fingerprint")
    monkeypatch.setattr(stages, "checkpoint_fingerprint", lambda _: "model-fingerprint")
    monkeypatch.setattr("gev.backends.torch.TorchBackend.load_checkpoint",
                        lambda self, *args, **kwargs:
                        (SimpleNamespace(head=SimpleNamespace(temperature=1.0)), metadata))
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_tokenizer",
                        lambda self, *args: object())
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_markers",
                        lambda self, *args: object())
    predictors = []
    monkeypatch.setattr("gev.backends.torch.TorchBackend.create_predictor",
                        lambda self, model, tokenizer, markers, **kwargs:
                        predictors.append(kwargs) or object())
    evaluations = []
    monkeypatch.setattr(stages, "evaluate_records", lambda *args: evaluations.append(args) or ({"clean": {}}, None))

    report = stages.evaluate_stage(
        str(tmp_path / "run"), suite="decision-v7", split="development",
        data_root=tmp_path / "data", output=output,
        config_path="configs/smoke.toml")

    assert loader_calls[0][0][1:] == ("decision-v7", "development")
    assert not loader_calls[0][1].get("allow_test", False)
    assert predictors[0]["temperature"] == 1.0
    assert predictors[0]["execution_mode"] == "rows"
    assert evaluations[0][3] == 1.0
    assert report["provenance"]["source_sha256"] == "source-hash"
    assert json.loads((output / "report.json").read_text())["provenance"]["suite"] == "decision-v7"


@pytest.mark.parametrize("local_exists", [False, True])
def test_locked_service_reserves_before_test_verify_or_fetch(monkeypatch, tmp_path, local_exists):
    from gev.evaluation import locked as locked_module

    order = []
    loader_calls = []
    suite_dir = tmp_path / "decision-v7"
    suite_dir.mkdir()
    test_path = suite_dir / "test.jsonl"
    if local_exists:
        test_path.write_bytes(b"fixture test bytes are never parsed by this test")
    monkeypatch.setattr(stages, "configure_runtime", lambda *args: None)
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_tokenizer",
                        lambda self, *args: object())
    monkeypatch.setattr("gev.models.families.Gemma3TextRuntime.load_markers",
                        lambda self, *args: object())
    monkeypatch.setattr("gev.backends.torch.TorchBackend.load_checkpoint",
                        lambda self, *args, **kwargs: (object(), {}))

    monkeypatch.setattr("gev.backends.torch.TorchBackend.create_predictor",
                        lambda *args, **kwargs: object())
    def load_verified(*args, **kwargs):
        loader_calls.append((args, kwargs))
        order.append("test-verify")
        return [{"id": "test/1"}], {}, "test-manifest"
    monkeypatch.setattr(stages, "load_verified_split", load_verified)
    from gev.data import suites
    monkeypatch.setattr(suites, "load_manifest", lambda suite: {"suite": suite})

    def fetch_locked_file(suite, destination, manifest):
        order.append("test-fetch")
        assert suite == "decision-v7"
        assert manifest == {"suite": suite}
        destination.write_bytes(b"fetched fixture bytes")
        return {"sha256": "fixture"}
    monkeypatch.setattr(suites, "_fetch_locked_test_file", fetch_locked_file)

    def locked_runner(**kwargs):
        order.append("reserved")
        rows, _predictor = kwargs["load_test"]("decision-v7")
        return {"status": "passed", "records": len(rows)}
    monkeypatch.setattr(locked_module, "run_locked", locked_runner)

    result = stages.locked_evaluation_stage(
        "run", selection="selection.json", suites=("decision-v7",),
        data_root=tmp_path, output=tmp_path / "locked", ledger=tmp_path / "ledger",
        config_path="configs/smoke.toml")
    assert result["status"] == "passed"
    assert order == (["reserved", "test-verify"] if local_exists else
                     ["reserved", "test-fetch", "test-verify"])
    assert loader_calls[0][0][2] == "test"
    assert loader_calls[0][1] == {"allow_test": True, "_locked_test": True}
