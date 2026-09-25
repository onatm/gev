import pytest
from gev.cli import main
from gev.application.evaluation import config_for_run, evaluation_provenance
from gev.diagnostics.environment import _torch_smoke


def test_validate_config(capsys):
    assert main(["diagnose", "config", "configs/smoke.toml"]) == 0
    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert '"backend": "torch"' in output
    assert '"model_family": "gemma3_text"' in output


def test_model_probe_has_one_qualification_protocol_and_base_only_is_not_qualified(
        monkeypatch, capsys):
    from types import SimpleNamespace
    from gev.commands import diagnose as handler

    monkeypatch.setattr(handler, "load_cli_config",
                        lambda _path: SimpleNamespace(backend=SimpleNamespace(id="mlx")))
    monkeypatch.setattr(handler, "_probe_mlx",
                        lambda *_args, **_kwargs: {"status": "base_structure_passed",
                                                   "inference_qualified": False})
    assert main(["diagnose", "model", "--probe", "--config",
                 "configs/gemma4-e2b-mlx-bf16.toml"]) == 2
    assert '"inference_qualified": false' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["diagnose", "model", "--probe", "--config",
              "configs/gemma4-e2b-mlx-bf16.toml", "--qualification", "v1"])


def test_train_rejects_unsupported_backend_before_data_or_model_loading(tmp_path):
    source = open("configs/smoke.toml", encoding="utf-8").read().replace('id = "torch"', 'id = "mlx"')
    config = tmp_path / "unsupported.toml"
    config.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="does not implement model family"):
        main(["train", str(config), "--data", str(tmp_path / "absent-data"),
              "--out", str(tmp_path / "run")])


def test_study_plan_and_run_dispatch_to_shared_orchestrator(monkeypatch, capsys, tmp_path):
    from gev.commands import study as handler

    calls = []
    monkeypatch.setattr(handler, "plan", lambda *args, **kwargs: calls.append(("plan", args, kwargs)) or {"status": "planned"})
    monkeypatch.setattr(handler, "run", lambda *args, **kwargs: calls.append(("run", args, kwargs)) or {"status": "diagnostic"})

    assert main(["study", "plan", "configs/smoke.toml", "--data", str(tmp_path), "--seeds", "2,3"]) == 0
    assert '"status": "planned"' in capsys.readouterr().out
    assert calls[-1][0] == "plan" and calls[-1][2]["seeds"] == (2, 3)

    assert main(["study", "run", "configs/smoke.toml", "--out", str(tmp_path / "run"),
                 "--data", str(tmp_path), "--seeds", "1", "--max-steps", "5"]) == 0
    assert calls[-1][0] == "run" and calls[-1][2]["max_steps"] == 5


def test_train_requires_resume_or_init_from_but_not_both(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "configs/smoke.toml", "--out", str(tmp_path / "run"),
              "--resume", "snapshot.pt", "--init-from", "base-run"])


def test_train_handler_delegates_resume_and_warm_start_to_distinct_services(monkeypatch, tmp_path):
    from gev.commands import train as handler

    calls = []
    monkeypatch.setattr(handler, "train_stage", lambda *args, **kwargs: calls.append(("resume", kwargs)) or {})
    monkeypatch.setattr(handler, "warm_start_stage", lambda *args, **kwargs: calls.append(("init", kwargs)) or {})
    main(["train", "configs/smoke.toml", "--out", str(tmp_path / "resume"), "--resume", "snapshot.pt"])
    main(["train", "configs/gemma3-1b-night2.toml", "--out", str(tmp_path / "warm"), "--init-from", "base-run"])
    assert calls[0][0] == "resume" and calls[0][1]["resume"] == "snapshot.pt"
    assert calls[1][0] == "init" and calls[1][1]["init_from"] == "base-run"


def test_train_rejects_nonpositive_max_steps_before_stage_loading(tmp_path):
    with pytest.raises(SystemExit, match="positive integer"):
        main(["train", "configs/smoke.toml", "--out", str(tmp_path / "run"),
              "--max-steps", "0"])


def test_calibrate_and_compare_handlers_delegate_to_saved_row_services(monkeypatch, tmp_path):
    import gev.evaluation.calibration as calibration
    import gev.evaluation.compare as compare_module

    calls = []
    monkeypatch.setattr(calibration, "calibrate", lambda *args, **kwargs:
                        calls.append(("calibrate", kwargs)) or {"status": "calibrated"})
    monkeypatch.setattr(compare_module, "compare", lambda *args, **kwargs:
                        calls.append(("compare", kwargs)) or {"status": "compared"})
    main(["calibrate", "--run", "rows", "--out", str(tmp_path / "temperature.json")])
    main(["compare", "--candidate", "candidate", "--reference", "reference",
          "--out", str(tmp_path / "comparison.json")])
    assert calls[0][0] == "calibrate" and calls[0][1].get("protocol") == "kev-screening"
    assert calls[1][0] == "compare" and calls[1][1].get("aggregation") == "micro"


def test_normal_evaluate_and_data_commands_cannot_select_test():
    with pytest.raises(SystemExit):
        main(["evaluate", "run", "--suite", "decision-v7", "--split", "test", "--out", "out"])
    with pytest.raises(SystemExit):
        main(["data", "fetch", "decision-v7", "test"])
    with pytest.raises(SystemExit):
        main(["data", "verify", "decision-v7", "test", "data/test.jsonl"])


def test_data_fetch_night2_uses_named_fetch_service(monkeypatch, tmp_path, capsys):
    from gev.data import continuation

    monkeypatch.setattr(continuation, "fetch_night2", lambda root: {"root": root})
    assert main(["data", "fetch", "night2", "--data-root", str(tmp_path)]) == 0
    assert str(tmp_path) in capsys.readouterr().out


def test_evaluate_uses_persisted_resolved_config_when_config_is_omitted(tmp_path):
    import dataclasses
    import hashlib
    import json
    from gev.configuration.config import load_config

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    expected = load_config("configs/smoke.toml")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "pointer.safetensors").write_bytes(b"pointer")
    def tensor(filename, payload):
        return {"filename": filename, "sha256": hashlib.sha256(payload).hexdigest(), "shapes": {"x": [1]}}
    (checkpoint / "manifest.json").write_text(json.dumps({
        "format": "gev.inference-checkpoint", "version": 1,
        "identity": {"family": "gemma3_text", "backend": "torch",
            "base": {"name": expected.model.name, "revision": expected.model.revision,
                     "type": expected.model.expected_model_type},
            "tokenizer": {"revision": expected.model.revision},
            "markers": {"ids": {"state": 1}, "strings": {"state": "s"}, "bos": False},
            "protocol": dataclasses.asdict(expected.protocol),
            "recipe": {"scientific_recipe": {}, "sha256": hashlib.sha256(b"{}").hexdigest()},
            "model_contract": {}},
        "lineage": {}, "training": {"metrics": {}, "config": {},
                                      "resolved_config": dataclasses.asdict(expected)},
        "execution": {}, "tensors": {"adapter": tensor("adapter_model.safetensors", b"adapter"),
                                      "pointer": tensor("pointer.safetensors", b"pointer")},
        "calibration": {"temperature": 1.0}}), encoding="utf-8")
    assert config_for_run(None, str(tmp_path)) == expected


def test_evaluate_handler_delegates_to_single_seed_stage(monkeypatch, tmp_path, capsys):
    from gev.commands import evaluate as handler

    calls = []
    monkeypatch.setattr(handler, "evaluate_stage", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or {"status": "evaluated"})
    assert main(["evaluate", "run", "--suite", "decision-v7", "--split", "development",
                 "--out", str(tmp_path / "eval")]) == 0
    assert calls[0][0] == ("run",)
    assert calls[0][1]["split"] == "development"
    assert '"status": "evaluated"' in capsys.readouterr().out


def test_locked_command_delegates_to_reserved_locked_service(monkeypatch, tmp_path):
    from gev.commands import locked as handler

    calls = []
    monkeypatch.setattr(handler, "locked_evaluation_stage", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or {"status": "reserved"})
    assert main(["locked", "evaluate", "--run", "run", "--selection", "selection.json",
                 "--out", str(tmp_path / "locked")]) == 0
    assert calls[0][1]["selection"] == "selection.json"


def test_diagnose_model_is_metadata_only_unless_probe_is_requested(monkeypatch, capsys):
    from gev.commands import diagnose as handler

    inspected = []
    monkeypatch.setattr("gev.diagnostics.inspect_model.inspect_model",
                        lambda config, output: inspected.append((config, output)) or {"status": "verified"})
    probes = []
    monkeypatch.setattr("gev.backends.torch.gemma3.check_model",
                        lambda **kwargs: probes.append(kwargs) or {"status": "passed"})
    assert main(["diagnose", "model", "--config", "configs/smoke.toml"]) == 0
    capsys.readouterr()
    assert inspected and not probes
    assert main(["diagnose", "model", "--probe", "--config", "configs/smoke.toml"]) == 0
    capsys.readouterr()
    assert probes == [{"tiny": False, "config_path": "configs/smoke.toml", "device": None}]


def test_data_prepare_night2_uses_shared_prepare_service(monkeypatch, tmp_path, capsys):
    from gev.data import continuation

    monkeypatch.setattr(continuation, "build_continuation",
                        lambda root, out=None: {"root": root, "out": str(out)})
    out = tmp_path / "night2-prepared"
    assert main(["data", "prepare", "night2", "--data-root", str(tmp_path),
                 "--out", str(out)]) == 0
    output = capsys.readouterr().out
    assert str(out) in output


def test_old_duplicate_cli_names_are_not_registered():
    for command in ("doctor", "eval", "experiment", "continue-training", "eval-locked"):
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_device_options_include_torch_cuda():
    from gev.commands.parser import build_parser

    args = build_parser().parse_args([
        "train", "configs/smoke.toml", "--out", "run", "--device", "cuda"])
    assert args.device == "cuda"


def test_cpu_runtime_smoke_is_real():
    result = _torch_smoke("cpu", "fp32")
    assert result["status"] == "passed"


def test_resume_rejects_existing_inference_checkpoint_before_loading_model(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint"
    checkpoint.mkdir(parents=True)
    with pytest.raises(SystemExit, match="fresh --out"):
        main(["train", "configs/smoke.toml", "--resume", str(tmp_path / "snapshot.pt"), "--out", str(tmp_path / "run")])


def test_eval_provenance_keeps_ood_identity_separate_from_incomplete_training():
    provenance = evaluation_provenance(
        suite="transfer-v4", split="development", suite_sha256="transfer-manifest",
        source_sha256="transfer-source", checkpoint_meta={
            "lineage": {"source_sha256": "train-source", "manifest_sha256": "train-manifest"},
            "training": {"metrics": {"complete": False}},
        }, checkpoint_hash="checkpoint", execution_mode="rows",
        trained_execution_mode="rows")
    assert provenance["suite_sha256"] == "transfer-manifest"
    assert provenance["training_manifest_sha256"] == "train-manifest"
    assert provenance["training_complete"] is False
