import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gev.application import training as stages
from gev.configuration.config import load_config
from gev.models.registry import resolve_model
from gev.domain.tokenization import MarkerMap


def test_train_stage_uses_verified_train_data_and_shared_checkpoint_metadata(monkeypatch, tmp_path):
    config = load_config("configs/smoke.toml")
    marker_path = tmp_path / "markers.json"
    marker_path.write_text(json.dumps({"tokenizer_sha256": "token-hash"}), encoding="utf-8")
    config = replace(config, model=replace(config.model, marker_artifact=str(marker_path)))
    resolved_model = resolve_model("gemma3_text", "torch")
    model = object()
    train_calls = []
    metrics = {"device": "cpu", "complete": True}
    resolved = SimpleNamespace(
        model=resolved_model,
        load_tokenizer=lambda: object(),
        load_markers=lambda _tokenizer: markers,
        encode_record=lambda tokenizer, record, marker_map, **caps: {"encoded": True},
        create_model=lambda **_: model,
        validate_runtime_available=lambda: None,
        seed_rng=lambda: None,
        save_checkpoint=lambda *args: saved.append(args),
        train=lambda model_arg, schedule_arg, output_arg, **kwargs:
        train_calls.append(((model_arg, schedule_arg, output_arg), kwargs)) or metrics,
        provenance=lambda **kwargs: {"study_id": config.experiment_id, **kwargs},
    )
    markers = MarkerMap({"state": 1, "question": 2, "option_start": 3,
                         "option_end": 4, "decide": 5},
                        {"state": "s", "question": "q", "option_start": "os",
                         "option_end": "oe", "decide": "d"}, config.model.revision)
    rows = [{"_meta": {"id": "train/1"}}]
    loader_calls = []
    monkeypatch.setattr(stages, "load_config", lambda _: config)
    monkeypatch.setattr(stages, "resolve_experiment_config", lambda _: resolved)
    monkeypatch.setattr(stages, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(stages, "load_verified_split",
                        lambda *args, **kwargs: loader_calls.append((args, kwargs)) or
                        (rows, {"files": {}}, "manifest-hash"))
    monkeypatch.setattr(stages, "validate_training_rows", lambda *args: None)
    monkeypatch.setattr(stages, "split_path", lambda *args: tmp_path / "train.jsonl")
    monkeypatch.setattr(stages, "file_digest", lambda _: "source-hash")
    monkeypatch.setattr("gev.domain.tokenization.MarkerMap.load", lambda *args: markers)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object())

    saved = []

    result = stages.train_stage("configs/smoke.toml", data_root=tmp_path,
                                output=tmp_path / "run")

    assert result is metrics
    assert loader_calls[0][0][1:] == ("decision-v7", "train")
    assert loader_calls[0][1]["training"] is True
    assert train_calls[0][0][0] is model
    assert train_calls[0][0][1].requests == rows
    assert train_calls[0][1]["source_hash"] == "source-hash"
    assert train_calls[0][1]["resume"] is None
    metadata = saved[0][2]
    assert metadata["study_id"] == config.experiment_id
    assert metadata["source_sha256"] == "source-hash"
    assert metadata["manifest_sha256"] == "manifest-hash"
    assert metadata["lora"]["targets"] == list(resolved_model.family.lora_targets)
    assert metadata["resume_input"] is None


def test_warm_start_stage_uses_fresh_optimizer_and_shared_metadata(monkeypatch, tmp_path):
    config = load_config("configs/gemma3-1b-night2.toml")
    marker_path = tmp_path / "markers.json"
    marker_path.write_text(json.dumps({"tokenizer_sha256": "token-hash"}), encoding="utf-8")
    config = replace(config, model=replace(config.model, marker_artifact=str(marker_path)))
    resolved_model = resolve_model("gemma3_text", "torch")
    train_calls = []
    metrics = {"device": "cpu", "complete": True}
    resolved = SimpleNamespace(
        model=resolved_model,
        load_tokenizer=lambda: object(),
        load_markers=lambda _tokenizer: markers,
        encode_record=lambda tokenizer, record, marker_map, **caps: {"encoded": True},
        validate_runtime_available=lambda: None,
        seed_rng=lambda: None,
        load_checkpoint=lambda *args, **kwargs: (model, {}),
        save_checkpoint=lambda *args: saved.append(args),
        checkpoint_fingerprint=lambda _: "init-fingerprint",
        trainable_fingerprint=lambda _: "initial-weights-fingerprint",
        train=lambda model_arg, schedule_arg, output_arg, **kwargs:
        train_calls.append(((model_arg, schedule_arg, output_arg), kwargs)) or metrics,
        provenance=lambda **kwargs: {"study_id": config.experiment_id, **kwargs},
    )
    markers = MarkerMap({"state": 1, "question": 2, "option_start": 3,
                         "option_end": 4, "decide": 5},
                        {"state": "s", "question": "q", "option_start": "os",
                         "option_end": "oe", "decide": "d"}, config.model.revision)
    run = tmp_path / "warm-start"
    prepared_path = run.with_name(run.name + "-data")
    rows = [{"_meta": {"id": "night2/1"}}]
    prepared = {"combined_sha256": "combined", "night2_sha256": "night2",
                "replay_ids_sha256": "replay"}
    model = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(stages, "load_config", lambda _: config)
    monkeypatch.setattr(stages, "resolve_experiment_config", lambda _: resolved)
    monkeypatch.setattr(stages, "validate_init_metadata", lambda *_: {"initializer_kind": "full-v7", "config": {}})
    monkeypatch.setattr(stages, "build_continuation", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(stages, "load_prepared", lambda *args, **kwargs: (rows, prepared))
    monkeypatch.setattr(stages, "file_digest", lambda _: "unused")
    monkeypatch.setattr("gev.domain.tokenization.MarkerMap.load", lambda *args: markers)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object())
    saved = []

    result = stages.warm_start_stage(config_path="configs/gemma3-1b-night2.toml",
                                     init_from="v7-run", data_root=tmp_path, output=run)

    assert result["init_checkpoint_fingerprint"] == "init-fingerprint"
    assert train_calls[0][0][0] is model
    assert "resume" not in train_calls[0][1]
    assert saved[0][2]["continuation"]["fresh_optimizer"] is True
    assert saved[0][2]["continuation"]["prepared"] == prepared


def test_mlx_bf16_diagnostic_reaches_development_gate_and_fails_closed(
        monkeypatch, tmp_path):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    base_qualification = {
        "revision": config.model.revision, "source_weights_dtype": "bf16",
        "compute_dtype": "bf16",
        "inventory": {"tensor_count": 2011, "text_tensor_count": 600,
                      "text_serialized_parameters": 4_647_449_891,
                      "effective_text_parameters": 4_628_569_344},
        "lora": {"layers": 205},
    }
    model = SimpleNamespace(provenance={"model_output_id": "gev-gemma4-e2b",
                                        "source_weights_dtype": "bf16",
                                        "compute_dtype": "bf16",
                                        "base_qualification": base_qualification},
                            eval=lambda: None)
    markers = MarkerMap(config.model.marker_ids,
                        {"state": "<unused0>", "question": "<unused1>",
                         "option_start": "<unused2>", "option_end": "<unused3>",
                         "decide": "<unused4>"}, config.model.revision)
    train_rows = [{"_meta": {"id": "train/one", "source": "fixture"}}]
    dev_rows = [{"_meta": {"id": "dev/one", "source": "fixture"}}]
    saved, observed = [], {}
    metrics = {"device": "gpu", "dtype": "bf16", "complete": False,
               "engineering_checks": {"bf16_frozen_decoder": True}}

    def fake_schedule(rows, training_config, *_args, **_kwargs):
        observed["max_steps"] = training_config.training.max_steps
        return SimpleNamespace(requests=rows)

    def fake_train(_model, _schedule, output, **_kwargs):
        Path(output).mkdir(parents=True, exist_ok=True)
        observed["trained"] = True
        return metrics

    resolved = SimpleNamespace(
        model=resolve_model("gemma4_e2b_text", "mlx"),
        validate_runtime_available=lambda: None,
        seed_rng=lambda: None,
        load_tokenizer=lambda: object(),
        load_markers=lambda _tokenizer: markers,
        create_model=lambda **_kwargs: model,
        encode_record=lambda _tokenizer, record, _markers: {"metadata": record["_meta"]},
        create_predictor=lambda *_args, **_kwargs: object(),
        trainable_fingerprint=lambda _model: "f" * 64,
        train=fake_train,
        save_checkpoint=lambda *args: saved.append(args),
    )
    monkeypatch.setattr(stages, "load_config", lambda _path: config)
    monkeypatch.setattr(stages, "resolve_experiment_config", lambda _config: resolved)
    monkeypatch.setattr(stages, "TrainingSchedule", fake_schedule)
    monkeypatch.setattr(stages, "load_verified_split",
                        lambda _root, _suite, split, **_kwargs:
                        ((train_rows if split == "train" else dev_rows), {}, "manifest"))
    monkeypatch.setattr(stages, "validate_training_rows", lambda *_args: None)
    monkeypatch.setattr(stages, "split_path", lambda *_args: tmp_path / "train.jsonl")
    monkeypatch.setattr(stages, "file_digest", lambda _path: "source")
    monkeypatch.setattr("gev.domain.tokenization.MarkerMap.load", lambda *_args: markers)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("gev.evaluation.development.select_development_rows",
                        lambda rows, **_kwargs: rows)
    monkeypatch.setattr("gev.backends.mlx.qualification.run_base_qualification",
                        lambda **_kwargs: pytest.fail("offline FP32 oracle must not run in training"))
    monkeypatch.setattr("gev.backends.mlx.qualification.load_torch_cpu_oracle",
                        lambda *_args, **_kwargs: pytest.fail("Torch oracle must not run in training"))
    monkeypatch.setattr("gev.domain.materialize.materialize", lambda row: row)

    def fake_evaluate(rows, _predictor, output, _temperature):
        Path(output).mkdir(parents=True, exist_ok=True)
        report = {"coverage": {"evaluated_records": len(rows),
                                "evaluated_questions": 1, "rejected_records": 0,
                                "truncated_records": 0},
                  "mechanism_checks": {"passed": True},
                  "clean": {"acc": 0.5, "nll": 0.6, "brier": 0.4, "ece": 0.1}}
        (Path(output) / "report.json").write_text(json.dumps(report) + "\n")
        return report, [{"source": "fixture"}]

    monkeypatch.setattr("gev.evaluation.benchmark.evaluate_records", fake_evaluate)
    monkeypatch.setattr("gev.evaluation.metrics.grouped_metrics", lambda *_args: {"fixture": {"nll": 0.6}})
    monkeypatch.setattr("gev.backends.mlx.qualification.collect_mlx_training_checks",
                        lambda *_args, **_kwargs: {
                            name: False for name in (
                                "source_inventory_exact", "decoder_compute_dtype",
                                "fp32_lora_and_pointer", "fp32_optimizer_state",
                                "finite_loss_gradients", "nonzero_trainable_update",
                                "boundary_gathers_valid", "decoder_masks_valid",
                                "shared_kv_mapping_exact")})
    monkeypatch.setattr(stages, "_row_isolation_check", lambda *_args, **_kwargs: (True, 0.0))
    output = tmp_path / "diagnostic-run"
    with pytest.raises(RuntimeError, match="qualification checks failed"):
        stages.train_stage("configs/gemma4-e2b-mlx-bf16.toml", data_root=tmp_path,
                           output=output, max_steps=12)

    assert observed == {"max_steps": 12, "trained": True}
    assert saved == []
    assert not (output / "checkpoint").exists()
    assert (output / "qualification.json").is_file()
    training_metrics = json.loads((output / "training_metrics.json").read_text())
    receipt = training_metrics["qualification"]
    assert receipt["status"] == "failed"
    assert receipt["backend"] == "mlx"
    assert "development_eligible" not in receipt
    assert receipt["checks"]["source_inventory_exact"] is False


@pytest.mark.parametrize(("backend", "compute_dtype", "checks_pass", "mechanisms_pass"), [
    ("mlx", "bf16", True, True), ("mlx", "fp32", True, True),
    ("torch", "bf16", True, True), ("torch", "fp32", True, True),
    ("torch", "bf16", False, True), ("torch", "bf16", True, False),
])
def test_gemma4_train_stage_builds_one_backend_neutral_checkpoint_receipt(
        monkeypatch, tmp_path, backend, compute_dtype, checks_pass, mechanisms_pass):
    from gev.models.policy import GEMMA4_POLICY

    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = replace(config, backend=replace(config.backend, id=backend),
                     training=replace(config.training, dtype=compute_dtype),
                     runtime=replace(config.runtime,
                                     device="gpu" if backend == "mlx" else "cpu"))
    inventory = {"tensor_count": 2011, "text_tensor_count": 600,
                 "text_serialized_parameters": 4_647_449_891,
                 "effective_text_parameters": 4_628_569_344,
                 "name_shape_sha256": "d" * 64}
    model = SimpleNamespace(
        provenance={"base_qualification": {"revision": config.model.revision,
                                            "source_weights_dtype": "bf16",
                                            "compute_dtype": compute_dtype,
                                            "inventory": inventory}},
        source_provenance={"model": config.model.name,
                           "revision": config.model.revision,
                           "source_weights_dtype": "bf16",
                           "compute_dtype": compute_dtype,
                           "source_text_tensor_count": 600,
                           "source_text_parameters": 4_647_449_891,
                           "effective_text_parameters": 4_628_569_344,
                           "source_name_shape_sha256": "d" * 64},
        eval=lambda: None)
    markers = MarkerMap(config.model.marker_ids,
                        {"state": "<unused0>", "question": "<unused1>",
                         "option_start": "<unused2>", "option_end": "<unused3>",
                         "decide": "<unused4>"}, config.model.revision)
    train_rows = [{"_meta": {"id": "train/one", "source": "fixture"}}]
    dev_rows = [{"_meta": {"id": "dev/one", "source": "fixture"},
                 "state": "fixture", "questions": {"q1": {}, "q2": {}}}]
    split_calls, saved = [], []
    checks = {name: True for name in
              GEMMA4_POLICY.required_qualification_checks(backend, compute_dtype)}
    if not checks_pass:
        checks["source_inventory_exact"] = False
    metrics = {"device": config.runtime.device, "dtype": compute_dtype,
               "compute_dtype": compute_dtype, "weights_dtype": compute_dtype,
               "source_weights_dtype": "bf16", "complete": False,
               "logical_steps": 3, "qualification_checks": checks}
    resolved_model = resolve_model("gemma4_e2b_text", backend)
    persisted_receipts = []

    def fake_save(_model, _directory, metadata, _tokenizer=None):
        from gev.models.qualification import bind_checkpoint_tensors

        saved.append(metadata)
        persisted_receipts.append(bind_checkpoint_tensors(
            metadata["qualification"], {"adapter": "a" * 64, "pointer": "b" * 64}))

    resolved = SimpleNamespace(
        model=resolved_model, validate_runtime_available=lambda: None,
        seed_rng=lambda: None, load_tokenizer=lambda: object(),
        load_markers=lambda _tokenizer: markers,
        create_model=lambda **_kwargs: model,
        encode_record=lambda _tokenizer, row, _markers: {"metadata": row["_meta"]},
        create_predictor=lambda *_args, **_kwargs: object(),
        train=lambda *_args, **_kwargs: metrics,
        trainable_fingerprint=lambda _model: "e" * 64,
        provenance=lambda **_kwargs: {
            "study_id": config.experiment_id, "model_output_id": resolved_model.output_model_id,
            "protocol": {"id": config.protocol.id, "version": config.protocol.version},
            "model_family": resolved_model.family.family_id,
            "family_runtime": "Gemma4E2BTextRuntime", "backend": backend,
            "scientific_recipe": {}, "scientific_recipe_sha256": "a" * 64,
            "resolved_config": {"resolved_config": {}, "operational_controls": {}},
        },
        save_checkpoint=fake_save,
    )
    monkeypatch.setattr(stages, "load_config", lambda _path: config)
    monkeypatch.setattr(stages, "resolve_experiment_config", lambda _config: resolved)
    monkeypatch.setattr(stages, "_configure_runtime", lambda *_args: None)
    monkeypatch.setattr(stages, "TrainingSchedule",
                        lambda rows, *_args, **_kwargs: SimpleNamespace(requests=rows))
    monkeypatch.setattr(stages, "load_verified_split",
                        lambda _root, _suite, split, **_kwargs:
                        split_calls.append(split)
                        or ((train_rows if split == "train" else dev_rows), {}, "c" * 64))
    monkeypatch.setattr(stages, "validate_training_rows", lambda *_args: None)
    monkeypatch.setattr(stages, "split_path", lambda *_args: tmp_path / "train.jsonl")
    monkeypatch.setattr(stages, "file_digest", lambda _path: "b" * 64)
    monkeypatch.setattr("gev.domain.tokenization.MarkerMap.load", lambda *_args: markers)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("gev.domain.materialize.materialize", lambda row: row)
    monkeypatch.setattr(stages, "_row_isolation_check", lambda *_args, **_kwargs: (True, 0.0))
    if backend == "mlx":
        monkeypatch.setattr("gev.backends.mlx.qualification.collect_mlx_training_checks",
                            lambda *_args, **_kwargs: checks)
    else:
        monkeypatch.setattr("gev.backends.torch.qualification.collect_torch_training_checks",
                            lambda *_args, **_kwargs: checks)
        monkeypatch.setattr("gev.backends.torch.qualification.causal_attention_masks_valid",
                            lambda *_args, **_kwargs: True)

    def evaluate(rows, _predictor, eval_output, _temperature):
        Path(eval_output).mkdir(parents=True, exist_ok=True)
        report = {"coverage": {"requested_records": len(rows),
                                "evaluated_records": len(rows),
                                "requested_questions": 2, "evaluated_questions": 2,
                                "rejected_records": 0,
                                "truncated_records": 0},
                  "mechanism_checks": {
                      "passed": mechanisms_pass,
                      "failures": 0 if mechanisms_pass else 1},
                  "clean": {"acc": 0.5, "nll": 0.7, "brier": 0.4, "ece": 0.1}}
        return report, [{"source": "fixture"}]

    monkeypatch.setattr("gev.evaluation.benchmark.evaluate_records", evaluate)
    monkeypatch.setattr("gev.evaluation.metrics.grouped_metrics",
                        lambda *_args: {"fixture": {"nll": 0.7}})
    monkeypatch.setattr("gev.artifacts.checkpoint_identity.read_checkpoint_manifest",
                        lambda _directory: {"qualification": persisted_receipts[0]})
    if checks_pass and mechanisms_pass:
        result = stages.train_stage("configs/gemma4-e2b-mlx-bf16.toml",
                                    data_root=tmp_path, output=tmp_path / "run")
    else:
        with pytest.raises(RuntimeError, match="qualification checks failed"):
            stages.train_stage("configs/gemma4-e2b-mlx-bf16.toml",
                               data_root=tmp_path, output=tmp_path / "run")
        failed = json.loads((tmp_path / "run" / "qualification.json").read_text())
        assert failed["status"] == "failed"
        assert saved == []
        assert not (tmp_path / "run" / "checkpoint").exists()
        if not checks_pass:
            assert failed["checks"]["source_inventory_exact"] is False
        if not mechanisms_pass:
            assert failed["checks"]["development_mechanisms"] is False
        return

    receipt = result["qualification"]
    assert receipt["status"] == "passed"
    assert receipt["backend"] == backend
    assert receipt["base"]["source_weights_dtype"] == "bf16"
    assert receipt["base"]["compute_dtype"] == compute_dtype
    assert receipt["training_complete"] is False
    assert receipt["checkpoint_ready"] is True
    assert receipt["development"]["scores"]["clean"]["nll"] == 0.7
    assert receipt["checks"]["development_mechanisms"] is True
    assert split_calls == ["train", "development"]
    assert saved[0]["qualification"]["checkpoint_ready"] is False
    assert json.loads((tmp_path / "run" / "qualification.json").read_text()) == receipt
    assert json.loads((tmp_path / "run" / "training_metrics.json").read_text())[
        "qualification"] == receipt
    assert "development_eligible" not in receipt
    assert "full_recipe_eligible" not in receipt
    assert "heldout_eligible" not in receipt
    assert not (tmp_path / "run" / "fp32-staging-receipt.json").exists()
