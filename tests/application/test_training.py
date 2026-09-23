import json
from dataclasses import replace
from types import SimpleNamespace

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
