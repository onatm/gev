import pytest
import dataclasses

from gev.configuration.config import BackendConfig, ConfigError, ModelConfig, ProtocolConfig, load_config
from gev.configuration.resolved import _recipe_digest, resolve_experiment_config, scientific_recipe


def test_smoke_config_is_valid():
    config = load_config("configs/smoke.toml")
    assert config.model.expected_model_type == "gemma3_text"
    assert config.model.family == "gemma3_text"
    assert config.backend.id == "torch"
    assert config.protocol.id == "kev-decision-v7"


def test_gemma4_backend_precision_examples_match_the_pinned_policy():
    examples = {
        "configs/gemma4-e2b-mlx-bf16.toml": ("mlx", "bf16"),
        "configs/gemma4-e2b-mlx-fp32.toml": ("mlx", "fp32"),
        "configs/gemma4-e2b-torch-bf16.toml": ("torch", "bf16"),
        "configs/gemma4-e2b-torch-fp32.toml": ("torch", "fp32"),
    }
    resolved = [resolve_experiment_config(load_config(path))
                for path in examples]
    configs = [item.config for item in resolved]

    assert len({config.experiment_id for config in configs}) == len(examples)
    assert all((config.backend.id, config.training.dtype) == examples[path]
               for path, config in zip(examples, configs, strict=True))
    assert all(config.runtime.device == "auto" for config in configs)
    assert all(config.model.name == "google/gemma-4-E2B"
               and config.model.revision == "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
               and config.model.marker_ids == {
                   "state": 6, "question": 7, "option_start": 8,
                   "option_end": 9, "decide": 10}
               for config in configs)


def test_gemma4_config_records_independent_bf16_mlx_recipe():
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    assert config.model.name == "google/gemma-4-E2B"
    assert config.model.revision == "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
    assert config.model.marker_ids == {
        "state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}
    assert config.backend.id == "mlx"
    assert config.training.dtype == "bf16"
    assert config.runtime.execution_mode == "rows"
    resolved = resolve_experiment_config(config)
    assert resolved.model.family.family_id == "gemma4_e2b_text"
    assert resolved.model.backend.backend_id == "mlx"
    assert resolved.provenance()["model_output_id"] == "gev-gemma4-e2b"
    assert resolved.provenance()["model_output_id"] != config.model.name
    assert resolved.recipe["model"]["source_weights_dtype"] == "bf16"
    assert resolved.recipe["model"]["compute_dtype"] == "bf16"


def test_gemma4_config_resolution_accepts_torch_and_staged_mlx_fp32(tmp_path):
    source = open("configs/gemma4-e2b-mlx-bf16.toml", encoding="utf-8").read()
    path = tmp_path / "gemma4-torch.toml"
    path.write_text(source.replace('id = "mlx"', 'id = "torch"'), encoding="utf-8")
    resolved = resolve_experiment_config(load_config(path))
    assert resolved.model.backend.backend_id == "torch"
    resolved.validate_runtime_available()
    path.write_text(source.replace('dtype = "bf16"', 'dtype = "fp32"'), encoding="utf-8")
    mlx_fp32 = resolve_experiment_config(load_config(path))
    assert mlx_fp32.model.backend.backend_id == "mlx"
    from gev.models.policy import GEMMA4_POLICY
    assert ("mlx", "gpu", "fp32") in GEMMA4_POLICY.implemented
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    with pytest.raises(ValueError, match="pinned PRETRAINED base"):
        resolve_experiment_config(dataclasses.replace(
            config, model=dataclasses.replace(config.model, name="un-pinned/model")))


@pytest.mark.parametrize(("device", "dtype"), [
    (device, dtype)
    for device in ("cpu", "mps", "cuda")
    for dtype in ("fp32", "bf16")
])
def test_gemma4_torch_policy_resolves_all_implemented_precision_device_pairs(device, dtype):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = dataclasses.replace(
        config,
        backend=BackendConfig("torch"),
        training=dataclasses.replace(config.training, dtype=dtype, microbatch=2),
        runtime=dataclasses.replace(config.runtime, device=device,
                                    gradient_checkpointing=True, mps_fallback=True),
    )
    resolved = resolve_experiment_config(config)
    assert resolved.config.backend.id == "torch"


def test_auto_policy_probes_only_when_device_can_change_precision_support():
    from gev.models.policy import auto_device_requires_probe

    gemma3 = load_config("configs/gemma3-1b-v7.toml")
    assert not auto_device_requires_probe(gemma3)
    gemma3_bf16 = dataclasses.replace(
        gemma3, training=dataclasses.replace(gemma3.training, dtype="bf16"))
    assert auto_device_requires_probe(gemma3_bf16)

    gemma4 = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    gemma4_torch = dataclasses.replace(gemma4, backend=BackendConfig("torch"))
    assert not auto_device_requires_probe(gemma4_torch)


@pytest.mark.parametrize("old,new,message", [
    ('mps_fallback = false', 'mps_fallback = true', "MPS fallback"),
    ('execution_mode = "rows"', 'execution_mode = "packed"', "rows execution only"),
])
def test_gemma4_config_rejects_unsupported_runtime_modes(tmp_path, old, new, message):
    source = open("configs/gemma4-e2b-mlx-bf16.toml", encoding="utf-8").read().replace(old, new)
    path = tmp_path / "unsupported-gemma4.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_experiment_config(load_config(path))


def test_resolved_config_selects_backend_without_constructing_a_model():
    config = load_config("configs/gemma3-1b-v7.toml")
    resolved = resolve_experiment_config(config)
    assert resolved.model.family.family_id == "gemma3_text"
    assert resolved.model.backend.backend_id == "torch"
    assert resolved.provenance()["scientific_recipe_sha256"] == resolved.recipe_sha256
    assert resolved.provenance(output_path="runs/example")["operational_controls"]["output_path"] == "runs/example"


def test_auto_device_selection_and_unavailable_explicit_mps_are_not_silent(monkeypatch):
    config = load_config("configs/gemma3-1b-v7.toml")
    resolved = resolve_experiment_config(config)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolved.select_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolved.select_device() == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolved.select_device() == "mps"

    mps_config = dataclasses.replace(
        config, runtime=dataclasses.replace(config.runtime, device="mps"))
    explicit = resolve_experiment_config(mps_config)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing fallback"):
        explicit.validate_runtime_available()


def test_scientific_recipe_excludes_operational_controls_but_tracks_training_policy():
    config = load_config("configs/gemma3-1b-v7.toml")
    first = resolve_experiment_config(config)
    operational_change = dataclasses.replace(
        config,
        experiment_id="another-run",
        training=dataclasses.replace(config.training, seed=7, max_steps=3, save_every=2),
        runtime=dataclasses.replace(config.runtime, device="cpu", output_root="elsewhere"),
    )
    assert resolve_experiment_config(operational_change).recipe_sha256 == first.recipe_sha256
    changed_policy = dataclasses.replace(
        config, training=dataclasses.replace(config.training, learning_rate=.0002))
    assert resolve_experiment_config(changed_policy).recipe_sha256 != first.recipe_sha256


def test_scientific_recipe_excludes_backend_and_execution_controls_but_tracks_recipe_choices():
    config = load_config("configs/gemma3-1b-v7.toml")
    baseline = _recipe_digest(scientific_recipe(config))
    backend_change = dataclasses.replace(config, backend=BackendConfig("alternate-backend"))
    runtime_change = dataclasses.replace(
        config,
        experiment_id="another-run",
        training=dataclasses.replace(config.training, seed=9, max_steps=5, save_every=3),
        runtime=dataclasses.replace(config.runtime, device="cpu", output_root="elsewhere",
                                    attn_implementation="sdpa", gradient_checkpointing=True,
                                    empty_cache=True),
    )
    assert _recipe_digest(scientific_recipe(backend_change)) == baseline
    assert _recipe_digest(scientific_recipe(runtime_change)) == baseline

    for changed in (
        dataclasses.replace(config, model=dataclasses.replace(config.model, name="other/model")),
        dataclasses.replace(config, training=dataclasses.replace(config.training, dtype="bf16")),
        dataclasses.replace(config, runtime=dataclasses.replace(config.runtime,
                                                                 execution_mode="packed")),
    ):
        assert _recipe_digest(scientific_recipe(changed)) != baseline


def test_unsupported_model_backend_fails_during_resolution():
    config = load_config("configs/smoke.toml")
    with pytest.raises(ValueError, match="does not implement model family"):
        resolve_experiment_config(dataclasses.replace(config, backend=BackendConfig("mlx")))
    with pytest.raises(ValueError, match="does not match family architecture"):
        resolve_experiment_config(dataclasses.replace(
            config, model=dataclasses.replace(config.model, expected_model_type="other")))
    with pytest.raises(ValueError, match="unsupported experiment protocol"):
        resolve_experiment_config(dataclasses.replace(
            config, protocol=ProtocolConfig("unversioned-protocol", 1)))


def test_marker_roles_are_validated_against_the_selected_family():
    config = load_config("configs/smoke.toml")
    wrong_gemma_markers = dataclasses.replace(
        config, model=dataclasses.replace(config.model, marker_ids={"start": 1, "answer": 2}))
    with pytest.raises(ValueError, match="marker ID roles do not match"):
        resolve_experiment_config(wrong_gemma_markers)


def test_marker_id_config_parsing_accepts_generic_distinct_semantic_roles(tmp_path):
    source = open("configs/smoke.toml", encoding="utf-8").read().replace(
        'expected_model_type = "gemma3_text"',
        'expected_model_type = "gemma3_text"\nmarker_ids = { start = 1, answer = 2 }')
    path = tmp_path / "alternate-markers.toml"
    path.write_text(source, encoding="utf-8")

    assert load_config(path).model.marker_ids == {"start": 1, "answer": 2}


def test_marker_id_config_rejects_duplicate_ids(tmp_path):
    source = open("configs/smoke.toml", encoding="utf-8").read().replace(
        'expected_model_type = "gemma3_text"',
        'expected_model_type = "gemma3_text"\nmarker_ids = { start = 1, answer = 1 }')
    path = tmp_path / "duplicate-markers.toml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="marker_ids"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[experiment]\nid='x'\ntypo=true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


@pytest.mark.parametrize("replacement, message", [("revision = 'main'", "40-character"), ("seed = true", "seed"), ("learning_rate = inf", "finite")])
def test_unsafe_values_are_rejected(tmp_path, replacement, message):
    text = open("configs/smoke.toml", encoding="utf-8").read()
    if replacement.startswith("revision"):
        text = text.replace('revision = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"', replacement)
    else:
        text = text.replace("seed = 0", replacement) if replacement.startswith("seed") else text.replace("learning_rate = 0.0001", replacement)
    path = tmp_path / "unsafe.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_bf16_cpu_is_rejected_by_model_policy_and_auto_checks_effective_device(tmp_path, monkeypatch):
    source = open("configs/smoke.toml", encoding="utf-8").read().replace(
        'dtype = "fp32"', 'dtype = "bf16"').replace('device = "auto"', 'device = "cpu"')
    path = tmp_path / "bf16-cpu.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="torch/cpu/bf16"):
        resolve_experiment_config(load_config(path))

    auto_path = tmp_path / "bf16-auto.toml"
    auto_path.write_text(source.replace('device = "cpu"', 'device = "auto"'), encoding="utf-8")
    resolved = resolve_experiment_config(load_config(auto_path))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="torch/cpu/bf16"):
        resolved.validate_runtime_available()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [('dtype = "fp16"', "compute dtype"), ('device = "gpu"', "does not support runtime device")],
)
def test_unsupported_dtype_and_device_are_rejected_during_resolution(
        tmp_path, replacement, message):
    source = open("configs/smoke.toml", encoding="utf-8").read()
    if replacement.startswith("dtype"):
        source = source.replace('dtype = "fp32"', replacement)
    else:
        source = source.replace('device = "auto"', replacement)
    path = tmp_path / "unsupported-policy.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_experiment_config(load_config(path))


@pytest.mark.parametrize(("replacement", "message"), [
    ("dtype = 4", "training.dtype must be a string"),
    ("device = false", "runtime.device must be a string"),
    ("execution_mode = [\"rows\"]", "execution_mode must be rows or packed"),
])
def test_config_keeps_strict_scalar_type_validation(tmp_path, replacement, message):
    source = open("configs/smoke.toml", encoding="utf-8").read()
    if replacement.startswith("dtype"):
        source = source.replace('dtype = "fp32"', replacement)
    elif replacement.startswith("device"):
        source = source.replace('device = "auto"', replacement)
    else:
        source = source.replace('empty_cache = false', 'empty_cache = false\n' + replacement)
    path = tmp_path / "bad-type.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_policy_records_future_gemma4_matrix_without_enabling_it():
    from gev.models.policy import GEMMA4_POLICY

    assert GEMMA4_POLICY.source_weights_dtype == "bf16"
    assert ("mlx", "gpu", "bf16") in GEMMA4_POLICY.implemented
    assert ("mlx", "gpu", "fp32") in GEMMA4_POLICY.implemented
    assert ("torch", "cuda", "bf16") in GEMMA4_POLICY.implemented
    assert GEMMA4_POLICY.bf16_trainable_state_dtype == "fp32"
    assert GEMMA4_POLICY.bf16_optimizer_state_dtype == "fp32"
    assert "decision-v7/development" in GEMMA4_POLICY.development_evaluation_scope
    assert "causal_attention_masks" in GEMMA4_POLICY.required_qualification_checks(
        "torch", "fp32")
    assert "shared_kv_mapping_exact" in GEMMA4_POLICY.required_qualification_checks(
        "mlx", "bf16")


def test_gemma4_development_evaluation_scope_is_centralized():
    from gev.models.policy import validate_development_evaluation_scope

    validate_development_evaluation_scope(
        "gemma4_e2b_text", "decision-v7", "development", "rows")
    with pytest.raises(ValueError, match="evaluation is development-only"):
        validate_development_evaluation_scope(
            "gemma4_e2b_text", "decision-v7", "calibration", "rows")
