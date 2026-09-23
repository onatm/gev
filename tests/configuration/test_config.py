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
    with pytest.raises(ValueError, match="unknown model backend"):
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


def test_bf16_is_not_configurable_for_cpu_or_auto(tmp_path):
    source = open("configs/smoke.toml", encoding="utf-8").read().replace(
        'dtype = "fp32"', 'dtype = "bf16"')
    path = tmp_path / "bf16-auto.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="runtime.device=mps"):
        load_config(path)
