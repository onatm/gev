import dataclasses
import tomllib

from gev.configuration.config import load_config
from gev.configuration.resolved import resolve_experiment_config
from gev.study.config import serialize_toml, write_seed_config


def test_seed_config_is_typed_roundtrip_and_operational_hash_is_stable(tmp_path):
    base = load_config("configs/gemma3-1b-v7.toml")
    first = write_seed_config(base, 7, tmp_path, save_every=100)
    second = write_seed_config(base, 8, tmp_path, save_every=200)

    assert tomllib.loads(first.read_text(encoding="utf-8"))["training"]["seed"] == 7
    assert load_config(first) == dataclasses.replace(
        base, training=dataclasses.replace(base.training, seed=7, save_every=100))
    assert resolve_experiment_config(load_config(first)).recipe_sha256 == \
        resolve_experiment_config(load_config(second)).recipe_sha256


def test_explicit_snapshot_interval_is_not_overwritten_by_study_default(tmp_path):
    config_text = open("configs/gemma3-1b-v7.toml", encoding="utf-8").read()
    config_text = config_text.replace(
        "p_none_pair = 0.25", "p_none_pair = 0.25\nsave_every = 17")
    config_path = tmp_path / "explicit.toml"
    config_path.write_text(config_text)
    config = load_config(config_path)

    explicit = write_seed_config(config, 1, tmp_path, save_every=100)
    short_dir = tmp_path / "short"
    short_dir.mkdir()
    short = write_seed_config(load_config("configs/gemma3-1b-v7.toml"), 2,
                              short_dir)

    assert "save_every = 17" in explicit.read_text()
    assert "save_every = 100" not in explicit.read_text()
    assert "save_every =" not in short.read_text()


def test_deterministic_toml_serializer_roundtrips_supported_scalars():
    value = {"section": {"text": 'quote " and newline\nnext', "enabled": True,
                         "ratio": 0.125, "items": ["a", False], "optional": None}}

    encoded = serialize_toml(value)

    assert encoded == serialize_toml(value)
    assert tomllib.loads(encoded) == {"section": {
        "text": 'quote " and newline\nnext', "enabled": True,
        "ratio": 0.125, "items": ["a", False]}}
