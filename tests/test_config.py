import pytest

from gev.config import ConfigError, load_config


def test_smoke_config_is_valid():
    config = load_config("configs/smoke.toml")
    assert config.model.expected_model_type == "gemma3_text"


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
