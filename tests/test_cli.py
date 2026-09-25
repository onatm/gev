import json
from pathlib import Path

import pytest

from gev import cli, config

CONFIGS = sorted(Path(__file__).parents[1].glob("configs/*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_checked_in_configs_load(path):
    loaded = config.load(path)
    assert config.from_dict(json.loads(json.dumps(loaded.to_dict()))) == loaded


def test_config_rejects_unknown_keys_and_bad_values(tmp_path):
    base = '[model]\nfamily = "gemma4"\nname = "m"\nrevision = "' + "a" * 40 + '"\n'
    path = tmp_path / "c.toml"
    path.write_text('name = "x"\n' + base + "[training]\nbogus = 1\n")
    with pytest.raises(config.ConfigError, match="unknown key"):
        config.load(path)
    path.write_text('name = "x"\nbackend = "mlx"\n' + base.replace("gemma4", "gemma3"))
    with pytest.raises(config.ConfigError, match="MLX backend supports gemma4"):
        config.load(path)


def test_overrides_reach_training_fields():
    loaded = config.load(CONFIGS[0]).replace(seed=5, max_steps=3, device="cpu")
    assert (loaded.training.seed, loaded.training.max_steps, loaded.device) == (5, 3, "cpu")


def test_parser_and_error_exit(tmp_path, capsys):
    args = cli.build_parser().parse_args(["evaluate", "run", "--suite", "decision-v7", "--split", "test", "--out", "o"])
    assert args.split == "test"
    assert cli.main(["calibrate", str(tmp_path / "missing")]) == 1
    assert cli.main(["train", str(CONFIGS[0]), "--out", str(tmp_path), "--data", str(tmp_path)]) == 1
    assert "exists" in capsys.readouterr().err
