from pathlib import Path

import pytest

from gev.study.runner import plan
import gev.study.runner as study_runner


def test_three_seed_plan_matches_pinned_steps_and_variant_counts():
    value = plan("configs/gemma3-1b-v7.toml", data="data")

    assert [trial["seed"] for trial in value["trials"]] == [0, 1, 2]
    assert [trial["steps"] for trial in value["trials"]] == [3144, 3144, 3144]
    assert [trial["variants"] for trial in value["trials"]] == [30428, 30530, 30370]
    assert value["no_test_loading"] is True


@pytest.mark.parametrize(("source", "new", "message"), [
    ("[backend]\nid = \"torch\"", "unknown", "unknown model backend"),
    ("family = \"gemma3_text\"", "unknown", "unknown model family"),
])
def test_plan_rejects_unregistered_model_selection_before_data_access(
        tmp_path, monkeypatch, source, new, message):
    config = Path("configs/smoke.toml").read_text()
    if source.startswith("[backend]"):
        config = config.replace(source, f'[backend]\nid = "{new}"')
    else:
        config = config.replace(source, f'family = "{new}"')
    path = tmp_path / "unsupported.toml"
    path.write_text(config)
    monkeypatch.setattr(study_runner, "_verified",
                        lambda *_: pytest.fail("data accessed before model resolution"))

    with pytest.raises(ValueError, match=message):
        plan(path, seeds=(0,), data=tmp_path / "missing-data")


def test_plan_persists_resolved_recipe_identity_from_verified_training_data(monkeypatch, tmp_path):
    manifest = {"files": {"train.jsonl": {"records": 1, "sha256": "train"}}}
    monkeypatch.setattr(study_runner, "_verified", lambda *_: ([{"state": "s", "questions": {}}], manifest))

    result = plan("configs/gemma3-1b-v7.toml", seeds=(0,), data=tmp_path)

    assert result["study_id"] == "gemma3-1b-v7"
    assert result["protocol"] == {"id": "kev-decision-v7", "version": 1}
    assert result["model_family"] == "gemma3_text"
    assert result["backend"] == "torch"
    assert result["scientific_recipe"]["training"]["epochs"] == 2
    assert result["resolved_config"]["runtime"]["output_root"] == "runs"
    assert list(tmp_path.iterdir()) == []


def test_save_interval_does_not_change_scientific_recipe_identity(tmp_path):
    source = Path("configs/gemma3-1b-v7.toml").read_text()
    configured = tmp_path / "save-every.toml"
    configured.write_text(source.replace(
        "p_none_pair = 0.25", "p_none_pair = 0.25\nsave_every = 7"))

    assert plan("configs/gemma3-1b-v7.toml", seeds=(0,), data="data")["config_sha256"] == \
        plan(configured, seeds=(0,), data="data")["config_sha256"]
