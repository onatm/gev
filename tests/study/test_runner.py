import json
from pathlib import Path

import pytest

import gev.study.runner as study_runner


def _fake_plan(*, smoke_child=False):
    return {"smoke_child": smoke_child, "selection_rule": "test",
            "trials": [{"seed": 0, "steps": 3144}]}


def _fake_child_process(calls, *, truncated=0, returncode=0,
                        question_shortfall=0, mechanisms_passed=True):
    def fake_run(command, *, log_path, on_update, progress_label, **kwargs):
        calls.append(command)
        on_update({"step": 4, "total_steps": 9, "latest_loss": 1.25}, .01, False)
        if command[3] == "train":
            output = Path(command[command.index("--out") + 1])
            assert not output.exists()
            if returncode == 0:
                output.mkdir(parents=True)
                (output / "training_metrics.json").write_text(json.dumps({"complete": True}))
                (output / "checkpoint").mkdir()
        elif command[3] == "evaluate":
            assert "/seed-" in command[4] or command[4].endswith("existing-run")
            assert command[command.index("--temperature") + 1] == "1"
            output = Path(command[command.index("--out") + 1])
            output.mkdir(parents=True)
            (output / "report.json").write_text(json.dumps({
                "coverage": {"requested_records": 2, "evaluated_records": 2,
                             "requested_questions": 3,
                             "evaluated_questions": 3 - question_shortfall,
                             "rejected_records": 0, "truncated_records": truncated},
                "mechanism_checks": {"passed": mechanisms_passed},
                "clean": {"acc": 1.0, "nll": 0.1, "brier": 0.1, "ece": 0.0},
                "tasks": {},
            }))
        elif command[3] == "calibrate":
            Path(command[command.index("--out") + 1]).write_text("{}")
        log_path.write_text("child output\n")
        output_excerpt = "Traceback (most recent call last):\nValueError: child failed\n" if returncode else ""
        is_training = command[3] == "train"
        return {"returncode": returncode, "step": 4 if is_training else None,
                "total_steps": 9 if is_training else None,
                "latest_loss": 1.25 if is_training else None, "elapsed_seconds": .01,
                "log_path": str(log_path), "output_excerpt": output_excerpt}
    return fake_run


def test_run_saves_seed_config_status_and_real_child_command_shape(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: _fake_plan())
    fake = _fake_child_process(calls)

    def check_initial_status(command, *, log_path, **kwargs):
        if command[3] == "train":
            status = json.loads((log_path.parents[2] / "status.json").read_text())
            assert status["step"] == 0
            assert status["total_steps"] == status["full_training_steps"] == 3144
            kwargs["on_update"]({"step": None, "total_steps": None, "latest_loss": None}, .01, True)
            status = json.loads((log_path.parents[2] / "status.json").read_text())
            assert status["step"] == 0
            assert status["total_steps"] == status["full_training_steps"] == 3144
        return fake(command, log_path=log_path, **kwargs)

    monkeypatch.setattr(study_runner, "run_child", check_initial_status)
    output = tmp_path / "study"
    result = study_runner.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=output)

    assert result["status"] == "completed"
    assert (output / "configs" / "seed-0.toml").exists()
    assert calls[0][3] == "train"
    assert calls[0][4].endswith("configs/seed-0.toml")
    assert not (output / "seed-0" / "configs").exists()
    assert "save_every = 100" in (output / "configs" / "seed-0.toml").read_text()
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["stage"] == "finished"
    assert (status["seed"], status["seed_index"], status["seed_total"]) == (0, 0, 1)
    assert status["step"] == 4 and status["latest_loss"] == 1.25
    terminal = capsys.readouterr().out
    assert str(output / "result.json") in terminal
    assert '"reports"' not in terminal


def test_next_seed_status_resets_before_its_first_heartbeat(tmp_path, monkeypatch):
    seeds = (2, 4)
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: {
        **_fake_plan(), "trials": [{"seed": seed, "steps": 3144} for seed in seeds],
    })
    calls = []
    fake = _fake_child_process(calls)
    training_children = 0

    def check_seed_status(command, *, log_path, **kwargs):
        nonlocal training_children
        if command[3] == "train":
            seed = seeds[training_children]
            status_path = log_path.parents[2] / "status.json"
            status = json.loads(status_path.read_text())
            assert status["seed"] == seed and status["step"] == 0
            assert status["total_steps"] == 3144
            kwargs["on_update"]({"step": None, "total_steps": None, "latest_loss": None}, .01, True)
            status = json.loads(status_path.read_text())
            assert status["seed"] == seed and status["step"] == 0
            assert status["total_steps"] == 3144
            training_children += 1
        return fake(command, log_path=log_path, **kwargs)

    monkeypatch.setattr(study_runner, "run_child", check_seed_status)
    result = study_runner.run("configs/gemma3-1b-v7.toml", seeds=seeds, data="data",
                              out=tmp_path / "multi-seed")

    assert result["status"] == "completed"
    assert training_children == 2


def test_every_evaluation_child_uses_the_trial_and_unit_temperature(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: _fake_plan())
    monkeypatch.setattr(study_runner, "run_child", _fake_child_process(calls))

    study_runner.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=tmp_path / "study")
    evaluations = [command for command in calls if command[3] == "evaluate"]

    assert len(evaluations) == 3
    assert all(command[4].endswith("seed-0") for command in evaluations)
    assert all("--config" not in command for command in evaluations)
    assert all(command[command.index("--temperature") + 1] == "1" for command in evaluations)


@pytest.mark.parametrize(("options", "reason"), [
    ({"truncated": 1}, "truncation"),
    ({"question_shortfall": 1}, "question coverage"),
    ({"mechanisms_passed": False}, "mechanism"),
])
def test_incomplete_evaluations_are_diagnostic_and_never_promoted(
        tmp_path, monkeypatch, options, reason):
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: _fake_plan())
    monkeypatch.setattr(study_runner, "run_child",
                        _fake_child_process([], **options))

    result = study_runner.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data",
                              out=tmp_path / reason.replace(" ", "-"))

    assert result["status"] == "diagnostic"
    assert result["promotion"]["selected_seed"] is None
    assert result["trials"][0]["eligible"] is False


def test_short_diagnostic_uses_one_step_without_full_study_snapshot_interval(
        tmp_path, monkeypatch):
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: _fake_plan())
    calls = []
    fake = _fake_child_process(calls)

    def incomplete_train(command, **kwargs):
        if command[3] == "train":
            status = json.loads((kwargs["log_path"].parents[2] / "status.json").read_text())
            assert status["step"] == 0 and status["total_steps"] == 1
            assert status["full_training_steps"] == 3144
        result = fake(command, **kwargs)
        if command[3] == "train":
            metrics = Path(command[command.index("--out") + 1]) / "training_metrics.json"
            metrics.write_text(json.dumps({"complete": False}))
        return result

    monkeypatch.setattr(study_runner, "run_child", incomplete_train)
    result = study_runner.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data",
                              out=tmp_path / "short-study", max_steps=1)
    seed_config = (tmp_path / "short-study" / "configs" / "seed-0.toml").read_text()

    assert "save_every =" not in seed_config
    assert "--max-steps" in calls[0]
    assert result["status"] == "diagnostic"


def test_failed_train_child_records_failure_and_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(study_runner, "plan", lambda *args, **kwargs: _fake_plan())
    monkeypatch.setattr(study_runner, "run_child",
                        _fake_child_process([], returncode=9))

    with pytest.raises(SystemExit) as exc:
        study_runner.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data",
                         out=tmp_path / "study")

    result = json.loads((tmp_path / "study" / "result.json").read_text())
    assert exc.value.code == 1
    assert result["status"] == "failed"
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][0]["stage"] == "train"
    assert result["trials"][0]["exit_code"] == 9
    assert "ValueError: child failed" in result["trials"][0]["output_excerpt"]
    assert not (tmp_path / "study" / "seed-0").exists()
