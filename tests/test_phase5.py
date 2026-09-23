import json
import os
from pathlib import Path
import sys

import pytest

from gev.evaluation.calibration import calibrate
from gev.evaluation.compare import compare
from gev.experiment import plan
import gev.experiment as experiment


def _rows(n=10):
    return [{"id": f"r{i}", "group": f"g{i}", "question": "q", "source": "s",
             "task": "task", "type": "choice", "variant": "clean", "keys": ["a", "b"],
             "label": i % 2, "p": [.8, .2] if i % 2 == 0 else [.2, .8],
             "logits": [2.0, 0.0] if i % 2 == 0 else [0.0, 2.0],
             "inference_temperature": 1.0} for i in range(n)]


def _run(path, split="calibration"):
    path.mkdir()
    rows = _rows()
    (path / "rows.json").write_text(json.dumps(rows))
    (path / "report.json").write_text(json.dumps({"coverage": {"rejected_records": 0,
        "evaluated_records": len(rows), "requested_records": len(rows)},
        "provenance": {"suite": "decision-v7", "split": split,
                        "suite_sha256": "suite", "source_sha256": "source"}}))
    return path


def test_plan_has_three_distinct_seeds_and_exact_steps():
    value = plan("configs/gemma3-1b-v7.toml", data="data")
    assert [x["seed"] for x in value["trials"]] == [0, 1, 2]
    assert all(x["steps"] == 3144 for x in value["trials"])
    assert value["no_test_loading"] is True


def test_plan_computes_seed_specific_variants():
    trials = plan("configs/gemma3-1b-v7.toml", data="data")["trials"]
    assert [x["variants"] for x in trials] == [30428, 30530, 30370]


def test_calibration_screening_writes_report(tmp_path):
    run = _run(tmp_path / "run")
    output = tmp_path / "calibration.json"
    value = calibrate(run, protocol="kev-screening", out=output)
    assert value["points"] == 81 and output.exists()


def test_calibration_rejects_wrong_split(tmp_path):
    with pytest.raises(ValueError, match="requires split=calibration"):
        calibrate(_run(tmp_path / "run", "development"), protocol="kev-screening")


def test_calibration_rejects_transfer_provenance(tmp_path):
    run = _run(tmp_path / "run")
    report = json.loads((run / "report.json").read_text())
    report["provenance"]["suite"] = "transfer-v4"
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="transfer"):
        calibrate(run)


def test_calibration_rejects_double_temperature(tmp_path):
    run = _run(tmp_path / "run")
    rows = json.loads((run / "rows.json").read_text())
    rows[0]["inference_temperature"] = 2.0
    (run / "rows.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="raw rows"):
        calibrate(run)


def test_compare_self_has_zero_interval(tmp_path):
    run = _run(tmp_path / "run", "development")
    value = compare(run, run, samples=20)
    assert value["paired"]["nll"]["ci95"] == [0.0, 0.0]


def test_compare_rejects_identity_mismatch(tmp_path):
    one, two = _run(tmp_path / "one", "development"), _run(tmp_path / "two", "development")
    report = json.loads((two / "report.json").read_text())
    report["provenance"]["source_sha256"] = "other"
    (two / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity mismatch"):
        compare(one, two)


def test_compare_rejects_incomplete_coverage(tmp_path):
    run = _run(tmp_path / "run", "development")
    report = json.loads((run / "report.json").read_text())
    report["coverage"]["rejected_records"] = 1
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete"):
        compare(run, run)


def test_compare_requires_explicit_calibrated_mode(tmp_path):
    one = _run(tmp_path / "one", "development")
    two = _run(tmp_path / "two", "development")
    rows = json.loads((one / "rows.json").read_text())
    rows[0]["inference_temperature"] = 2.0
    (one / "rows.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="raw comparison"):
        compare(one, two)


def _fake_experiment_plan(*, smoke_child=False):
    return {"smoke_child": smoke_child, "selection_rule": "test",
            "trials": [{"seed": 0, "steps": 3144}]}


def _fake_child_process(calls, *, truncated=0, returncode=0):
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
        elif command[3] == "eval":
            assert "--config" in command
            assert command[command.index("--temperature") + 1] == "1"
            output = Path(command[command.index("--out") + 1])
            output.mkdir(parents=True)
            (output / "report.json").write_text(json.dumps({
                "coverage": {"requested_records": 2, "evaluated_records": 2,
                             "requested_questions": 3, "evaluated_questions": 3,
                             "rejected_records": 0, "truncated_records": truncated},
                "mechanism_checks": {"passed": True},
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


def test_run_persists_configs_and_starts_train_with_fresh_trial(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: _fake_experiment_plan())
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

    monkeypatch.setattr(experiment, "_run_child", check_initial_status)
    result = experiment.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=tmp_path / "study")
    study = tmp_path / "study"
    assert result["status"] == "completed"
    assert (study / "configs" / "seed-0.toml").exists()
    assert calls[0][3] == "train"
    assert not (study / "seed-0" / "configs").exists()
    assert "save_every = 100" in (study / "configs" / "seed-0.toml").read_text()
    assert (study / "status.json").exists()
    status = json.loads((study / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["stage"] == "finished"
    assert status["seed"] == 0 and status["seed_index"] == 0 and status["seed_total"] == 1
    assert status["step"] == 4
    assert status["latest_loss"] == 1.25
    terminal = capsys.readouterr().out
    assert str(study / "result.json") in terminal
    assert '"reports"' not in terminal


def test_next_seed_status_resets_before_its_first_heartbeat(tmp_path, monkeypatch):
    seeds = (2, 4)
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: {
        **_fake_experiment_plan(),
        "trials": [{"seed": seed, "steps": 3144} for seed in seeds],
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
            assert status["seed"] == seed
            assert status["step"] == 0
            assert status["total_steps"] == 3144
            kwargs["on_update"]({"step": None, "total_steps": None, "latest_loss": None}, .01, True)
            status = json.loads(status_path.read_text())
            assert status["seed"] == seed
            assert status["step"] == 0
            assert status["total_steps"] == 3144
            training_children += 1
        return fake(command, log_path=log_path, **kwargs)

    monkeypatch.setattr(experiment, "_run_child", check_seed_status)
    result = experiment.run("configs/gemma3-1b-v7.toml", seeds=seeds, data="data",
                            out=tmp_path / "multi-seed")
    assert result["status"] == "completed"
    assert training_children == 2


def test_run_passes_config_and_unit_temperature_to_every_eval_child(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: _fake_experiment_plan())
    monkeypatch.setattr(experiment, "_run_child", _fake_child_process(calls))
    experiment.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=tmp_path / "study")
    evals = [command for command in calls if command[3] == "eval"]
    assert len(evals) == 3
    assert all(command[command.index("--config") + 1].endswith("configs/seed-0.toml") for command in evals)
    assert all(command[command.index("--temperature") + 1] == "1" for command in evals)


def test_run_requires_record_question_and_truncation_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: _fake_experiment_plan())
    monkeypatch.setattr(experiment, "_run_child", _fake_child_process([], truncated=1))
    result = experiment.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=tmp_path / "study")
    assert result["status"] == "diagnostic"
    assert result["promotion"]["selected_seed"] is None


def test_short_diagnostic_does_not_add_default_snapshot_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: _fake_experiment_plan())
    calls = []
    fake = _fake_child_process(calls)

    def incomplete_train(command, **kwargs):
        if command[3] == "train":
            status = json.loads((kwargs["log_path"].parents[2] / "status.json").read_text())
            assert status["step"] == 0
            assert status["total_steps"] == 1
            assert status["full_training_steps"] == 3144
        result = fake(command, **kwargs)
        if command[3] == "train":
            metrics = Path(command[command.index("--out") + 1]) / "training_metrics.json"
            metrics.write_text(json.dumps({"complete": False}))
        return result

    monkeypatch.setattr(experiment, "_run_child", incomplete_train)
    result = experiment.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data",
                            out=tmp_path / "short-study", max_steps=1)
    seed_config = (tmp_path / "short-study" / "configs" / "seed-0.toml").read_text()
    assert "save_every =" not in seed_config
    assert "--max-steps" in calls[0]
    assert result["status"] == "diagnostic"


def test_save_interval_does_not_change_recipe_fingerprint(tmp_path):
    source = Path("configs/gemma3-1b-v7.toml").read_text()
    configured = tmp_path / "save-every.toml"
    configured.write_text(source.replace("p_none_pair = 0.25", "p_none_pair = 0.25\nsave_every = 7"))
    assert plan("configs/gemma3-1b-v7.toml", seeds=(0,), data="data")["config_sha256"] == \
        plan(configured, seeds=(0,), data="data")["config_sha256"]


def test_run_records_failed_child_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "plan", lambda *args, **kwargs: _fake_experiment_plan())
    monkeypatch.setattr(experiment, "_run_child", _fake_child_process([], returncode=9))
    with pytest.raises(SystemExit) as exc:
        experiment.run("configs/gemma3-1b-v7.toml", seeds=(0,), data="data", out=tmp_path / "study")
    assert exc.value.code == 1
    result = json.loads((tmp_path / "study" / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][0]["stage"] == "train"
    assert result["trials"][0]["exit_code"] == 9
    assert "ValueError: child failed" in result["trials"][0]["output_excerpt"]
    assert (tmp_path / "study" / "seed-0").exists() is False


def test_run_child_streams_combined_output_and_logs_exact_bytes(tmp_path):
    log_path = tmp_path / "stage.log"
    updates = []
    script = "import sys; print('stdout marker', flush=True); print('stderr marker', file=sys.stderr, flush=True)"
    result = experiment._run_child([sys.executable, "-c", script], log_path=log_path,
                                   on_update=lambda *args: updates.append(args),
                                   heartbeat_interval=.05)
    logged = log_path.read_bytes()
    assert b"stdout marker\n" in logged
    assert b"stderr marker\n" in logged
    assert result["returncode"] == 0


def test_run_child_parses_live_training_progress(tmp_path):
    updates = []
    script = "import time; print('step 1/3 loss 2.5', flush=True); time.sleep(1.05); print('step 2/3 loss 1.25', flush=True)"
    result = experiment._run_child([sys.executable, "-c", script], log_path=tmp_path / "train.log",
                                   on_update=lambda progress, elapsed, heartbeat: updates.append(progress.copy()))
    assert result["step"] == 2 and result["latest_loss"] == 1.25
    assert any(update.get("step") == 1 for update in updates)


def test_run_child_reports_heartbeat_while_child_is_quiet(tmp_path):
    updates = []
    script = "import time; time.sleep(.3)"
    experiment._run_child([sys.executable, "-c", script], log_path=tmp_path / "quiet.log",
                          on_update=lambda progress, elapsed, heartbeat: updates.append(heartbeat),
                          heartbeat_interval=.05)
    assert True in updates


@pytest.mark.parametrize(("stage", "expected"), [
    ("train", "loading; still running"),
    ("development", "working; elapsed"),
])
def test_run_child_heartbeat_message_matches_idle_stage(tmp_path, capsys, stage, expected):
    experiment._run_child([sys.executable, "-c", "import time; time.sleep(.2)"],
                          log_path=tmp_path / f"{stage}.log", on_update=lambda *args: None,
                          heartbeat_interval=.04, stage=stage, progress_label=f"seed 2 (1/1) {stage}")
    output = capsys.readouterr().out
    assert expected in output
    assert "loadingstill running" not in output


def test_run_child_heartbeat_shows_last_training_step(tmp_path, capsys):
    script = "import time; print('step 1/3 loss 2.5', flush=True); time.sleep(.2)"
    experiment._run_child([sys.executable, "-c", script], log_path=tmp_path / "progress.log",
                          on_update=lambda *args: None, heartbeat_interval=.04, stage="train",
                          progress_label="seed 2 (1/1) train")
    assert "step 1/3; still running" in capsys.readouterr().out


def test_run_child_failure_keeps_traceback_and_exit_code(tmp_path):
    log_path = tmp_path / "failure.log"
    script = "raise RuntimeError('diagnostic failure')"
    result = experiment._run_child([sys.executable, "-c", script], log_path=log_path,
                                   on_update=lambda *args: None)
    assert result["returncode"] != 0
    assert "Traceback (most recent call last)" in result["output_excerpt"]
    assert "RuntimeError: diagnostic failure" in log_path.read_text()


def test_run_child_interrupt_terminates_and_reaps_child(tmp_path):
    log_path = tmp_path / "interrupt.log"
    script = "import os,time; print(os.getpid(), flush=True); time.sleep(30)"

    def interrupt_on_heartbeat(progress, elapsed, heartbeat):
        if heartbeat:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        experiment._run_child([sys.executable, "-c", script], log_path=log_path,
                              on_update=interrupt_on_heartbeat, heartbeat_interval=.05)
    pid = int(log_path.read_text().splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_full_study_save_interval_preserves_explicit_setting(tmp_path, monkeypatch):
    config = Path("configs/gemma3-1b-v7.toml").read_text()
    config = config.replace("p_none_pair = 0.25", "p_none_pair = 0.25\nsave_every = 17")
    config_path = tmp_path / "explicit.toml"
    config_path.write_text(config)
    directory = tmp_path / "configs"
    directory.mkdir()
    explicit = experiment._toml_for_seed(config_path, 1, directory, save_every=100)
    assert "save_every = 17" in explicit.read_text()
    assert "save_every = 100" not in explicit.read_text()
    short = experiment._toml_for_seed(Path("configs/gemma3-1b-v7.toml"), 1, directory)
    assert "save_every =" not in short.read_text()
