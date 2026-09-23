import os
import sys

import pytest

from gev.study.process import run_child


def test_child_streams_combined_output_and_logs_exact_bytes(tmp_path):
    log_path = tmp_path / "stage.log"
    script = "import sys; print('stdout marker', flush=True); print('stderr marker', file=sys.stderr, flush=True)"

    result = run_child([sys.executable, "-c", script], log_path=log_path,
                       on_update=lambda *args: None, heartbeat_interval=.05)

    assert b"stdout marker\n" in log_path.read_bytes()
    assert b"stderr marker\n" in log_path.read_bytes()
    assert result["returncode"] == 0


def test_child_parses_and_reports_live_training_progress(tmp_path):
    updates = []
    script = "import time; print('step 1/3 loss 2.5', flush=True); time.sleep(1.05); print('step 2/3 loss 1.25', flush=True)"

    result = run_child([sys.executable, "-c", script], log_path=tmp_path / "train.log",
                       on_update=lambda progress, elapsed, heartbeat: updates.append(progress.copy()))

    assert result["step"] == 2 and result["latest_loss"] == 1.25
    assert any(update.get("step") == 1 for update in updates)


def test_quiet_child_emits_heartbeat(tmp_path):
    updates = []
    run_child([sys.executable, "-c", "import time; time.sleep(.3)"],
              log_path=tmp_path / "quiet.log",
              on_update=lambda progress, elapsed, heartbeat: updates.append(heartbeat),
              heartbeat_interval=.05)

    assert True in updates


@pytest.mark.parametrize(("stage", "expected"), [
    ("train", "loading; still running"),
    ("development", "working; elapsed"),
])
def test_heartbeat_text_describes_idle_child_stage(tmp_path, capsys, stage, expected):
    run_child([sys.executable, "-c", "import time; time.sleep(.2)"],
              log_path=tmp_path / f"{stage}.log", on_update=lambda *args: None,
              heartbeat_interval=.04, stage=stage,
              progress_label=f"seed 2 (1/1) {stage}")

    output = capsys.readouterr().out
    assert expected in output
    assert "loadingstill running" not in output


def test_training_heartbeat_includes_latest_progress(tmp_path, capsys):
    script = "import time; print('step 1/3 loss 2.5', flush=True); time.sleep(.2)"
    run_child([sys.executable, "-c", script], log_path=tmp_path / "progress.log",
              on_update=lambda *args: None, heartbeat_interval=.04, stage="train",
              progress_label="seed 2 (1/1) train")

    assert "step 1/3; still running" in capsys.readouterr().out


def test_failure_keeps_traceback_in_log_and_excerpt(tmp_path):
    log_path = tmp_path / "failure.log"
    result = run_child([sys.executable, "-c", "raise RuntimeError('diagnostic failure')"],
                       log_path=log_path, on_update=lambda *args: None)

    assert result["returncode"] != 0
    assert "Traceback (most recent call last)" in result["output_excerpt"]
    assert "RuntimeError: diagnostic failure" in log_path.read_text()


def test_interrupt_terminates_and_reaps_child(tmp_path):
    log_path = tmp_path / "interrupt.log"
    script = "import os,time; print(os.getpid(), flush=True); time.sleep(30)"

    def interrupt_on_heartbeat(progress, elapsed, heartbeat):
        if heartbeat:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_child([sys.executable, "-c", script], log_path=log_path,
                  on_update=interrupt_on_heartbeat, heartbeat_interval=.5)

    pid = int(log_path.read_text().splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
