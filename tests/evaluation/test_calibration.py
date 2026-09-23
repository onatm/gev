import json

import pytest

from gev.evaluation.calibration import calibrate


def _rows(count=10):
    return [{"id": f"r{i}", "group": f"g{i}", "question": "q", "source": "s",
             "task": "task", "type": "choice", "variant": "clean", "keys": ["a", "b"],
             "label": i % 2, "p": [.8, .2] if i % 2 == 0 else [.2, .8],
             "logits": [2.0, 0.0] if i % 2 == 0 else [0.0, 2.0],
             "inference_temperature": 1.0} for i in range(count)]


def _run(path, split="calibration"):
    path.mkdir()
    rows = _rows()
    (path / "rows.json").write_text(json.dumps(rows))
    (path / "report.json").write_text(json.dumps({
        "coverage": {"rejected_records": 0, "evaluated_records": len(rows),
                     "requested_records": len(rows)},
        "provenance": {"suite": "decision-v7", "split": split,
                       "suite_sha256": "suite", "source_sha256": "source"},
    }))
    return path


def test_screening_calibration_writes_report_and_uses_fixed_search_grid(tmp_path):
    run = _run(tmp_path / "run")
    output = tmp_path / "calibration.json"

    result = calibrate(run, protocol="kev-screening", out=output)

    assert result["points"] == 81
    assert result["checkpoint_updated"] is False
    assert json.loads(output.read_text())["provenance"]["raw_rows"] is True


def test_screening_calibration_requires_calibration_split(tmp_path):
    with pytest.raises(ValueError, match="requires split=calibration"):
        calibrate(_run(tmp_path / "run", "development"), protocol="kev-screening")


def test_calibration_rejects_transfer_provenance(tmp_path):
    run = _run(tmp_path / "run")
    report = json.loads((run / "report.json").read_text())
    report["provenance"]["suite"] = "transfer-v4"
    (run / "report.json").write_text(json.dumps(report))

    with pytest.raises(ValueError, match="transfer"):
        calibrate(run)


def test_calibration_rejects_rows_already_temperature_scaled(tmp_path):
    run = _run(tmp_path / "run")
    rows = json.loads((run / "rows.json").read_text())
    rows[0]["inference_temperature"] = 2.0
    (run / "rows.json").write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="raw rows"):
        calibrate(run)
