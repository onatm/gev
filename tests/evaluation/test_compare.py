import json

import pytest

from gev.evaluation.compare import compare


def _rows(count=10):
    return [{"id": f"r{i}", "group": f"g{i}", "question": "q", "source": "s",
             "task": "task", "type": "choice", "variant": "clean", "keys": ["a", "b"],
             "label": i % 2, "p": [.8, .2] if i % 2 == 0 else [.2, .8],
             "logits": [2.0, 0.0] if i % 2 == 0 else [0.0, 2.0],
             "inference_temperature": 1.0} for i in range(count)]


def _run(path, split="development"):
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


def test_self_comparison_has_zero_paired_confidence_interval(tmp_path):
    run = _run(tmp_path / "run")

    result = compare(run, run, samples=20)

    assert result["paired"]["nll"]["ci95"] == [0.0, 0.0]


def test_comparison_rejects_source_identity_mismatch(tmp_path):
    candidate = _run(tmp_path / "candidate")
    reference = _run(tmp_path / "reference")
    report = json.loads((reference / "report.json").read_text())
    report["provenance"]["source_sha256"] = "other"
    (reference / "report.json").write_text(json.dumps(report))

    with pytest.raises(ValueError, match="identity mismatch"):
        compare(candidate, reference)


def test_comparison_requires_complete_row_coverage(tmp_path):
    run = _run(tmp_path / "run")
    report = json.loads((run / "report.json").read_text())
    report["coverage"]["rejected_records"] = 1
    (run / "report.json").write_text(json.dumps(report))

    with pytest.raises(ValueError, match="incomplete"):
        compare(run, run)


def test_calibrated_rows_require_explicit_comparison_mode(tmp_path):
    candidate = _run(tmp_path / "candidate")
    reference = _run(tmp_path / "reference")
    rows = json.loads((candidate / "rows.json").read_text())
    rows[0]["inference_temperature"] = 2.0
    (candidate / "rows.json").write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="raw comparison"):
        compare(candidate, reference)
