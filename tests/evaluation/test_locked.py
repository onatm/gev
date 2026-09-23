"""Protocol tests use authored rows only; they never fetch or load held-out data."""
from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path

import pytest

from gev.evaluation import locked


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    manifests = {
        "decision-v7": ("manifest-v7", {"files": {"test.jsonl": {"sha256": "data-v7"}}}),
        "transfer-v4": ("manifest-v4", {"files": {"test.jsonl": {"sha256": "data-v4"}}}),
    }
    monkeypatch.setattr(locked, "_manifest_info", lambda suite: manifests[suite])
    study = tmp_path / "locked-result.json"
    study.write_text(json.dumps({"promotion": {"selected_seed": 0}, "trials": []}))
    meta = {"lineage": {"source_sha256": locked.V7_TRAIN_SHA256,
                         "manifest_sha256": locked.V7_MANIFEST_SHA256},
            "training": {"metrics": {"complete": True, "logical_steps": 3144,
                                       "processed_records": 25152, "source_count": 12576},
                         "config": {"experiment_id": "gemma3-1b-v7", "training": {
                             "epochs": 2, "logical_batch": 8, "context_length": 384, "seed": 0}}},
            "calibration": {"temperature": 1.0}}
    selection = {
        "protocol": "locked-eval-v7", "model_fingerprint": "weights-a", "temperature": 1.0,
        "study": {"path": str(study), "sha256": locked._sha(study)},
        "checkpoint": {"path": str(tmp_path / "checkpoint")},
        "suites": {suite: {"manifest_sha256": value[0], "data_sha256": value[1]["files"]["test.jsonl"]["sha256"]}
                   for suite, value in manifests.items()},
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    return tmp_path, selection_path, meta


def _record(identifier="r1"):
    return {"state": "state", "_meta": {"id": identifier, "source": "known", "group_id": identifier},
            "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "task"}}}


def _predictor(calls, temperature=1.0):
    def predict(_record):
        calls.append(1)
        return {"probabilities": {"q": {"a": .75, "b": .25}}, "logits": {"q": {"a": 1., "b": 0.}},
                "inference_temperature": temperature}
    return predict


def _run(protocol, suites=("decision-v7",)):
    root, selection, meta = protocol
    calls = []
    result = locked.run_locked(selection=selection, suites=suites, data_root=root / "unused",
                               output=root / "out", ledger=root / "ledger.jsonl",
                               load_test=lambda _suite: ([_record()], _predictor(calls)),
                               checkpoint_meta=meta, model_fingerprint="weights-a")
    return root, result, calls


def test_one_suite_writes_separate_raw_and_calibrated_report(protocol):
    root, result, calls = _run(protocol)
    report = json.loads((root / "out/decision-v7/report.json").read_text())
    assert result["status"] == "complete"
    assert len(calls) == 1
    assert report["temperature"] == 1.0
    assert report["raw_clean"] == report["calibrated_clean"]
    assert report["rows_sha256"] == locked._sha(root / "out/decision-v7/rows.json")


def test_combined_reservation_blocks_later_individual_calls(protocol):
    root, _, _ = _run(protocol, suites=("decision-v7", "transfer-v4"))
    selection = root / "selection.json"
    with pytest.raises(ValueError, match="already reserved"):
        locked.run_locked(selection=selection, suites=("transfer-v4",), data_root=root,
                          output=root / "out2", ledger=root / "ledger.jsonl",
                          load_test=lambda _: ([_record()], _predictor([])), checkpoint_meta=protocol[2],
                          model_fingerprint="weights-a")


def test_reversed_order_is_still_duplicate(protocol):
    root, _, _ = _run(protocol, suites=("decision-v7", "transfer-v4"))
    with pytest.raises(ValueError, match="already reserved"):
        locked.run_locked(selection=root / "selection.json", suites=("transfer-v4", "decision-v7"),
                          data_root=root, output=root / "out2", ledger=root / "ledger.jsonl",
                          load_test=lambda _: ([_record()], _predictor([])), checkpoint_meta=protocol[2],
                          model_fingerprint="weights-a")


def test_duplicate_and_empty_suite_requests_rejected_before_loader(protocol):
    root, selection, meta = protocol
    loader = lambda _: pytest.fail("loader must not run")
    with pytest.raises(ValueError, match="unique"):
        locked.run_locked(selection=selection, suites=("decision-v7", "decision-v7"), data_root=root,
                          output=root / "out", ledger=root / "ledger.jsonl", load_test=loader,
                          checkpoint_meta=meta, model_fingerprint="weights-a")
    with pytest.raises(ValueError, match="non-empty"):
        locked.run_locked(selection=selection, suites=(), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=loader, checkpoint_meta=meta,
                          model_fingerprint="weights-a")


def test_wrong_fingerprint_and_incomplete_lineage_are_preflight_failures(protocol):
    root, selection, meta = protocol
    with pytest.raises(ValueError, match="fingerprint"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail(),
                          checkpoint_meta=meta, model_fingerprint="wrong")
    incomplete = {**meta, "training": {**meta["training"], "metrics": {
        **meta["training"]["metrics"], "logical_steps": 32}}}
    with pytest.raises(ValueError, match="lineage"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail(),
                          checkpoint_meta=incomplete, model_fingerprint="weights-a")


def test_existing_output_does_not_reserve(protocol):
    root, selection, meta = protocol
    (root / "out").mkdir()
    with pytest.raises(FileExistsError):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail(),
                          checkpoint_meta=meta, model_fingerprint="weights-a")
    assert not (root / "ledger.jsonl").exists()


def test_callback_failure_is_recorded_and_cannot_retry(protocol):
    root, selection, meta = protocol
    with pytest.raises(RuntimeError):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
                          checkpoint_meta=meta, model_fingerprint="weights-a")
    statuses = [json.loads(line)["status"] for line in (root / "ledger.jsonl").read_text().splitlines()]
    assert statuses == ["started", "failed"]
    with pytest.raises(ValueError, match="already reserved"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out2",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail(),
                          checkpoint_meta=meta, model_fingerprint="weights-a")


def test_nonraw_predictor_is_rejected_without_second_call(protocol):
    root, selection, meta = protocol
    calls = []
    with pytest.raises(ValueError, match="raw temperature"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: ([_record()], _predictor(calls, 2.0)),
                          checkpoint_meta=meta, model_fingerprint="weights-a")
    assert len(calls) == 1


def test_temperature_fit_and_temperature_change_cannot_reuse_key(protocol):
    root, selection, meta = protocol
    value = json.loads(selection.read_text())
    value["temperature"] = 2.0
    value["temperature_fit"] = {"suite": "decision-v7", "split": "development", "temperature": 2.0}
    selection.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="temperature"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail(),
                          checkpoint_meta=meta, model_fingerprint="weights-a")


def test_concurrent_reservation_has_one_winner(protocol):
    root, selection, meta = protocol
    barrier = threading.Barrier(2)
    results = []
    def invoke(index):
        try:
            barrier.wait()
            results.append(locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root,
                                             output=root / f"out-{index}", ledger=root / "ledger.jsonl",
                                             load_test=lambda _: ([_record()], _predictor([])),
                                             checkpoint_meta=meta, model_fingerprint="weights-a"))
        except Exception as exc:
            results.append(exc)
    threads = [threading.Thread(target=invoke, args=(i,)) for i in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sum(isinstance(item, dict) for item in results) == 1
    assert sum(isinstance(item, ValueError) for item in results) == 1


def test_no_verified_identity_means_no_reservation(protocol):
    root, selection, _ = protocol
    with pytest.raises(ValueError, match="actual checkpoint"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root, output=root / "out",
                          ledger=root / "ledger.jsonl", load_test=lambda _: pytest.fail())


def test_normal_split_loader_refuses_test(tmp_path):
    from gev.data.access import load_verified_split
    from gev.data.suites import SuiteError
    with pytest.raises(SuiteError, match="locked"):
        load_verified_split(str(tmp_path), "decision-v7", "test", allow_test=True)


def _checkpoint_fixture(root, *, seed=0, smoke=False):
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
    (checkpoint / "pointer.safetensors").write_bytes(b"fixture-pointer")
    meta = {"format": "gev.inference-checkpoint", "version": 1,
        "identity": {"family": "gemma3_text", "backend": "torch",
            "base": {"name": "fixture", "revision": "f" * 40, "type": "gemma3_text"},
            "tokenizer": {"revision": "f" * 40},
            "markers": {"ids": {"state": 1}, "strings": {"state": "s"}, "bos": False},
            "protocol": {"id": "kev-decision-v7", "version": 1},
            "recipe": {"scientific_recipe": {}, "sha256": hashlib.sha256(b"{}").hexdigest()},
            "model_contract": {"head_width": 256, "lora": {}, "representation_version": 1}},
        "lineage": {"source_sha256": locked.V7_TRAIN_SHA256,
                    "manifest_sha256": locked.V7_MANIFEST_SHA256},
        "training": {"metrics": {"complete": True, "source_count": 12576,
                                  "processed_records": 25152, "logical_steps": 3144},
                     "config": {"experiment_id": "gemma3-1b-v7", "training": {
                         "seed": seed, "epochs": 2, "logical_batch": 8,
                         "context_length": 384, "p_none_pair": .25}}},
        "execution": {}, "calibration": {"temperature": 1.0},
        "tensors": {name: {"filename": filename, "sha256": locked._sha(checkpoint / filename), "shapes": {"x": [1]}}
                    for name, filename in (("adapter", "adapter_model.safetensors"), ("pointer", "pointer.safetensors"))}}
    if smoke:
        meta["lineage"]["continuation"] = {"diagnostic_smoke_init": True}
    (checkpoint / "manifest.json").write_text(json.dumps(meta))
    return checkpoint, meta


def _study_for(path, trial_reports=True, seed=0):
    coverage = {"rejected_records": 0, "evaluated_records": 1, "requested_records": 1}
    report = {"coverage": coverage, "mechanism_checks": {"passed": True}}
    return {"promotion": {"selected_seed": seed}, "trials": [{
        "seed": seed, "path": str(path), "eligible": True, "completed": True,
        "reports": {name: report for name in ("calibration", "development", "transfer")} if trial_reports else {},
    }]}


def test_register_consumes_actual_promotion_trial_shape_and_run_locked_matches(protocol, monkeypatch):
    root, _, _ = protocol
    trial = root / "trial"
    checkpoint, _meta = _checkpoint_fixture(trial)
    study = root / "actual-study.json"
    study.write_text(json.dumps(_study_for(trial)))
    selection_path = root / "registered.json"
    value = locked.register_candidate(run=trial, study=study, out=selection_path)
    assert value["study"]["path"] == str(study.resolve())
    assert value["checkpoint"]["path"] == str(checkpoint)
    result = locked.run_locked(selection=selection_path, suites=("decision-v7",), data_root=root,
                               output=root / "registered-out", ledger=root / "registered-ledger.jsonl",
                               load_test=lambda _: ([_record()], _predictor([])), run=trial)
    assert result["status"] == "complete"


def test_register_rejects_unrelated_checkpoint_before_any_test_path(protocol):
    root, _, _ = protocol
    selected = root / "selected"
    unrelated = root / "unrelated"
    _checkpoint_fixture(selected)
    _checkpoint_fixture(unrelated)
    study = root / "actual-study.json"
    study.write_text(json.dumps(_study_for(selected)))
    with pytest.raises(ValueError, match="selected trial"):
        locked.register_candidate(run=unrelated, study=study, out=root / "unrelated-selection.json")


def test_nested_smoke_continuation_is_rejected_before_reservation(protocol):
    root, selection, _ = protocol
    meta = {**protocol[2], "lineage": {**protocol[2]["lineage"],
                                         "continuation": {"diagnostic_smoke_init": True}}}
    with pytest.raises(ValueError, match="smoke|lineage"):
        locked.run_locked(selection=selection, suites=("decision-v7",), data_root=root,
                          output=root / "smoke-out", ledger=root / "smoke-ledger.jsonl",
                          load_test=lambda _: pytest.fail("test loader must not run"),
                          checkpoint_meta=meta, model_fingerprint="weights-a")
    assert not (root / "smoke-ledger.jsonl").exists()
