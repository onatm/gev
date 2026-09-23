import json
from copy import deepcopy

import numpy as np
import pytest

from gev.evaluation.benchmark import evaluate_records, prediction_rows, validate_distribution
from gev.evaluation.mechanism import paired_flip
from gev.evaluation.predictors import LocalPredictor
from gev.evaluation.metrics import (cross_validated_temperature, fit_temperature, grouped_folds, metrics, nll_at_temperature,
                                    paired_bootstrap, probabilities_at_temperature, raw_row, select_threshold)
from gev.models.gemma import build_tiny_model
from gev.tokenization import MarkerMap


FIXTURE = "tests/fixtures/kev_metrics_reference.json"


def row(**changes):
    value = {"p": [0.8, 0.2], "label": 0, "type": "choice", "variant": "clean", "source": "s", "task": "t",
             "id": "r", "question": "q", "keys": ["a", "b"], "group": "g"}
    value.update(changes)
    return value


def record(identifier="r", variant="clean"):
    return {"state": {"case": "x"}, "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "task"}},
            "_meta": {"id": identifier, "group_id": identifier, "source": "source", "variant": variant}}


def test_pinned_fixture_matches_independent_values():
    data = json.load(open(FIXTURE))
    got = metrics(data["rows"])
    for key, expected in data["expected"].items(): assert got[key] == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["probability", "logit"])
def test_binary_and_multiclass_nll_are_stable_for_zero_and_logits(kind):
    values = row(p=[0.0, 1.0]) if kind == "probability" else row(p=[0.0, 1.0], logits=[-1000.0, 1000.0])
    assert np.isfinite(nll_at_temperature(values))
    multi = row(p=[0.2, 0.3, 0.5], keys=["a", "b", "c"], label=2)
    assert probabilities_at_temperature(multi, 2).sum() == pytest.approx(1)


def test_metrics_exposes_selective_top_bins_and_score_metrics():
    scored = row(p=[0.1, 0.2, 0.7], keys=["0", "1", "2"], label=2, type="score")
    result = metrics([scored, row()])
    assert set(result["top_bins"]) == {"0.9", "0.95", "0.99"}
    assert set(result["selective"]) == {"0.5", "0.8"}
    assert "ranked_probability_score" in result


def test_equal_confidence_is_selected_as_a_whole_group():
    threshold = select_threshold(np.array([.8, .8, .2]), np.array([True, True, False]), 0)
    assert threshold == .8


def test_temperature_validation_and_precalibrated_fit_guard():
    with pytest.raises(ValueError): probabilities_at_temperature(row(), 0)
    with pytest.raises(ValueError): fit_temperature([row(inference_temperature=2, logits=[1, 0])])


def test_macro_and_micro_calibration_can_differ():
    rows = [row(id="a", task="large", p=[.6, .4], logits=[.4, 0]), row(id="b", task="large", p=[.6, .4], logits=[.4, 0]),
            row(id="c", task="small", p=[.1, .9], label=0, logits=[0, .2])]
    assert fit_temperature(rows, "macro", 11) != fit_temperature(rows, "micro", 11)


def test_grouped_folds_keep_siblings_together_and_cover_folds():
    rows = [row(id=f"{i}-{j}", group=f"g{i}", question=str(j), source="s") for i in range(6) for j in range(2)]
    folds = grouped_folds(rows, 3, 7)
    assert len(set(folds)) == 3
    assert all(len({folds[i * 2], folds[i * 2 + 1]}) == 1 for i in range(6))


def test_paired_bootstrap_rejects_id_mismatch():
    with pytest.raises(ValueError): paired_bootstrap([row(id="a")], [row(id="b")], samples=2)


def test_raw_row_restores_recorded_inference_temperature():
    calibrated = row(p=[.8, .2], logits=[2, 0], inference_temperature=2)
    restored = raw_row(calibrated)
    assert restored["inference_temperature"] == 1
    assert restored["p"][0] > .8


def test_validate_distribution_requires_exact_keys_and_finite_values():
    with pytest.raises(ValueError): validate_distribution({"a": .5}, ["a", "b"])
    with pytest.raises(ValueError): validate_distribution({"a": float("nan"), "b": 1}, ["a", "b"])
    p, total = validate_distribution({"a": .6, "b": .4}, ["a", "b"])
    assert total == pytest.approx(1) and p.sum() == pytest.approx(1)


def test_prediction_rows_validates_question_and_logit_keys():
    rec = record(); pred = {"probabilities": {"q": {"a": .6, "b": .4}}, "logits": {"q": {"a": 1, "b": 0}}}
    assert prediction_rows(rec, pred)[0]["logits"] == [1, 0]
    with pytest.raises(ValueError): prediction_rows(rec, {"probabilities": {"other": {"a": 1, "b": 0}}})


def test_evaluate_writes_strict_json_and_failure_record(tmp_path):
    def predictor(_): return {"probabilities": {"q": {"a": .7, "b": .3}}, "latency_ms": 4}
    report, rows = evaluate_records([record()], predictor, tmp_path / "ok")
    assert report["coverage"]["evaluated_records"] == 1 and rows[0]["p"] == [.7, .3]
    assert json.loads((tmp_path / "ok" / "predictions.jsonl").read_text())["request_sha256"]


def test_evaluate_aborts_with_record_id_on_invalid_prediction(tmp_path):
    def predictor(_): return {"probabilities": {"q": {"a": float("nan"), "b": 0}}}
    with pytest.raises(ValueError): evaluate_records([record("bad")], predictor, tmp_path / "bad")
    assert json.loads((tmp_path / "bad" / "failure.json").read_text())["record_id"] == "bad"


def test_paired_flip_reports_contrastive_change():
    a = row(id="a", pair_id="p", sibling="a", p=[.9, .1], label=0)
    b = row(id="b", pair_id="p", sibling="b", p=[.1, .9], label=1)
    assert paired_flip([a, b])["flip_rate"] == 1


def test_cross_validated_temperature_returns_provenance():
    rows = [row(id=str(i), group=str(i), logits=[1, 0], p=[.7, .3]) for i in range(4)]
    result = cross_validated_temperature(rows, folds=2, samples=2, points=3)
    assert result["folds"] == 2 and result["samples"] == 2 and len(result["temperatures"]) == 2


@pytest.mark.parametrize("temperature", [float("nan"), 0.0, -1.0, float("inf")])
def test_local_predictor_rejects_nonfinite_or_nonpositive_temperature(temperature):
    with pytest.raises(ValueError): LocalPredictor(object(), object(), object(), state_cap=10, branch_cap=10, packed_cap=10, temperature=temperature)


def test_request_hash_uses_public_kev_request_shape():
    from gev.evaluation.benchmark import _request_hash
    first = record(); second = deepcopy(first)
    second["questions"]["q"]["label"] = "b"; second["_meta"]["id"] = "different"
    assert _request_hash(first) == _request_hash(second)


def test_request_hash_preserves_criteria_order():
    from gev.evaluation.benchmark import _request_hash
    first = record(); second = deepcopy(first)
    second["questions"]["q"]["criteria"] = {"b": "B", "a": "A"}
    assert _request_hash(first) != _request_hash(second)


def test_local_predictor_rows_and_evaluation_apply_temperature_once(tmp_path):
    class SyntheticTokenizer:
        all_special_tokens = ()
        def __call__(self, text, add_special_tokens=False):
            class Encoded: pass
            encoded = Encoded(); encoded.input_ids = [10 + (ord(char) % 200) for char in str(text)][:20] or [10]
            return encoded

    markers = MarkerMap({"state": 220, "question": 221, "option_start": 222, "option_end": 223, "decide": 224},
                        {"state": "<|state|>", "question": "<|question|>", "option_start": "<|option_start|>",
                         "option_end": "<|option_end|>", "decide": "<|decide|>"}, "synthetic")
    questions = {
        "choice": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "choice"},
        "noul": {"type": "noul", "instructions": "is it", "label": True, "src": "noul"},
        "score": {"type": "score", "instructions": "score", "criteria": ["low", "high"], "label": 1, "src": "score"},
    }
    rec = {"state": "synthetic state", "questions": questions,
           "_meta": {"id": "synthetic", "group_id": "synthetic", "source": "synthetic", "variant": "clean"}}
    model = build_tiny_model(layers=6, hidden_size=32)
    predictor = LocalPredictor(model, SyntheticTokenizer(), markers, state_cap=64, branch_cap=128, packed_cap=256, temperature=2.0)
    prediction = predictor(rec)
    assert model.head.temperature == 2.0
    for qid, scaled in prediction["logits"].items():
        assert prediction["raw_logits"][qid].keys() == scaled.keys()
        assert np.allclose(list(prediction["raw_logits"][qid].values()), np.asarray(list(scaled.values())) * 2)
        assert sum(prediction["probabilities"][qid].values()) == pytest.approx(1)
    rows = prediction_rows(rec, prediction)
    assert all("raw_logits" in item and len(item["raw_logits"]) == len(item["keys"]) for item in rows)
    restored = raw_row(rows[0])
    assert restored["inference_temperature"] == 1 and np.asarray(restored["logits"]).shape == (2,)
    report, evaluated = evaluate_records([rec], predictor, tmp_path / "integration")
    assert report["coverage"]["evaluated_questions"] == 3 and len(evaluated) == 3
    assert report["nll_provenance"]["raw_logits"]
    with pytest.raises(ValueError, match="applied twice"):
        evaluate_records([rec], predictor, tmp_path / "double", temperature=2.0)
