import json
import math
from pathlib import Path

import numpy as np
import pytest

from gev import metrics


def row(logits=(1.0, 0.0), label=0, **changes):
    value = {"logits": list(logits), "label": label, "type": "choice", "variant": "clean", "source": "s",
             "task": "t", "id": "r", "question": "q", "keys": ["a", "b"], "group": "g"}
    value.update(changes)
    return value


def test_matches_pinned_kev_fixture():
    fixture = json.loads((Path(__file__).parent / "kev_metrics_reference.json").read_text())
    rows = [{**r, "logits": [math.log(p) for p in r.pop("p")]} for r in fixture["rows"]]
    got = metrics.metrics(rows)
    for key, expected in fixture["expected"].items():
        assert got[key] == pytest.approx(expected)


def test_nll_is_stable_for_extreme_logits():
    assert np.isfinite(metrics.nll(row(logits=(-1000.0, 1000.0))))


def test_temperature_fit_recovers_overconfidence():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        label = int(rng.integers(2))
        margin = 6.0 if rng.random() < 0.7 else -6.0  # 70% correct but ~100% confident
        logits = [margin, 0.0] if label == 0 else [0.0, margin]
        rows.append(row(logits=logits, label=label, id=str(i), group=str(i)))
    temperature = metrics.fit_temperature(rows)
    assert temperature > 2
    assert metrics.metrics(rows, temperature)["nll"] < metrics.metrics(rows)["nll"]
    assert metrics.metrics(rows, temperature)["acc"] == metrics.metrics(rows)["acc"]


def test_paired_flip_counts_contrastive_changes():
    a = row(logits=(2.0, 0.0), label=0, pair_id="p", sibling="a")
    b = row(logits=(0.0, 2.0), label=1, pair_id="p", sibling="b", id="r2")
    assert metrics.paired_flip([a, b]) == {"pairs": 1, "flip_rate": 1.0, "both_correct_rate": 1.0}


def test_paired_bootstrap_is_zero_for_identical_rows():
    rows = [row(logits=(float(i % 3), 0.0), label=i % 2, id=str(i), group=str(i)) for i in range(20)]
    result = metrics.paired_bootstrap(rows, rows, metric="nll", samples=50)
    assert result["delta"] == 0 and result["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        metrics.paired_bootstrap(rows, rows[:-1])
