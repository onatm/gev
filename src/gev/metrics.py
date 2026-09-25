"""Scoring of evaluation rows, adapted from Kev ``metrics.py``.

Source: https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/metrics.py

A row is one question: ``{"logits": raw T=1 logits, "label": int, "type", "task",
"source", "group", "id", "question", "variant", ...}``. Temperature is applied here,
never baked into saved rows.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def probabilities(row: dict, temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(row["logits"], dtype=float) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def nll(row: dict, temperature: float = 1.0) -> float:
    z = np.asarray(row["logits"], dtype=float) / temperature
    z = z - z.max()
    return float(np.log(np.exp(z).sum()) - z[row["label"]])


def ece(conf, correct, bins: int = 10) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def _risk_curve(confidence, correct):
    order = np.argsort(-confidence, kind="stable")
    confidence, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(confidence[1:] != confidence[:-1]), len(order) - 1]
    return ends + 1, errors[ends]


def coverage_at_error(confidence, correct, budget: float) -> float:
    """Largest accepted fraction (whole confidence groups) whose error rate is within budget."""
    accepted, errors = _risk_curve(np.asarray(confidence, float), np.asarray(correct, bool))
    ok = np.flatnonzero(errors <= budget * accepted)
    return float(accepted[ok[-1]] / accepted[-1]) if len(ok) else 0.0


def aurc(confidence, correct) -> float:
    accepted, errors = _risk_curve(np.asarray(confidence, float), np.asarray(correct, bool))
    return float(np.sum(np.diff(np.r_[0, accepted]) * errors / accepted) / accepted[-1])


def metrics(rows: list[dict], temperature: float = 1.0) -> dict:
    if not rows:
        raise ValueError("cannot score an empty population")
    nlls, acc, conf, brier, mae, rps = [], [], [], [], [], []
    for row in rows:
        p, y = probabilities(row, temperature), row["label"]
        target = np.eye(len(p))[y]
        nlls.append(nll(row, temperature)); acc.append(bool(p.argmax() == y))
        conf.append(float(p.max())); brier.append(float(((p - target) ** 2).sum()))
        if row["type"] == "score":
            mae.append(abs(float(p @ np.arange(len(p))) - y))
            rps.append(float(((p.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    confidence, correct = np.asarray(conf), np.asarray(acc, dtype=bool)
    high = confidence >= 0.9
    result = {"n": len(rows), "acc": float(correct.mean()), "nll": float(np.mean(nlls)),
              "brier": float(np.mean(brier)), "ece": ece(confidence, correct),
              "mean_conf": float(confidence.mean()),
              "confident_error_rate": float(np.mean(high & ~correct)),
              "coverage_at_0_9": float(high.mean()),
              "error_rate_at_0_9": float((~correct[high]).mean()) if high.any() else None,
              "coverage_at_5pct_error": coverage_at_error(confidence, correct, 0.05),
              "coverage_at_1pct_error": coverage_at_error(confidence, correct, 0.01),
              "aurc": aurc(confidence, correct)}
    if mae:
        result.update(score_mae=float(np.mean(mae)), ranked_probability_score=float(np.mean(rps)))
    return result


def grouped_metrics(rows: list[dict], key: str, temperature: float = 1.0) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {str(name): metrics(group, temperature) for name, group in sorted(groups.items(), key=lambda x: str(x[0]))}


def scored_rows(rows: list[dict]) -> list[dict]:
    """Clean, knowable questions: the population headline metrics are computed on."""
    return [row for row in rows if row["variant"] == "clean" and row["source"] != "unknowable"]


def fit_temperature(rows: list[dict], points: int = 121) -> float:
    """Minimize mean NLL over a log grid on [0.25, 4] (Kev's release fit)."""
    clean = scored_rows(rows)
    if not clean:
        raise ValueError("cannot fit a temperature without clean labelled rows")
    candidates = np.exp(np.linspace(np.log(.25), np.log(4), points))
    losses = [np.mean([nll(row, float(t)) for row in clean]) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def paired_flip(rows: list[dict]) -> dict | None:
    """Contrastive pairs whose answer changes: does the prediction follow?"""
    by_pair: dict = {}
    for row in rows:
        if row.get("pair_id"):
            pair = by_pair.setdefault((row["pair_id"], row["question"]), {})
            if row.get("sibling") in pair:
                raise ValueError("duplicate contrastive sibling")
            pair[row.get("sibling")] = row
    if not by_pair:
        return None
    if any(set(pair) != {"a", "b"} for pair in by_pair.values()):
        raise ValueError("incomplete contrastive pair")
    predicted = lambda row: int(np.argmax(row["logits"]))
    truth = lambda row: row["keys"][row["label"]]
    answer = lambda row: row["keys"][predicted(row)]
    relevant = [p for p in by_pair.values() if truth(p["a"]) != truth(p["b"])]
    invariant = [p for p in by_pair.values() if truth(p["a"]) == truth(p["b"])]
    both = lambda pairs: (sum(all(answer(r) == truth(r) for r in p.values()) for p in pairs) / len(pairs)
                          if pairs else None)
    result = {"pairs": len(relevant),
              "flip_rate": (sum(answer(p["a"]) != answer(p["b"]) for p in relevant) / len(relevant)
                            if relevant else None),
              "both_correct_rate": both(relevant)}
    if invariant:
        result.update(invariant_pairs=len(invariant),
                      invariance_rate=sum(answer(p["a"]) == answer(p["b"]) for p in invariant) / len(invariant),
                      invariant_both_correct_rate=both(invariant))
    return result


def unknowable_report(rows: list[dict]) -> dict | None:
    unknowable = [r for r in rows if r["source"] == "unknowable"]
    controls = [r for r in rows if r["source"] == "unknowable_control"]
    if not unknowable:
        return None
    top = lambda group: [float(probabilities(r).max()) for r in group]
    by_id = {r["id"]: r for r in controls}
    paired = [(float(probabilities(r).max()), float(probabilities(by_id[r["control_id"]]).max()))
              for r in unknowable if r.get("control_id") in by_id]
    return {"n": len(unknowable), "mean_max_p": float(np.mean(top(unknowable))),
            "share_at_0_9": float(np.mean([c >= .9 for c in top(unknowable)])),
            "control_mean_max_p": float(np.mean(top(controls))) if controls else None,
            "control_acc": (float(np.mean([int(np.argmax(r["logits"]) == r["label"]) for r in controls]))
                            if controls else None),
            "paired_confidence_drop": float(np.mean([c - u for u, c in paired])) if paired else None}


def summarize(rows: list[dict], temperature: float = 1.0) -> dict:
    """The report: clean metrics (raw T=1 and at ``temperature``), per task/source/variant."""
    clean = [r for r in rows if r["variant"] == "clean"]
    knowable = scored_rows(rows)
    report = {"clean": metrics(knowable) if knowable else None,
              "tasks": grouped_metrics(clean, "task"),
              "sources": grouped_metrics(clean, "source"),
              "variants": grouped_metrics(rows, "variant"),
              "paired_flip": paired_flip(clean),
              "unknowable": unknowable_report(clean)}
    if temperature != 1.0 and knowable:
        report["temperature"] = temperature
        report["clean_calibrated"] = metrics(knowable, temperature)
    return report


def paired_bootstrap(candidate: list[dict], reference: list[dict], *, metric: str = "nll",
                     samples: int = 1000, seed: int = 0) -> dict:
    """Candidate − reference with a source-stratified cluster bootstrap over original records."""
    def index(rows):
        indexed = {(r["id"], r["question"]): r for r in scored_rows(rows)}
        if len(indexed) != len(scored_rows(rows)):
            raise ValueError("duplicate paired example")
        return indexed

    a, b = index(candidate), index(reference)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical clean examples")
    keys = sorted(a)
    if any(a[k]["label"] != b[k]["label"] or a[k]["keys"] != b[k]["keys"] for k in keys):
        raise ValueError("paired comparison labels or option order differ")
    clusters: dict = defaultdict(lambda: defaultdict(list))
    for i, key in enumerate(keys):
        clusters[a[key]["source"]][a[key]["group"]].append(i)
    units = {source: [np.asarray(v) for v in groups.values()] for source, groups in clusters.items()}

    def arrays(indexed):
        rows = [indexed[k] for k in keys]
        conf = np.asarray([probabilities(r).max() for r in rows])
        correct = np.asarray([np.argmax(r["logits"]) == r["label"] for r in rows])
        per_row = {"acc": correct.astype(float), "nll": np.asarray([nll(r) for r in rows]),
                   "brier": np.asarray([float(((probabilities(r) - np.eye(len(r["logits"]))[r["label"]]) ** 2).sum())
                                        for r in rows])}
        return conf, correct, per_row

    sides = [arrays(a), arrays(b)]

    def statistic(indices, side):
        conf, correct, per_row = side
        if metric in per_row:
            return float(per_row[metric][indices].mean())
        if metric == "ece":
            return ece(conf[indices], correct[indices])
        if metric == "coverage_at_5pct_error":
            return coverage_at_error(conf[indices], correct[indices], 0.05)
        raise ValueError(f"unsupported bootstrap metric: {metric}")

    delta = lambda idx: statistic(idx, sides[0]) - statistic(idx, sides[1])
    rng, values = np.random.default_rng(seed), []
    for _ in range(samples):
        drawn = [groups[i] for groups in units.values() for i in rng.integers(0, len(groups), len(groups))]
        values.append(delta(np.concatenate(drawn)))
    return {"delta": delta(np.arange(len(keys))), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "samples": samples, "clusters": sum(len(v) for v in units.values())}
