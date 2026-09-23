"""Mechanism and contrastive checks for evaluation-only scoring."""
from __future__ import annotations

import copy


def paired_flip(rows):
    by_pair = {}
    for row in rows:
        if row.get("pair_id"):
            pair = by_pair.setdefault((row["pair_id"], row.get("question", "decision")), {})
            if row.get("sibling") in pair: raise ValueError("duplicate contrastive sibling")
            pair[row.get("sibling")] = row
    if not by_pair: return None
    if any(set(pair) != {"a", "b"} for pair in by_pair.values()): raise ValueError("incomplete contrastive pair")
    prediction = lambda row: row["keys"][max(range(len(row["p"])), key=row["p"].__getitem__)]
    truth = lambda row: row["keys"][row["label"]]
    relevant = [pair for pair in by_pair.values() if truth(pair["a"]) != truth(pair["b"])]
    invariant = [pair for pair in by_pair.values() if truth(pair["a"]) == truth(pair["b"])]
    both = lambda pairs: sum(all(prediction(row) == truth(row) for row in pair.values()) for pair in pairs) / len(pairs) if pairs else None
    result = {"pairs": len(relevant), "flip_rate": sum(prediction(p["a"]) != prediction(p["b"]) for p in relevant) / len(relevant) if relevant else None,
              "both_correct_rate": both(relevant)}
    if invariant:
        result.update(invariant_pairs=len(invariant), invariance_rate=sum(prediction(p["a"]) == prediction(p["b"]) for p in invariant) / len(invariant),
                      invariant_both_correct_rate=both(invariant))
    return result


def mechanism_checks(records, predictor, tolerance=1e-3):
    clean = [record for record in records if record.get("_meta", {}).get("variant", "clean") == "clean"][:8]
    maximum = 0.0; comparisons = 0; failures = 0; reasons = []
    for record in clean:
        original = copy.deepcopy(record)
        packed = predictor(record)["probabilities"]
        if record != original: raise AssertionError("mechanism predictor mutated a record")
        for qid, question in record["questions"].items():
            alone_record = copy.deepcopy(record); alone_record["questions"] = {qid: copy.deepcopy(question)}
            alone = predictor(alone_record)["probabilities"][qid]
            delta = max(abs(float(alone[k]) - float(packed[qid][k])) for k in alone)
            maximum = max(maximum, delta); comparisons += 1
            if delta > tolerance: failures += 1
        if record["questions"]:
            fake_id = "__unrelated_mechanism_sibling__"
            fake_question = copy.deepcopy(next(iter(record["questions"].values())))
            with_fake = copy.deepcopy(record); with_fake["questions"][fake_id] = fake_question
            fake_prediction = predictor(with_fake)["probabilities"]
            for qid in record["questions"]:
                delta = max(abs(float(fake_prediction[qid][k]) - float(packed[qid][k])) for k in packed[qid])
                maximum = max(maximum, delta); comparisons += 1
                if delta > tolerance: failures += 1
    if not clean: reasons.append("no clean records")
    if not comparisons: reasons.append("no mechanism comparisons")
    if failures: reasons.append(f"{failures} comparisons exceeded tolerance")
    return {"records": len(clean), "comparisons": comparisons, "failures": failures, "max_delta": maximum,
            "tolerance": tolerance, "passed": bool(comparisons) and not failures, "reason": "; ".join(reasons) if reasons else None}
