"""Exact labelled-request augmentation from pinned Kev ``data.py``."""

from __future__ import annotations

import hashlib
import random

NONE_OPTIONS = [
    ("other", "None of the above"), ("other", "A reason that fits none of the above"),
    ("none", "None of these"), ("other", "Something else"), ("not_listed", "Not listed here"),
    ("none_of_the_above", None), ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"), ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"), ("none", None), ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple",
               "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}


def source_seed(seed: int, source: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def item_rng(seed: int, epoch: int, identifier: str) -> random.Random:
    return random.Random(source_seed(seed, f"{epoch}:{identifier}"))


def augment(req: dict, rng: random.Random, p_none=.1, p_none_distract=.12, p_distract=.15) -> dict:
    """Kev's mutually-exclusive choice augmentation on the labelled API shape."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:
            keys = list(crit); rng.shuffle(keys)
            out["questions"][qid] = {**q, "criteria": {key: crit[key] for key in keys}}
            continue
        r = rng.random()
        none_options = [(key, value) for key, value in NONE_OPTIONS if key not in crit]
        distractors = [key for key in DISTRACTORS if key not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            key, value = rng.choice(none_options); crit.pop(y); crit[key] = value; y = key
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            key, value = rng.choice(none_options); crit[key] = value
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            key = rng.choice(distractors); crit[key] = DISTRACTORS[key]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {key: crit[key] for key in keys}, "label": y}
    return out


def none_pair(req: dict, rng: random.Random) -> list[dict]:
    eligible = [(qid, q) for qid, q in req["questions"].items()
                if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    choices = [(key, value) for key, value in NONE_OPTIONS if key not in q["criteria"]]
    key, value = rng.choice(choices or [("none_of_these", None)])
    keys = list(q["criteria"]) + [key]; rng.shuffle(keys)
    present_q = {**q, "criteria": {item: (value if item == key else q["criteria"][item]) for item in keys}}
    absent_q = {**present_q, "criteria": {item: val for item, val in present_q["criteria"].items() if item != q["label"]}, "label": key}
    return [{"state": req["state"], "questions": {qid: present_q}},
            {"state": req["state"], "questions": {qid: absent_q}}]
