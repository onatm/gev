"""Pure reference excerpts copied from Kev at 08ab0b87d27cb5577a3b371ad7ed4e4686b0502b.

This fixture is intentionally independent of ``gev`` imports; it is used only
for parity tests and does not download or execute the complete upstream app.
"""

import random

NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"),
                ("none", "None of these"), ("other", "Something else"), ("not_listed", "Not listed here"),
                ("none_of_the_above", None), ("other", "A category that fits none of the above"),
                ("other", "None of the listed options apply"), ("unknown", "Cannot be determined from the options given"),
                ("other", "Other"), ("none", None), ("other", "An answer not covered by the other options"),
                ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}


def render(v, indent=0):
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def augment(req, rng, p_none=.1, p_none_distract=.12, p_distract=.15):
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice": out["questions"][qid] = q; continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:
            keys = list(crit); rng.shuffle(keys); out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}; continue
        r = rng.random(); ns = [(k, v) for k, v in NONE_OPTIONS if k not in crit]; ds = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and ns:
            k, v = rng.choice(ns); crit.pop(y); crit[k] = v; y = k
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and ns:
            k, v = rng.choice(ns); crit[k] = v
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and ds:
            k = rng.choice(ds); crit[k] = DISTRACTORS[k]
        keys = list(crit); rng.shuffle(keys); out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out
