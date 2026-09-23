"""Materialize labelled Kev-shaped requests without mutating their input.

Source: https://raw.githubusercontent.com/jaredpalmer/kev/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/data.py
"""

from __future__ import annotations

import copy
import math

from .api_request import api_request
from .representation import question_keys, render, to_record, validate_label, validate_target


def materialize(request: dict) -> dict:
    source = copy.deepcopy(request)
    public = api_request(source)
    questions = []
    for qid, original in source["questions"].items():
        kind = original["type"]
        criteria = original.get("criteria") or ({} if kind == "noul" else [])
        keys = question_keys(kind, criteria)
        label = validate_label(original, keys)
        converted, _ = to_record(public)
        options = converted["questions"][len(questions)]["options"]
        target = original.get("target")
        vector = None
        if target is not None:
            vector = validate_target(target, keys)
        questions.append({"id": qid, "instr": render(original.get("instructions")), "options": options,
                          "label": label, "keys": keys, "qtype": kind, "src": original.get("src"),
                          **({"target": vector} if vector is not None else {})})
    return {"state": render(public["state"]), "questions": questions, "metadata": copy.deepcopy(source.get("_meta", {}))}
