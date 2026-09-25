"""Kev request representation: rendering, option keys, labels, and materialization.

Adapted from Kev ``api.py``/``data.py`` at 08ab0b87d27cb5577a3b371ad7ed4e4686b0502b.
"""

from __future__ import annotations

import copy
import math
from typing import Any

QUESTION_FIELDS = ("type", "instructions", "criteria")


def render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}" if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def _criteria(question: dict) -> Any:
    return question.get("criteria") or ({} if question.get("type") == "noul" else [])


def question_keys(qtype: str, criteria: Any) -> list[str]:
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unknown question type: {qtype}")


def keys_of(question: dict) -> list[str]:
    return question_keys(question["type"], _criteria(question))


def _option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def options_of(question: dict) -> list[str]:
    kind, criteria = question["type"], _criteria(question)
    if kind == "noul":
        return [_option_text("no", criteria.get("false")), _option_text("yes", criteria.get("true"))]
    if kind == "choice":
        return [_option_text(key, criteria[key]) for key in criteria]
    return [render(item) for item in criteria]


def validate_label(question: dict, keys: list[str]) -> int:
    kind, label = question["type"], question.get("label")
    if kind == "choice":
        if label not in keys:
            raise ValueError(f"label {label!r} is not an option")
        return keys.index(label)
    if kind == "noul":
        if not isinstance(label, bool):
            raise ValueError("noul label must be boolean")
        return int(label)
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(keys):
        raise ValueError("score label must be an in-range level index")
    return label


def validate_target(target: Any, keys: list[str]) -> list[float]:
    if not isinstance(target, dict):
        raise ValueError("target must be a keyed object")
    values = [float(target.get(key, 0.0)) for key in keys]
    if any(not math.isfinite(value) or value < 0 for value in values) or not sum(values) > 0:
        raise ValueError("target must have finite non-negative positive mass")
    total = sum(values)
    return [value / total for value in values]


def validate_request(request: dict, *, labelled: bool) -> None:
    if not isinstance(request, dict) or "state" not in request:
        raise ValueError("request needs a state")
    if not isinstance(request.get("questions"), dict) or not request["questions"]:
        raise ValueError("request needs non-empty questions")
    for question in request["questions"].values():
        kind, criteria = question.get("type"), question.get("criteria")
        if kind not in {"choice", "noul", "score"}:
            raise ValueError(f"unknown question type: {kind}")
        if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise ValueError("choice criteria must be a non-empty object")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise ValueError("score criteria must be a non-empty list")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object")
        if labelled:
            keys = keys_of(question)
            validate_label(question, keys)
            if question.get("target") is not None:
                validate_target(question["target"], keys)


def public_request(request: dict) -> dict:
    """Strip labels and provenance: the shape a served model receives."""
    validate_request(request, labelled=False)
    return {"state": request["state"],
            "questions": {qid: {key: value for key, value in question.items() if key in QUESTION_FIELDS}
                          for qid, question in request["questions"].items()}}


def materialize(request: dict, *, labelled: bool = True) -> dict:
    """Convert a Kev request into rendered text plus option keys, labels, and soft targets."""
    validate_request(request, labelled=labelled)
    questions = []
    for qid, original in request["questions"].items():
        keys = keys_of(original)
        question = {"id": qid, "type": original["type"], "keys": keys,
                    "instr": render(original.get("instructions")), "options": options_of(original),
                    "src": original.get("src")}
        if labelled:
            question["label"] = validate_label(original, keys)
            if original.get("target") is not None:
                question["target"] = validate_target(original["target"], keys)
        questions.append(question)
    return {"state": render(request["state"]), "questions": questions,
            "metadata": copy.deepcopy(request.get("_meta", {}))}
