"""Pinned Kev API representation functions.

Adapted from Kev ``api.py`` at 08ab0b87d27cb5577a3b371ad7ed4e4686b0502b.
"""

from __future__ import annotations

from typing import Any


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


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria: Any) -> list[str]:
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unknown question type: {qtype}")


def to_record(request: dict) -> tuple[dict, list[dict]]:
    """Convert a labelled API request while keeping IDs and labels out of text."""
    if not isinstance(request.get("state"), (str, dict, list, int, float, bool, type(None))):
        raise ValueError("invalid state")
    questions, metadata = [], []
    for qid, question in request["questions"].items():
        kind = question.get("type")
        criteria = question.get("criteria") or ({} if kind == "noul" else [])
        keys = question_keys(kind, criteria)
        if kind == "noul":
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
        elif kind == "choice":
            options = [option_text(key, criteria[key]) for key in keys]
        else:
            options = [render(item) for item in criteria]
        questions.append({"instr": render(question.get("instructions")), "options": options,
                          "label": question.get("label"), "keys": keys, "qid": qid,
                          "qtype": kind})
        metadata.append({"id": qid, "type": kind, "keys": keys,
                         **({"legend": dict(zip(keys, options))} if kind == "score" else {})})
    return {"state": render(request["state"]), "questions": questions}, metadata


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
    import math
    if not isinstance(target, dict):
        raise ValueError("target must be a keyed object")
    values = [float(target.get(key, 0.0)) for key in keys]
    if any(not math.isfinite(value) or value < 0 for value in values) or not sum(values) > 0:
        raise ValueError("target must have finite non-negative positive mass")
    total = sum(values)
    return [value / total for value in values]
