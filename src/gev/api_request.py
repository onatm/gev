"""Boundary that removes labels and provenance before a request leaves Gev.

Source: https://raw.githubusercontent.com/jaredpalmer/kev/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/data.py
"""

from __future__ import annotations


QUESTION_FIELDS = ("type", "instructions", "criteria")


def api_request(record: dict) -> dict:
    if "state" not in record or not record.get("questions"):
        raise ValueError("record needs state and non-empty questions")
    return {
        "state": record["state"],
        "questions": {
            qid: {key: value for key, value in question.items() if key in QUESTION_FIELDS}
            for qid, question in record["questions"].items()
        },
    }
