"""Deterministic, source-balanced selection from the approved development split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict


def selected_ids_sha256(rows: list[dict]) -> str:
    identifiers = sorted(row["_meta"]["id"] for row in rows)
    return hashlib.sha256(json.dumps(
        identifiers, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def select_development_rows(rows: list[dict], *, require_multi_question_row: bool = True) -> list[dict]:
    """Choose deterministic per-source length extremes and a multi-question row."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("_meta", {})
        identifier, source = metadata.get("id"), metadata.get("source")
        if (not isinstance(identifier, str) or not identifier
                or not isinstance(source, str) or not source):
            raise ValueError("selection requires verified development row IDs and sources")
        by_source[source].append(row)
    if not by_source:
        raise ValueError("development selection is empty")

    selected: dict[str, dict] = {}
    for source in sorted(by_source):
        ordered = sorted(by_source[source], key=lambda row: (
            len(json.dumps(row, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")),
            row["_meta"]["id"]))
        for row in (ordered[0], ordered[-1]):
            selected[row["_meta"]["id"]] = row

    if require_multi_question_row and not any(
            len(row.get("questions", {})) > 1 for row in selected.values()):
        multi_question = [row for row in rows if len(row.get("questions", {})) > 1]
        if not multi_question:
            raise ValueError("development data has no multi-question row")
        chosen = min(multi_question, key=lambda row: hashlib.sha256(
            row["_meta"]["id"].encode("utf-8")).hexdigest())
        selected[chosen["_meta"]["id"]] = chosen
    return [selected[identifier] for identifier in sorted(selected)]
