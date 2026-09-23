"""Backend-neutral representative selection and measurement-file I/O."""

from __future__ import annotations

import json
from pathlib import Path


def select_representative_records(records: list[dict], limit: int) -> list[dict]:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("records must be a positive integer")
    if not records:
        raise ValueError("development selection is empty")
    ordered = sorted(records, key=lambda row: row.get("_meta", {}).get("id", ""))
    if limit >= 8:
        quotas = {"agnews": 3, "yelp": 2, "banking77": 3}
        selected = []
        for source, quota in quotas.items():
            candidates = [row for row in ordered if row.get("_meta", {}).get("source") == source]
            candidates.sort(key=lambda row: (-len(row.get("questions", {})), row.get("_meta", {}).get("id", "")))
            if len(candidates) < quota:
                raise ValueError(f"development selection lacks {source} representatives")
            selected.extend(candidates[:quota])
        remaining = [row for row in ordered if row not in selected]
        selected.extend(remaining[:limit - len(selected)])
        return selected
    return ordered[:min(limit, len(ordered))]


def write_measurement(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
