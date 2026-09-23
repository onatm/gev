"""Small, transport-independent contracts shared by rendering and materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: Any
    options: tuple[str, ...]
    keys: tuple[str, ...]
    label: int
    target: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Record:
    state: str
    questions: tuple[Question, ...]
    metadata: dict[str, Any]
