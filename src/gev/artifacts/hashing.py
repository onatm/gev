"""Small, shared helpers for byte-level provenance hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256(path.read_bytes())
