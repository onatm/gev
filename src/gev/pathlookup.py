"""Package-resource lookup for immutable reference artifacts."""

from __future__ import annotations

from importlib.resources import files


def reference_bytes(*parts: str) -> bytes:
    """Read an immutable artifact bundled below ``gev/resources``."""
    return files("gev").joinpath("resources", *parts).read_bytes()
