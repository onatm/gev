"""Small CLI boundary helpers shared by command handlers."""

from __future__ import annotations

from ..configuration.config import ConfigError, load_config


def load_cli_config(path: str):
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
