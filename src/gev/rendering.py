"""Kev-compatible deterministic rendering.

Source: https://raw.githubusercontent.com/jaredpalmer/kev/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/api.py
"""

from __future__ import annotations

from .representation import option_text, question_keys, render


__all__ = ["render", "option_text", "question_keys"]
