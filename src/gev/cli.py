"""Small executable entry point for Gev's command handlers."""

from __future__ import annotations

import sys

from .commands.dispatch import dispatch
from .commands.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    return dispatch(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
