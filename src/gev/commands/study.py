"""Porcelain study planning and orchestration commands."""

from __future__ import annotations

import json

from ..study.runner import plan, run


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit("--seeds must be a comma-separated list of non-negative integers") from exc
    if not result:
        raise SystemExit("--seeds must contain at least one seed")
    return result


def handle_study(args) -> int:
    seeds = _seeds(args.seeds)
    if args.study_action == "plan":
        result = plan(args.config, seeds=seeds, data=args.data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    run(args.config, seeds=seeds, data=args.data, out=args.out,
        max_steps=args.max_steps, smoke=args.smoke, existing_run=args.existing_run)
    return 0
