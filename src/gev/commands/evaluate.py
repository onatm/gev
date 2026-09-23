"""Thin non-test evaluation command handler."""

from __future__ import annotations

import json

from ..application.evaluation import evaluate_stage


def handle_evaluate(args) -> int:
    report = evaluate_stage(
        args.run, suite=args.suite, split=args.split, data_root=args.data,
        output=args.out, config_path=args.config, temperature=args.temperature,
        device=args.device, execution=args.execution)
    print(json.dumps(report, indent=2))
    return 0
