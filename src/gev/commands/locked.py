"""Thin explicit locked-evaluation command handlers."""

from __future__ import annotations

import json

from ..application.evaluation import locked_evaluation_stage


def handle_locked_register(args) -> int:
    from ..evaluation.locked import register_candidate

    result = register_candidate(run=args.run, study=args.study, out=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def handle_locked_evaluate(args) -> int:
    suites = tuple(suite.strip() for suite in args.suites.split(",") if suite.strip())
    result = locked_evaluation_stage(
        args.run, selection=args.selection, suites=suites, data_root=args.data,
        output=args.out, ledger=args.ledger, config_path=args.config, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
