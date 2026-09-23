"""Saved-row calibration and comparison command handlers."""

from __future__ import annotations

import json


def handle_calibrate(args) -> int:
    from ..evaluation.calibration import calibrate

    result = calibrate(args.run, rows=args.rows, protocol=args.protocol, out=args.out,
                       update_checkpoint=args.update_checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def handle_compare(args) -> int:
    from ..evaluation.compare import compare

    result = compare(args.candidate, args.reference, aggregation=args.aggregation,
                     samples=args.samples, seed=args.seed, out=args.out,
                     calibrated=args.calibrated, allow_test_compare=args.allow_test_compare)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
