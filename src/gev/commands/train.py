"""Thin ``gev train`` command handler."""

from __future__ import annotations

import json

from ..application.training import train_stage, warm_start_stage


def handle_train(args) -> int:
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer")
    try:
        if args.init_from:
            result = warm_start_stage(
                args.config, init_from=args.init_from, data_root=args.data,
                output=args.out, max_steps=args.max_steps, device=args.device)
        else:
            result = train_stage(
                args.config, data_root=args.data, output=args.out,
                max_steps=args.max_steps, device=args.device, resume=args.resume)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
