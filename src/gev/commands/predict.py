"""Thin single-request prediction command handler."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from ..application.prediction import predict_stage


def handle_predict(args) -> int:
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise SystemExit("prediction temperature must be finite and positive")
    try:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        request = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"prediction input error: {exc}") from exc
    if not isinstance(request, dict):
        raise SystemExit("prediction input must be one Kev-style request object")
    result = predict_stage(args.run, request, config_path=args.config,
                           temperature=args.temperature, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
