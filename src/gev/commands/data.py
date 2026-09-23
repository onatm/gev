"""Data plumbing command handlers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..data.suites import (SuiteError, audit, fetch_file, fetch_manifest, smoke,
                           token_length_audit, verify)


def handle_data(args) -> int:
    try:
        if args.data_action == "fetch":
            if args.target == "night2":
                if args.split is not None:
                    raise SuiteError("night2 fetch does not take a suite split")
                from ..data.continuation import fetch_night2
                result = fetch_night2(args.data_root)
            else:
                if not args.split:
                    raise SuiteError("suite fetch requires a split")
                with tempfile.TemporaryDirectory() as directory:
                    manifest = fetch_manifest(args.target, Path(directory) / "manifest.json")
                    result = fetch_file(args.split, args.target,
                                        Path(args.data_root) / args.target / f"{args.split}.jsonl",
                                        manifest)
        elif args.data_action == "prepare":
            if args.target != "night2":
                raise SuiteError("data prepare currently supports only night2")
            from ..data.continuation import build_continuation
            result = build_continuation(args.data_root, out=args.out)
        elif args.data_action == "sample":
            result = smoke(Path(args.data_root), args.train_records, args.dev_records,
                           Path(args.out), args.seed)
        elif args.data_action == "verify":
            with tempfile.TemporaryDirectory() as directory:
                manifest = fetch_manifest(args.suite, Path(directory) / "manifest.json")
                result = verify(Path(args.path), manifest, args.split, suite=args.suite)
        elif args.data_action == "audit":
            if args.target == "tokens":
                if not args.config or not args.markers or args.split or args.path:
                    raise SuiteError("token audit requires --config and --markers and takes no split/path")
                result = token_length_audit(Path(args.data_root), args.config, Path(args.markers),
                                            augment_train=args.augment,
                                            seeds=tuple(int(seed) for seed in args.seeds.split(",")),
                                            output=Path(args.out))
            else:
                if not args.split or not args.path:
                    raise SuiteError("suite audit requires a split and path")
                if args.config or args.markers or args.augment:
                    raise SuiteError("token-length options require `data audit tokens`")
                with tempfile.TemporaryDirectory() as directory:
                    manifest = fetch_manifest(args.target, Path(directory) / "manifest.json")
                    result = audit(Path(args.path), manifest, args.split, suite=args.target)
        else:  # pragma: no cover - argparse prevents this
            raise SuiteError(f"unknown data action: {args.data_action}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("valid", True) else 1
    except (SuiteError, OSError, ValueError) as exc:
        raise SystemExit(f"data error: {exc}") from exc
