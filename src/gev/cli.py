"""``gev`` command line: data, train, evaluate, calibrate, compare, predict, push."""

from __future__ import annotations

import argparse
import json
import sys

from . import config as config_module
from . import data

DEVICES = ("auto", "cpu", "mps", "cuda", "gpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gev", description="Train and serve Jev-like decision models.")
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="fetch or sample pinned Kev suites")
    data_commands = data_parser.add_subparsers(dest="action", required=True)
    fetch = data_commands.add_parser("fetch", help="download and verify suite splits")
    fetch.add_argument("suite", choices=sorted(data.SUITES))
    fetch.add_argument("splits", nargs="+", choices=data.SPLITS)
    fetch.add_argument("--data", default="data")
    sample = data_commands.add_parser("sample", help="write a small decision-v7 subset for smoke runs")
    sample.add_argument("--data", default="data")
    sample.add_argument("--out", default="data/smoke")
    sample.add_argument("--train-records", type=int, default=128)
    sample.add_argument("--dev-records", type=int, default=64)
    sample.add_argument("--seed", type=int, default=0)

    train = commands.add_parser("train", help="train a LoRA adapter and pointer head")
    train.add_argument("config")
    train.add_argument("--out", required=True)
    train.add_argument("--data", default="data")
    train.add_argument("--resume", action="store_true", help="continue the run in --out from its last saved state")
    train.add_argument("--device", choices=DEVICES)
    train.add_argument("--seed", type=int)
    train.add_argument("--max-steps", type=int)

    evaluate = commands.add_parser("evaluate", help="score a checkpoint on a suite split")
    evaluate.add_argument("model", help="run directory, checkpoint directory, or Hub repo id")
    evaluate.add_argument("--suite", required=True, choices=sorted(data.SUITES))
    evaluate.add_argument("--split", required=True, choices=data.SPLITS)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--data", default="data")
    evaluate.add_argument("--backend", choices=config_module.BACKENDS)
    evaluate.add_argument("--device", choices=DEVICES)

    calibrate = commands.add_parser("calibrate", help="fit a serving temperature from an evaluation")
    calibrate.add_argument("evaluation", help="evaluation output directory (calibration or development split)")
    calibrate.add_argument("--update", action="store_true", help="store the temperature in the checkpoint")

    compare = commands.add_parser("compare", help="paired bootstrap of two evaluations on the same data")
    compare.add_argument("candidate")
    compare.add_argument("reference")
    compare.add_argument("--samples", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=0)

    predict = commands.add_parser("predict", help="answer one unlabeled Kev request (JSON)")
    predict.add_argument("model", help="run directory, checkpoint directory, or Hub repo id")
    predict.add_argument("--input", default="-", help="request JSON file, or - for stdin")
    predict.add_argument("--temperature", type=float)
    predict.add_argument("--backend", choices=config_module.BACKENDS)
    predict.add_argument("--device", choices=DEVICES)

    push = commands.add_parser("push", help="upload a checkpoint to the Hugging Face Hub")
    push.add_argument("run", help="run or checkpoint directory")
    push.add_argument("--repo", required=True, help="e.g. user/gev-gemma4-e2b")
    push.add_argument("--public", action="store_true")
    push.add_argument("--report", action="append", default=[], help="evaluation report.json for the model card")
    return parser


def run(args: argparse.Namespace):
    if args.command == "data":
        if args.action == "fetch":
            return {split: str(data.fetch(args.suite, split, args.data)) for split in args.splits}
        return data.sample(args.data, args.out, train_records=args.train_records,
                           dev_records=args.dev_records, seed=args.seed)
    if args.command == "train":
        from .train import train

        config = config_module.load(args.config)
        overrides = {"device": args.device, "seed": args.seed, "max_steps": args.max_steps}
        config = config.replace(**{k: v for k, v in overrides.items() if v is not None})
        return train(config, data_root=args.data, out=args.out, resume=args.resume)
    if args.command in ("evaluate", "predict"):
        from .evaluate import Model, evaluate

        model = Model(args.model, backend=args.backend, device=args.device)
        if args.command == "predict":
            text = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
            return model.predict(json.loads(text), args.temperature)
        report = evaluate(model, suite=args.suite, split=args.split, data_root=args.data, out=args.out)
        return {"clean": report["clean"], "clean_calibrated": report.get("clean_calibrated")}
    if args.command == "calibrate":
        from .evaluate import calibrate

        return calibrate(args.evaluation, update=args.update)
    if args.command == "compare":
        from .evaluate import compare

        return compare(args.candidate, args.reference, samples=args.samples, seed=args.seed)
    if args.command == "push":
        from .checkpoint import push

        return {"commit": push(args.run, args.repo, private=not args.public, reports=args.report)}
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        print(f"gev: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
