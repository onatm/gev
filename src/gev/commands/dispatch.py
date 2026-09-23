"""Route parsed commands to their workflow-specific handlers."""

from __future__ import annotations

def dispatch(args) -> int:
    if args.command == "study":
        from .study import handle_study
        return handle_study(args)
    if args.command == "train":
        from .train import handle_train
        return handle_train(args)
    if args.command == "evaluate":
        from .evaluate import handle_evaluate
        return handle_evaluate(args)
    if args.command == "predict":
        from .predict import handle_predict
        return handle_predict(args)
    if args.command == "calibrate":
        from .analysis import handle_calibrate
        return handle_calibrate(args)
    if args.command == "compare":
        from .analysis import handle_compare
        return handle_compare(args)
    if args.command == "data":
        from .data import handle_data
        if args.data_action == "verify":
            args.target = args.suite
        return handle_data(args)
    if args.command == "locked":
        from .locked import handle_locked_evaluate, handle_locked_register
        if args.locked_action == "register":
            return handle_locked_register(args)
        return handle_locked_evaluate(args)
    if args.command == "diagnose":
        from .diagnose import handle_diagnose
        return handle_diagnose(args)
    raise ValueError(f"unhandled CLI command: {args.command}")
