"""Backend-neutral training loop with a one-cycle schedule and step-boundary resume."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

from . import checkpoint, data, hub
from .config import Config
from .encoding import Markers, encode, token_count
from .records import materialize


def load_tokenizer(config: Config):
    from transformers import AutoTokenizer

    hub.configure_hub()
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
    return tokenizer, Markers.resolve(tokenizer, config.model.markers)


def make_runner(config: Config, **kwargs):
    if config.backend == "mlx":
        from .mlx_backend import MlxRunner

        return MlxRunner(config, **kwargs)
    from .torch_backend import TorchRunner

    return TorchRunner(config, **kwargs)


def one_cycle(step: int, total: int, peak: float) -> float:
    """Cosine one-cycle: warm up from peak/25 over 10% of steps, then anneal to peak/1e4."""
    warmup = max(1, round(total * 0.1))
    start, end = peak / 25.0, peak / 1e4
    if step < warmup:
        return start + (peak - start) * (1 - math.cos(math.pi * step / warmup)) / 2
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + (peak - end) * (1 + math.cos(math.pi * fraction)) / 2


def batch_variants(batch: list[dict], config: Config, *, epoch: int, tokenizer, markers) -> list[dict]:
    """Augment, materialize, and encode one logical batch, longest first to minimize padding."""
    t = config.training
    variants = []
    for request in batch:
        for variant in data.training_variants(request, seed=t.seed, epoch=epoch, p_none=t.p_none,
                                              p_none_distract=t.p_none_distract, p_distract=t.p_distract,
                                              p_none_pair=t.p_none_pair):
            record = materialize(variant)
            variants.append({"encoding": encode(tokenizer, record, markers, state_cap=t.state_cap,
                                                branch_cap=t.branch_cap),
                             "questions": record["questions"]})
    return sorted(variants, key=lambda v: -max(len(v["encoding"]["state"]) + len(r["ids"])
                                              for r in v["encoding"]["rows"]))


def _save_state(out: Path, runner, progress: dict) -> None:
    """Atomically replace ``out/state`` with the adapter, optimizer, and progress cursor."""
    temporary = out / "state.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    runner.save(temporary)
    runner.save_optimizer(temporary)
    (temporary / "progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
    shutil.rmtree(out / "state", ignore_errors=True)
    temporary.rename(out / "state")


def train(config: Config, *, data_root: str | Path, out: str | Path, resume: bool = False,
          runner=None, tokenizer=None, log=print) -> dict:
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"{out} exists; pass --resume to continue it or choose a new --out")
    if resume and not (out / "state" / "progress.json").exists():
        raise FileNotFoundError(f"nothing to resume in {out}")
    requests, train_sha256 = data.load_split(data_root, "decision-v7", "train")
    if tokenizer is None:
        tokenizer, markers = load_tokenizer(config)
    else:
        markers = Markers.resolve(tokenizer, config.model.markers)
    runner = make_runner(config) if runner is None else runner
    runner.init_optimizer()

    t = config.training
    steps_per_epoch = math.ceil(len(requests) / t.logical_batch)
    total_steps = t.epochs * steps_per_epoch
    limit = min(t.max_steps or total_steps, total_steps)
    progress = {"step": 0, "seconds": 0.0, "train_sha256": train_sha256}
    if resume:
        progress = json.loads((out / "state" / "progress.json").read_text(encoding="utf-8"))
        if progress["train_sha256"] != train_sha256:
            raise ValueError("training data changed since this run started")
        runner.load(out / "state")
        runner.load_optimizer(out / "state")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    log(f"training {config.name}: {len(requests)} records, {total_steps} steps "
        f"({config.backend}/{runner.device}/{config.dtype})")

    started = time.perf_counter() - progress["seconds"]
    orders: dict[int, list[dict]] = {}
    with (out / "log.jsonl").open("a", encoding="utf-8") as history:
        while progress["step"] < limit:
            step = progress["step"]
            epoch, index = divmod(step, steps_per_epoch)
            order = orders.setdefault(epoch, data.epoch_order(requests, t.seed, epoch))
            batch = order[index * t.logical_batch:(index + 1) * t.logical_batch]
            variants = batch_variants(batch, config, epoch=epoch, tokenizer=tokenizer, markers=markers)
            lr, head_lr = (one_cycle(step, total_steps, peak)
                           for peak in (t.learning_rate, config.head_learning_rate))
            step_started = time.perf_counter()
            loss = runner.train_step(variants, lr, head_lr)
            progress.update(step=step + 1, seconds=time.perf_counter() - started)
            entry = {"step": step + 1, "epoch": epoch, "loss": loss, "lr": lr, "variants": len(variants),
                     "tokens": sum(token_count(v["encoding"]) for v in variants),
                     "step_seconds": time.perf_counter() - step_started}
            peak = runner.peak_memory()
            if peak is not None:
                entry["peak_memory_gb"] = peak / 1e9
            history.write(json.dumps(entry) + "\n")
            history.flush()
            log(f"step {step + 1}/{limit} loss {loss:.4f} lr {lr:.2e} "
                f"{entry['tokens'] / entry['step_seconds']:.0f} tok/s"
                + (f" peak {entry['peak_memory_gb']:.1f} GB" if peak is not None else ""))
            if t.save_every and (step + 1) % t.save_every == 0 and step + 1 < limit:
                _save_state(out, runner, progress)

    summary = {"steps": progress["step"], "total_steps": total_steps, "complete": progress["step"] == total_steps,
               "seconds": progress["seconds"], "train_sha256": train_sha256, "train_records": len(requests)}
    _save_state(out, runner, progress)
    shutil.rmtree(out / "checkpoint", ignore_errors=True)
    runner.save(out / "checkpoint")
    checkpoint.write_metadata(out / "checkpoint", config, markers, training=summary)
    log(f"saved {out / 'checkpoint'}")
    return summary
