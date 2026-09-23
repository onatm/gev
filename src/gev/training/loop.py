"""Deterministic logical training and exact, boundary-only continuation."""
from __future__ import annotations

import contextlib, dataclasses, hashlib, json, math, os, random, tempfile, time
from pathlib import Path

import torch

from ..training.batching import logical_batches, physical_token_count, variant_count_for_request, variants_for_request
from ..training.objective import logical_loss


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _rng_state() -> dict:
    import numpy as np
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    state["numpy"] = np.random.get_state()
    return state


def _restore_rng(state: dict) -> None:
    import numpy as np
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    np.random.set_state(state["numpy"])


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            # Adam's scalar step is intentionally kept on CPU for the default
            # (non-capturable) optimizer.  Moving it to MPS can make the next
            # optimizer step fail on otherwise valid checkpoints.
            if isinstance(value, torch.Tensor) and not (key == "step" and value.ndim == 0):
                state[key] = value.to(device)


def _optimizer_state_cpu(optimizer):
    """Copy only Adam state to CPU; never clone the frozen backbone."""
    source = optimizer.state_dict()
    # Optimizer state dictionaries can retain references to live tensors.
    # Construct a separate nested mapping before converting snapshot values.
    state = {"state": {}, "param_groups": [dict(group) for group in source["param_groups"]]}
    for parameter_id, values in source["state"].items():
        copied = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                dtype = torch.float32 if value.is_floating_point() else value.dtype
                copied[key] = value.detach().to(device="cpu", dtype=dtype).clone()
            elif isinstance(value, dict):
                copied[key] = dict(value)
            elif isinstance(value, list):
                copied[key] = list(value)
            elif isinstance(value, tuple):
                copied[key] = tuple(value)
            else:
                copied[key] = value
        state["state"][parameter_id] = copied
    return state


def save_training_state(path: str | Path, *, recipe, model, optimizer, scheduler, rng,
                        shuffle_rng, epoch, next_batch, order, global_step, metrics,
                        augmentation_digest, complete=False) -> None:
    """Atomically save a resumable boundary state.

    The state deliberately contains trainable adapter/head tensors only.  The
    base model is reconstructed from its pinned revision on resume.
    """
    trainable = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if "lora_" in k or k.startswith("head.")}
    state = {"format_version": 2, "schema": "gev.logical-resume.v2", "complete": bool(complete),
             "recipe": recipe, "trainable_state": trainable,
             "optimizer": _optimizer_state_cpu(optimizer), "scheduler": scheduler.state_dict(),
             "rng": rng, "shuffle_rng": shuffle_rng, "epoch": epoch,
             "next_batch": next_batch, "order": order, "global_step": global_step,
             "metrics": metrics, "augmentation_digest": augmentation_digest}
    _atomic_torch_save(state, Path(path))


def load_training_state(path: str | Path) -> dict:
    """Load and validate a resume snapshot before applying anything."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"format_version", "schema", "complete", "recipe", "trainable_state", "optimizer",
                "scheduler", "rng", "shuffle_rng", "epoch", "next_batch", "order", "global_step",
                "metrics", "augmentation_digest"}
    missing = sorted(required - set(state))
    if missing or state.get("format_version") != 2 or state.get("schema") != "gev.logical-resume.v2":
        raise ValueError("invalid or incomplete resume snapshot" + (f": missing {', '.join(missing)}" if missing else ""))
    recipe = state["recipe"]
    for field in ("config_sha256", "source_sha256", "model_revision", "tokenizer_revision", "marker_ids", "marker_strings", "marker_bos"):
        if field not in recipe:
            raise ValueError(f"resume snapshot missing recipe field: {field}")
    return state


def _recipe(config, source_hash, manifest, model, tokenizer, markers):
    # max_steps and save_every are operational controls, not the recipe identity.
    value = dataclasses.asdict(config)
    value["training"].pop("max_steps", None)
    value["training"].pop("save_every", None)
    return {"config": value, "config_sha256": _json_hash(value),
            "source_sha256": source_hash, "manifest": manifest,
            "model_name": config.model.name, "model_revision": config.model.revision,
            "tokenizer_revision": getattr(markers, "tokenizer_revision", config.model.revision),
            "marker_ids": getattr(markers, "ids", None), "marker_strings": getattr(markers, "strings", None),
            "marker_bos": getattr(markers, "bos", False), "representation_version": 1}


def train(model, requests, tokenizer, markers, config, output: str | Path, *, source_hash=None,
          manifest=None, progress=True, resume: str | Path | None = None):
    output = Path(output)
    resume_path = Path(resume) if resume else None
    if output.exists() and resume_path is None: raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if config.runtime.device == "auto" else config.runtime.device
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
    if config.training.dtype == "bf16" and device != "mps": raise RuntimeError("bf16 training is explicitly supported only on MPS")
    model.to(device)
    head = list(model.head.parameters()); head_ids = {id(p) for p in head}
    lora = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head_lr = config.training.head_learning_rate or config.training.learning_rate
    opt = torch.optim.AdamW([{"params": lora, "lr": config.training.learning_rate}, {"params": head, "lr": head_lr}], weight_decay=config.training.weight_decay)
    full_steps = config.training.epochs * math.ceil(len(requests) / config.training.logical_batch)
    sched_steps = max(full_steps, 1)
    if sched_steps <= 10: raise ValueError(f"OneCycleLR requires more than 10 full scheduled steps; got {sched_steps}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[config.training.learning_rate, head_lr], total_steps=sched_steps, pct_start=.1)
    recipe = _recipe(config, source_hash, manifest, model, tokenizer, markers)
    ordered = list(requests)
    shuffle_rng = random.Random(config.training.seed)
    digest = hashlib.sha256(b"gev-augmentation-chain-v1").hexdigest()
    metrics = {"source_count": len(requests), "requested_records": len(requests) * config.training.epochs,
               "planned_variant_count": sum(variant_count_for_request(r, seed=config.training.seed, epoch=e, p_none=config.training.p_none, p_none_distract=config.training.p_none_distract, p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair) for e in range(config.training.epochs) for r in requests),
               "processed_records": 0, "variant_count": 0, "logical_steps": 0, "physical_microbatches": 0,
               "logical_tokens": 0, "physical_tokens": 0, "losses": [], "learning_rates": [], "wall_seconds": 0.0,
               "device": device, "dtype": config.training.dtype, "master_dtype": "fp32", "execution_mode": config.runtime.execution_mode, "optimizer_betas": list(opt.defaults["betas"]), "full_sched_steps": sched_steps, "complete": True}
    epoch = batch_cursor = 0
    start = time.perf_counter()
    if resume_path:
        state = load_training_state(resume_path)
        if state["recipe"] != recipe: raise ValueError("resume source/config/model/tokenizer/marker contract mismatch")
        current = {k: v for k, v in model.state_dict().items() if any(x in k for x in ("lora_", "head."))}
        if set(current) != set(state["trainable_state"]): raise ValueError("resume trainable model state mismatch")
        model.load_state_dict(state["trainable_state"], strict=False)
        opt.load_state_dict(state["optimizer"]); _move_optimizer_state(opt, device)
        scheduler.load_state_dict(state["scheduler"])
        epoch, batch_cursor = state["epoch"], state["next_batch"]
        ordered = [next(r for r in requests if r.get("_meta", {}).get("id", r.get("id", "")) == i) for i in state["order"]]
        shuffle_rng.setstate(state["shuffle_rng"]); metrics = state["metrics"]; digest = state["augmentation_digest"]
        # RNG must be restored only after model and optimizer construction/loading.
        _restore_rng(state["rng"])
    def snapshot():
        save_training_state(output / "last_good.resume.pt", recipe=recipe, model=model,
                            optimizer=opt, scheduler=scheduler, rng=_rng_state(),
                            shuffle_rng=shuffle_rng.getstate(), epoch=epoch,
                            next_batch=batch_cursor,
                            order=[r.get("_meta", {}).get("id", r.get("id", "")) for r in ordered],
                            global_step=metrics["logical_steps"], metrics=metrics,
                            augmentation_digest=digest, complete=metrics.get("complete", False))
    model.train(); opt.zero_grad(set_to_none=True)
    limit = config.training.max_steps or full_steps
    for epoch in range(epoch, config.training.epochs):
        if batch_cursor == 0: shuffle_rng.shuffle(ordered)
        batches = list(logical_batches(ordered, config.training.logical_batch))
        for batch_index in range(batch_cursor, len(batches)):
            batch_cursor = batch_index
            batch = batches[batch_index]
            variants = [v for req in batch for v in variants_for_request(req, seed=config.training.seed, epoch=epoch, tokenizer=tokenizer, markers=markers, caps=(config.training.state_cap, config.training.branch_cap, config.training.packed_cap), p_none=config.training.p_none, p_none_distract=config.training.p_none_distract, p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair)]
            for order, variant in enumerate(variants):
                payload = json.dumps({"id": variant.original_source_id, "epoch": epoch, "order": order, "record": variant.record}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                digest = hashlib.sha256(bytes.fromhex(digest) + payload).hexdigest()
            losses = []
            for chunk_start in range(0, len(variants), config.training.microbatch):
                part = variants[chunk_start:chunk_start + config.training.microbatch]
                autocast = torch.autocast(device_type="mps", dtype=torch.bfloat16) if config.training.dtype == "bf16" else contextlib.nullcontext()
                with autocast:
                    logits = (model.forward_packed_batch([v.encoding for v in part])
                              if config.runtime.execution_mode == "packed"
                              else model.forward_batch([v.encoding for v in part]))
                part_loss = logical_loss(logits, [{"record": v.record, "encoding": v.encoding} for v in part])
                if not torch.isfinite(part_loss).item(): raise FloatingPointError(f"non-finite loss at epoch={epoch} step={metrics['logical_steps'] + 1}")
                (part_loss * len(part) / len(variants)).backward(); metrics["physical_microbatches"] += 1; metrics["physical_tokens"] += sum(physical_token_count(v.encoding) for v in part); losses.append(part_loss.detach() * len(part) / len(variants))
            loss = torch.stack(losses).sum(); trainable = lora + head
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in trainable): raise FloatingPointError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
            metrics["logical_steps"] += 1; metrics["processed_records"] += len(batch); metrics["variant_count"] += len(variants); metrics["logical_tokens"] += sum(len(v.encoding["ids"]) for v in variants); metrics["losses"].append(float(loss.cpu())); metrics["learning_rates"].append([g["lr"] for g in opt.param_groups]); batch_cursor += 1
            if progress: print(f"step {metrics['logical_steps']}/{limit} loss {metrics['losses'][-1]:.6f}", flush=True)
            if config.training.save_every and metrics["logical_steps"] % config.training.save_every == 0: snapshot()
            if metrics["logical_steps"] >= limit: break
        boundary = batch_cursor >= len(batches)
        if boundary:
            batch_cursor = 0
        if metrics["logical_steps"] >= limit:
            # When stopping exactly at an epoch boundary, persist the next
            # epoch.  A mid-epoch stop must retain its cursor and repeat no
            # completed batch on continuation.
            if boundary:
                epoch += 1
            break
    metrics["complete"] = metrics["logical_steps"] >= full_steps; metrics["wall_seconds"] += time.perf_counter() - start
    snapshot()
    provenance = {**recipe, "training_args": dataclasses.asdict(config.training), "variant_count": metrics["variant_count"], "augmentation_digest": digest, "complete": metrics["complete"]}
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "training_config.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    return metrics
