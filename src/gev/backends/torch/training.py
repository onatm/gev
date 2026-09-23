"""Deterministic logical training and exact, boundary-only continuation."""
from __future__ import annotations

import contextlib, dataclasses, hashlib, json, math, os, random, tempfile, time
from pathlib import Path

import torch

from ...training.batching import physical_token_count, variant_count_for_request
from .objective import logical_loss
from ...configuration.resolved import resolve_experiment_config
from ...training.schedule import TrainingSchedule

RESUME_VERSION = 1


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
    state = {"format": "gev.logical-resume", "version": RESUME_VERSION,
             "identity": recipe,
             "progress": {"complete": bool(complete), "shuffle_rng": shuffle_rng,
                          "epoch": epoch, "next_batch": next_batch, "order": order,
                          "global_step": global_step, "metrics": metrics,
                          "augmentation_digest": augmentation_digest},
             "torch_state": {"trainable_state": trainable,
                             "optimizer": _optimizer_state_cpu(optimizer),
                             "scheduler": scheduler.state_dict(), "rng": rng}}
    _atomic_torch_save(state, Path(path))


def load_training_state(path: str | Path) -> dict:
    """Load and validate a resume snapshot before applying anything."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"format", "version", "identity", "progress", "torch_state"}
    if not isinstance(state, dict):
        raise ValueError("invalid or incomplete resume snapshot")
    missing = sorted(required - set(state))
    unexpected = sorted(set(state) - required)
    version = state.get("version")
    if (missing or unexpected or state.get("format") != "gev.logical-resume"
            or isinstance(version, bool) or not isinstance(version, int)
            or version != RESUME_VERSION):
        details = []
        if missing: details.append(f"missing {', '.join(missing)}")
        if unexpected: details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError("invalid or incomplete resume snapshot" +
                         (f": {'; '.join(details)}" if details else ""))
    recipe = state["identity"]
    progress = state["progress"]
    torch_state = state["torch_state"]
    if not isinstance(recipe, dict) or not isinstance(progress, dict) or not isinstance(torch_state, dict):
        raise ValueError("invalid or incomplete resume snapshot: contract sections must be objects")
    required_identity = {"recipe_sha256", "recipe", "identity", "source", "training", "execution"}
    required_progress = {"complete", "shuffle_rng", "epoch", "next_batch", "order", "global_step", "metrics", "augmentation_digest"}
    required_torch = {"trainable_state", "optimizer", "scheduler", "rng"}
    missing = sorted((required_identity - set(recipe)) |
                     (required_progress - set(progress)) |
                     (required_torch - set(torch_state)))
    unexpected = sorted((set(recipe) - required_identity) |
                        (set(progress) - required_progress) |
                        (set(torch_state) - required_torch))
    if missing or unexpected:
        details = []
        if missing: details.append(f"missing {', '.join(missing)}")
        if unexpected: details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError(f"invalid or incomplete resume snapshot: {'; '.join(details)}")
    if (not isinstance(recipe["recipe_sha256"], str)
            or len(recipe["recipe_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in recipe["recipe_sha256"])
            or any(not isinstance(recipe[field], dict)
                   for field in ("recipe", "identity", "source", "training", "execution"))):
        raise ValueError("invalid or incomplete resume snapshot: identity fields are invalid")
    if (not isinstance(progress["complete"], bool)
            or any(isinstance(progress[field], bool) or not isinstance(progress[field], int)
                   or progress[field] < 0 for field in ("epoch", "next_batch", "global_step"))
            or not isinstance(progress["shuffle_rng"], tuple)
            or not isinstance(progress["order"], list)
            or any(not isinstance(identifier, str) for identifier in progress["order"])
            or not isinstance(progress["metrics"], dict)
            or not isinstance(progress["augmentation_digest"], str)
            or len(progress["augmentation_digest"]) != 64
            or any(char not in "0123456789abcdef"
                   for char in progress["augmentation_digest"])):
        raise ValueError("invalid or incomplete resume snapshot: progress fields are invalid")
    if (not isinstance(torch_state["trainable_state"], dict)
            or any(not isinstance(torch_state[field], dict)
                   for field in ("optimizer", "scheduler", "rng"))):
        raise ValueError("invalid or incomplete resume snapshot: Torch state fields are invalid")
    return state


def _recipe(config, source_hash, manifest, model, tokenizer, markers, *, resolved=None):
    resolved = resolved or resolve_experiment_config(config)
    # The seed is replicate identity, not recipe identity. max_steps and
    # save_every may change across an exact boundary resume.
    return {
        "recipe_sha256": resolved.recipe_sha256,
        "recipe": resolved.recipe,
        "identity": {"family": resolved.model.family.family_id,
                     "backend": resolved.model.backend.backend_id,
                     "base": {"name": config.model.name, "revision": config.model.revision},
                     "tokenizer": {"revision": getattr(markers, "tokenizer_revision", config.model.revision)},
                     "markers": {"ids": getattr(markers, "ids", None),
                                 "strings": getattr(markers, "strings", None),
                                 "bos": getattr(markers, "bos", False)},
                     "protocol": dataclasses.asdict(config.protocol),
                     "representation_version": 1},
        "source": {"sha256": source_hash, "manifest": manifest},
        "training": {"study_id": config.experiment_id, "replicate_seed": config.training.seed},
        "execution": {"device": config.runtime.device,
                      "attn_implementation": config.runtime.attn_implementation,
                      "gradient_checkpointing": config.runtime.gradient_checkpointing,
                      "mps_fallback": config.runtime.mps_fallback},
    }


def create_optimizer(model, config):
    """Construct the production adapter/head AdamW parameter groups."""
    head = list(model.head.parameters())
    head_ids = {id(parameter) for parameter in head}
    lora = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids]
    head_lr = config.training.head_learning_rate or config.training.learning_rate
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": config.training.learning_rate},
         {"params": head, "lr": head_lr}],
        weight_decay=config.training.weight_decay)
    return optimizer, lora + head, head_lr


def optimizer_step(model, variants, config, optimizer, *, device: str,
                   scheduler=None, trainable_parameters=None):
    """Run the production microbatch/backward/clip/update step.

    The caller supplies policy-ordered variants. This method owns tensor work,
    autocast, gradients, clipping, optimizer update, and optional scheduler step.
    """
    if not variants:
        raise ValueError("cannot perform an optimizer step with no variants")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    physical_tokens = 0
    microbatches = 0
    trainable = (list(trainable_parameters) if trainable_parameters is not None else
                 [parameter for parameter in model.parameters() if parameter.requires_grad])
    for start in range(0, len(variants), config.training.microbatch):
        part = variants[start:start + config.training.microbatch]
        autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if config.training.dtype == "bf16" else contextlib.nullcontext())
        with autocast:
            logits = (model.forward_packed_batch([variant.encoding for variant in part])
                      if config.runtime.execution_mode == "packed"
                      else model.forward_batch([variant.encoding for variant in part]))
        part_loss = logical_loss(
            logits, [{"record": variant.record, "encoding": variant.encoding}
                     for variant in part])
        if not torch.isfinite(part_loss).item():
            raise FloatingPointError("non-finite loss")
        weighted_loss = part_loss * len(part) / len(variants)
        weighted_loss.backward()
        microbatches += 1
        physical_tokens += sum(physical_token_count(variant.encoding) for variant in part)
        losses.append(weighted_loss.detach())
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
           for parameter in trainable):
        raise FloatingPointError("non-finite gradient")
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return torch.stack(losses).sum(), microbatches, physical_tokens


def train(model, schedule: TrainingSchedule, config, output: str | Path, *, source_hash=None,
          manifest=None, progress=True, resume: str | Path | None = None):
    output = Path(output)
    requests = schedule.requests
    tokenizer = schedule.tokenizer
    markers = schedule.markers
    resume_path = Path(resume) if resume else None
    if output.exists() and resume_path is None: raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolve_experiment_config(config)
    device = resolved.select_device()
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
    if config.training.dtype == "bf16" and device != "mps": raise RuntimeError("bf16 training is explicitly supported only on MPS")
    model.to(device)
    opt, trainable, head_lr = create_optimizer(model, config)
    full_steps = config.training.epochs * math.ceil(len(requests) / config.training.logical_batch)
    sched_steps = max(full_steps, 1)
    if sched_steps <= 10: raise ValueError(f"OneCycleLR requires more than 10 full scheduled steps; got {sched_steps}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[config.training.learning_rate, head_lr], total_steps=sched_steps, pct_start=.1)
    recipe = _recipe(config, source_hash, manifest, model, tokenizer, markers, resolved=resolved)
    metrics = {"source_count": len(requests), "requested_records": len(requests) * config.training.epochs,
               "planned_variant_count": sum(variant_count_for_request(r, seed=config.training.seed, epoch=e, p_none=config.training.p_none, p_none_distract=config.training.p_none_distract, p_distract=config.training.p_distract, p_none_pair=config.training.p_none_pair) for e in range(config.training.epochs) for r in requests),
               "processed_records": 0, "variant_count": 0, "logical_steps": 0, "physical_microbatches": 0,
               "logical_tokens": 0, "physical_tokens": 0, "losses": [], "learning_rates": [], "wall_seconds": 0.0,
               "device": device, "dtype": config.training.dtype, "master_dtype": "fp32", "execution_mode": config.runtime.execution_mode, "optimizer_betas": list(opt.defaults["betas"]), "full_sched_steps": sched_steps, "complete": True}
    start = time.perf_counter()
    if resume_path:
        state = load_training_state(resume_path)
        if state["identity"] != recipe: raise ValueError("resume source/config/model/tokenizer/marker contract mismatch")
        torch_state = state["torch_state"]
        progress_state = state["progress"]
        current = {k: v for k, v in model.state_dict().items() if any(x in k for x in ("lora_", "head."))}
        if set(current) != set(torch_state["trainable_state"]): raise ValueError("resume trainable model state mismatch")
        model.load_state_dict(torch_state["trainable_state"], strict=False)
        opt.load_state_dict(torch_state["optimizer"]); _move_optimizer_state(opt, device)
        scheduler.load_state_dict(torch_state["scheduler"])
        schedule.restore(state)
        metrics = progress_state["metrics"]
        # RNG must be restored only after model and optimizer construction/loading.
        _restore_rng(torch_state["rng"])
    def snapshot():
        save_training_state(output / "last_good.resume.pt", recipe=recipe, model=model,
                            optimizer=opt, scheduler=scheduler, rng=_rng_state(),
                            shuffle_rng=schedule.shuffle_rng.getstate(), epoch=schedule.epoch,
                            next_batch=schedule.batch_cursor,
                            order=schedule.order_ids(),
                            global_step=metrics["logical_steps"], metrics=metrics,
                            augmentation_digest=schedule.augmentation_digest,
                            complete=metrics.get("complete", False))
    model.train(); opt.zero_grad(set_to_none=True)
    limit = config.training.max_steps or full_steps
    for epoch in range(schedule.epoch, config.training.epochs):
        schedule.epoch = epoch
        batches = schedule.batches_for_epoch()
        for batch_index in range(schedule.batch_cursor, len(batches)):
            schedule.batch_cursor = batch_index
            batch = batches[batch_index]
            variants = schedule.variants_for_batch(batch, epoch=epoch)
            try:
                loss, microbatches, physical_tokens = optimizer_step(
                    model, variants, config, opt, device=device, scheduler=scheduler,
                    trainable_parameters=trainable)
            except FloatingPointError as exc:
                if str(exc) == "non-finite loss":
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch} step={metrics['logical_steps'] + 1}") from exc
                raise
            metrics["physical_microbatches"] += microbatches
            metrics["physical_tokens"] += physical_tokens
            metrics["logical_steps"] += 1; metrics["processed_records"] += len(batch); metrics["variant_count"] += len(variants); metrics["logical_tokens"] += sum(len(v.encoding["ids"]) for v in variants); metrics["losses"].append(float(loss.cpu())); metrics["learning_rates"].append([g["lr"] for g in opt.param_groups]); schedule.complete_batch()
            if progress: print(f"step {metrics['logical_steps']}/{limit} loss {metrics['losses'][-1]:.6f}", flush=True)
            if config.training.save_every and metrics["logical_steps"] % config.training.save_every == 0: snapshot()
            if metrics["logical_steps"] >= limit: break
        schedule.finish_epoch(len(batches), stopped=metrics["logical_steps"] >= limit)
        if metrics["logical_steps"] >= limit:
            break
    metrics["complete"] = metrics["logical_steps"] >= full_steps; metrics["wall_seconds"] += time.perf_counter() - start
    snapshot()
    provenance = {**recipe, "resolved_config": resolved.provenance(output_path=str(output)),
                  "training_args": dataclasses.asdict(config.training),
                  "variant_count": metrics["variant_count"],
                  "augmentation_digest": schedule.augmentation_digest,
                  "complete": metrics["complete"]}
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "training_config.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    return metrics
