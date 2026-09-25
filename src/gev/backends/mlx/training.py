"""Deterministic MLX row training with boundary-exact resume snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from ...training.batching import physical_token_count, variant_count_for_request
from ...training.policy import KEV_OBJECTIVE_POLICY
from ...training.schedule import TrainingSchedule

RESUME_FORMAT = "gev.mlx.logical-resume"
RESUME_VERSION = 1


def _step_seed(seed: int, step: int, variant: int = 0) -> int:
    digest = hashlib.sha256(f"gev-mlx-rng-v1:{seed}:{step}:{variant}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _one_cycle(step: int, total_steps: int, peak: float) -> float:
    warmup = max(1, round(total_steps * 0.1))
    initial, final = peak / 25.0, peak / 10000.0
    if step < warmup:
        fraction = step / warmup
        return initial + (peak - initial) * (1 - math.cos(math.pi * fraction)) / 2
    fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return final + (peak - final) * (1 + math.cos(math.pi * fraction)) / 2


def _objective(model, variant):
    record = variant.record
    logits = model.forward_one(variant.encoding)
    questions = list(record["questions"].values()) if isinstance(record["questions"], dict) else record["questions"]
    if len(logits) != len(questions) or not logits:
        raise ValueError("model returned a different number of questions")
    losses = []
    for values, question in zip(logits, questions, strict=True):
        scores = values.astype(mx.float32)
        if question.get("target") is not None:
            target = mx.array(question["target"], dtype=mx.float32)
            losses.append(-(target * mx.log_softmax(scores, axis=-1)).sum())
        else:
            label = int(question["label"])
            losses.append(mx.logsumexp(scores, axis=-1) - scores[label])
    return mx.stack(losses).mean()


def _group(tree, *, head: bool):
    return tree_unflatten([(name, value) for name, value in tree_flatten(tree)
                           if name.startswith("head.") == head])


def _finite(tree) -> bool:
    return all(bool(mx.all(mx.isfinite(value)).item()) for _, value in tree_flatten(tree))


def _tree_norm(tree) -> float:
    flat = tree_flatten(tree)
    return (float(mx.sqrt(sum((value.astype(mx.float32) ** 2).sum()
                              for _, value in flat)).item()) if flat else 0.0)


def _group_changed(before: dict, after: dict, *, head: bool) -> bool:
    names = [name for name in before if name.startswith("head.") == head]
    return any(bool(mx.any(before[name] != after[name]).item()) for name in names)


def _precision_contract(model, config) -> dict:
    compute_dtype = config.training.dtype
    if compute_dtype not in {"bf16", "fp32"}:
        raise ValueError("Gemma 4 MLX compute dtype must be bf16 or fp32")
    provenance = getattr(model, "provenance", None)
    if not isinstance(provenance, dict):
        raise RuntimeError("Gemma 4 MLX model lacks pinned BF16 provenance")
    base = provenance.get("base_qualification", {})
    if (provenance.get("model_output_id") != "gev-gemma4-e2b"
            or provenance.get("source_weights_dtype") != "bf16"
            or provenance.get("compute_dtype") != compute_dtype
            or base.get("revision") != "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
            or base.get("source_weights_dtype") != "bf16"
            or base.get("compute_dtype") != compute_dtype):
        raise ValueError("Gemma 4 training requires the pinned BF16 source and selected compute identity")
    inventory = base.get("inventory", {})
    if (inventory.get("tensor_count") != 2011 or inventory.get("text_tensor_count") != 600
            or inventory.get("text_serialized_parameters") != 4_647_449_891
            or inventory.get("effective_text_parameters") != 4_628_569_344):
        raise ValueError("Gemma 4 source tensor inventory does not match the pinned BF16 model")
    if base.get("lora", {}).get("layers") != 205:
        raise ValueError("Gemma 4 LoRA target inventory must contain 205 modules")
    trainable = tree_flatten(model.trainable_parameters())
    lora = [(name, value) for name, value in trainable if not name.startswith("head.")]
    head = [(name, value) for name, value in trainable if name.startswith("head.")]
    if (not lora or any(value.dtype != mx.float32 for _, value in lora)
            or len(head) != 4 or any(value.dtype != mx.float32 for _, value in head)
            or model.head.query.weight.shape[0] != 256
            or model.head.key.weight.shape[0] != 256):
        raise ValueError("Gemma 4 trainable state requires 205 FP32 LoRA masters and an FP32 pointer")
    expected_decoder_dtype = mx.bfloat16 if compute_dtype == "bf16" else mx.float32
    frozen = [(name, value) for name, value in tree_flatten(model.decoder.parameters())
              if ".lora_" not in name]
    if not frozen or any(value.dtype != expected_decoder_dtype for _, value in frozen):
        raise ValueError(
            f"Gemma 4 frozen decoder weights must remain {compute_dtype.upper()} compute")
    return {"source_inventory_exact": True,
            "bf16_frozen_decoder": compute_dtype == "bf16",
            "fp32_frozen_decoder": compute_dtype == "fp32",
            "fp32_lora_master": True, "fp32_pointer": True,
            "trained_lora_modules": 205,
            "trainable_tensor_count": len(trainable)}


def _training_memory_snapshot() -> dict:
    import psutil

    return {"process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
            "mlx_active_bytes": int(mx.get_active_memory())}


def _optimizer_state_fp32(optimizers: dict) -> bool:
    floating = {mx.float16, mx.bfloat16, mx.float32}
    values = [value for optimizer in optimizers.values()
              for _, value in tree_flatten(optimizer.state)
              if getattr(value, "dtype", None) in floating]
    return bool(values) and all(value.dtype == mx.float32 for value in values)


def _clip(tree, maximum: float = 1.0):
    flat = tree_flatten(tree)
    norm = mx.sqrt(sum((value.astype(mx.float32) ** 2).sum() for _, value in flat))
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), maximum / (norm + 1e-6))
    return tree_unflatten([(name, value * scale) for name, value in flat]), float(norm.item())


def _encode_tree(value, tensors: dict, prefix: str):
    if isinstance(value, mx.array):
        name = f"t{len(tensors)}"
        tensors[name] = value
        return {"tensor": name}
    if isinstance(value, dict):
        return {"dict": [[key, _encode_tree(item, tensors, prefix)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode_tree(item, tensors, prefix) for item in value]}
    if isinstance(value, list):
        return {"list": [_encode_tree(item, tensors, prefix) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": value}
    raise TypeError(f"unsupported MLX resume state value: {type(value).__name__}")


def _decode_tree(value, tensors):
    if "tensor" in value:
        return tensors[value["tensor"]]
    if "dict" in value:
        return {key: _decode_tree(item, tensors) for key, item in value["dict"]}
    if "tuple" in value:
        return tuple(_decode_tree(item, tensors) for item in value["tuple"])
    if "list" in value:
        return [_decode_tree(item, tensors) for item in value["list"]]
    return value.get("value")


def _tuple_state(value):
    return tuple(_tuple_state(item) for item in value) if isinstance(value, list) else value


def _write_resume(directory: Path, *, identity: dict, progress: dict,
                  model, optimizers: dict) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    backup = directory.with_name(f".{directory.name}.previous")
    try:
        tensors = {}
        model_state = {}
        for name, value in tree_flatten(model.trainable_parameters()):
            key = f"model.{len(tensors)}"
            tensors[key] = value
            model_state[name] = key
        optimizer_state = {name: _encode_tree(optimizer.state, tensors, name)
                           for name, optimizer in optimizers.items()}
        mx.save_safetensors(str(temporary / "state.safetensors"), tensors)
        state = {"format": RESUME_FORMAT, "version": RESUME_VERSION,
                 "identity": identity, "progress": progress,
                 "model_state": model_state, "optimizer_state": optimizer_state,
                 "scheduler": {"step": progress["global_step"], "policy": "one-cycle-cosine-v1"},
                 "rng": {"policy": "logical-step-seed-v1", "seed": identity["training"]["seed"]}}
        (temporary / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        if backup.exists():
            shutil.rmtree(backup)
        if directory.exists():
            os.replace(directory, backup)
        os.replace(temporary, directory)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not directory.exists():
            os.replace(backup, directory)
        raise


def _read_resume(directory: Path, *, identity: dict, model, optimizers: dict) -> dict:
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    required = {"format", "version", "identity", "progress", "model_state",
                "optimizer_state", "scheduler", "rng"}
    if (not isinstance(state, dict) or set(state) != required
            or state.get("format") != RESUME_FORMAT
            or isinstance(state.get("version"), bool)
            or not isinstance(state.get("version"), int)
            or state.get("version") != RESUME_VERSION
            or state.get("identity") != identity):
        raise ValueError("MLX resume source/config/model/tokenizer/marker contract mismatch")
    progress = state.get("progress")
    progress_fields = {"complete", "shuffle_rng", "epoch", "next_batch", "order",
                       "global_step", "metrics", "augmentation_digest"}
    if (not isinstance(progress, dict) or set(progress) != progress_fields
            or not isinstance(progress["complete"], bool)
            or any(isinstance(progress[field], bool) or not isinstance(progress[field], int)
                   or progress[field] < 0 for field in ("epoch", "next_batch", "global_step"))
            or not isinstance(progress["order"], list)
            or any(not isinstance(value, str) for value in progress["order"])
            or not isinstance(progress["metrics"], dict)
            or not isinstance(progress["augmentation_digest"], str)
            or len(progress["augmentation_digest"]) != 64
            or any(value not in "0123456789abcdef" for value in progress["augmentation_digest"])):
        raise ValueError("invalid or incomplete MLX resume progress")
    if (not isinstance(state.get("model_state"), dict)
            or any(not isinstance(name, str) or not isinstance(key, str)
                   for name, key in state["model_state"].items())
            or not isinstance(state.get("optimizer_state"), dict)):
        raise ValueError("invalid or incomplete MLX resume tensor state")
    if (state.get("scheduler") != {"step": progress["global_step"],
                                    "policy": "one-cycle-cosine-v1"}
            or state.get("rng") != {"policy": "logical-step-seed-v1",
                                     "seed": identity["training"]["seed"]}
            or progress["metrics"].get("logical_steps") != progress["global_step"]):
        raise ValueError("MLX resume scheduler/RNG state mismatch")
    progress["shuffle_rng"] = _tuple_state(progress["shuffle_rng"])
    try:
        random.Random().setstate(progress["shuffle_rng"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid MLX resume shuffle RNG state") from exc
    tensors = mx.load(str(directory / "state.safetensors"))
    current = dict(tree_flatten(model.trainable_parameters()))
    if set(current) != set(state.get("model_state", {})):
        raise ValueError("MLX resume trainable model state mismatch")
    weights = []
    for name, key in state["model_state"].items():
        if (key not in tensors or tuple(current[name].shape) != tuple(tensors[key].shape)
                or current[name].dtype != tensors[key].dtype):
            raise ValueError("MLX resume trainable tensor shape/name mismatch")
        weights.append((name, tensors[key]))
    weights = tree_unflatten(weights)
    model.update(weights)
    if set(optimizers) != set(state.get("optimizer_state", {})):
        raise ValueError("MLX resume optimizer group mismatch")
    for name, optimizer in optimizers.items():
        optimizer.state = _decode_tree(state["optimizer_state"][name], tensors)
        optimizer._initialized = True
    return state


def _identity(config, source_hash, manifest, schedule, resolved):
    return {
        "recipe_sha256": resolved.recipe_sha256,
        "recipe": resolved.recipe,
        "identity": {"family": resolved.model.family.family_id,
                     "model_output_id": resolved.model.output_model_id,
                     "backend": resolved.model.backend.backend_id,
                     "base": {"name": config.model.name, "revision": config.model.revision},
                     "tokenizer": {"revision": schedule.markers.tokenizer_revision},
                     "markers": {"ids": schedule.markers.ids, "strings": schedule.markers.strings,
                                 "bos": schedule.markers.bos},
                     "protocol": {"id": config.protocol.id, "version": config.protocol.version},
                     "representation_version": 1},
        "source": {"sha256": source_hash, "manifest": manifest},
         "training": {"study_id": config.experiment_id, "seed": config.training.seed,
                      "source_weights_dtype": "bf16", "compute_dtype": config.training.dtype,
                      "trainable_master_dtype": "fp32"},
        "execution": {"device": config.runtime.device, "dtype": config.training.dtype,
                      "execution_mode": config.runtime.execution_mode},
    }


def _optimizer_groups(model, config):
    trainable = model.trainable_parameters()
    head = _group(trainable, head=True)
    adapters = _group(trainable, head=False)
    if not tree_flatten(head) or not tree_flatten(adapters):
        raise ValueError("MLX model must expose both trainable LoRA adapters and pointer head")
    head_lr = config.training.head_learning_rate or config.training.learning_rate
    return {"adapters": optim.AdamW(config.training.learning_rate,
                                    weight_decay=config.training.weight_decay),
            "head": optim.AdamW(head_lr, weight_decay=config.training.weight_decay)}, head_lr


def train(model, schedule: TrainingSchedule, config, output: str | Path, *, source_hash=None,
          manifest=None, progress=True, resume: str | Path | None = None):
    if config.runtime.execution_mode != "rows":
        raise ValueError("Gemma 4 MLX supports rows execution only")
    if config.training.microbatch != 1:
        raise ValueError("Gemma 4 MLX currently supports microbatch=1 only")
    precision_contract = _precision_contract(model, config)
    output = Path(output)
    resume_path = Path(resume) if resume is not None else None
    resume_state = (resume_path / "state.json" if resume_path and resume_path.is_dir()
                    else resume_path)
    if output.exists() and resume_path is None:
        raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from ...configuration.resolved import resolve_experiment_config

    resolved = resolve_experiment_config(config)
    device = resolved.select_device()
    if device != "gpu":
        raise RuntimeError("Gemma 4 MLX training requires Metal GPU")
    model.train()
    optimizers, head_lr = _optimizer_groups(model, config)
    full_steps = config.training.epochs * math.ceil(len(schedule.requests) / config.training.logical_batch)
    if full_steps <= 10:
        raise ValueError(f"OneCycle schedule requires more than 10 full scheduled steps; got {full_steps}")
    identity = _identity(config, source_hash, manifest or {}, schedule, resolved)
    metrics = {"source_count": len(schedule.requests),
               "requested_records": len(schedule.requests) * config.training.epochs,
               "planned_variant_count": sum(variant_count_for_request(
                   row, seed=config.training.seed, epoch=epoch, p_none=config.training.p_none,
                   p_none_distract=config.training.p_none_distract, p_distract=config.training.p_distract,
                   p_none_pair=config.training.p_none_pair)
                   for epoch in range(config.training.epochs) for row in schedule.requests),
               "processed_records": 0, "variant_count": 0, "logical_steps": 0,
               "physical_microbatches": 0, "logical_tokens": 0, "physical_tokens": 0,
                "losses": [], "learning_rates": [], "step_seconds": [], "wall_seconds": 0.0,
                "device": device, "dtype": config.training.dtype,
                 "compute_dtype": config.training.dtype,
                 "weights_dtype": config.training.dtype, "source_weights_dtype": "bf16",
                "execution_mode": "rows", "full_sched_steps": full_steps,
                "optimizer_betas": [0.9, 0.999], "rng_policy": "logical-step-seed-v1",
                "engineering_checks": {
                    **precision_contract, "finite_loss_gradients": False,
                    "nonzero_adapter_and_head_gradients": False,
                    "nonzero_trainable_update": False,
                    "finite_optimizer_state": False,
                    "optimizer_master_state_fp32": False,
                },
                "complete": False}
    if resume_path:
        if resume_state is None or not resume_state.is_file():
            raise ValueError("MLX resume snapshot is missing")
        state = _read_resume(resume_path.parent if resume_path.name == "state.json" else resume_path,
                             identity=identity, model=model, optimizers=optimizers)
        schedule.restore(state)
        metrics = state["progress"]["metrics"]
        if metrics.get("logical_steps") != state["progress"].get("global_step"):
            raise ValueError("MLX resume progress cursor and scheduler step differ")

    objective_grad = nn.value_and_grad(model, _objective)
    started = time.perf_counter()
    limit = config.training.max_steps or full_steps
    memory_before = _training_memory_snapshot()
    mx.reset_peak_memory()
    process_peak = memory_before["process_rss_bytes"]
    active_peak = memory_before["mlx_active_bytes"]
    update_checked = False

    def snapshot(complete):
        progress_state = {"complete": complete, "shuffle_rng": schedule.shuffle_rng.getstate(),
                          "epoch": schedule.epoch, "next_batch": schedule.batch_cursor,
                          "order": schedule.order_ids(), "global_step": metrics["logical_steps"],
                          "metrics": metrics, "augmentation_digest": schedule.augmentation_digest}
        _write_resume(output / "last_good.resume.mlx", identity=identity,
                      progress=progress_state, model=model, optimizers=optimizers)

    for epoch in range(schedule.epoch, config.training.epochs):
        schedule.epoch = epoch
        batches = schedule.batches_for_epoch()
        for batch_index in range(schedule.batch_cursor, len(batches)):
            step_started = time.perf_counter()
            schedule.batch_cursor = batch_index
            batch = batches[batch_index]
            variants = schedule.variants_for_batch(batch, epoch=epoch)
            if not variants:
                raise ValueError("logical batch produced no training variants")
            flat_gradients = None
            loss_total = 0.0
            physical_tokens = 0
            for variant_index, variant in enumerate(variants):
                mx.random.seed(_step_seed(config.training.seed, metrics["logical_steps"], variant_index))
                loss, gradients = objective_grad(model, variant)
                mx.eval(loss, gradients)
                if not bool(mx.isfinite(loss).item()) or not _finite(gradients):
                    raise FloatingPointError(f"non-finite MLX loss/gradient at step {metrics['logical_steps'] + 1}")
                adapter_grad_norm = _tree_norm(_group(gradients, head=False))
                head_grad_norm = _tree_norm(_group(gradients, head=True))
                if adapter_grad_norm <= 0.0 or head_grad_norm <= 0.0:
                    raise FloatingPointError(
                        f"zero Gemma 4 adapter/head gradient at step {metrics['logical_steps'] + 1}")
                loss_total += float(loss.item()) / len(variants)
                factor = 1.0 / len(variants)
                contribution = [(name, value * factor) for name, value in tree_flatten(gradients)]
                if flat_gradients is None:
                    flat_gradients = contribution
                else:
                    previous = dict(flat_gradients)
                    flat_gradients = [(name, previous[name] + value) for name, value in contribution]
                physical_tokens += physical_token_count(variant.encoding)
            gradients, _grad_norm = _clip(tree_unflatten(flat_gradients))
            before_update = (dict(tree_flatten(model.trainable_parameters()))
                             if not update_checked else None)
            adapter_lr = _one_cycle(metrics["logical_steps"], full_steps, config.training.learning_rate)
            head_step_lr = _one_cycle(metrics["logical_steps"], full_steps, head_lr)
            optimizers["adapters"].learning_rate = mx.array(adapter_lr, dtype=mx.float32)
            optimizers["head"].learning_rate = mx.array(head_step_lr, dtype=mx.float32)
            optimizers["adapters"].update(model, _group(gradients, head=False))
            optimizers["head"].update(model, _group(gradients, head=True))
            mx.eval(model.parameters(), optimizers["adapters"].state, optimizers["head"].state)
            if not _finite(optimizers["adapters"].state) or not _finite(optimizers["head"].state):
                raise FloatingPointError("non-finite Gemma 4 optimizer state after update")
            if not _optimizer_state_fp32(optimizers):
                raise FloatingPointError("Gemma 4 AdamW master moments must remain FP32")
            engineering = metrics["engineering_checks"]
            engineering["finite_loss_gradients"] = True
            engineering["nonzero_adapter_and_head_gradients"] = True
            engineering["finite_optimizer_state"] = True
            engineering["optimizer_master_state_fp32"] = True
            if before_update is not None:
                after_update = dict(tree_flatten(model.trainable_parameters()))
                adapter_updated = _group_changed(before_update, after_update, head=False)
                head_updated = _group_changed(before_update, after_update, head=True)
                if not adapter_updated or not head_updated:
                    raise FloatingPointError("Gemma 4 step did not update both adapters and pointer head")
                engineering["nonzero_trainable_update"] = True
                update_checked = True
            metrics["logical_steps"] += 1
            metrics["processed_records"] += len(batch)
            metrics["variant_count"] += len(variants)
            metrics["logical_tokens"] += sum(len(variant.encoding["ids"]) for variant in variants)
            metrics["physical_tokens"] += physical_tokens
            metrics["physical_microbatches"] += len(variants)
            metrics["losses"].append(loss_total)
            metrics["learning_rates"].append([adapter_lr, head_step_lr])
            metrics["step_seconds"].append(time.perf_counter() - step_started)
            schedule.complete_batch()
            if progress:
                print(f"step {metrics['logical_steps']}/{limit} loss {loss_total:.6f}", flush=True)
            sample = _training_memory_snapshot()
            process_peak = max(process_peak, sample["process_rss_bytes"])
            active_peak = max(active_peak, sample["mlx_active_bytes"])
            if config.training.save_every and metrics["logical_steps"] % config.training.save_every == 0:
                snapshot(False)
            if metrics["logical_steps"] >= limit:
                break
        schedule.finish_epoch(len(batches), stopped=metrics["logical_steps"] >= limit)
        if metrics["logical_steps"] >= limit:
            break
    metrics["complete"] = metrics["logical_steps"] >= full_steps
    metrics["wall_seconds"] += time.perf_counter() - started
    metrics["training_performance"] = {
        "scope": "training loop only; model loading and offline diagnostics excluded",
        "training_phase_seconds": metrics["wall_seconds"],
        "logical_steps_per_second": (metrics["logical_steps"] / metrics["wall_seconds"]
                                      if metrics["wall_seconds"] else 0.0),
        "physical_tokens_per_second": (metrics["physical_tokens"] / metrics["wall_seconds"]
                                       if metrics["wall_seconds"] else 0.0),
        "step_seconds_mean": (sum(metrics["step_seconds"]) / len(metrics["step_seconds"])
                              if metrics["step_seconds"] else 0.0),
    }
    metrics["training_memory"] = {
        "scope": "training loop only; excludes model loading and offline oracle",
        "process_rss_before_bytes": memory_before["process_rss_bytes"],
        "process_rss_peak_bytes": process_peak,
        "mlx_active_before_bytes": memory_before["mlx_active_bytes"],
        "mlx_active_peak_observed_bytes": active_peak,
        "mlx_peak_bytes": int(mx.get_peak_memory()),
    }
    snapshot(metrics["complete"])
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    return metrics
