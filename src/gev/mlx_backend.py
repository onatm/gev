"""MLX backend (Apple Silicon GPU) for Gemma 4 text: mlx-lm decoder + LoRA + pointer head.

It reads and writes the same PEFT-format checkpoint as the Torch backend. MLX
LoRA stores ``lora_a`` as (in, r) and ``lora_b`` as (r, out); PEFT stores the
transposes, so tensors are transposed at the boundary.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from . import checkpoint, hub
from .config import Config
from .encoding import flatten_rows

_PEFT = {"lora_a": "lora_A.weight", "lora_b": "lora_B.weight"}


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, width)
        self.key = nn.Linear(hidden_size, width)

    def __call__(self, decision, options):
        """decision: [rows, hidden]; options: [rows, max_options, hidden] -> [rows, max_options]."""
        query = self.query(decision.astype(mx.float32))
        keys = self.key(options.astype(mx.float32))
        return (keys @ query[:, :, None]).squeeze(-1) / math.sqrt(keys.shape[-1])


class RowModel(nn.Module):
    def __init__(self, decoder, head):
        super().__init__()
        self.decoder = decoder
        self.head = head

    def __call__(self, ids, decide, options, valid):
        # Rows are right-padded; causal attention keeps padding out of every real position.
        hidden = self.decoder(ids)
        rows = mx.arange(ids.shape[0])
        scores = self.head(hidden[rows, decide], hidden[rows[:, None], options])
        return mx.where(valid, scores, mx.array(-1e9, dtype=mx.float32))


def checkpoint_layers(decoder) -> None:
    """Recompute each decoder layer's activations in the backward pass instead of storing them.

    Each layer's class is swapped for a subclass (parameter paths are unchanged), so only
    this decoder is affected. ``mx.checkpoint`` takes arrays only: the mask, cache, and
    offset are closed over, and the offset comes back through a side channel.
    """
    for layer in decoder.layers:
        layer.__class__ = _checkpointed(type(layer))


def _checkpointed(cls):
    if getattr(cls, "_gev_checkpointed", False):
        return cls

    class Checkpointed(cls):
        _gev_checkpointed = True

        def __call__(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
            extra = {}

            def inner(params, x, arrays):
                self.update(params)
                h, kvs, extra["offset"] = cls.__call__(self, x, mask, cache, per_layer_input=arrays.get("input"),
                                                       shared_kv=arrays.get("kv"), offset=offset)
                return h, kvs

            arrays = {k: v for k, v in (("input", per_layer_input), ("kv", shared_kv)) if v is not None}
            h, kvs = mx.checkpoint(inner)(self.trainable_parameters(), x, arrays)
            return h, kvs, extra["offset"]

    Checkpointed.__name__ = Checkpointed.__qualname__ = f"Checkpointed{cls.__name__}"
    return Checkpointed


def load_decoder(config: Config):
    """Load the pinned multimodal Gemma 4 checkpoint and keep only its text decoder."""
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import _get_classes, load_config

    hub.configure_hub()
    path = Path(snapshot_download(config.model.name, revision=config.model.revision,
                                  allow_patterns=["*.json", "*.safetensors"]))
    model_config = load_config(path)
    model_class, args_class = _get_classes(model_config)
    base = model_class(args_class.from_dict(model_config))
    weights = {}
    for filename in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(filename)))
    weights = base.sanitize(weights) if hasattr(base, "sanitize") else weights
    expected = dict(tree_flatten(base.parameters()))
    # KV-shared layers reuse earlier layers' keys/values; their k/v weights have no destination.
    extra = [k for k in weights if k not in expected]
    if any(k.split(".")[-2] not in ("k_proj", "v_proj", "k_norm") for k in extra):
        raise ValueError(f"unexpected Gemma 4 tensors: {extra[:3]}")
    base.load_weights([(k, v) for k, v in weights.items() if k in expected], strict=True)
    return base.language_model.model


class MlxRunner:
    def __init__(self, config: Config, *, decoder=None, device: str | None = None):
        if (device or config.device) not in ("auto", "gpu"):
            raise ValueError("the MLX backend runs on the Apple GPU; use device 'gpu' or 'auto'")
        if not mx.metal.is_available():
            raise RuntimeError("MLX requires an Apple Silicon Metal GPU")
        from mlx_lm.tuner.utils import linear_to_lora_layers

        self.config = config
        self.device = "gpu"
        mx.random.seed(config.training.seed)
        decoder = load_decoder(config) if decoder is None else decoder
        decoder.set_dtype(mx.bfloat16 if config.dtype == "bf16" else mx.float32)
        decoder.freeze()
        model = config.model
        keys = {name for layer in decoder.layers for name, module in layer.named_modules()
                if name.rsplit(".", 1)[-1] in model.lora_targets and isinstance(module, nn.Linear)}
        linear_to_lora_layers(decoder, len(decoder.layers), {
            "rank": model.lora_rank, "scale": model.lora_alpha / model.lora_rank,
            "dropout": model.lora_dropout, "keys": keys})
        if config.gradient_checkpointing:
            checkpoint_layers(decoder)
        hidden = decoder.embed_tokens.weight.shape[1]
        self.model = RowModel(decoder, PointerHead(hidden, model.pointer_width))
        self.optimizers = None

    # --- forward ---------------------------------------------------------------------------

    @staticmethod
    def _batch(encodings: list[dict]):
        rows = flatten_rows(encodings)
        width = max(len(ids) for _, ids, _, _ in rows)
        widest = max(len(options) for _, _, _, options in rows)
        ids = np.zeros((len(rows), width), dtype=np.int32)
        decide = np.zeros(len(rows), dtype=np.int32)
        options = np.zeros((len(rows), widest), dtype=np.int32)
        valid = np.zeros((len(rows), widest), dtype=bool)
        for i, (_, row_ids, row_decide, row_options) in enumerate(rows):
            ids[i, :len(row_ids)] = row_ids
            decide[i] = row_decide
            options[i, :len(row_options)] = row_options
            options[i, len(row_options):] = row_decide  # any in-row index; masked out below
            valid[i, :len(row_options)] = True
        return rows, (mx.array(ids), mx.array(decide), mx.array(options), mx.array(valid))

    def logits(self, encodings: list[dict]) -> list[list[np.ndarray]]:
        self.model.eval()
        rows, batch = self._batch(encodings)
        scores = np.asarray(self.model(*batch).astype(mx.float32))
        result: list[list[np.ndarray]] = [[] for _ in encodings]
        for i, (record, _, _, options) in enumerate(rows):
            result[record].append(scores[i, :len(options)])
        return result

    # --- training --------------------------------------------------------------------------

    def init_optimizer(self) -> None:
        # bias_correction=True matches torch.optim.AdamW; MLX defaults to False.
        decay = self.config.training.weight_decay
        self.optimizers = {name: optim.AdamW(lr, weight_decay=decay, bias_correction=True)
                           for name, lr in (("adapter", self.config.training.learning_rate),
                                            ("head", self.config.head_learning_rate))}

    @staticmethod
    def _loss(model, ids, decide, options, valid, targets, weights):
        log_p = nn.log_softmax(model(ids, decide, options, valid), axis=-1)
        return (-(targets * log_p).sum(axis=-1) * weights).sum()

    def train_step(self, variants: list[dict], lr: float, head_lr: float) -> float:
        """One optimizer step; loss = mean over variants of mean question cross-entropy."""
        self.model.train()
        loss_and_grad = nn.value_and_grad(self.model, self._loss)
        chunk, total, gradients = self.config.training.microbatch, mx.array(0.0), None
        for start in range(0, len(variants), chunk):
            part = variants[start:start + chunk]
            rows, batch = self._batch([variant["encoding"] for variant in part])
            widest = batch[2].shape[1]
            targets = np.zeros((len(rows), widest), dtype=np.float32)
            weights = np.zeros(len(rows), dtype=np.float32)
            questions = [q for variant in part for q in variant["questions"]]
            counts = [len(variant["questions"]) for variant in part]
            for i, ((record, _, _, options), question) in enumerate(zip(rows, questions)):
                if question.get("target") is not None:
                    targets[i, :len(options)] = question["target"]
                else:
                    targets[i, question["label"]] = 1.0
                weights[i] = 1.0 / (counts[record] * len(variants))
            loss, grads = loss_and_grad(self.model, *batch, mx.array(targets), mx.array(weights))
            gradients = grads if gradients is None else tree_map(mx.add, gradients, grads)
            total = total + loss
            mx.eval(gradients, total)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        if not (math.isfinite(float(total)) and math.isfinite(float(norm))):
            raise FloatingPointError("non-finite loss or gradient")
        flat = tree_flatten(gradients)
        self.optimizers["adapter"].learning_rate = lr
        self.optimizers["head"].learning_rate = head_lr
        self.optimizers["adapter"].update(self.model, tree_unflatten([(k, v) for k, v in flat if not k.startswith("head.")]))
        self.optimizers["head"].update(self.model, tree_unflatten([(k, v) for k, v in flat if k.startswith("head.")]))
        mx.eval(self.model.trainable_parameters(), [o.state for o in self.optimizers.values()])
        return float(total)

    def peak_memory(self) -> int:
        """Peak MLX memory in bytes since the last call."""
        peak = mx.get_peak_memory()
        mx.reset_peak_memory()
        return peak

    # --- persistence -----------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint.lora_config(self.config, with_base=True).save_pretrained(directory)
        adapter = {}
        for name, value in tree_flatten(self.model.decoder.trainable_parameters()):
            module, kind = name.rsplit(".", 1)
            adapter[f"{checkpoint.ADAPTER_PREFIX}{module}.{_PEFT[kind]}"] = value.astype(mx.float32).T
        mx.save_safetensors(str(directory / "adapter_model.safetensors"), adapter)
        mx.save_safetensors(str(directory / "pointer.safetensors"), dict(tree_flatten(self.model.head.parameters())))

    def load(self, directory: Path) -> None:
        reverse = {peft: mlx for mlx, peft in _PEFT.items()}
        adapter = []
        for name, value in mx.load(str(directory / "adapter_model.safetensors")).items():
            module, kind, suffix = name.removeprefix(checkpoint.ADAPTER_PREFIX).rsplit(".", 2)
            adapter.append((f"{module}.{reverse[f'{kind}.{suffix}']}", value.T))
        expected = {k for k, _ in tree_flatten(self.model.decoder.trainable_parameters())}
        if {k for k, _ in adapter} != expected:
            raise ValueError("adapter tensors do not match this model's LoRA layout")
        self.model.decoder.update(tree_unflatten(adapter))
        self.model.head.update(tree_unflatten(list(mx.load(str(directory / "pointer.safetensors")).items())))
        mx.eval(self.model.parameters())

    def save_optimizer(self, directory: Path) -> None:
        for name, optimizer in self.optimizers.items():
            mx.save_safetensors(str(directory / f"optimizer-{name}.safetensors"), dict(tree_flatten(optimizer.state)))

    def load_optimizer(self, directory: Path) -> None:
        for name, optimizer in self.optimizers.items():
            optimizer.state = tree_unflatten(list(mx.load(str(directory / f"optimizer-{name}.safetensors")).items()))
