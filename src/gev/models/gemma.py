"""Actual Hugging Face Gemma 3 row encoding with PEFT adapters.

The tiny constructor is deliberately an actual ``Gemma3TextModel`` with a
randomly initialized reduced configuration.  It is a diagnostic, not a
replacement for the pinned gated checkpoint.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import load_config
from ..materialize import materialize
from ..network import use_system_ssl
from ..tokenization import MarkerMap, encode, rows_of
from .masks import continuation_attention_mask, packed_attention_mask
from .pointer import PointerHead

_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Prefix:
    """An immutable description of a state-only inference cache."""
    state_token_ids: tuple[int, ...]
    state_positions: tuple[int, ...]
    cache: Any
    model_identity: int
    device: str
    dtype: str
    parameter_versions: tuple[int, ...]
    autocast_dtype: str | None = None


def _parameter_versions(model: nn.Module) -> tuple[int, ...]:
    return tuple(int(getattr(parameter, "_version", 0)) for parameter in model.parameters())


def _autocast_dtype(device: torch.device) -> str | None:
    enabled = torch.is_autocast_enabled(device_type=device.type)
    return str(torch.get_autocast_dtype(device.type)) if enabled else None


def _tiny_config(layers: int = 6, hidden_size: int = 64, window: int = 8) -> Any:
    from transformers import Gemma3TextConfig

    if layers not in (6, 12) or hidden_size not in (32, 64):
        raise ValueError("tiny layers must be 6 or 12 and hidden_size must be 32 or 64")
    head_dim = 16
    return Gemma3TextConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=128,
        sliding_window=window,
        layer_types=["sliding_attention" if i % 6 != 5 else "full_attention" for i in range(layers)],
        rope_parameters={"sliding_attention": {"rope_theta": 10000.0},
                         "full_attention": {"rope_theta": 1000000.0}},
        query_pre_attn_scalar=head_dim,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=1,
        tie_word_embeddings=False,
    )


def load_real_backbone(name: str, revision: str, *, attn_implementation: str = "eager",
                       gradient_checkpointing: bool = False) -> nn.Module:
    """Load the pinned real checkpoint; authentication failures are not hidden."""
    use_system_ssl()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float32, attn_implementation=attn_implementation, use_cache=False
    )
    if not hasattr(model, "model") or getattr(model.config, "model_type", None) != "gemma3_text":
        raise ValueError("pinned checkpoint is not a Gemma 3 text causal model")
    model = model.model
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model


def _with_peft(backbone: nn.Module) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(_TARGETS),
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(backbone, config)


class GemmaRowModel(nn.Module):
    """Independent state/question rows over a real Gemma decoder."""

    def __init__(self, backbone: nn.Module, *, temperature: float = 1.0,
                 use_peft: bool = True, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if gradient_checkpointing:
            if not hasattr(backbone, "gradient_checkpointing_enable"):
                raise TypeError("backbone does not support gradient checkpointing")
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone = _with_peft(backbone) if use_peft else backbone
        self.backbone.config.use_cache = False
        self.head = PointerHead(int(self.backbone.config.hidden_size), temperature=temperature)

    @property
    def decoder(self) -> nn.Module:
        return self.backbone

    def _rows_hidden(self, encodings: list[dict]) -> tuple[list[tuple[int, int, list[int], list[int], list[int]]], torch.Tensor]:
        """Run all logical rows together and retain offsets for each record."""
        device = next(self.parameters()).device
        rows: list[tuple[int, int, list[int], list[int], list[int]]] = []
        for record_index, encoded in enumerate(encodings):
            state_ids, state_pos, question_rows = rows_of(encoded)
            for question in question_rows:
                ids = state_ids + question["ids"]
                positions = state_pos + question["pos"]
                decide = len(state_ids) + question["decide"]
                options = [len(state_ids) + option for option in question["opts"]]
                rows.append((record_index, decide, options, ids, positions))
        if not rows:
            return [], torch.empty((0, 0, int(self.backbone.config.hidden_size)), device=device)
        width = max(len(row[3]) for row in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        positions = torch.zeros_like(ids)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            length = len(row[3])
            ids[index, :length] = torch.tensor(row[3], dtype=torch.long, device=device)
            positions[index, :length] = torch.tensor(row[4], dtype=torch.long, device=device)
            attention[index, :length] = 1
        output = self.backbone(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
        return rows, output.last_hidden_state

    def forward_rows_batch(self, encodings: list[dict]) -> list[list[torch.Tensor]]:
        rows, hidden = self._rows_hidden(encodings)
        result: list[list[torch.Tensor]] = [[] for _ in encodings]
        for row_index, (record, decide, options, _ids, _positions) in enumerate(rows):
            result[record].append(self.head(hidden[row_index, decide], hidden[row_index, options]))
        return result

    def forward_one(self, encoding: dict) -> list[torch.Tensor]:
        return self.forward_rows_batch([encoding])[0]

    def forward_batch(self, encodings: list[dict], *, execution_mode: str = "rows") -> list[list[torch.Tensor]]:
        if execution_mode == "rows":
            return self.forward_rows_batch(encodings)
        if execution_mode == "packed":
            return self.forward_packed_batch(encodings)
        raise ValueError("execution_mode must be rows or packed")

    def _packed_hidden(self, encodings: list[dict]) -> torch.Tensor:
        """Run independently packed records with one mask per Gemma layer kind."""
        if not encodings:
            return torch.empty((0, 0, int(self.backbone.config.hidden_size)),
                               device=next(self.parameters()).device)
        device = next(self.parameters()).device
        width = max(len(value["ids"]) for value in encodings)
        pad_id = getattr(self.backbone.config, "pad_token_id", 0) or 0
        ids = torch.full((len(encodings), width), pad_id, dtype=torch.long, device=device)
        segments = torch.full_like(ids, -1)
        positions = torch.zeros_like(ids)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row, value in enumerate(encodings):
            length = len(value["ids"])
            ids[row, :length] = torch.as_tensor(value["ids"], dtype=torch.long, device=device)
            segments[row, :length] = torch.as_tensor(value["seg"], dtype=torch.long, device=device)
            positions[row, :length] = torch.as_tensor(value["pos"], dtype=torch.long, device=device)
            valid[row, :length] = True
        mask_dtype = torch.bool if getattr(self.backbone.config, "_attn_implementation", "eager") == "sdpa" else next(self.parameters()).dtype
        masks = {kind: packed_attention_mask(segments, positions, kind,
                 window=int(self.backbone.config.sliding_window), valid=valid, dtype=mask_dtype)
                 for kind in {"full_attention", "sliding_attention"}}
        output = self.backbone(input_ids=ids, attention_mask=masks, position_ids=positions, use_cache=False)
        return output.last_hidden_state

    def forward_packed_batch(self, encodings: list[dict]) -> list[list[torch.Tensor]]:
        """Return pointer logits in the original record/question order."""
        hidden = self._packed_hidden(encodings)
        result: list[list[torch.Tensor]] = []
        for row, encoding in enumerate(encodings):
            result.append([self.head(hidden[row, decide], hidden[row, options])
                           for decide, options in zip(encoding["decide_idx"], encoding["opt_idx"])])
        return result

    def forward_packed_one(self, encoding: dict) -> list[torch.Tensor]:
        return self.forward_packed_batch([encoding])[0]

    @torch.no_grad()
    def prefill_prefix(self, state_token_ids: list[int] | tuple[int, ...],
                       state_positions: list[int] | tuple[int, ...] | None = None) -> Prefix:
        """Prefill a state-only native DynamicCache; never mutate it on answers."""
        if self.training:
            raise RuntimeError("prefix caching is inference-only")
        device = next(self.parameters()).device
        ids = tuple(int(value) for value in state_token_ids)
        positions = tuple(range(len(ids))) if state_positions is None else tuple(int(value) for value in state_positions)
        if len(ids) != len(positions) or not ids:
            raise ValueError("prefix token IDs and positions must be non-empty and equal length")
        if positions != tuple(range(len(positions))):
            raise ValueError("prefix state positions must be contiguous from zero")
        from transformers import DynamicCache
        cache = DynamicCache(config=self.backbone.config)
        tensor_ids = torch.tensor([ids], dtype=torch.long, device=device)
        tensor_pos = torch.tensor([positions], dtype=torch.long, device=device)
        self.backbone(input_ids=tensor_ids, attention_mask=torch.ones_like(tensor_ids),
                      position_ids=tensor_pos, past_key_values=cache, use_cache=True)
        return Prefix(ids, positions, cache, id(self), str(device), str(next(self.parameters()).dtype),
                      _parameter_versions(self), _autocast_dtype(device))

    def _check_prefix(self, prefix: Prefix) -> None:
        if self.training:
            raise RuntimeError("prefix caching is inference-only")
        parameter = next(self.parameters())
        if (prefix.model_identity != id(self) or prefix.device != str(parameter.device)
                or prefix.dtype != str(parameter.dtype) or prefix.parameter_versions != _parameter_versions(self)
                or prefix.autocast_dtype != _autocast_dtype(parameter.device)):
            raise ValueError("prefix cache is stale or belongs to another model/device/dtype")

    @torch.no_grad()
    def forward_prefix_batch(self, prefix: Prefix, questions: list[dict]) -> list[torch.Tensor]:
        """Answer independent branch rows using private clones of a prefix cache."""
        self._check_prefix(prefix)
        device = next(self.parameters()).device
        result = []
        for question in questions:
            ids = torch.tensor([question["ids"]], dtype=torch.long, device=device)
            positions = torch.tensor([question["pos"]], dtype=torch.long, device=device)
            cache = copy.deepcopy(prefix.cache)
            mask_dtype = (torch.bool if getattr(self.backbone.config, "_attn_implementation", "eager") == "sdpa"
                          else next(self.parameters()).dtype)
            masks = {}
            for kind in {"full_attention", "sliding_attention"}:
                layer = next(i for i, value in enumerate(self.backbone.config.layer_types) if value == kind)
                kv_length, kv_offset = cache.get_mask_sizes(ids.shape[1], layer)
                key_positions = torch.arange(kv_offset, kv_offset + kv_length, device=device).unsqueeze(0)
                key_segments = (key_positions >= len(prefix.state_token_ids)).long()
                query_segments = torch.ones_like(positions)
                masks[kind] = continuation_attention_mask(
                    query_segments, positions, key_segments, key_positions, kind,
                    window=int(self.backbone.config.sliding_window), dtype=mask_dtype)
            output = self.backbone(input_ids=ids, attention_mask=masks,
                                   position_ids=positions, past_key_values=cache, use_cache=True)
            result.append(self.head(output.last_hidden_state[0, question["decide"]],
                                    output.last_hidden_state[0, question["opts"]]))
        return result

    @torch.no_grad()
    def forward_with_prefix(self, encoding: dict, prefix: Prefix) -> list[torch.Tensor]:
        """Validate the complete encoded state before using a prefix cache."""
        self._check_prefix(prefix)
        state_ids = tuple(int(value) for value in encoding["ids"][:encoding["state_length"]])
        state_positions = tuple(int(value) for value in encoding["pos"][:encoding["state_length"]])
        if state_ids != prefix.state_token_ids or state_positions != prefix.state_positions:
            raise ValueError("encoding state does not match prefix cache")
        _, _, questions = rows_of(encoding)
        return self.forward_prefix_batch(prefix, questions)

    def forward(self, encodings: list[dict] | dict) -> list[list[torch.Tensor]] | list[torch.Tensor]:
        if isinstance(encodings, dict):
            return self.forward_one(encodings)
        return self.forward_rows_batch(encodings)

    @torch.no_grad()
    def probs(self, encodings: list[dict] | dict) -> list[list[torch.Tensor]] | list[torch.Tensor]:
        logits = self.forward(encodings)
        if isinstance(logits, list) and (not logits or isinstance(logits[0], torch.Tensor)):
            return [self.head.probabilities(value) for value in logits]
        return [[self.head.probabilities(value) for value in record] for record in logits]


def build_tiny_model(*, layers: int = 6, hidden_size: int = 64, window: int = 8,
                     temperature: float = 1.0) -> GemmaRowModel:
    from transformers import Gemma3TextModel

    torch.manual_seed(0)
    return GemmaRowModel(Gemma3TextModel(_tiny_config(layers, hidden_size, window)), temperature=temperature)


def check_model(*, tiny: bool = False, config_path: str | None = None, device: str | None = None) -> dict[str, Any]:
    """CLI diagnostic with an explicit boundary between random and pretrained modes."""
    if tiny and config_path:
        return {"status": "failed", "error": "--tiny and --config are mutually exclusive"}
    if tiny:
        model = build_tiny_model()
        return _run_diagnostic(model, pretrained=False, model_name="actual Gemma3TextModel", device=device)
    config = load_config(config_path or "configs/gemma3-1b-v7.toml")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
        markers = MarkerMap.load("runs/reference/model-marker-map.json", tokenizer)
        model = GemmaRowModel(load_real_backbone(config.model.name, config.model.revision))
        return _run_diagnostic(model, pretrained=True, model_name=config.model.name,
                               tokenizer=tokenizer, markers=markers, device=device)
    except Exception as exc:
        message = str(exc)
        gated = any(word in message.lower() for word in (
            "401", "403", "gated", "authorized", "token", "access", "couldn't connect", "offline",
            "certificate verify failed",
        ))
        return {"status": "blocked" if gated else "failed", "pretrained": False,
                "error": f"{type(exc).__name__}: {message}"}


def _diagnostic_record() -> dict[str, Any]:
    return {"state": "A short validated state.", "questions": {
        "truth": {"type": "noul", "instructions": "Is this statement supported?", "label": True},
        "choice": {"type": "choice", "instructions": "Choose the supported option.",
                   "criteria": {"alpha": "The first option.", "beta": "The second option."},
                   "label": "alpha"},
    }}


def _run_diagnostic(model: GemmaRowModel, *, pretrained: bool, model_name: str,
                    tokenizer: Any | None = None, markers: MarkerMap | None = None,
                    device: str | None = None) -> dict[str, Any]:
    """Exercise the decoder, row isolation, gradients, and a single optimizer step."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if tokenizer is None:
        # Tiny IDs are intentionally synthetic and stay inside the tiny vocabulary.
        encoded_record = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1],
                          "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, -1, -1, 0, 1, -2],
                          "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    else:
        encoded_record = encode(tokenizer, materialize(_diagnostic_record()), markers,
                                state_cap=10**9, branch_cap=10**9, packed_cap=10**9)
    model = model.to(device)
    model.eval()
    solo = model.probs(encoded_record)
    joint = model.probs([encoded_record, encoded_record])
    max_delta = max(float((a - b).abs().max()) for a, b in zip(solo, joint[0]))
    model.train()
    labels = encoded_record["labels"]
    logits = model.forward_one(encoded_record)
    loss = sum(torch.nn.functional.cross_entropy(value.unsqueeze(0), torch.tensor([label], device=device))
               for value, label in zip(logits, labels))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients)
    head_gradients = [parameter.grad for parameter in model.head.parameters()]
    lora_b_gradients = [parameter.grad for name, parameter in model.named_parameters()
                        if "lora_B" in name and parameter.requires_grad]
    nonzero = lambda values: sum(int(torch.count_nonzero(value).item()) for value in values if value is not None)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    optimizer.step()
    frozen_unchanged = all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                           if name in frozen)
    marker_stats = None
    if markers is not None:
        embedding = model.decoder.base_model.model.embed_tokens.weight.detach().float()
        vectors = [embedding[markers.ids[role]].cpu() for role in ("state", "question", "option_start", "option_end", "decide")]
        distances = [float(torch.linalg.vector_norm(vectors[i] - vectors[j])) for i in range(5) for j in range(i)]
        marker_stats = {"ids": markers.ids, "norms": {role: float(vectors[i].norm()) for i, role in enumerate(markers.ids)},
                        "pairwise_distance_min": min(distances), "pairwise_distance_max": max(distances),
                        "aliases": len(set(markers.ids.values())) != 5}
    return {"status": "passed" if finite_gradients and frozen_unchanged and nonzero(lora_b_gradients) > 0 else "failed",
            "model": model_name, "pretrained": pretrained, "device": device,
            "max_joint_solo_probability_delta": max_delta, "loss": float(loss.detach().cpu()),
            "finite_gradients": finite_gradients, "head_gradient_nonzero": nonzero(head_gradients),
            "lora_B_gradient_nonzero": nonzero(lora_b_gradients), "frozen_weights_unchanged": frozen_unchanged,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "markers": marker_stats}
