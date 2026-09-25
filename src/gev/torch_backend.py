"""PyTorch backend (CUDA, MPS, CPU): frozen Gemma decoder + PEFT LoRA + pointer head."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import checkpoint
from .config import Config
from .encoding import flatten_rows


class PointerHead(nn.Module):
    """Score each option-end hidden state against the question's decide state (FP32)."""

    def __init__(self, hidden_size: int, width: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.key = nn.Linear(hidden_size, width, dtype=torch.float32)

    def forward(self, decision: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        query, keys = self.query(decision.float()), self.key(options.float())
        return (keys @ query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.key.out_features)


def select_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if not available.get(requested):
        raise RuntimeError(f"torch device {requested!r} is unavailable")
    return requested


def _gemma4_text_decoder(name: str, revision: str, dtype: torch.dtype, attn_implementation: str) -> nn.Module:
    """Load only the text decoder of the multimodal Gemma 4 checkpoint."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, Gemma4TextModel

    path = Path(snapshot_download(name, revision=revision, allow_patterns=["config.json", "*.safetensors*"]))
    text_config = AutoConfig.from_pretrained(path).text_config
    text_config._attn_implementation = attn_implementation
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Gemma4TextModel(text_config)
    finally:
        torch.set_default_dtype(default)
    load_text_weights(model, sorted(path.glob("*.safetensors")))
    return model


def load_text_weights(model: nn.Module, files: list[Path], prefix: str = "model.language_model.") -> None:
    """Stream ``prefix``-ed tensors into ``model``, one at a time, requiring an exact match.

    KV-shared layers reuse earlier layers' keys/values, so the checkpoint's k/v
    weights for those layers have no destination and are skipped.
    """
    from safetensors import safe_open

    expected, seen = model.state_dict(), set()
    with torch.no_grad():
        for filename in files:
            with safe_open(filename, framework="pt") as handle:
                for key in handle.keys():
                    name = key.removeprefix(prefix)
                    if not key.startswith(prefix):
                        continue
                    if name not in expected:
                        if name.split(".")[-2] in ("k_proj", "v_proj", "k_norm"):
                            continue
                        raise ValueError(f"unexpected Gemma text tensor: {name}")
                    expected[name].copy_(handle.get_tensor(key))
                    seen.add(name)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} text tensors, e.g. {sorted(missing)[:3]}")


def load_backbone(config: Config) -> nn.Module:
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float32
    model = config.model
    if model.family == "gemma4":
        backbone = _gemma4_text_decoder(model.name, model.revision, dtype, config.attn_implementation)
    else:
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(model.name, revision=model.revision, dtype=dtype,
                                                        attn_implementation=config.attn_implementation).model
    backbone.config.use_cache = False
    return backbone


class TorchRunner:
    """Owns the model and optimizer; exchanges plain encodings and NumPy logits."""

    def __init__(self, config: Config, *, backbone: nn.Module | None = None, device: str | None = None):
        from peft import get_peft_model

        self.config = config
        self.device = select_device(device or config.device)
        backbone = load_backbone(config) if backbone is None else backbone
        if config.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone = get_peft_model(backbone, checkpoint.lora_config(config))
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()  # FP32 adapter masters over a BF16 base
        self.head = PointerHead(int(backbone.config.hidden_size), config.model.pointer_width)
        self.backbone.to(self.device)
        self.head.to(self.device)
        self.pad_id = getattr(backbone.config, "pad_token_id", None) or 0
        self.optimizer = None

    # --- forward ---------------------------------------------------------------------------

    def _forward(self, encodings: list[dict]) -> list[list[torch.Tensor]]:
        rows = flatten_rows(encodings)
        width = max(len(ids) for _, ids, _, _ in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (_, row_ids, _, _) in enumerate(rows):
            ids[i, :len(row_ids)] = torch.tensor(row_ids)
            mask[i, :len(row_ids)] = 1
        hidden = self.backbone(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        result: list[list[torch.Tensor]] = [[] for _ in encodings]
        for i, (record, _, decide, options) in enumerate(rows):
            result[record].append(self.head(hidden[i, decide], hidden[i, options]))
        return result

    def logits(self, encodings: list[dict]) -> list[list[np.ndarray]]:
        self.backbone.eval(); self.head.eval()
        with torch.inference_mode():
            return [[values.float().cpu().numpy() for values in record] for record in self._forward(encodings)]

    # --- training --------------------------------------------------------------------------

    def _parameter_groups(self):
        lora = [p for p in self.backbone.parameters() if p.requires_grad]
        return lora, list(self.head.parameters())

    def init_optimizer(self) -> None:
        lora, head = self._parameter_groups()
        self.optimizer = torch.optim.AdamW(
            [{"params": lora, "lr": self.config.training.learning_rate},
             {"params": head, "lr": self.config.head_learning_rate}],
            weight_decay=self.config.training.weight_decay)

    def train_step(self, variants: list[dict], lr: float, head_lr: float) -> float:
        """One optimizer step over a logical batch; loss = mean over variants of mean question CE."""
        self.backbone.train(); self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        chunk = self.config.training.microbatch
        total = torch.zeros((), device=self.device)
        for start in range(0, len(variants), chunk):
            part = variants[start:start + chunk]
            logits = self._forward([variant["encoding"] for variant in part])
            losses = [torch.stack([_question_loss(z, q) for z, q in zip(record, variant["questions"])]).mean()
                      for record, variant in zip(logits, part)]
            loss = torch.stack(losses).sum() / len(variants)
            loss.backward()
            total += loss.detach()
        lora, head = self._parameter_groups()
        norm = torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
        if not torch.isfinite(norm) or not torch.isfinite(total):
            raise FloatingPointError("non-finite loss or gradient")
        for group, value in zip(self.optimizer.param_groups, (lr, head_lr)):
            group["lr"] = value
        self.optimizer.step()
        return float(total)

    def peak_memory(self) -> int | None:
        """Peak CUDA memory in bytes since the last call; None on MPS and CPU."""
        if self.device != "cuda":
            return None
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return peak

    # --- persistence -----------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        checkpoint.lora_config(self.config, with_base=True).save_pretrained(directory)
        adapter = {k: v.detach().float().cpu().contiguous()
                   for k, v in get_peft_model_state_dict(self.backbone).items()}
        save_file(adapter, directory / "adapter_model.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  directory / "pointer.safetensors")

    def load(self, directory: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter = load_file(directory / "adapter_model.safetensors")
        result = set_peft_model_state_dict(self.backbone, adapter)
        expected = {k for k, _ in self.backbone.named_parameters() if "lora_" in k}
        if getattr(result, "unexpected_keys", None) or len(adapter) != len(expected):
            raise ValueError("adapter tensors do not match this model's LoRA layout")
        self.head.load_state_dict(load_file(directory / "pointer.safetensors"))

    def save_optimizer(self, directory: Path) -> None:
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")

    def load_optimizer(self, directory: Path) -> None:
        self.optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location=self.device))


def _question_loss(logits: torch.Tensor, question: dict) -> torch.Tensor:
    z = logits.float()
    if question.get("target") is not None:
        target = torch.as_tensor(question["target"], dtype=z.dtype, device=z.device)
        return -(target * torch.log_softmax(z, dim=-1)).sum()
    return torch.nn.functional.cross_entropy(z.unsqueeze(0), torch.tensor([question["label"]], device=z.device))
