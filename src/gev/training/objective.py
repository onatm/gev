"""The per-question and per-variant objective used by the trainer."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def question_loss(logits: torch.Tensor, question: dict) -> torch.Tensor:
    z = logits.float()
    if question.get("target") is not None:
        target = torch.as_tensor(question["target"], dtype=z.dtype, device=z.device)
        return -(target * F.log_softmax(z, dim=-1)).sum()
    return F.cross_entropy(z.unsqueeze(0), torch.tensor([question["label"]], device=z.device))

def variant_loss(logits: list[torch.Tensor], questions: list[dict]) -> torch.Tensor:
    if len(logits) != len(questions):
        raise ValueError("model returned a different number of questions")
    if not logits:
        raise ValueError("variant has no questions")
    return torch.stack([question_loss(z, q) for z, q in zip(logits, questions)]).mean()

def logical_loss(logits: list[list[torch.Tensor]], variants: list[dict]) -> torch.Tensor:
    if len(logits) != len(variants):
        raise ValueError("model returned a different number of variants")
    return torch.stack([variant_loss(z, v["record"]["questions"]) for z, v in zip(logits, variants)]).sum() / len(variants)
