"""The two-projection pointer readout."""

from __future__ import annotations

import math

import torch
from torch import nn


class PointerHead(nn.Module):
    """Score option-end states against the question decision state."""

    def __init__(self, hidden_size: int, width: int = 256, temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.query = nn.Linear(hidden_size, width, bias=True, dtype=torch.float32)
        self.key = nn.Linear(hidden_size, width, bias=True, dtype=torch.float32)
        self._temperature = 1.0
        self.temperature = temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be finite and positive")
        self._temperature = float(value)

    def forward(self, decision: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        query = self.query(decision.float())
        keys = self.key(options.float())
        scores = (keys @ query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.key.out_features)
        return scores if self.training or self.temperature == 1.0 else scores / self.temperature

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)
