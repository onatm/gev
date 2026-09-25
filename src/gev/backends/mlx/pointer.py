"""Float32 pointer readout kept small and backend-local."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class MlxPointerHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 256, temperature: float = 1.0):
        super().__init__()
        self.query = nn.Linear(hidden_size, width, bias=True)
        self.key = nn.Linear(hidden_size, width, bias=True)
        self.temperature = float(temperature)

    def __call__(self, decision, options):
        query = self.query(decision.astype(mx.float32))
        keys = self.key(options.astype(mx.float32))
        return (keys @ query) / math.sqrt(keys.shape[-1])

    def probabilities(self, logits):
        return mx.softmax(logits.astype(mx.float32) / self.temperature, axis=-1)
