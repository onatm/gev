"""Reusable local checkpoint predictors."""
from __future__ import annotations

import time
import math

import torch

from ...domain.materialize import materialize
from ...domain.tokenization import encode


class TorchPredictor:
    """Serve one record with exactly one inference-temperature application."""

    def __init__(self, model, tokenizer, markers, *, state_cap, branch_cap, packed_cap,
                 temperature=1.0, execution_mode="rows", encoder=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("inference temperature must be finite and positive")
        self.model = model.eval()
        self.model.head.temperature = temperature
        self.tokenizer = tokenizer
        self.markers = markers
        self.caps = (state_cap, branch_cap, packed_cap)
        self.encoder = encode if encoder is None else encoder
        self.temperature = float(temperature)
        if execution_mode not in {"rows", "packed"}:
            raise ValueError("execution_mode must be rows or packed")
        self.execution_mode = execution_mode

    def __call__(self, record):
        encoded = self.encoder(self.tokenizer, materialize(record), self.markers,
                               state_cap=self.caps[0], branch_cap=self.caps[1],
                               packed_cap=self.caps[2])
        device = next(self.model.parameters()).device
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            # PointerHead applies this temperature in eval mode.  Do not pass
            # it again to evaluate_records; its additional temperature is 1.
            served = (self.model.forward_packed_one(encoded) if self.execution_mode == "packed"
                      else self.model.forward_one(encoded))
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        probabilities, logits, raw_logits = {}, {}, {}
        for qid, question, values in zip(record["questions"], record["questions"].values(), served):
            keys = list(question.get("criteria", {})) if question["type"] == "choice" else (
                ["false", "true"] if question["type"] == "noul" else [str(i) for i in range(len(question["criteria"]))])
            scaled = [float(v) for v in values.detach().cpu()]
            raw = [v * self.temperature for v in scaled]
            logits[qid] = dict(zip(keys, scaled))
            raw_logits[qid] = dict(zip(keys, raw))
            probabilities[qid] = {key: float(p) for key, p in zip(keys, torch.softmax(torch.tensor(scaled), -1))}
        return {"probabilities": probabilities, "logits": logits, "raw_logits": raw_logits,
                "inference_temperature": self.temperature, "latency_ms": latency,
                "token_count": len(encoded["ids"]), "question_count": len(served)}
