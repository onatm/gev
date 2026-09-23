"""Reusable local checkpoint predictors."""
from __future__ import annotations

import time
import math

import torch

from ..api_request import api_request
from ..materialize import materialize
from ..representation import to_record
from ..tokenization import encode


def predict_request(request, model, tokenizer, markers, *, state_cap, branch_cap,
                    packed_cap, temperature=1.0):
    """Predict an unlabeled Kev-style request without entering evaluation code."""
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("inference temperature must be finite and positive")
    public_request = api_request(request)
    record, _ = to_record(public_request)
    encoded = encode(tokenizer, record, markers, state_cap=state_cap,
                     branch_cap=branch_cap, packed_cap=packed_cap)

    model.eval()
    model.head.temperature = float(temperature)
    with torch.no_grad():
        probabilities = model.probs(encoded)
    questions = record["questions"]
    if len(probabilities) != len(questions):
        raise ValueError("model returned a different number of questions than the request")

    results = []
    for question, values in zip(questions, probabilities, strict=True):
        scores = [float(value) for value in values.detach().cpu().tolist()]
        if len(scores) != len(question["keys"]):
            raise ValueError(f"model returned an invalid option count for question {question['qid']!r}")
        if any(not math.isfinite(score) or score < 0 for score in scores):
            raise ValueError(f"model returned invalid probabilities for question {question['qid']!r}")
        winner_index = max(range(len(scores)), key=scores.__getitem__)
        winner = question["keys"][winner_index]
        results.append({
            "id": question["qid"],
            "type": question["qtype"],
            "probabilities": dict(zip(question["keys"], scores, strict=True)),
            "winner": winner,
            "winner_probability": scores[winner_index],
        })
    return {"inference_temperature": float(temperature), "questions": results}


class LocalPredictor:
    """Serve one record with exactly one inference-temperature application."""

    def __init__(self, model, tokenizer, markers, *, state_cap, branch_cap, packed_cap,
                 temperature=1.0, execution_mode="rows"):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("inference temperature must be finite and positive")
        self.model = model.eval()
        self.model.head.temperature = temperature
        self.tokenizer = tokenizer
        self.markers = markers
        self.caps = (state_cap, branch_cap, packed_cap)
        self.temperature = float(temperature)
        if execution_mode not in {"rows", "packed"}:
            raise ValueError("execution_mode must be rows or packed")
        self.execution_mode = execution_mode

    def __call__(self, record):
        encoded = encode(self.tokenizer, materialize(record), self.markers,
                         state_cap=self.caps[0], branch_cap=self.caps[1], packed_cap=self.caps[2])
        device = next(self.model.parameters()).device
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            # PointerHead applies this temperature in eval mode.  Do not pass
            # it again to evaluate_records; its additional temperature is 1.
            served = (self.model.forward_packed_one(encoded) if self.execution_mode == "packed"
                      else self.model.forward_one(encoded))
        if device.type == "mps":
            torch.mps.synchronize()
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
