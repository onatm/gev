"""Backend-native Gemma 4 predictor using plain Python probabilities."""

from __future__ import annotations

import math
import time

import mlx.core as mx

from ...domain.materialize import materialize
from ...domain.tokenization import encode


class MlxPredictor:
    def __init__(self, model, tokenizer, markers, *, state_cap, branch_cap, packed_cap,
                 temperature=1.0, execution_mode="rows", encoder=None):
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("inference temperature must be finite and positive")
        if execution_mode != "rows":
            raise ValueError("Gemma 4 MLX supports rows execution only")
        self.model = model.eval()
        self.model.head.temperature = float(temperature)
        self.tokenizer, self.markers = tokenizer, markers
        self.caps = (state_cap, branch_cap, packed_cap)
        self.encoder = encode if encoder is None else encoder
        self.temperature = float(temperature)
        self.execution_mode = execution_mode

    def __call__(self, record):
        encoded = self.encoder(self.tokenizer, materialize(record), self.markers,
                               state_cap=self.caps[0], branch_cap=self.caps[1],
                               packed_cap=self.caps[2])
        started = time.perf_counter()
        raw_outputs = self.model.forward_one(encoded)
        scaled_outputs = [values.astype(mx.float32) / self.temperature for values in raw_outputs]
        probabilities = [mx.softmax(values, axis=-1) for values in scaled_outputs]
        mx.eval(raw_outputs, scaled_outputs, probabilities)
        latency = (time.perf_counter() - started) * 1000
        result, logits, raw_logits = {}, {}, {}
        for (qid, question), scaled, raw, values in zip(
                record["questions"].items(), scaled_outputs, raw_outputs, probabilities, strict=True):
            keys = list(question.get("criteria", {})) if question["type"] == "choice" else (
                ["false", "true"] if question["type"] == "noul"
                else [str(index) for index in range(len(question["criteria"]))])
            scores = [float(value) for value in values.tolist()]
            scaled_values = [float(value) for value in scaled.tolist()]
            raw_values = [float(value) for value in raw.tolist()]
            if (len(scores) != len(keys) or len(scaled_values) != len(keys)
                    or len(raw_values) != len(keys)
                    or any(not math.isfinite(value) or value < 0 for value in scores)
                    or any(not math.isfinite(value) for value in scaled_values + raw_values)):
                raise ValueError(f"MLX predictor returned invalid probabilities for {qid!r}")
            result[qid] = dict(zip(keys, scores, strict=True))
            logits[qid] = dict(zip(keys, scaled_values, strict=True))
            raw_logits[qid] = dict(zip(keys, raw_values, strict=True))
        return {"probabilities": result, "logits": logits, "raw_logits": raw_logits,
                "inference_temperature": self.temperature,
                "latency_ms": latency, "token_count": len(encoded["ids"]),
                "question_count": len(raw_outputs)}
