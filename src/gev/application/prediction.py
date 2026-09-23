"""Inference on one unlabeled request, outside the evaluation pipeline."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from ..artifacts.checkpoint_identity import read_checkpoint_manifest
from ..backends.torch.environment import configure_runtime
from ..configuration.resolved import resolve_experiment_config
from ..domain.api_request import api_request
from ..domain.representation import to_record
from .evaluation import config_for_run


def predict_request(request: dict, model, tokenizer, markers, *, encoder,
                    state_cap: int, branch_cap: int, packed_cap: int,
                    temperature: float = 1.0) -> dict:
    """Predict without labels, suite access, or evaluation artifacts."""
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("inference temperature must be finite and positive")
    record, _ = to_record(api_request(request))
    encoded = encoder(tokenizer, record, markers, state_cap=state_cap,
                      branch_cap=branch_cap, packed_cap=packed_cap)
    model.eval()
    model.head.temperature = float(temperature)
    import torch

    with torch.no_grad():
        probabilities = model.probs(encoded)
    if len(probabilities) != len(record["questions"]):
        raise ValueError("model returned a different number of questions than the request")

    results = []
    for question, values in zip(record["questions"], probabilities, strict=True):
        scores = [float(value) for value in values.detach().cpu().tolist()]
        if len(scores) != len(question["keys"]):
            raise ValueError(f"model returned an invalid option count for question {question['qid']!r}")
        if any(not math.isfinite(score) or score < 0 for score in scores):
            raise ValueError(f"model returned invalid probabilities for question {question['qid']!r}")
        winner_index = max(range(len(scores)), key=scores.__getitem__)
        results.append({
            "id": question["qid"],
            "type": question["qtype"],
            "probabilities": dict(zip(question["keys"], scores, strict=True)),
            "winner": question["keys"][winner_index],
            "winner_probability": scores[winner_index],
        })
    return {"inference_temperature": float(temperature), "questions": results}


def predict_stage(run: str | Path, request: dict, *, config_path: str | Path | None = None,
                  temperature: float = 1.0, device: str | None = None) -> dict:
    """Load a current-format checkpoint through its registered model/backend."""
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("inference temperature must be finite and positive")
    config = config_for_run(config_path, run)
    if device is not None:
        config = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, device=device))
    resolved = resolve_experiment_config(config)
    configure_runtime(config.runtime.device, config.runtime.mps_fallback)
    resolved.validate_runtime_available()
    checkpoint = Path(run) / "checkpoint" if (Path(run) / "checkpoint").exists() else Path(run)
    metadata = read_checkpoint_manifest(checkpoint)
    if any(metadata["execution"].get(key) != getattr(config.training, key)
           for key in ("state_cap", "branch_cap", "packed_cap")):
        raise ValueError("prediction config caps do not match checkpoint contract")
    tokenizer = resolved.load_tokenizer()
    markers = resolved.load_markers(tokenizer)
    model, _ = resolved.load_checkpoint(checkpoint, device=resolved.select_device(),
                                        tokenizer=tokenizer, expected_marker_map=markers)
    return predict_request(request, model, tokenizer, markers,
                           encoder=resolved.model.family_runtime.encode_record,
                           state_cap=config.training.state_cap,
                           branch_cap=config.training.branch_cap,
                           packed_cap=config.training.packed_cap, temperature=temperature)
