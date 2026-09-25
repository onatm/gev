"""Load a trained checkpoint, predict requests, and evaluate suites."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import checkpoint, data, metrics
from .encoding import encode
from .records import materialize, public_request


class Model:
    """A checkpoint bound to a backend runner, its tokenizer, and its serving temperature."""

    def __init__(self, source: str | Path, *, backend: str | None = None, device: str | None = None,
                 runner=None, tokenizer=None):
        from .train import load_tokenizer, make_runner

        path = checkpoint.resolve(source)
        self.path = path if (path / "gev.json").exists() else checkpoint.download(str(source))
        self.metadata = checkpoint.read_metadata(self.path)
        config = checkpoint.config_of(self.path)
        if backend and backend != config.backend:
            config = config.replace(backend=backend, device="auto")
        if device:
            config = config.replace(device=device)
        self.config = config
        if tokenizer is None:
            self.tokenizer, self.markers = load_tokenizer(config)
        else:
            from .encoding import Markers

            self.tokenizer, self.markers = tokenizer, Markers.resolve(tokenizer, config.model.markers)
        checkpoint.check_markers(self.metadata, self.markers)
        self.runner = make_runner(config) if runner is None else runner
        self.runner.load(self.path)
        self.temperature = float(self.metadata["temperature"])

    def encode(self, record: dict) -> dict:
        return encode(self.tokenizer, record, self.markers,
                      state_cap=self.metadata["state_cap"], branch_cap=self.metadata["branch_cap"])

    def raw_logits(self, record: dict) -> list[np.ndarray]:
        """Raw (T=1) pointer logits for a materialized record, one array per question.

        Each question row runs alone: in BF16, batching rows of different lengths
        changes kernel shapes and shifts logits (up to ~0.5 on Gemma 4 E2B), so a
        score would otherwise depend on its neighbours. Training still batches.
        """
        encoding = self.encode(record)
        return [self.runner.logits([{"state": encoding["state"], "rows": [row]}])[0][0]
                for row in encoding["rows"]]

    def predict(self, request: dict, temperature: float | None = None) -> dict:
        temperature = self.temperature if temperature is None else temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        record = materialize(public_request(request), labelled=False)
        answers = {}
        for question, logits in zip(record["questions"], self.raw_logits(record), strict=True):
            p = metrics.probabilities({"logits": logits}, temperature)
            answers[question["id"]] = {"probabilities": dict(zip(question["keys"], p.tolist())),
                                       "answer": question["keys"][int(p.argmax())]}
        return {"temperature": temperature, "questions": answers}


def rows_for(request: dict, record: dict, logits: list[np.ndarray]) -> list[dict]:
    """One scoring row per question, carrying raw logits and the metadata metrics group by."""
    meta = request.get("_meta", {})
    return [{"id": meta["id"], "group": meta.get("group_id", meta["id"]), "question": question["id"],
             "source": meta.get("source", "unknown"), "task": question.get("src"), "type": question["type"],
             "variant": meta.get("variant", "clean"), "keys": question["keys"], "label": question["label"],
             "logits": [float(v) for v in values], "pair_id": meta.get("pair_id"),
             "sibling": meta.get("sibling"), "control_id": meta.get("control_id")}
            for question, values in zip(record["questions"], logits, strict=True)]


def evaluate(model: Model, *, suite: str, split: str, data_root: str | Path, out: str | Path,
             log=print) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    requests, data_sha256 = data.load_split(data_root, suite, split)
    rows = []
    for index, request in enumerate(requests, 1):
        record = materialize(request)
        rows.extend(rows_for(request, record, model.raw_logits(record)))
        if index % 100 == 0 or index == len(requests):
            log(f"{index}/{len(requests)} records")
    report = {"suite": suite, "split": split, "data_sha256": data_sha256, "checkpoint": str(model.path),
              "base_model": model.metadata["base_model"], "backend": model.config.backend,
              "records": len(requests), "questions": len(rows),
              **metrics.summarize(rows, model.temperature)}
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def load_evaluation(directory: str | Path) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return report, rows


def calibrate(evaluation: str | Path, *, update: bool = False) -> dict:
    """Fit a serving temperature on saved calibration/development rows; optionally store it."""
    report, rows = load_evaluation(evaluation)
    if report["split"] == "test":
        raise ValueError("never fit a temperature on test data")
    temperature = metrics.fit_temperature(rows)
    clean = metrics.scored_rows(rows)
    result = {"temperature": temperature, "fitted_on": f"{report['suite']}/{report['split']}",
              "raw": metrics.metrics(clean), "calibrated": metrics.metrics(clean, temperature)}
    if update:
        checkpoint.set_temperature(report["checkpoint"], temperature)
    return result


def compare(candidate: str | Path, reference: str | Path, *, samples: int = 1000, seed: int = 0) -> dict:
    (a_report, a_rows), (b_report, b_rows) = load_evaluation(candidate), load_evaluation(reference)
    if (a_report["suite"], a_report["split"], a_report["data_sha256"]) != \
            (b_report["suite"], b_report["split"], b_report["data_sha256"]):
        raise ValueError("evaluations are on different data")
    return {"suite": a_report["suite"], "split": a_report["split"],
            "candidate": a_report["clean"], "reference": b_report["clean"],
            "paired": {m: metrics.paired_bootstrap(a_rows, b_rows, metric=m, samples=samples, seed=seed)
                       for m in ("acc", "nll", "brier", "ece")}}
