import dataclasses
import hashlib
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma4TextModel

from gev.application import evaluation as evaluation_stage
from gev.application import prediction as prediction_stage
from gev.application import training as training_stage
from gev.backends.torch import TorchBackend
from gev.backends.torch.gemma4 import _tiny_config
from gev.backends.torch.training import create_optimizer, optimizer_step
from gev.configuration.config import load_config
from gev.configuration.resolved import resolve_experiment_config
from gev.domain.tokenization import MarkerMap
from gev.models.families import Gemma4E2BTextRuntime
from gev.training.batching import Variant


def _encoding(question_count=2):
    ids, seg, pos, options, decides = [6, 2], [0, 0], [0, 1], [], []
    for index in range(question_count):
        start = len(ids)
        branch = [7, 3, 8, 12, 9, 8, 13, 9, 10]
        ids.extend(branch)
        seg.extend([index + 1] * len(branch))
        pos.extend(range(2, 2 + len(branch)))
        options.append([start + 4, start + 7])
        decides.append(start + 8)
    return {"ids": ids, "seg": seg, "pos": pos, "opt": [],
            "decide_idx": decides, "opt_idx": options,
            "labels": [0, 1][:question_count], "state_length": 2}


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_gemma4_train_stage_checkpoint_evaluate_and_predict_with_tiny_torch_model(
        monkeypatch, tmp_path, compute_dtype):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = dataclasses.replace(
        config,
        backend=dataclasses.replace(config.backend, id="torch"),
        training=dataclasses.replace(config.training, dtype=compute_dtype),
        runtime=dataclasses.replace(config.runtime, device="cpu"),
    )
    source_provenance = {
        "model": config.model.name, "revision": config.model.revision,
        "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
        "source_text_tensor_count": 600,
        "source_text_parameters": 4_647_449_891,
        "effective_text_parameters": 4_628_569_344,
        "source_name_shape_sha256": "d" * 64,
    }
    torch.manual_seed(7)
    template = Gemma4TextModel(_tiny_config())
    base_state = {name: value.detach().clone() for name, value in template.state_dict().items()}

    def tiny_base(_name=None, _revision=None, *, compute_dtype):
        backbone = Gemma4TextModel(_tiny_config())
        backbone.load_state_dict(base_state)
        backbone._gev_source = {**source_provenance, "compute_dtype": compute_dtype}
        return backbone

    create_model = TorchBackend.create_model
    load_checkpoint = TorchBackend.load_checkpoint

    def create_tiny_model(self, family, **kwargs):
        if family.family_id == "gemma4_e2b_text" and kwargs.get("backbone") is None:
            kwargs["backbone"] = tiny_base(
                kwargs["model_name"], kwargs["revision"],
                compute_dtype=kwargs["compute_dtype"])
        return create_model(self, family, **kwargs)

    def load_tiny_checkpoint(self, directory, **kwargs):
        if kwargs.get("backbone_loader") is None:
            kwargs["backbone_loader"] = tiny_base
        return load_checkpoint(self, directory, **kwargs)

    monkeypatch.setattr(TorchBackend, "create_model", create_tiny_model)
    monkeypatch.setattr(TorchBackend, "load_checkpoint", load_tiny_checkpoint)

    markers = MarkerMap(
        config.model.marker_ids,
        {role: f"<unused{index}>" for index, role in enumerate(config.model.marker_ids)},
        config.model.revision,
    )
    tokenizer = SimpleNamespace(save_pretrained=lambda _directory: None)
    request = {"state": "tiny state", "_meta": {
        "id": "fixture/one", "source": "fixture", "group_id": "fixture/one"},
        "questions": {
            "q1": {"type": "choice", "src": "fixture", "instructions": "choose",
                   "criteria": {"a": "A", "b": "B"}, "label": "a"},
            "q2": {"type": "choice", "src": "fixture", "instructions": "choose",
                   "criteria": {"a": "A", "b": "B"}, "label": "b"},
        }}
    train_rows, development_rows = [request], [request]

    monkeypatch.setattr(Gemma4E2BTextRuntime, "load_tokenizer",
                        lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(Gemma4E2BTextRuntime, "load_markers",
                        lambda *_args, **_kwargs: markers)
    monkeypatch.setattr(Gemma4E2BTextRuntime, "encode_record",
                        lambda _self, _tokenizer, record, _markers, **_caps:
                        {**_encoding(len(record["questions"])),
                         "metadata": record.get("_meta", {})})
    monkeypatch.setattr(training_stage, "load_config", lambda _path: config)
    monkeypatch.setattr(training_stage, "_configure_runtime", lambda *_args: None)
    monkeypatch.setattr(training_stage, "load_verified_split",
                        lambda _root, _suite, split, **_kwargs:
                        (train_rows if split == "train" else development_rows, {},
                         ("a" if split == "train" else "b") * 64))
    monkeypatch.setattr(training_stage, "validate_training_rows", lambda *_args: None)
    monkeypatch.setattr(training_stage, "split_path", lambda *_args: tmp_path / "train.jsonl")
    monkeypatch.setattr(
        training_stage, "file_digest",
        lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if Path(path).is_file() else "a" * 64)

    def one_real_update(_self, model, _schedule, training_config, _output, **_kwargs):
        optimizer, trainable, _head_lr = create_optimizer(model, training_config)
        variant = Variant({"questions": [{"label": 0}, {"label": 1}]},
                          _encoding(), "fixture/one", "fixture")
        before = [parameter.detach().clone() for parameter in trainable]
        loss, _microbatches, _tokens = optimizer_step(
            model, [variant], training_config, optimizer, device="cpu",
            trainable_parameters=trainable)
        updated = any(not torch.equal(old, new.detach())
                      for old, new in zip(before, trainable, strict=True))
        metrics = {
            "device": "cpu", "dtype": compute_dtype, "compute_dtype": compute_dtype,
            "weights_dtype": compute_dtype, "source_weights_dtype": "bf16",
            "complete": False, "logical_steps": 1,
            "qualification_checks": {
                "finite_loss_gradients": bool(torch.isfinite(loss)),
                "nonzero_trainable_update": updated,
                "fp32_optimizer_state": all(
                    value.dtype == torch.float32
                    for state in optimizer.state.values() for value in state.values()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()),
            },
        }
        from gev.backends.torch.qualification import collect_torch_training_checks

        metrics["qualification_checks"].update(
            collect_torch_training_checks(model, metrics))
        return metrics

    monkeypatch.setattr(TorchBackend, "train", one_real_update)
    output = tmp_path / "tiny-run"
    training_result = training_stage.train_stage(
        "tiny-gemma4.toml", data_root=tmp_path, output=output)

    receipt = training_result["qualification"]
    assert receipt["status"] == "passed"
    assert receipt["backend"] == "torch"
    assert receipt["base"]["source_weights_dtype"] == "bf16"
    assert receipt["base"]["compute_dtype"] == compute_dtype
    assert receipt["checkpoint_ready"] is True
    assert (output / "checkpoint" / "manifest.json").is_file()
    manifest = json.loads((output / "checkpoint" / "manifest.json").read_text())
    run_receipt = json.loads((output / "qualification.json").read_text())
    checkpoint_receipt = json.loads(
        (output / "checkpoint" / "qualification.json").read_text())
    assert manifest["qualification"] == run_receipt == checkpoint_receipt

    monkeypatch.setattr(evaluation_stage, "config_for_run", lambda *_args: config)
    monkeypatch.setattr(evaluation_stage, "_configure_runtime", lambda *_args: None)
    monkeypatch.setattr(evaluation_stage, "load_verified_split",
                        lambda *_args, **_kwargs: (development_rows, {}, "b" * 64))
    monkeypatch.setattr(evaluation_stage, "split_path", lambda *_args: tmp_path / "development.jsonl")
    monkeypatch.setattr(evaluation_stage, "file_digest", lambda _path: "b" * 64)
    evaluation = evaluation_stage.evaluate_stage(
        output, suite="decision-v7", split="development", data_root=tmp_path,
        output=tmp_path / "evaluated")
    assert evaluation["provenance"]["backend"] == "torch"
    assert evaluation["coverage"]["evaluated_records"] == 1

    monkeypatch.setattr(prediction_stage, "config_for_run", lambda *_args: config)
    monkeypatch.setattr(prediction_stage, "_configure_runtime", lambda *_args: None)
    prediction = prediction_stage.predict_stage(
        output, {"state": "tiny state", "questions": {
            "q": {"type": "choice", "src": "fixture", "instructions": "choose",
                  "criteria": {"a": "A", "b": "B"}}}}, device="cpu")
    assert prediction["questions"][0]["winner"] in {"a", "b"}
