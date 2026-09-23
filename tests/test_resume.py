import dataclasses
import pytest

import torch

from gev.config import ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig
from gev.models.gemma import build_tiny_model
from gev.training.batching import Variant
from gev.training.loop import _optimizer_state_cpu, train
from gev.tokenization import MarkerMap


def _request(i):
    return {"state": "state", "questions": {"q": {"type": "choice", "instructions": "choose", "criteria": {"a": "a", "b": "b"}, "label": "a"}}, "_meta": {"id": f"r{i}", "source": "fixture", "group_id": f"g{i}"}}


@pytest.mark.parametrize("record_count,logical_batch,interrupt,full_steps", [(12, 1, 7, 24), (12, 1, 12, 24), (25, 2, 7, 26)])
def test_resume_uses_actual_train_and_preserves_boundary(monkeypatch, tmp_path, record_count, logical_batch, interrupt, full_steps):
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": list(range(7)), "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    def variants(request, **_):
        return [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")]
    monkeypatch.setattr("gev.training.loop.variants_for_request", variants)
    config = ExperimentConfig("resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(3, 2, 1e-3, "fp32", 384, logical_batch=logical_batch, microbatch=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("cpu", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(record_count)]
    full = tmp_path / "full"
    train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, config, full, source_hash="source", manifest={"m": "manifest"}, progress=False)
    partial_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=interrupt, save_every=1))
    partial = tmp_path / "partial"
    train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, partial_config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False)
    resumed_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=full_steps, save_every=1))
    train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, resumed_config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=partial / "last_good.resume.pt")
    full_state = torch.load(full / "last_good.resume.pt", weights_only=False)["trainable_state"]
    resumed_state = torch.load(partial / "last_good.resume.pt", weights_only=False)["trainable_state"]
    for key in full_state:
        torch.testing.assert_close(full_state[key], resumed_state[key], rtol=0, atol=0)
    full_snapshot = torch.load(full / "last_good.resume.pt", weights_only=False)
    resumed_snapshot = torch.load(partial / "last_good.resume.pt", weights_only=False)
    assert resumed_snapshot["complete"] is True
    assert full_snapshot["augmentation_digest"] == resumed_snapshot["augmentation_digest"]
    assert full_snapshot["metrics"]["learning_rates"] == resumed_snapshot["metrics"]["learning_rates"]
    assert full_snapshot["metrics"]["logical_steps"] == resumed_snapshot["metrics"]["logical_steps"] == full_steps
    assert full_snapshot["scheduler"] == resumed_snapshot["scheduler"]
    assert full_snapshot["optimizer"]["param_groups"] == resumed_snapshot["optimizer"]["param_groups"]
    assert full_snapshot["shuffle_rng"] == resumed_snapshot["shuffle_rng"]


def test_resume_rejects_source_config_and_marker_changes(monkeypatch, tmp_path):
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": list(range(7)), "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    monkeypatch.setattr("gev.training.loop.variants_for_request", lambda request, **_: [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")])
    config = ExperimentConfig("resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(3, 2, 1e-3, "fp32", 384, logical_batch=1, max_steps=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("cpu", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(6)]
    out = tmp_path / "partial"
    train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, config, out, source_hash="source", manifest={"m": "manifest"}, progress=False)
    with pytest.raises(ValueError, match="mismatch"):
        train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, config, out, source_hash="different", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")
    with pytest.raises(ValueError, match="mismatch"):
        train(build_tiny_model(layers=6, hidden_size=32), requests, None, dataclasses.replace(markers, ids={**markers.ids, "state": 11}), config, out, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")
    with pytest.raises(ValueError, match="mismatch"):
        altered = dataclasses.replace(config, training=dataclasses.replace(config.training, learning_rate=2e-3))
        train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, altered, out, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")


def test_optimizer_snapshot_does_not_move_or_alias_live_state():
    model = build_tiny_model(layers=6, hidden_size=32)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    (model.head.query.weight.square().mean()).backward()
    optimizer.step()

    live = {key: value for state in optimizer.state.values() for key, value in state.items() if isinstance(value, torch.Tensor)}
    copied = _optimizer_state_cpu(optimizer)
    for key, value in live.items():
        assert value.device.type == "cpu"
        assert all(value is not copied_value for copied_state in copied["state"].values() for copied_key, copied_value in copied_state.items() if copied_key == key and isinstance(copied_value, torch.Tensor))
    (model.head.query.weight.square().mean()).backward()
    optimizer.step()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_resume_keeps_adam_step_scalar_cpu(monkeypatch, tmp_path):
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": list(range(7)), "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    monkeypatch.setattr("gev.training.loop.variants_for_request", lambda request, **_: [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")])
    config = ExperimentConfig("mps-resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(11, 1, 1e-3, "fp32", 384, logical_batch=1, microbatch=1, max_steps=1, save_every=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("mps", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(11)]
    partial = tmp_path / "mps-partial"
    train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False)
    snapshot = torch.load(partial / "last_good.resume.pt", weights_only=False)
    assert all(value.device.type == "cpu" for state in snapshot["optimizer"]["state"].values() for value in state.values() if isinstance(value, torch.Tensor))
    resumed = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=11))
    metrics = train(build_tiny_model(layers=6, hidden_size=32), requests, None, markers, resumed, partial, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=partial / "last_good.resume.pt")
    assert metrics["complete"] and metrics["logical_steps"] == 11
