import dataclasses
import pytest

import torch

from gev.configuration.config import ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig
from gev.backends.torch.gemma3 import build_tiny_model
from gev.training.batching import Variant
from gev.backends.torch.training import _optimizer_state_cpu, load_training_state, train
from gev.training.schedule import TrainingSchedule
from gev.domain.tokenization import MarkerMap


def _request(i):
    return {"state": "state", "questions": {"q": {"type": "choice", "instructions": "choose", "criteria": {"a": "a", "b": "b"}, "label": "a"}}, "_meta": {"id": f"r{i}", "source": "fixture", "group_id": f"g{i}"}}


def _train(model, requests, markers, config, output, **kwargs):
    schedule = TrainingSchedule(requests, config, None, markers)
    return train(model, schedule, config, output, **kwargs)


@pytest.mark.parametrize("record_count,logical_batch,interrupt,full_steps", [(12, 1, 7, 24), (12, 1, 12, 24), (25, 2, 7, 26)])
def test_resume_uses_actual_train_and_preserves_boundary(monkeypatch, tmp_path, record_count, logical_batch, interrupt, full_steps):
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": list(range(7)), "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    def variants(request, **_):
        return [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")]
    monkeypatch.setattr("gev.training.schedule.variants_for_request", variants)
    config = ExperimentConfig("resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(3, 2, 1e-3, "fp32", 384, logical_batch=logical_batch, microbatch=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("cpu", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(record_count)]
    full = tmp_path / "full"
    _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, config, full, source_hash="source", manifest={"m": "manifest"}, progress=False)
    partial_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=interrupt, save_every=1))
    partial = tmp_path / "partial"
    _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, partial_config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False)
    resumed_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=full_steps, save_every=1))
    _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, resumed_config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=partial / "last_good.resume.pt")
    full_state = torch.load(full / "last_good.resume.pt", weights_only=False)["torch_state"]["trainable_state"]
    resumed_state = torch.load(partial / "last_good.resume.pt", weights_only=False)["torch_state"]["trainable_state"]
    for key in full_state:
        torch.testing.assert_close(full_state[key], resumed_state[key], rtol=0, atol=0)
    full_snapshot = torch.load(full / "last_good.resume.pt", weights_only=False)
    resumed_snapshot = torch.load(partial / "last_good.resume.pt", weights_only=False)
    assert full_snapshot["format"] == "gev.logical-resume"
    assert full_snapshot["version"] == 1
    assert load_training_state(full / "last_good.resume.pt")["version"] == 1
    assert set(full_snapshot) == {"format", "version", "identity", "progress", "torch_state"}
    assert resumed_snapshot["progress"]["complete"] is True
    assert full_snapshot["progress"]["augmentation_digest"] == resumed_snapshot["progress"]["augmentation_digest"]
    assert full_snapshot["progress"]["metrics"]["learning_rates"] == resumed_snapshot["progress"]["metrics"]["learning_rates"]
    assert full_snapshot["progress"]["metrics"]["logical_steps"] == resumed_snapshot["progress"]["metrics"]["logical_steps"] == full_steps
    assert full_snapshot["torch_state"]["scheduler"] == resumed_snapshot["torch_state"]["scheduler"]
    assert full_snapshot["torch_state"]["optimizer"]["param_groups"] == resumed_snapshot["torch_state"]["optimizer"]["param_groups"]
    assert full_snapshot["progress"]["shuffle_rng"] == resumed_snapshot["progress"]["shuffle_rng"]


def test_resume_snapshot_rejects_non_v1_and_unknown_schema_fields(tmp_path):
    snapshot = {
        "format": "gev.logical-resume",
        "version": 1,
        "identity": {"recipe_sha256": "a" * 64, "recipe": {}, "identity": {},
                     "source": {}, "training": {}, "execution": {}},
        "progress": {"complete": False, "shuffle_rng": (), "epoch": 0,
                     "next_batch": 0, "order": [], "global_step": 0,
                     "metrics": {}, "augmentation_digest": "b" * 64},
        "torch_state": {"trainable_state": {}, "optimizer": {}, "scheduler": {}, "rng": {}},
    }
    path = tmp_path / "snapshot.pt"
    torch.save(snapshot, path)
    assert load_training_state(path) == snapshot

    invalid_snapshots = []
    invalid_snapshots.extend((dict(snapshot, version=3), dict(snapshot, version=True),
                              dict(snapshot, version=1.0)))
    extra_top_level = dict(snapshot, future_field=True)
    invalid_snapshots.append(extra_top_level)
    extra_progress = dict(snapshot, progress={**snapshot["progress"], "future_field": True})
    invalid_snapshots.append(extra_progress)
    extra_torch_state = dict(snapshot, torch_state={**snapshot["torch_state"], "future_field": True})
    invalid_snapshots.append(extra_torch_state)
    invalid_shuffle_state = dict(snapshot, progress={**snapshot["progress"], "shuffle_rng": []})
    invalid_snapshots.append(invalid_shuffle_state)
    for invalid_snapshot in invalid_snapshots:
        torch.save(invalid_snapshot, path)
        with pytest.raises(ValueError, match="invalid or incomplete resume snapshot"):
            load_training_state(path)


def test_resume_rejects_source_config_and_marker_changes(monkeypatch, tmp_path):
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": list(range(7)), "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    monkeypatch.setattr("gev.training.schedule.variants_for_request", lambda request, **_: [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")])
    config = ExperimentConfig("resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(3, 2, 1e-3, "fp32", 384, logical_batch=1, max_steps=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("cpu", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(6)]
    out = tmp_path / "partial"
    _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, config, out, source_hash="source", manifest={"m": "manifest"}, progress=False)
    with pytest.raises(ValueError, match="mismatch"):
        _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, config, out, source_hash="different", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")
    with pytest.raises(ValueError, match="mismatch"):
        _train(build_tiny_model(layers=6, hidden_size=32), requests, dataclasses.replace(markers, ids={**markers.ids, "state": 11}), config, out, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")
    with pytest.raises(ValueError, match="mismatch"):
        altered = dataclasses.replace(config, training=dataclasses.replace(config.training, learning_rate=2e-3))
        _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, altered, out, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=out / "last_good.resume.pt")


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
    monkeypatch.setattr("gev.training.schedule.variants_for_request", lambda request, **_: [Variant({"questions": [{"label": 0}]}, encoding, request["_meta"]["id"], "fixture")])
    config = ExperimentConfig("mps-resume", ModelConfig("tiny", "0" * 40, "gemma3_text"), TrainingConfig(11, 1, 1e-3, "fp32", 384, logical_batch=1, microbatch=1, max_steps=1, save_every=1, p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0), RuntimeConfig("mps", False, "runs"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10}, {"state": "s", "question": "q", "option_start": "os", "option_end": "oe", "decide": "d"}, "0" * 40)
    requests = [_request(i) for i in range(11)]
    partial = tmp_path / "mps-partial"
    _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, config, partial, source_hash="source", manifest={"m": "manifest"}, progress=False)
    snapshot = torch.load(partial / "last_good.resume.pt", weights_only=False)
    assert all(value.device.type == "cpu" for state in snapshot["torch_state"]["optimizer"]["state"].values() for value in state.values() if isinstance(value, torch.Tensor))
    resumed = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=11))
    metrics = _train(build_tiny_model(layers=6, hidden_size=32), requests, markers, resumed, partial, source_hash="source", manifest={"m": "manifest"}, progress=False, resume=partial / "last_good.resume.pt")
    assert metrics["complete"] and metrics["logical_steps"] == 11
