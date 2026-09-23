import torch
import pytest
import random
from dataclasses import replace

from gev.backends.torch.training import create_optimizer, optimizer_step
from gev.configuration.config import load_config
from gev.backends.torch.gemma3 import build_tiny_model
from gev.training.batching import Variant, physical_token_count


def _variant():
    encoding = {"ids": [2, 3, 4, 5, 6, 7, 8],
                "seg": [0, 0, 1, 1, 1, 1, 1],
                "pos": [0, 1, 2, 3, 4, 5, 6],
                "opt": [-1, -1, -1, -1, 0, 1, -2],
                "decide_idx": [6], "opt_idx": [[4, 5]],
                "labels": [0], "state_length": 2}
    record = {"questions": [{"label": 0}]}
    return Variant(record, encoding, "fixture/1", "fixture")


def test_torch_production_optimizer_step_is_shared_with_profile_and_updates_trainables():
    from gev.backends.torch import profiling

    assert profiling.optimizer_step is optimizer_step
    config = load_config("configs/smoke.toml")
    model = build_tiny_model(layers=6, hidden_size=32).train()
    optimizer, trainable, _head_lr = create_optimizer(model, config)
    before = [parameter.detach().clone() for parameter in trainable]
    variant = _variant()

    loss, microbatches, tokens = optimizer_step(
        model, [variant], config, optimizer, device="cpu",
        trainable_parameters=trainable)

    assert torch.isfinite(loss)
    assert microbatches == 1
    assert tokens == physical_token_count(variant.encoding)
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in trainable if parameter.grad is not None)


def test_training_runner_builds_onecycle_from_full_schedule_and_uses_real_update(tmp_path):
    from gev.backends.torch.training import train

    config = load_config("configs/smoke.toml")
    config = replace(
        config,
        training=replace(config.training, epochs=11, logical_batch=1,
                         max_steps=1, microbatch=1),
        runtime=replace(config.runtime, device="cpu"),
    )
    variant = _variant()

    class OneBatchSchedule:
        requests = [{"state": "state", "questions": {
            "q": {"type": "choice", "criteria": {"a": "A", "b": "B"},
                  "label": "a"}}}]
        tokenizer = None
        markers = None
        epoch = 0
        batch_cursor = 0
        augmentation_digest = "0" * 64
        shuffle_rng = random.Random(0)

        def batches_for_epoch(self):
            return [self.requests]

        def variants_for_batch(self, batch, *, epoch):
            return [variant]

        def complete_batch(self):
            self.batch_cursor += 1

        def finish_epoch(self, batch_count, *, stopped):
            if self.batch_cursor >= batch_count:
                self.batch_cursor = 0
                if stopped:
                    self.epoch += 1
            return self.batch_cursor == 0

        def order_ids(self):
            return ["fixture/1"]

    model = build_tiny_model(layers=6, hidden_size=32)

    metrics = train(model, OneBatchSchedule(), config, tmp_path / "run", progress=False)
    snapshot = torch.load(tmp_path / "run/last_good.resume.pt",
                          map_location="cpu", weights_only=False)
    scheduler = snapshot["torch_state"]["scheduler"]

    assert metrics["full_sched_steps"] == 11
    assert metrics["logical_steps"] == 1 and metrics["complete"] is False
    assert scheduler["total_steps"] == 11
    assert scheduler["last_epoch"] == 1
    assert scheduler["_schedule_phases"][0]["end_step"] == pytest.approx(.1)
