import json

import numpy as np
import pytest

from gev.encoding import Markers
from gev.evaluate import Model, evaluate
from gev.torch_backend import TorchRunner
from gev.train import batch_variants, train
from tests.conftest import MARKERS, gemma3_backbone, gemma4_backbone, request, tiny_config

BACKBONES = {"gemma3": gemma3_backbone, "gemma4": gemma4_backbone}


def runner_for(family, **training):
    config = tiny_config(family, **training)
    return config, TorchRunner(config, backbone=BACKBONES[family]())


def variants(config, tokenizer, count=4):
    markers = Markers.resolve(tokenizer, MARKERS)
    return batch_variants([request(i) for i in range(count)], config, epoch=0, tokenizer=tokenizer, markers=markers)


@pytest.mark.parametrize("family", ["gemma3", "gemma4"])
def test_rows_are_isolated_under_padding_and_batching(family, tokenizer):
    config, runner = runner_for(family)
    encodings = [v["encoding"] for v in variants(config, tokenizer)]
    batched = runner.logits(encodings)
    for encoding, together in zip(encodings, batched):
        for row, values in zip(encoding["rows"], together):
            alone = runner.logits([{"state": encoding["state"], "rows": [row]}])[0][0]
            np.testing.assert_allclose(alone, values, atol=1e-5)


@pytest.mark.parametrize("family", ["gemma3", "gemma4"])
def test_training_reduces_loss_and_round_trips(family, tokenizer, tmp_path):
    config, runner = runner_for(family)
    batch = variants(config, tokenizer)
    runner.init_optimizer()
    losses = [runner.train_step(batch, 3e-3, 3e-3) for _ in range(8)]
    assert losses[-1] < losses[0]
    before = runner.logits([v["encoding"] for v in batch])
    runner.save(tmp_path / "ckpt")
    _, fresh = runner_for(family)
    fresh.load(tmp_path / "ckpt")
    after = fresh.logits([v["encoding"] for v in batch])
    for x, y in zip(before, after):
        for a, b in zip(x, y):
            np.testing.assert_allclose(a, b, atol=1e-6)
    adapter = json.loads((tmp_path / "ckpt" / "adapter_config.json").read_text())
    assert adapter["r"] == 4 and adapter["base_model_name_or_path"] == config.model.name


def test_train_resume_evaluate_predict(tokenizer, sample_root, tmp_path):
    config = tiny_config("gemma3", epochs=2, save_every=2)
    straight = train(config, data_root=sample_root, out=tmp_path / "a", tokenizer=tokenizer,
                     runner=TorchRunner(config, backbone=gemma3_backbone()), log=lambda *_: None)
    assert straight["complete"] and straight["steps"] == 8

    first = config.replace(max_steps=4)
    train(first, data_root=sample_root, out=tmp_path / "b", tokenizer=tokenizer,
          runner=TorchRunner(first, backbone=gemma3_backbone()), log=lambda *_: None)
    resumed = train(config, data_root=sample_root, out=tmp_path / "b", resume=True, tokenizer=tokenizer,
                    runner=TorchRunner(config, backbone=gemma3_backbone()), log=lambda *_: None)
    assert resumed["steps"] == 8
    a = [json.loads(line)["loss"] for line in (tmp_path / "a" / "log.jsonl").read_text().splitlines()]
    b = [json.loads(line)["loss"] for line in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    np.testing.assert_allclose(a, b, rtol=1e-4)

    model = Model(tmp_path / "a", tokenizer=tokenizer, runner=TorchRunner(config, backbone=gemma3_backbone()))
    report = evaluate(model, suite="decision-v7", split="development", data_root=sample_root,
                      out=tmp_path / "eval", log=lambda *_: None)
    assert report["clean"]["n"] == 12 and report["questions"] == 12
    answer = model.predict({"state": "it is b", "questions": {"q": {"type": "noul", "instructions": "is it b"}}})
    assert set(answer["questions"]["q"]["probabilities"]) == {"false", "true"}

    from gev.evaluate import calibrate
    fit = calibrate(tmp_path / "eval", update=True)
    assert json.loads((tmp_path / "a" / "checkpoint" / "gev.json").read_text())["temperature"] == fit["temperature"]
