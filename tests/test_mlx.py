import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX needs an Apple Silicon GPU", allow_module_level=True)

from gev.encoding import Markers  # noqa: E402
from gev.mlx_backend import MlxRunner  # noqa: E402
from gev.torch_backend import TorchRunner  # noqa: E402
from gev.train import batch_variants, train  # noqa: E402
from tests.conftest import MARKERS, gemma4_backbone, gemma4_text_config, request, tiny_config  # noqa: E402


def mlx_decoder(torch_backbone):
    """An mlx-lm Gemma 4 text decoder holding the same weights as a tiny Torch Gemma4TextModel."""
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes
    from transformers import Gemma4Config

    config = Gemma4Config(text_config=gemma4_text_config().to_dict()).to_dict()
    config["text_config"]["global_head_dim"] = config["text_config"]["head_dim"]
    model_class, args_class = _get_classes(config)
    base = model_class(args_class.from_dict(config))
    weights = base.sanitize({f"model.language_model.{k}": mx.array(v.float().numpy())
                             for k, v in torch_backbone.state_dict().items()})
    expected = dict(tree_flatten(base.parameters()))
    base.load_weights([(k, v) for k, v in weights.items() if k in expected], strict=False)
    return base.language_model.model


def mlx_runner(**training):
    config = tiny_config("gemma4", backend="mlx", **training)
    return config, MlxRunner(config, decoder=mlx_decoder(gemma4_backbone()))


def variants(config, tokenizer):
    markers = Markers.resolve(tokenizer, MARKERS)
    return batch_variants([request(i) for i in range(4)], config, epoch=0, tokenizer=tokenizer, markers=markers)


def test_rows_are_isolated_under_padding_and_batching(tokenizer):
    config, runner = mlx_runner()
    encodings = [v["encoding"] for v in variants(config, tokenizer)]
    for encoding, together in zip(encodings, runner.logits(encodings)):
        for row, values in zip(encoding["rows"], together):
            alone = runner.logits([{"state": encoding["state"], "rows": [row]}])[0][0]
            np.testing.assert_allclose(alone, values, atol=1e-5)


def test_training_reduces_loss_and_round_trips(tokenizer, tmp_path):
    config, runner = mlx_runner()
    batch = variants(config, tokenizer)
    runner.init_optimizer()
    losses = [runner.train_step(batch, 3e-3, 3e-3) for _ in range(8)]
    assert losses[-1] < losses[0]
    runner.save(tmp_path / "ckpt")
    runner.save_optimizer(tmp_path / "ckpt")
    _, fresh = mlx_runner()
    fresh.load(tmp_path / "ckpt")
    fresh.init_optimizer()
    fresh.load_optimizer(tmp_path / "ckpt")
    encodings = [v["encoding"] for v in batch]
    for x, y in zip(runner.logits(encodings), fresh.logits(encodings)):
        for a, b in zip(x, y):
            np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.isclose(runner.train_step(batch, 1e-3, 1e-3), fresh.train_step(batch, 1e-3, 1e-3), rtol=1e-4)


def test_checkpoints_move_between_torch_and_mlx(tokenizer, tmp_path):
    """Train on Torch, serve on MLX (and back): same checkpoint, same logits."""
    torch_config = tiny_config("gemma4")
    backbone = gemma4_backbone()
    torch_runner = TorchRunner(torch_config, backbone=backbone)
    batch = variants(torch_config, tokenizer)
    torch_runner.init_optimizer()
    for _ in range(3):
        torch_runner.train_step(batch, 1e-2, 1e-2)
    torch_runner.save(tmp_path / "torch")

    _, runner = mlx_runner()
    runner.load(tmp_path / "torch")
    encodings = [v["encoding"] for v in batch]
    for x, y in zip(torch_runner.logits(encodings), runner.logits(encodings)):
        for a, b in zip(x, y):
            np.testing.assert_allclose(a, b, atol=1e-3)

    runner.save(tmp_path / "mlx")
    back = TorchRunner(torch_config, backbone=gemma4_backbone())
    back.load(tmp_path / "mlx")
    for x, y in zip(torch_runner.logits(encodings), back.logits(encodings)):
        for a, b in zip(x, y):
            np.testing.assert_allclose(a, b, atol=1e-6)


def test_train_and_resume(tokenizer, sample_root, tmp_path):
    config = tiny_config("gemma4", backend="mlx", epochs=2, save_every=2)
    first = config.replace(max_steps=4)
    train(first, data_root=sample_root, out=tmp_path / "run", tokenizer=tokenizer,
          runner=MlxRunner(first, decoder=mlx_decoder(gemma4_backbone())), log=lambda *_: None)
    summary = train(config, data_root=sample_root, out=tmp_path / "run", resume=True, tokenizer=tokenizer,
                    runner=MlxRunner(config, decoder=mlx_decoder(gemma4_backbone())), log=lambda *_: None)
    assert summary["complete"] and summary["steps"] == 8
    metadata = json.loads((tmp_path / "run" / "checkpoint" / "gev.json").read_text())
    assert metadata["trained_with"]["backend"] == "mlx"


def test_torch_and_mlx_optimize_identically(tokenizer, tmp_path):
    """From the same weights, both backends produce the same losses and updates."""
    import torch

    torch_config = tiny_config("gemma4")
    torch_runner = TorchRunner(torch_config, backbone=gemma4_backbone())
    with torch.no_grad():
        for name, parameter in torch_runner.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.05)
    torch_runner.save(tmp_path / "start")
    _, runner = mlx_runner()
    runner.load(tmp_path / "start")
    batch = variants(torch_config, tokenizer)
    torch_runner.init_optimizer()
    runner.init_optimizer()
    for _ in range(3):
        assert np.isclose(torch_runner.train_step(batch, 1e-3, 1e-3), runner.train_step(batch, 1e-3, 1e-3), rtol=1e-4)
