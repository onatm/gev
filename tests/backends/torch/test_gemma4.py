import pytest
import torch
import json
import random
from pathlib import Path
from safetensors.torch import save_file
from transformers import Gemma4TextModel

from gev.backends.torch import gemma4
from gev.backends.torch.gemma4 import (
    _load_gemma4_snapshot, _tiny_config, build_tiny_gemma4_model,
    load_gemma4_backbone)
from gev.backends.torch.training import create_optimizer, optimizer_step
from gev.configuration.config import load_config
from gev.training.batching import Variant


def _encoding():
    return {
        "ids": [2, 3, 4, 5, 6, 7, 8],
        "seg": [0, 0, 1, 1, 1, 1, 1],
        "pos": [0, 1, 2, 3, 4, 5, 6],
        "opt": [-1, -1, -1, -1, 0, 1, -2],
        "decide_idx": [6],
        "opt_idx": [[4, 5]],
        "labels": [0],
        "state_length": 2,
    }


@pytest.mark.parametrize(("compute_dtype", "decoder_dtype"), [
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
])
def test_tiny_gemma4_text_rows_keep_trainables_fp32_and_backpropagate(
        compute_dtype, decoder_dtype):
    model = build_tiny_gemma4_model(compute_dtype=compute_dtype)
    assert model.decoder.embed_tokens.weight.dtype == decoder_dtype
    assert all(parameter.dtype == torch.float32 for parameter in model.head.parameters())
    lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
    assert lora and all(parameter.dtype == torch.float32 for parameter in lora)

    logits = model.forward_one(_encoding())
    assert len(logits) == 1 and logits[0].shape == (2,)
    torch.nn.functional.cross_entropy(logits[0].unsqueeze(0), torch.tensor([0])).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.head.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "lora_B" in name)


def test_tiny_gemma4_rejects_packed_execution():
    model = build_tiny_gemma4_model()
    with pytest.raises(ValueError, match="rows execution only"):
        model.forward_batch([_encoding()], execution_mode="packed")


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_torch_gemma4_causal_attention_mask_check(compute_dtype):
    from gev.backends.torch.qualification import causal_attention_masks_valid

    model = build_tiny_gemma4_model(compute_dtype=compute_dtype).eval()
    assert causal_attention_masks_valid(
        model, [_encoding()], compute_dtype=compute_dtype)


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_torch_gemma4_qualification_collector_checks_source_and_masters(compute_dtype):
    from gev.backends.torch.qualification import collect_torch_training_checks

    model = build_tiny_gemma4_model(compute_dtype=compute_dtype)
    model.source_provenance = {
        "model": "google/gemma-4-E2B",
        "revision": gemma4.GEMMA4_E2B_REVISION,
        "source_weights_dtype": "bf16",
        "compute_dtype": compute_dtype,
        "source_text_tensor_count": 600,
        "source_text_parameters": 4_647_449_891,
        "effective_text_parameters": 4_628_569_344,
        "source_name_shape_sha256": "a" * 64,
    }
    checks = collect_torch_training_checks(model, {"qualification_checks": {
        "fp32_optimizer_state": True, "finite_loss_gradients": True,
        "nonzero_trainable_update": True}})

    assert all(checks[name] for name in (
        "source_inventory_exact", "decoder_compute_dtype",
        "fp32_lora_and_pointer", "fp32_optimizer_state",
        "finite_loss_gradients", "nonzero_trainable_update"))


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_torch_gemma4_predictor_serves_native_precision_rows(compute_dtype):
    from gev.backends.torch.predictor import TorchPredictor

    model = build_tiny_gemma4_model(compute_dtype=compute_dtype)
    predictor = TorchPredictor(
        model, tokenizer=None, markers=None, state_cap=32, branch_cap=32,
        packed_cap=32, temperature=1.5,
        encoder=lambda *_args, **_kwargs: _encoding())
    record = {"state": "fixture", "questions": {
        "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a"}}}

    result = predictor(record)

    assert list(result["probabilities"]) == ["q"]
    assert sum(result["probabilities"]["q"].values()) == pytest.approx(1.0)
    assert all(torch.isfinite(torch.tensor(value))
               for value in result["probabilities"]["q"].values())


@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_tiny_gemma4_takes_optimizer_step_with_fp32_master_and_pointer_state(compute_dtype):
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    from dataclasses import replace
    config = replace(
        config,
        backend=replace(config.backend, id="torch"),
        training=replace(config.training, dtype=compute_dtype, logical_batch=1, microbatch=1),
        runtime=replace(config.runtime, device="cpu", gradient_checkpointing=False),
    )
    model = build_tiny_gemma4_model(compute_dtype=compute_dtype).train()
    optimizer, trainable, _head_lr = create_optimizer(model, config)
    before = [parameter.detach().clone() for parameter in trainable]
    loss, microbatches, _tokens = optimizer_step(
        model, [Variant({"questions": [{"label": 0}]}, _encoding(), "fixture/1", "fixture")],
        config, optimizer, device="cpu", trainable_parameters=trainable)

    assert torch.isfinite(loss)
    assert microbatches == 1
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    assert all(parameter.dtype == torch.float32 for state in optimizer.state.values()
               for key, parameter in state.items()
               if isinstance(parameter, torch.Tensor) and parameter.is_floating_point())


def _write_source_fixture(root: Path, *, shared_layers=0, corrupt=None):
    config = _tiny_config()
    config.num_kv_shared_layers = shared_layers
    (root / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "text_config": config.to_dict(),
    }), encoding="utf-8")
    source_model = Gemma4TextModel(config)
    tensors = {f"model.language_model.{name}": value.detach().cpu().to(torch.bfloat16).contiguous()
               for name, value in source_model.state_dict().items()}
    if shared_layers:
        for name in ("k_proj", "v_proj", "k_norm"):
            source = tensors[f"model.language_model.layers.0.self_attn.{name}.weight"]
            tensors[f"model.language_model.layers.1.self_attn.{name}.weight"] = source.clone()
    if corrupt == "missing":
        tensors.pop(next(iter(tensors)))
    elif corrupt == "unexpected":
        tensors["model.language_model.unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif corrupt == "dtype":
        tensors[next(iter(tensors))] = tensors[next(iter(tensors))].float()
    save_file(tensors, str(root / "model.safetensors"))


@pytest.mark.parametrize(("compute_dtype", "expected"), [
    ("fp32", torch.float32), ("bf16", torch.bfloat16),
])
def test_gemma4_loader_validates_bf16_source_and_loads_text_only_dtype(
        tmp_path, compute_dtype, expected):
    _write_source_fixture(tmp_path, shared_layers=1)

    model = _load_gemma4_snapshot(
        tmp_path, "google/gemma-4-E2B", gemma4.GEMMA4_E2B_REVISION,
        compute_dtype=compute_dtype, attn_implementation="eager",
        gradient_checkpointing=False, enforce_pinned_inventory=False)

    assert model.embed_tokens.weight.dtype == expected
    assert model._gev_source["source_weights_dtype"] == "bf16"
    assert model._gev_source["source_text_tensor_count"] == len(model.state_dict()) + 3


@pytest.mark.parametrize(("corrupt", "message"), [
    ("missing", "tensor inventory mismatch"),
    ("unexpected", "tensor inventory mismatch"),
    ("dtype", "is not BF16"),
])
def test_gemma4_loader_rejects_non_strict_source_inventory(
        tmp_path, corrupt, message):
    _write_source_fixture(tmp_path, corrupt=corrupt)

    with pytest.raises(ValueError, match=message):
        _load_gemma4_snapshot(
            tmp_path, "google/gemma-4-E2B", gemma4.GEMMA4_E2B_REVISION,
            compute_dtype="fp32", attn_implementation="eager",
            gradient_checkpointing=False, enforce_pinned_inventory=False)


def test_production_torch_loader_rejects_tiny_inventory_without_test_seam(
        tmp_path, monkeypatch):
    _write_source_fixture(tmp_path)
    monkeypatch.setattr(gemma4, "_snapshot", lambda *_: tmp_path)

    with pytest.raises(ValueError, match="pinned Gemma 4 text (config dimensions|inventory) mismatch"):
        load_gemma4_backbone(
            "google/gemma-4-E2B", gemma4.GEMMA4_E2B_REVISION, compute_dtype="fp32")


def test_gemma4_loader_rejects_wrong_source_identity_before_snapshot_access(monkeypatch):
    monkeypatch.setattr(gemma4, "_snapshot", lambda *_: pytest.fail("snapshot requested"))
    with pytest.raises(ValueError, match="pinned google/gemma-4-E2B"):
        load_gemma4_backbone("other/model", gemma4.GEMMA4_E2B_REVISION)


def test_snapshot_download_filters_vision_only_shards(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub

    revision = gemma4.GEMMA4_E2B_REVISION
    root = tmp_path / revision
    root.mkdir()
    siblings = ["config.json", "model.safetensors.index.json",
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    monkeypatch.setattr("gev.infrastructure.network.use_system_ssl", lambda: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda *_args, **_kwargs: SimpleNamespace(
            sha=revision, siblings=[SimpleNamespace(rfilename=name) for name in siblings])))
    calls = []

    def download(_model_name, *, revision, allow_patterns):
        calls.append(list(allow_patterns))
        if len(calls) == 1:
            weight_map = {
                **{f"model.language_model.fixture_{index}": siblings[2]
                   for index in range(600)},
                **{f"model.vision_tower.fixture_{index}": siblings[3]
                   for index in range(1411)},
            }
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": weight_map
            }), encoding="utf-8")
        return root

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    assert gemma4._snapshot("google/gemma-4-E2B", revision) == root
    assert calls[0] == ["config.json", "model.safetensors.index.json"]
    assert calls[1] == ["config.json", "model.safetensors.index.json", siblings[2]]


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("compute_dtype", ["fp32", "bf16"])
def test_gemma4_accelerator_step_when_available(device, compute_dtype):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    from dataclasses import replace
    config = replace(
        config,
        backend=replace(config.backend, id="torch"),
        training=replace(config.training, dtype=compute_dtype, logical_batch=1, microbatch=1),
        runtime=replace(config.runtime, device=device, gradient_checkpointing=False),
    )
    model = build_tiny_gemma4_model(compute_dtype=compute_dtype).to(device).train()
    optimizer, trainable, _head_lr = create_optimizer(model, config)
    loss, _microbatches, _tokens = optimizer_step(
        model, [Variant({"questions": [{"label": 0}]}, _encoding(), "fixture/1", "fixture")],
        config, optimizer, device=device, trainable_parameters=trainable)
    assert torch.isfinite(loss)
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    model.eval()
    assert all(torch.isfinite(logits).all() for logits in model.forward_one(_encoding()))


def test_torch_gemma4_bf16_resume_restores_fp32_trainable_and_optimizer_state(tmp_path):
    from dataclasses import replace
    from gev.backends.torch.training import load_training_state, train

    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = replace(
        config,
        backend=replace(config.backend, id="torch"),
        training=replace(config.training, dtype="bf16", epochs=11, max_steps=1,
                         logical_batch=1, microbatch=1),
        runtime=replace(config.runtime, device="cpu"),
    )
    variant = Variant({"questions": [{"label": 0}]}, _encoding(), "fixture/1", "fixture")

    class OneBatchSchedule:
        requests = [{"id": "fixture/1", "state": "fixture state", "questions": {
            "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a"}}}]
        tokenizer = None
        markers = None
        epoch = 0
        batch_cursor = 0
        augmentation_digest = "0" * 64
        shuffle_rng = random.Random(0)

        def batches_for_epoch(self):
            return [self.requests]

        def variants_for_batch(self, _batch, *, epoch):
            return [variant]

        def complete_batch(self):
            self.batch_cursor += 1

        def finish_epoch(self, batch_count, *, stopped):
            boundary = self.batch_cursor >= batch_count
            if boundary:
                self.batch_cursor = 0
                if stopped:
                    self.epoch += 1
            return boundary

        def order_ids(self):
            return ["fixture/1"]

        def restore(self, state):
            self.epoch = state["progress"]["epoch"]
            self.batch_cursor = state["progress"]["next_batch"]
            self.shuffle_rng.setstate(state["progress"]["shuffle_rng"])
            self.augmentation_digest = state["progress"]["augmentation_digest"]

    output = tmp_path / "resume"
    train(build_tiny_gemma4_model(compute_dtype="bf16"), OneBatchSchedule(),
          config, output, progress=False)
    resume = output / "last_good.resume.pt"
    snapshot = load_training_state(resume)
    assert all(value.dtype == torch.float32
               for value in snapshot["torch_state"]["trainable_state"].values())

    resumed_config = replace(
        config, training=replace(config.training, max_steps=2))
    metrics = train(build_tiny_gemma4_model(compute_dtype="bf16"), OneBatchSchedule(),
                     resumed_config, output, progress=False, resume=resume)
    resumed_state = load_training_state(resume)
    assert metrics["logical_steps"] == 2
    assert all(value.dtype == torch.float32 for state in resumed_state["torch_state"][
        "optimizer"]["state"].values() for key, value in state.items()
               if isinstance(value, torch.Tensor) and value.is_floating_point())
