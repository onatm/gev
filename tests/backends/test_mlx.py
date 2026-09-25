import os
import hashlib
import json
import pytest

os.environ.setdefault("MLX_ENABLE_TF32", "0")

pytest.importorskip("mlx.core", reason="MLX backend tests require the optional MLX extra on Apple Silicon")
pytest.importorskip("mlx_lm", reason="MLX backend tests require the optional mlx-lm package")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

from gev.backends.mlx.gemma4 import (Gemma4RowModel, _install_lora,
                                     _TorchCompatibleLinear,
                                     _shared_kv_extra_names,
                                     _TorchCompatibleRMSNorm,
                                     _verify_marker_embeddings)
from gev.backends.mlx.pointer import MlxPointerHead
from gev.backends.mlx.predictor import MlxPredictor
from gev.backends.mlx.qualification import (BOUNDARY_IDS, POLICY,
                                           _compare_selected_layers)
from gev.models.qualification import (qualification_code_sha256,
                                      qualification_policy_sha256,
                                      seal_qualification_receipt,
                                      trainable_content_sha256)
from gev.backends.mlx.training import train
from gev.backends.mlx.checkpoint import load_checkpoint, save_checkpoint
from gev.application.prediction import predict_request
from gev.configuration.config import (BackendConfig, ExperimentConfig, ModelConfig,
                                      RuntimeConfig, TrainingConfig)
from gev.domain.tokenization import MarkerMap
from gev.training.batching import Variant
from gev.training.schedule import TrainingSchedule


def encoded_row():
    state = [6, 3]
    branch = [7, 4, 8, 11, 9, 8, 12, 9, 10]
    return {"ids": state + branch, "seg": [0, 0] + [1] * len(branch),
            "pos": list(range(2 + len(branch))), "opt": [-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -2],
            "decide_idx": [len(state) + len(branch) - 1],
            "opt_idx": [[len(state) + 4, len(state) + 7]], "labels": [0], "state_length": 2}


def _base_qualification(compute_dtype="bf16"):
    return {"revision": POLICY.base_revision, "source_weights_dtype": "bf16",
            "compute_dtype": compute_dtype,
            "inventory": {"tensor_count": 2011, "text_tensor_count": 600,
                          "text_serialized_parameters": 4_647_449_891,
                          "effective_text_parameters": 4_628_569_344,
                          "name_shape_sha256": "d" * 64},
            "lora": {"layers": POLICY.expected_trainable_lora_matrices},
            "marker_embeddings": {"main": {"finite": True}}}


def test_mlx_pointer_is_float32_and_returns_probabilities():
    head = MlxPointerHead(8)
    decision = mx.ones((8,), dtype=mx.bfloat16)
    options = mx.ones((3, 8), dtype=mx.bfloat16)
    logits = head(decision, options)
    probabilities = head.probabilities(logits)
    mx.eval(logits, probabilities)
    assert logits.dtype == mx.float32
    assert probabilities.dtype == mx.float32
    assert np.allclose(np.asarray(probabilities).sum(), 1.0)


def test_mlx_row_model_isolates_questions_and_records():
    class TextDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 1536)

        def __call__(self, ids):
            return self.embedding(ids)

    decoder = TextDecoder()
    base = nn.Module()
    base.language_model = nn.Module()
    base.language_model.model = decoder
    model = Gemma4RowModel({"model": base, "revision": "fixture"})
    row = encoded_row()
    solo = model.probs(row)
    joint = model.probs([row, row])
    mx.eval(*solo, *joint[0], *joint[1])
    assert all(np.allclose(np.asarray(left), np.asarray(right))
               for left, right in zip(solo, joint[0], strict=True))
    assert len(joint) == 2
    try:
        model.forward_batch([row], execution_mode="packed")
    except ValueError as exc:
        assert "rows" in str(exc)
    else:
        raise AssertionError("packed mode must fail closed for Gemma 4")


def test_independent_question_rows_preserve_sibling_order_and_gradient_determinism():
    class CausalFixtureDecoder(nn.Module):
        def __call__(self, ids):
            cumulative = mx.cumsum(ids.astype(mx.float32), axis=1)
            hidden = mx.broadcast_to(cumulative[:, :, None], (*ids.shape, 1536))
            return hidden.astype(mx.bfloat16)

    base = nn.Module()
    base.language_model = nn.Module()
    base.language_model.model = CausalFixtureDecoder()
    model = Gemma4RowModel({"model": base})
    state = [6, 3]
    branches = [[7, 1, 8, 11, 9, 8, 12, 9, 10],
                [7, 2, 3, 4, 5, 8, 13, 9, 8, 14, 9, 10]]

    def encoding(ordered):
        ids, decide, options = list(state), [], []
        segments, positions = [0] * len(state), list(range(len(state)))
        for question_index, branch in enumerate(ordered, 1):
            start = len(ids)
            ids.extend(branch)
            segments.extend([question_index] * len(branch))
            positions.extend(range(len(state), len(state) + len(branch)))
            decide.append(len(ids) - 1)
            options.append([start + 3, start + 6])
        return {"ids": ids, "seg": segments, "pos": positions,
                "state_length": len(state), "decide_idx": decide,
                "opt_idx": options, "metadata": {"id": "batch-fixture"}}

    original = encoding(branches)
    scores = model.forward_one(original)
    assert len(scores) == 2
    reversed_scores = model.forward_one(encoding(list(reversed(branches))))
    mx.eval(reversed_scores)
    assert all(np.array_equal(np.asarray(scores[index]), np.asarray(reversed_scores[-1 - index]))
               for index in range(2))

    def row_loss(module):
        return mx.stack([value.astype(mx.float32).square().mean()
                         for value in module.forward_one(original)]).mean()

    value1, grad1 = nn.value_and_grad(model, row_loss)(model)
    value2, grad2 = nn.value_and_grad(model, row_loss)(model)
    mx.eval(value1, grad1, value2, grad2)
    assert np.array_equal(np.asarray(value1), np.asarray(value2))
    grad2_by_name = dict(tree_flatten(grad2))
    assert dict(tree_flatten(grad1)).keys() == grad2_by_name.keys()
    assert all(np.array_equal(np.asarray(value), np.asarray(grad2_by_name[name]))
               for name, value in tree_flatten(grad1))


def test_mlx_predictor_preserves_scaled_and_raw_logits_at_unit_and_nonunit_temperature():
    class TextDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 1536)

        def __call__(self, ids):
            return self.embedding(ids)

    base = nn.Module()
    base.language_model = nn.Module()
    base.language_model.model = TextDecoder()
    model = Gemma4RowModel({"model": base})
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10},
                        {"state": "<unused0>", "question": "<unused1>", "option_start": "<unused2>",
                         "option_end": "<unused3>", "decide": "<unused4>"}, "fixture")
    predictor_args = {"state_cap": 10, "branch_cap": 20, "packed_cap": 30,
                      "encoder": lambda *_args, **_kwargs: encoded_row()}
    record = {"state": "state", "_meta": {"id": "r1", "source": "fixture", "group_id": "r1"},
              "questions": {"choice": {"type": "choice", "instructions": "choose",
                                         "criteria": {"a": "A", "b": "B"}, "label": "a"}}}
    unit = MlxPredictor(model, object(), markers, temperature=1.0, **predictor_args)(record)
    assert unit["question_count"] == 1
    assert unit["logits"]["choice"] == unit["raw_logits"]["choice"]
    assert list(unit["probabilities"]["choice"]) == ["a", "b"]
    assert sum(unit["probabilities"]["choice"].values()) == pytest.approx(1.0)

    scaled = MlxPredictor(model, object(), markers, temperature=2.0, **predictor_args)(record)
    raw_values = np.array(list(scaled["raw_logits"]["choice"].values()))
    scaled_values = np.array(list(scaled["logits"]["choice"].values()))
    probs = np.array(list(scaled["probabilities"]["choice"].values()))
    assert np.allclose(scaled_values, raw_values / 2)
    assert np.allclose(probs, np.exp(scaled_values) / np.exp(scaled_values).sum())
    from gev.evaluation.benchmark import prediction_rows
    row = prediction_rows(record, scaled)[0]
    assert np.allclose(row["logits"], scaled_values)
    assert np.allclose(row["raw_logits"], raw_values)


def test_backend_neutral_predict_request_accepts_mlx_arrays_without_torch_conversion(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "no_grad", lambda: pytest.fail("MLX prediction entered Torch no_grad"))

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 1536)

        def __call__(self, ids):
            return self.embedding(ids)

    base = nn.Module()
    base.language_model = nn.Module()
    base.language_model.model = TinyDecoder()
    model = Gemma4RowModel({"model": base})
    assert model.backend_id == "mlx"

    request = {"state": "small state", "questions": {
        "q": {"type": "choice", "instructions": "choose",
              "criteria": {"a": "A", "b": "B"}}}}
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8,
                         "option_end": 9, "decide": 10},
                        {"state": "<unused0>", "question": "<unused1>",
                         "option_start": "<unused2>", "option_end": "<unused3>",
                         "decide": "<unused4>"}, "fixture")
    result = predict_request(request, model, object(), markers,
                             encoder=lambda *_args, **_kwargs: encoded_row(), state_cap=10,
                             branch_cap=20, packed_cap=30, temperature=2.0)
    assert result["inference_temperature"] == 2.0
    assert sum(result["questions"][0]["probabilities"].values()) == pytest.approx(1.0)


def test_explicit_all_layer_lora_inventory_and_nonzero_gradient():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, nn.Linear(4, 4))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self, name, nn.Linear(4, 4))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = Mlp()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer() for _ in range(35)]

    decoder = Decoder()
    decoder.freeze()
    facts = _install_lora(decoder)
    assert facts["layers"] == 35 * 7
    assert sum(facts["target_counts"].values()) == 35 * 7
    assert facts["rank"] == 16 and facts["alpha"] == 32 and facts["dropout"] == 0.05
    assert len(decoder.trainable_parameters()) > 0

    def loss(module):
        return module.layers[0].self_attn.q_proj(mx.ones((1, 4), dtype=mx.float32)).sum()

    value, gradients = nn.value_and_grad(decoder, loss)(decoder)
    mx.eval(value, gradients)
    assert any(np.count_nonzero(np.asarray(value)) for _, value in tree_flatten(gradients))


def test_fp32_diagnostic_declares_boundary_rows_without_a_quality_threshold():
    assert BOUNDARY_IDS == (6, 7, 8, 9, 10, 239673, 239674, 239675, 239676, 262143)
    assert POLICY.gate_for("fp32").gate_kind == "independent_implementation_diagnostic"
    assert POLICY.gate_for("fp32").max_abs_probability is None
    assert len(_shared_kv_extra_names(layers=35, shared_layers=20)) == 60


def test_bf16_loader_checks_exact_marker_and_high_boundary_embedding_rows():
    class Embeddings:
        def __init__(self, width, *, corrupt_last=False):
            prefix = mx.arange(262143, dtype=mx.float32).reshape(-1, 1)
            prefix = mx.broadcast_to(prefix + mx.arange(width, dtype=mx.float32),
                                     (262143, width)).astype(mx.bfloat16)
            last = mx.zeros((1, width), dtype=mx.bfloat16) if corrupt_last else mx.ones(
                (1, width), dtype=mx.bfloat16) * 262143
            self.weight = mx.concatenate((prefix, last), axis=0)

    class Decoder:
        def __init__(self, corrupt_ple=False):
            self.embed_tokens = Embeddings(4)
            self.embed_tokens_per_layer = Embeddings(5, corrupt_last=corrupt_ple)

    summary = _verify_marker_embeddings(Decoder())
    assert summary["main"]["boundary_ids"] == list(BOUNDARY_IDS)
    assert summary["main"]["high_boundary_nonzero"] is True
    assert summary["per_layer"]["high_boundary_nonzero"] is True
    with pytest.raises(ValueError, match="boundary embeddings"):
        _verify_marker_embeddings(Decoder(corrupt_ple=True))


def test_selected_layer_diagnostic_reports_error_and_rejects_shape_mismatch():
    exact = np.ones((2, 3), dtype=np.float32)
    different = np.zeros_like(exact)
    result = _compare_selected_layers({0: exact, 14: exact, 34: exact},
                                      {0: exact.copy(), 14: different, 34: exact.copy()})
    assert result["match"] is True
    assert result["selected_layers"]["0"]["max_abs"] == 0
    assert result["selected_layers"]["14"]["max_abs"] == 1
    mismatched = _compare_selected_layers({0: exact, 14: exact, 34: exact},
                                          {0: exact, 14: exact[:1], 34: exact})
    assert mismatched["match"] is False


def test_gemma4_rmsnorm_matches_transformers_float32_power_order():
    import torch

    source = nn.RMSNorm(4, eps=1e-6)
    source.weight = mx.array([0.25, 0.75, 1.25, 2.0], dtype=mx.bfloat16)
    norm = _TorchCompatibleRMSNorm(source)
    inputs = mx.array([[0.5, -1.25, 3.0, 7.0]], dtype=mx.bfloat16)
    actual = norm(inputs)
    value = torch.tensor(np.asarray(inputs.astype(mx.float32)), dtype=torch.float32)
    weight = torch.tensor(np.asarray(source.weight.astype(mx.float32)), dtype=torch.float32)
    expected = (value * torch.pow(value.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
                * weight).to(torch.bfloat16)
    mx.eval(actual)
    assert np.array_equal(np.asarray(actual.astype(mx.float32)), expected.float().numpy())


def test_gemma4_ple_linear_uses_torch_cpu_bf16_accumulation_order():
    import torch
    import torch.nn.functional as F

    source = nn.Linear(4, 3, bias=False)
    weight = mx.array([[.25, .125, -.5, .75], [1., -1., .5, .25],
                       [.1, .2, .3, .4]], dtype=mx.bfloat16)
    source.weight = weight
    layer = _TorchCompatibleLinear(source)
    inputs = mx.array([[.125, -.5, 2., 3.], [1., .25, -.75, .5]], dtype=mx.bfloat16)
    actual = layer(inputs)
    expected = F.linear(torch.tensor(np.asarray(inputs.astype(mx.float32))).to(torch.bfloat16),
                        torch.tensor(np.asarray(weight.astype(mx.float32))).to(torch.bfloat16))
    mx.eval(actual)
    assert np.array_equal(np.asarray(actual.astype(mx.float32)), expected.float().numpy())


def test_mlx_training_fails_closed_without_native_bf16_model_provenance(tmp_path):
    from gev.backends.mlx.training import train
    from gev.configuration.config import load_config

    output = tmp_path / "must-not-start"
    with pytest.raises(RuntimeError, match="lacks pinned BF16 provenance"):
        train(None, None, load_config("configs/gemma4-e2b-mlx-bf16.toml"), output)
    assert not output.exists()


def test_mlx_fp32_factory_passes_compute_dtype_and_attaches_staging_receipt(monkeypatch):
    from gev.backends.mlx import MlxBackend
    from gev.models.specs import GEMMA4_E2B, GEMMA4_E2B_REVISION

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 1536)
            self.frozen = mx.ones((2,), dtype=mx.float32)

        def __call__(self, ids):
            return self.embedding(ids)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = Decoder()

    loaded = {"model": Base(), "revision": GEMMA4_E2B_REVISION,
              "source_weights_dtype": "bf16", "compute_dtype": "fp32",
              "base_qualification": _base_qualification("fp32")}
    observed = {}
    backend = MlxBackend()
    monkeypatch.setattr(backend, "select_device", lambda requested: requested)
    monkeypatch.setattr("gev.backends.mlx.gemma4.load_gemma4_backbone",
                        lambda _name, _revision, *, compute_dtype:
                        observed.update(compute_dtype=compute_dtype) or loaded)

    model = backend.create_model(
        GEMMA4_E2B, model_name="google/gemma-4-E2B", revision=GEMMA4_E2B_REVISION,
        compute_dtype="fp32")

    assert observed == {"compute_dtype": "fp32"}
    assert model.decoder.frozen.dtype == mx.float32
    assert model.provenance["source_weights_dtype"] == "bf16"
    assert model.provenance["compute_dtype"] == "fp32"
    assert "fp32_staging_receipt" not in model.provenance


@pytest.mark.parametrize("compute_dtype", ["bf16", "fp32"])
def test_mlx_qualification_collector_checks_selected_decoder_precision(
        monkeypatch, compute_dtype):
    from gev.backends.mlx.qualification import collect_mlx_training_checks

    class Head:
        query = SimpleNamespace(weight=mx.ones((256, 1), dtype=mx.float32))
        key = SimpleNamespace(weight=mx.ones((256, 1), dtype=mx.float32))

        def parameters(self):
            return {"query.weight": self.query.weight, "key.weight": self.key.weight,
                    "query.bias": mx.ones((256,), dtype=mx.float32),
                    "key.bias": mx.ones((256,), dtype=mx.float32)}

    class Decoder:
        def parameters(self):
            return {"frozen.weight": mx.ones(
                (2,), dtype=mx.bfloat16 if compute_dtype == "bf16" else mx.float32),
                    "layer.lora_A": mx.ones((1,), dtype=mx.float32)}

    class Model:
        decoder = Decoder()
        head = Head()
        provenance = {"source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
                      "base_qualification": _base_qualification(compute_dtype)}

        def trainable_parameters(self):
            return {"layer.lora_A": mx.ones((1,), dtype=mx.float32),
                    "head.query.weight": self.head.query.weight,
                    "head.query.bias": mx.ones((256,), dtype=mx.float32),
                    "head.key.weight": self.head.key.weight,
                    "head.key.bias": mx.ones((256,), dtype=mx.float32)}

    monkeypatch.setattr("gev.backends.mlx.qualification._base_boundary_values",
                        lambda *_args, **_kwargs: {
                            "main": np.ones((10, 2), dtype=np.float32),
                            "ple": np.ones((10, 3), dtype=np.float32)})
    monkeypatch.setattr("gev.backends.mlx.qualification._mlx_attention_masks_exact",
                        lambda *_args: True)
    monkeypatch.setattr("gev.backends.mlx.qualification._mlx_shared_source_layers",
                        lambda _model: dict(POLICY.expected_shared_kv_sources))
    checks = collect_mlx_training_checks(
        Model(), {"engineering_checks": {
            "finite_loss_gradients": True, "finite_optimizer_state": True,
            "nonzero_adapter_and_head_gradients": True,
            "nonzero_trainable_update": True,
            "optimizer_master_state_fp32": True}}, development_encodings=[{}])

    assert checks["source_inventory_exact"] is True
    assert checks["decoder_compute_dtype"] is True
    assert checks["fp32_lora_and_pointer"] is True
    assert checks["shared_kv_mapping_exact"] is True


def test_resolved_mlx_model_factory_passes_configured_fp32_compute_dtype(monkeypatch):
    from gev.backends.mlx import MlxBackend
    from gev.backends.mlx.gemma4 import Gemma4RowModel
    from gev.configuration.config import load_config
    from gev.configuration.resolved import resolve_experiment_config
    from gev.models.specs import GEMMA4_E2B_REVISION

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 1536)

        def __call__(self, ids):
            return self.embedding(ids)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = Decoder()

    loaded = {"model": Base(), "revision": GEMMA4_E2B_REVISION,
              "source_weights_dtype": "bf16", "compute_dtype": "fp32",
              "base_qualification": _base_qualification("fp32")}
    monkeypatch.setattr(MlxBackend, "select_device", lambda _self, _requested: "gpu")
    monkeypatch.setattr("gev.backends.mlx.gemma4.load_gemma4_backbone",
                        lambda _name, _revision, *, compute_dtype: loaded)
    config = replace(
        load_config("configs/gemma4-e2b-mlx-bf16.toml"),
        training=replace(load_config("configs/gemma4-e2b-mlx-bf16.toml").training,
                         dtype="fp32"))
    resolved = resolve_experiment_config(config)

    model = resolved.create_model()

    assert isinstance(model, Gemma4RowModel)
    assert model.provenance["compute_dtype"] == "fp32"
    assert "fp32_staging_receipt" not in model.provenance


@pytest.mark.parametrize("compute_dtype", ["bf16", "fp32"])
def test_mlx_checkpoint_resume_restores_optimizer_schedule_rng_and_precision(
        monkeypatch, tmp_path, compute_dtype):
    class TinyTrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.provenance = {"model_output_id": "gev-gemma4-e2b",
                               "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
                               "base_qualification": _base_qualification(compute_dtype)}
            self.adapter = nn.Linear(4, 4)
            self.dropout = nn.Dropout(0.2)
            self.decoder = nn.Module()
            self.decoder.frozen_embedding = mx.ones(
                (4,), dtype=mx.bfloat16 if compute_dtype == "bf16" else mx.float32)
            self.decoder.freeze()
            self.head = MlxPointerHead(4, width=256)

        def forward_one(self, _encoding):
            decoded = (self.decoder.frozen_embedding * mx.array(
                0.5, dtype=self.decoder.frozen_embedding.dtype)).astype(mx.float32)
            query = self.adapter(self.dropout(decoded))
            options = self.adapter(self.dropout(mx.stack((decoded, decoded * 0.5))))
            return [self.head(query, options)]

    encoding = {"ids": [6, 7, 8, 11, 9, 8, 12, 9, 10], "seg": [0, 1, 1, 1, 1, 1, 1, 1, 1],
                "pos": list(range(9)), "opt": [-1, -1, 0, 0, 0, 1, 1, 1, -2],
                "decide_idx": [8], "opt_idx": [[4, 7]], "labels": [0], "state_length": 1}
    requests = [{"state": "fixture", "questions": {
        "q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"},
              "label": "a"}}, "_meta": {"id": f"r{index}"}} for index in range(12)]
    monkeypatch.setattr("gev.training.schedule.variants_for_request",
                        lambda request, **_: [Variant(
                            {"questions": [{"label": index % 2}]}, encoding,
                            request["_meta"]["id"], "fixture") for index in [int(request["_meta"]["id"][1:])]])
    config = ExperimentConfig(
        "gemma4-resume", ModelConfig("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
                                     "gemma4", marker_ids={"state": 6, "question": 7,
                                         "option_start": 8, "option_end": 9, "decide": 10},
                                     family="gemma4_e2b_text"),
        TrainingConfig(11, 1, 1e-3, "bf16", 384, state_cap=384, branch_cap=1024,
         packed_cap=2048, logical_batch=1, microbatch=1,
         p_none=0, p_none_distract=0, p_distract=0, p_none_pair=0),
        RuntimeConfig("gpu", False, "runs", execution_mode="rows"),
        backend=BackendConfig("mlx"))
    markers = MarkerMap({"state": 6, "question": 7, "option_start": 8, "option_end": 9, "decide": 10},
                        {"state": "<unused0>", "question": "<unused1>", "option_start": "<unused2>",
                         "option_end": "<unused3>", "decide": "<unused4>"},
                        "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")

    def run(directory: Path, *, steps=None, resume=None):
        mx.random.seed(config.training.seed)
        changed = replace(config, training=replace(config.training, max_steps=steps,
                                                    save_every=2 if steps else None))
        changed = replace(changed, training=replace(changed.training, dtype=compute_dtype))
        schedule = TrainingSchedule(requests, changed, None, markers)
        return train(TinyTrainModel(), schedule, changed, directory,
                     source_hash="train-data", manifest={"source": "fixture"},
                     progress=False, resume=resume)

    full, partial = tmp_path / "full", tmp_path / "partial"
    full_metrics = run(full)
    run(partial, steps=6)
    resumed_metrics = run(partial, steps=12, resume=partial / "last_good.resume.mlx")
    assert full_metrics["complete"] and resumed_metrics["complete"]
    assert full_metrics["logical_steps"] == resumed_metrics["logical_steps"] == 12
    assert full_metrics["engineering_checks"]["fp32_frozen_decoder"] == (
        compute_dtype == "fp32")
    assert full_metrics["engineering_checks"]["fp32_lora_master"] is True
    assert full_metrics["engineering_checks"]["optimizer_master_state_fp32"] is True
    full_tensors = mx.load(str(full / "last_good.resume.mlx" / "state.safetensors"))
    resumed_tensors = mx.load(str(partial / "last_good.resume.mlx" / "state.safetensors"))
    assert all(value.dtype == mx.float32 for value in full_tensors.values()
               if value.dtype in {mx.float16, mx.bfloat16, mx.float32})
    assert set(full_tensors) == set(resumed_tensors)
    different = [name for name in full_tensors
                 if not np.array_equal(np.asarray(full_tensors[name]), np.asarray(resumed_tensors[name]))]
    import json
    full_state = json.loads((full / "last_good.resume.mlx" / "state.json").read_text())
    resumed_state = json.loads((partial / "last_good.resume.mlx" / "state.json").read_text())
    model_different = [name for name, key in full_state["model_state"].items()
                       if not np.array_equal(np.asarray(full_tensors[key]),
                                             np.asarray(resumed_tensors[resumed_state["model_state"][name]]))]

    def tagged(value, prefix=""):
        if "tensor" in value:
            return {prefix: value["tensor"]}
        if "dict" in value:
            result = {}
            for key, item in value["dict"]:
                result.update(tagged(item, f"{prefix}.{key}"))
            return result
        return {}

    optimizer_different = []
    for group, value in full_state["optimizer_state"].items():
        expected = tagged(value)
        observed = tagged(resumed_state["optimizer_state"][group])
        optimizer_different.extend(
            f"{group}:{name}" for name, key in expected.items()
            if not np.array_equal(np.asarray(full_tensors[key]), np.asarray(resumed_tensors[observed[name]])))
    assert not model_different and not optimizer_different, {
        "model": model_different, "optimizer": optimizer_different, "raw_keys": different}
    assert full_state["scheduler"] == resumed_state["scheduler"]
    assert full_state["progress"]["augmentation_digest"] == resumed_state["progress"]["augmentation_digest"]
    assert full_state["progress"]["order"] == resumed_state["progress"]["order"]

    if compute_dtype == "fp32":
        full_state["identity"]["training"]["compute_dtype"] = "bf16"
        (full / "last_good.resume.mlx" / "state.json").write_text(
            json.dumps(full_state), encoding="utf-8")
        with pytest.raises(ValueError, match="contract mismatch"):
            run(full, steps=12, resume=full / "last_good.resume.mlx")


@pytest.mark.parametrize("compute_dtype", ["bf16", "fp32"])
def test_mlx_checkpoint_roundtrip_is_backend_strict(tmp_path, compute_dtype):
    from gev.application.training import _checkpoint_metadata
    from gev.configuration.config import load_config
    from gev.configuration.resolved import resolve_experiment_config

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_fixture = mx.ones((2,), dtype=mx.float32)

        def __call__(self, ids):
            dtype = mx.bfloat16 if compute_dtype == "bf16" else mx.float32
            return mx.ones((ids.shape[0], ids.shape[1], 1536), dtype=dtype)

    class Base(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = decoder

    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = replace(config, training=replace(config.training, dtype=compute_dtype))
    resolved = resolve_experiment_config(config)
    base_qualification = {**_base_qualification(compute_dtype),
                          "marker_embeddings": {"main": {"finite": True}},
                          "rmsnorm_policy": "torch-fp32-pow-v1", "rmsnorm_count": 1,
                          "ple_projection_policy": (
                              "torch-fp32-accumulate-bf16-output-v1" if compute_dtype == "bf16"
                              else "torch-fp32-accumulate-fp32-output-v1")}
    loaded_base = {"model": Base(Decoder()), "path": "fake-snapshot",
                   "revision": config.model.revision, "lora": {"layers": 1},
                   "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
                   "base_qualification": base_qualification}
    model = Gemma4RowModel(loaded_base)
    markers = MarkerMap(config.model.marker_ids,
                        {"state": "<unused0>", "question": "<unused1>", "option_start": "<unused2>",
                         "option_end": "<unused3>", "decide": "<unused4>"}, config.model.revision)
    from gev.evaluation.development import selected_ids_sha256
    from gev.models.policy import GEMMA4_POLICY

    selected_ids = ["dev/one"]
    checks = {name: True for name in
              GEMMA4_POLICY.required_qualification_checks("mlx", compute_dtype)}
    report = {"coverage": {"requested_records": 1, "evaluated_records": 1,
                            "requested_questions": 1, "evaluated_questions": 1,
                            "rejected_records": 0, "truncated_records": 0},
              "clean": {"nll": 0.7}}
    report_sha256 = hashlib.sha256(
        (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    qualification = seal_qualification_receipt({
        "format": "gev.trained-checkpoint-qualification", "version": 1,
        "status": "passed", "model_output_id": "gev-gemma4-e2b",
        "family": "gemma4_e2b_text", "backend": "mlx",
        "base": {"name": config.model.name, "revision": config.model.revision,
                 "source_weights_dtype": "bf16", "compute_dtype": compute_dtype},
        "policy_sha256": qualification_policy_sha256(
            "gemma4_e2b_text", "mlx", compute_dtype),
        "code_sha256": qualification_code_sha256(
            "mlx", Path(__file__).resolve().parents[2]),
        "checks": checks, "training_complete": True, "training_steps": 1,
        "development": {
            "suite": "decision-v7", "split": "development",
            "manifest_sha256": "b" * 64, "selected_ids": selected_ids,
            "selected_ids_sha256": selected_ids_sha256(
                [{"_meta": {"id": value}} for value in selected_ids]),
            "selected_record_count": 1,
            "coverage": report["coverage"], "report_sha256": report_sha256,
            "mechanism_checks": {"passed": True, "failures": 0},
            "scores": {"clean": {"nll": 0.7}},
        },
        "trainable_parameters_sha256": trainable_content_sha256("mlx", model),
        "checkpoint_ready": False, "checkpoint_tensor_sha256": None,
    })
    metrics = {"device": "gpu", "dtype": compute_dtype, "compute_dtype": compute_dtype,
               "weights_dtype": compute_dtype, "source_weights_dtype": "bf16",
               "complete": True, "qualification": qualification}
    metrics["base_qualification"] = base_qualification
    metadata = _checkpoint_metadata(config=config, resolved=resolved, markers=markers,
                                     source_hash="a" * 64, manifest_hash="b" * 64,
                                     metrics=metrics, output=tmp_path / "run",
                                     extra={"development_report": report})
    checkpoint = save_checkpoint(model, tmp_path / "checkpoint", metadata)
    loaded, manifest = resolved.load_checkpoint(
        checkpoint, device="gpu", expected_marker_map=markers,
        backbone_loader=lambda *_args, **_kwargs: {
            "model": Base(Decoder()), "path": "fake-snapshot",
            "revision": config.model.revision, "lora": {"layers": 1},
            "source_weights_dtype": "bf16", "compute_dtype": compute_dtype,
            "base_qualification": base_qualification})
    assert manifest["identity"]["backend"] == "mlx"
    assert manifest["identity"]["model_output_id"] == "gev-gemma4-e2b"
    assert manifest["execution"]["compute_dtype"] == compute_dtype
    assert manifest["identity"]["model_contract"]["source_weights_dtype"] == "bf16"
    assert manifest["qualification"]["status"] == "passed"
    assert manifest["qualification"]["checkpoint_ready"] is True
    original = dict(tree_flatten(model.trainable_parameters()))
    restored = dict(tree_flatten(loaded.trainable_parameters()))
    assert original.keys() == restored.keys()
    assert all(np.array_equal(np.asarray(original[name]), np.asarray(restored[name]))
               for name in original)

    manifest_path = checkpoint / "manifest.json"
    value = json.loads(manifest_path.read_text())
    qualification_path = checkpoint / "qualification.json"
    qualification_sidecar = json.loads(qualification_path.read_text())
    qualification_sidecar["status"] = "failed"
    qualification_path.write_text(json.dumps(qualification_sidecar))
    with pytest.raises(ValueError, match="sidecar qualification differs"):
        resolved.load_checkpoint(
            checkpoint, device="gpu",
            backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded before sidecar check"))
    qualification_path.write_text(json.dumps(value["qualification"], indent=2, sort_keys=True,
                                             allow_nan=False) + "\n", encoding="utf-8")
    report_path = checkpoint / "development_report.json"
    report_bytes = report_path.read_bytes()
    report_path.write_bytes(report_bytes + b" ")
    with pytest.raises(ValueError, match="development report digest mismatch"):
        resolved.load_checkpoint(
            checkpoint, device="gpu",
            backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded before report check"))
    report_path.write_bytes(report_bytes)
    receipt_tamper = json.loads(json.dumps(value))
    receipt_tamper["qualification"]["checks"]["source_inventory_exact"] = False
    manifest_path.write_text(json.dumps(receipt_tamper))
    with pytest.raises(ValueError, match="qualification receipt hash mismatch"):
        resolved.load_checkpoint(
            checkpoint, device="gpu",
            backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded before receipt check"))
    manifest_path.write_text(json.dumps(value))
    value["identity"]["backend"] = "torch"
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="backend mismatch"):
        load_checkpoint(checkpoint, config=config, device="gpu",
                        compute_dtype=compute_dtype,
                        backbone_loader=lambda *_args, **_kwargs: pytest.fail("base loaded before backend check"))
