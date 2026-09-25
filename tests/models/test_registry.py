import json
from pathlib import Path

import pytest
from dataclasses import replace

from gev.configuration.config import BackendConfig, load_config
from gev.models.registry import ModelRegistry, resolve_model
from gev.models.specs import BackendCapability, GEMMA3_TEXT, ModelFamilySpec
from gev.configuration.resolved import resolve_experiment_config
from gev.backends.torch.gemma3 import GemmaRowModel, _tiny_config
from gev.backends.torch.pointer import PointerHead
from gev.domain.tokenization import ROLES
from transformers import Gemma3TextModel


class FixtureFamilyRuntime:
    family_id = "fixture_family"

    def load_tokenizer(self, model_name, revision):
        return {"provider": "fixture", "model": model_name, "revision": revision}

    def load_markers(self, artifact_path, tokenizer):
        return {"provider": "fixture-markers", "path": artifact_path}

    def encode_record(self, tokenizer, record, markers, **caps):
        return {"encoded_by": "fixture", "record": record, "caps": caps}


def test_default_registry_resolves_torch_gemma3_text_composition():
    resolved = resolve_model("gemma3_text", "torch")
    assert resolved.family == GEMMA3_TEXT
    assert resolved.backend.backend_id == "torch"
    assert {BackendCapability.AUTOGRAD, BackendCapability.CAUSAL_LM,
            BackendCapability.LORA_ADAPTERS} <= resolved.backend.capabilities
    assert {BackendCapability.CPU, BackendCapability.MPS} <= resolved.backend.capabilities
    assert BackendCapability.CUDA in resolved.backend.capabilities
    assert resolved.output_model_id == "gev-gemma3-1b"
    assert ROLES == GEMMA3_TEXT.marker_roles
    assert GemmaRowModel.__name__ == "GemmaRowModel"
    assert PointerHead.__name__ == "PointerHead"


def test_default_registry_resolves_mlx_gemma4_composition():
    from gev.models.specs import GEMMA4_E2B

    resolved = resolve_model("gemma4_e2b_text", "mlx")
    assert resolved.family == GEMMA4_E2B
    assert not {
        BackendCapability.MLX_GPU, BackendCapability.BF16_SOURCE_WEIGHTS
    } & GEMMA4_E2B.required_backend_capabilities
    assert resolved.backend.backend_id == "mlx"
    assert resolved.backend.capabilities >= GEMMA4_E2B.required_backend_capabilities
    assert resolved.family_runtime.family_id == GEMMA4_E2B.family_id
    assert resolved.output_model_id == "gev-gemma4-e2b"
    assert resolved.output_model_id != "google/gemma-4-E2B"


def test_models_lock_keeps_gemma3_and_gemma4_as_symmetric_model_entries():
    lock = json.loads(Path("models.lock.json").read_text(encoding="utf-8"))
    assert "model" not in lock and "additional_models" not in lock
    entries = {entry["output_model_id"]: entry for entry in lock["models"]}
    assert set(entries) == {"gev-gemma3-1b", "gev-gemma4-e2b"}
    gemma3, gemma4 = entries["gev-gemma3-1b"], entries["gev-gemma4-e2b"]
    assert (gemma3["model"], gemma3["revision"], gemma3["expected_parameter_count"]) == (
        "google/gemma-3-1b-pt", "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29", 999885952)
    assert (gemma4["model"], gemma4["revision"], gemma4["source_weights_dtype"]) == (
        "google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f", "bf16")
    assert "compute_dtype" not in gemma4
    assert {backend["backend"] for backend in gemma4["backend_matrix"]} == {"torch", "mlx"}
    implemented = {
        (entry["backend"], device, dtype)
        for entry in gemma4["backend_matrix"]
        for device in entry["devices"]
        for dtype in entry["compute_dtypes"]
    }
    from gev.models.policy import GEMMA4_POLICY
    assert implemented == GEMMA4_POLICY.implemented
    assert gemma4["expected_text_tensor_count"] == 600
    assert gemma4["expected_text_serialized_parameter_count"] == 4_647_449_891
    assert gemma4["expected_text_parameter_count"] == 4_628_569_344
    assert gemma3["output_model_id"] != gemma3["model"]
    assert gemma4["output_model_id"] != gemma4["model"]


@pytest.mark.parametrize(
    ("family", "backend", "message"),
    [("unknown_family", "torch", "unknown model family"),
     ("gemma3_text", "mlx", "does not implement model family")],
)
def test_registry_rejects_unregistered_components(family, backend, message):
    with pytest.raises(ValueError, match=message):
        resolve_model(family, backend)


def test_registry_composes_a_synthetic_family_with_its_own_backend_factory():
    synthetic_family = ModelFamilySpec(
        "fixture_family", "fixture_arch", ("start", "answer"), (), frozenset())

    class FixtureBackend:
        backend_id = "fixture_backend"
        capabilities = frozenset({BackendCapability.CPU})
        families = frozenset({"fixture_family"})
        architectures = frozenset({"fixture_arch"})

        def create_model(self, family, *, model_name, revision, temperature=1.0,
                         backbone=None, attn_implementation="eager",
                         gradient_checkpointing=False, seed=None):
            return {"family": family.family_id, "backend": self.backend_id,
                    "name": model_name, "revision": revision}

        def create_predictor(self, model, tokenizer, markers, **contract):
            return {"backend": self.backend_id, "model": model, "contract": contract}

        def select_device(self, requested):
            return "fixture-device"

        def seed_rng(self, seed):
            return None

        def train(self, model, schedule, config, output, **kwargs):
            return {"backend": self.backend_id, "step_count": 1}

        def save_checkpoint(self, model, directory, metadata, tokenizer=None):
            return directory

        def load_checkpoint(self, directory, **kwargs):
            return {"loaded": directory}

        def checkpoint_fingerprint(self, directory):
            return "fixture-fingerprint"

        def trainable_fingerprint(self, model):
            return "fixture-trainable-fingerprint"

    family_runtime = FixtureFamilyRuntime()
    composition = ModelRegistry(families=(synthetic_family,), backends=(FixtureBackend(),),
                                family_runtimes=(family_runtime,))
    resolved = composition.resolve("fixture_family", "fixture_backend")
    assert resolved.create_model(model_name="fixture", revision="r0") == {
        "family": "fixture_family", "backend": "fixture_backend",
        "name": "fixture", "revision": "r0"}
    assert resolved.select_device("auto") == "fixture-device"
    assert resolved.train(object(), [], object(), "run")["backend"] == "fixture_backend"
    assert resolved.create_predictor(object(), None, None, state_cap=1,
                                     branch_cap=2, packed_cap=3)["backend"] == "fixture_backend"

    config = load_config("configs/smoke.toml")
    config = replace(config,
                     model=replace(config.model, family="fixture_family",
                                   expected_model_type="fixture_arch",
                                   marker_ids={"start": 1, "answer": 2}),
                     backend=BackendConfig("fixture_backend"))
    resolved_config = resolve_experiment_config(config, registry=composition)
    assert resolved_config.create_model()["backend"] == "fixture_backend"
    tokenizer = resolved_config.load_tokenizer()
    marker_map = resolved_config.load_markers(tokenizer)
    assert tokenizer["provider"] == "fixture"
    assert marker_map["provider"] == "fixture-markers"
    assert resolved_config.encode_record(tokenizer, {"state": "x"}, marker_map)["encoded_by"] == "fixture"
    assert resolved_config.select_device() == "fixture-device"
    assert resolved_config.train(object(), [], "run")["backend"] == "fixture_backend"
    predictor_spec = resolved_config.create_predictor(object(), tokenizer, marker_map)
    assert predictor_spec["backend"] == "fixture_backend"
    assert predictor_spec["contract"]["encoder"].__self__ is family_runtime
    with pytest.raises(ValueError, match="train_profiling"):
        resolved_config.profile_train(0, 1, "profile-output")


def test_torch_device_selection_prefers_cuda_then_mps_then_cpu(monkeypatch):
    import torch
    backend = resolve_model("gemma3_text", "torch").backend
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert backend.select_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.select_device("auto") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert backend.select_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        backend.select_device("cuda")


def test_torch_registry_constructs_tiny_gemma4_text_composition():
    from transformers import Gemma4TextModel
    from gev.backends.torch.gemma4 import Gemma4RowModel, _tiny_config

    config = load_config("configs/gemma4-e2b-mlx-bf16.toml")
    config = replace(config,
                     backend=BackendConfig("torch"),
                     runtime=replace(config.runtime, device="cpu"))
    resolved = resolve_experiment_config(config)
    tiny_backbone = Gemma4TextModel(_tiny_config())
    model = resolved.create_model(backbone=tiny_backbone)
    assert isinstance(model, Gemma4RowModel)
    assert model.compute_dtype == "bf16"


def test_registry_rejects_unsupported_architecture_pair():
    other_family = ModelFamilySpec("gemma3_text", "other_arch", (), (), frozenset())
    registry = ModelRegistry(families=(other_family,))
    with pytest.raises(ValueError, match="does not support architecture"):
        registry.resolve("gemma3_text", "torch")


def test_registry_requires_an_explicit_family_backend_implementation():
    family = ModelFamilySpec("unregistered_family", "gemma3_text", (), (), frozenset())
    runtime = FixtureFamilyRuntime()
    runtime.family_id = "unregistered_family"
    registry = ModelRegistry(families=(family,), family_runtimes=(runtime,))
    with pytest.raises(ValueError, match="does not implement model family"):
        registry.resolve("unregistered_family", "torch")


def test_registry_rejects_missing_backend_capabilities():
    class LimitedBackend:
        backend_id = "limited"
        capabilities = frozenset({BackendCapability.CAUSAL_LM, BackendCapability.LORA_ADAPTERS})
        families = frozenset({"gemma3_text"})
        architectures = frozenset({"gemma3_text"})

        def create_model(self, family, *, model_name, revision, temperature=1.0, backbone=None,
                         attn_implementation="eager", gradient_checkpointing=False, seed=None):
            raise AssertionError("resolution must reject this backend before model creation")

    limited_backend = LimitedBackend()
    registry = ModelRegistry(backends=(limited_backend,))
    with pytest.raises(ValueError, match="autograd"):
        registry.resolve("gemma3_text", "limited")


def test_registry_rejects_supported_capabilities_without_backend_operations():
    backend = type("IncompleteBackend", (), {
        "backend_id": "incomplete",
        "capabilities": GEMMA3_TEXT.required_backend_capabilities,
        "families": frozenset({"gemma3_text"}),
        "architectures": frozenset({"gemma3_text"}),
        "create_model": lambda *args, **kwargs: None,
    })()
    registry = ModelRegistry(backends=(backend,))
    with pytest.raises(ValueError, match="missing operation"):
        registry.resolve("gemma3_text", "incomplete")


def test_torch_backend_factory_constructs_through_selected_composition(monkeypatch):
    backbone = Gemma3TextModel(_tiny_config(layers=6, hidden_size=32))
    monkeypatch.setattr("gev.backends.torch.gemma3.load_real_backbone", lambda *args, **kwargs: backbone)
    model = resolve_model("gemma3_text", "torch").create_model(
        model_name="tiny", revision="0" * 40)
    assert isinstance(model, GemmaRowModel)
    assert model.head.query.in_features == 32
