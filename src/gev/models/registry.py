"""Resolve a model-family/backend composition before constructing a model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..backends.torch import TorchBackend
from ..backends.mlx import MlxBackend
from ..training.schedule import TrainingSchedule
from .families import Gemma3TextRuntime, Gemma4E2BTextRuntime, ModelFamilyRuntime
from .specs import BackendCapability, GEMMA3_TEXT, GEMMA4_E2B, ModelFamilySpec

_BACKEND_OPERATIONS = (
    "create_model", "create_predictor", "select_device", "seed_rng", "save_checkpoint",
    "load_checkpoint", "checkpoint_fingerprint", "trainable_fingerprint",
    "train",
)


class ModelBackend(Protocol):
    backend_id: str
    capabilities: frozenset[BackendCapability]
    families: frozenset[str]
    architectures: frozenset[str]

    def create_model(self, family: ModelFamilySpec, *, model_name: str, revision: str,
                     temperature: float = 1.0, backbone: object | None = None,
                     attn_implementation: str = "eager",
                     gradient_checkpointing: bool = False,
                     compute_dtype: str | None = None,
                     seed: int | None = None) -> object: ...

    def create_predictor(self, model: object, tokenizer: object, markers: object, *,
                         state_cap: int, branch_cap: int, packed_cap: int,
                         temperature: float = 1.0, execution_mode: str = "rows",
                         encoder: Callable[..., dict] | None = None) -> object: ...

    def select_device(self, requested: str) -> str: ...

    def seed_rng(self, seed: int) -> None: ...

    def save_checkpoint(self, model: object, directory, metadata: dict, tokenizer=None) -> object: ...

    def load_checkpoint(self, directory, *, config, device="cpu", tokenizer=None,
                        expected_marker_map=None, backbone_loader=None,
                        attn_implementation=None, compute_dtype=None) -> object: ...

    def checkpoint_fingerprint(self, directory) -> str: ...

    def trainable_fingerprint(self, model: object) -> str: ...

    def train(self, model: object, schedule: TrainingSchedule, config, output, **kwargs) -> dict: ...

@dataclass(frozen=True)
class ResolvedModel:
    family: ModelFamilySpec
    backend: ModelBackend
    family_runtime: ModelFamilyRuntime

    @property
    def output_model_id(self) -> str:
        return self.family.output_model_id or self.family.family_id

    def require_capability(self, capability: BackendCapability) -> None:
        if capability not in self.backend.capabilities:
            raise ValueError(
                f"backend {self.backend.backend_id} does not provide {capability.value}"
            )

    def load_tokenizer(self, model_name: str, revision: str):
        return self.family_runtime.load_tokenizer(model_name, revision)

    def load_markers(self, artifact_path: str | None, tokenizer):
        return self.family_runtime.load_markers(artifact_path, tokenizer)

    def encode_record(self, tokenizer, record: dict, markers, *,
                      state_cap: int, branch_cap: int, packed_cap: int) -> dict:
        return self.family_runtime.encode_record(
            tokenizer, record, markers, state_cap=state_cap,
            branch_cap=branch_cap, packed_cap=packed_cap)

    def create_model(self, *, model_name: str, revision: str, temperature: float = 1.0,
                     backbone: object | None = None, attn_implementation: str = "eager",
                     gradient_checkpointing: bool = False, seed: int | None = None,
                     compute_dtype: str | None = None) -> object:
        arguments = {
            "model_name": model_name, "revision": revision,
            "temperature": temperature, "backbone": backbone,
            "attn_implementation": attn_implementation,
            "gradient_checkpointing": gradient_checkpointing, "seed": seed,
        }
        # Preserve third-party/synthetic backend signatures; only the Gemma 4
        # built-in family owns this additional precision argument.
        if self.family.family_id == "gemma4_e2b_text":
            arguments["compute_dtype"] = (
                ("bf16" if self.backend.backend_id == "mlx" else "fp32")
                if compute_dtype is None else compute_dtype)
        return self.backend.create_model(self.family, **arguments)

    def create_predictor(self, model: object, tokenizer: object, markers: object, *,
                         state_cap: int, branch_cap: int, packed_cap: int,
                         temperature: float = 1.0, execution_mode: str = "rows") -> object:
        return self.backend.create_predictor(
            model, tokenizer, markers, state_cap=state_cap, branch_cap=branch_cap,
            packed_cap=packed_cap, temperature=temperature, execution_mode=execution_mode,
            encoder=self.family_runtime.encode_record)

    def select_device(self, requested: str) -> str:
        return self.backend.select_device(requested)

    def seed_rng(self, seed: int) -> None:
        self.backend.seed_rng(seed)

    def save_checkpoint(self, model: object, directory, metadata: dict, tokenizer=None) -> object:
        return self.backend.save_checkpoint(model, directory, metadata, tokenizer)

    def load_checkpoint(self, directory, *, config, device="cpu", tokenizer=None,
                        expected_marker_map=None, backbone_loader=None,
                        attn_implementation=None, compute_dtype=None) -> object:
        arguments = {
            "config": config, "device": device, "tokenizer": tokenizer,
            "expected_marker_map": expected_marker_map,
            "backbone_loader": backbone_loader,
            "attn_implementation": attn_implementation,
        }
        if self.family.family_id == "gemma4_e2b_text":
            arguments["compute_dtype"] = compute_dtype
        return self.backend.load_checkpoint(directory, **arguments)

    def checkpoint_fingerprint(self, directory) -> str:
        return self.backend.checkpoint_fingerprint(directory)

    def trainable_fingerprint(self, model: object) -> str:
        return self.backend.trainable_fingerprint(model)

    def train(self, model: object, schedule: TrainingSchedule, config, output, **kwargs) -> dict:
        return self.backend.train(model, schedule, config, output, **kwargs)

    def profile_train(self, config, warmup_steps: int, measure_steps: int,
                      output, *, data_root: str = "data") -> dict:
        self.require_capability(BackendCapability.TRAIN_PROFILING)
        operation = getattr(self.backend, "profile_train", None)
        if operation is None:
            raise ValueError(f"backend {self.backend.backend_id} lacks train profiling support")
        return operation(config, warmup_steps, measure_steps, output, data_root=data_root)

    def compare_precision_profiles(self, model, encodings, record_ids, *,
                                   device: str, include_bf16: bool = True):
        self.require_capability(BackendCapability.PRECISION_DIAGNOSTICS)
        operation = getattr(self.backend, "compare_precision_profiles", None)
        if operation is None:
            raise ValueError(f"backend {self.backend.backend_id} lacks precision diagnostics")
        return operation(
            model, encodings, record_ids, device=device, include_bf16=include_bf16)

    def measure_execution(self, model, encodings, *, records: int, warmup: int = 2) -> dict:
        self.require_capability(BackendCapability.EXECUTION_DIAGNOSTICS)
        operation = getattr(self.backend, "measure_execution", None)
        if operation is None:
            raise ValueError(f"backend {self.backend.backend_id} lacks execution diagnostics")
        return operation(model, encodings, records=records, warmup=warmup)

class ModelRegistry:
    """Small explicit registry; unsupported pairings fail before model loading."""

    def __init__(self, *, families: tuple[ModelFamilySpec, ...] = (GEMMA3_TEXT, GEMMA4_E2B),
                 backends: tuple[ModelBackend, ...] | None = None,
                 family_runtimes: tuple[ModelFamilyRuntime, ...] | None = None) -> None:
        self._families = {family.family_id: family for family in families}
        active_backends = backends if backends is not None else (TorchBackend(), MlxBackend())
        self._backends = {backend.backend_id: backend for backend in active_backends}
        active_family_runtimes = (family_runtimes if family_runtimes is not None else
                                   tuple(runtime for family in families for runtime in (
                                       (Gemma3TextRuntime(),) if family.family_id == GEMMA3_TEXT.family_id
                                       else (Gemma4E2BTextRuntime(),) if family.family_id == GEMMA4_E2B.family_id
                                       else ())))
        self._family_runtimes = {runtime.family_id: runtime for runtime in active_family_runtimes}
        if (len(self._families) != len(families)
                or len(self._backends) != len(active_backends)
                or len(self._family_runtimes) != len(active_family_runtimes)):
            raise ValueError("model family and backend IDs must be unique")

    def register_family(self, family: ModelFamilySpec, runtime: ModelFamilyRuntime) -> None:
        if family.family_id in self._families:
            raise ValueError(f"model family is already registered: {family.family_id}")
        if runtime.family_id != family.family_id:
            raise ValueError("family runtime ID does not match model family ID")
        if runtime.family_id in self._family_runtimes:
            raise ValueError(f"model family runtime is already registered: {runtime.family_id}")
        self._families[family.family_id] = family
        self._family_runtimes[runtime.family_id] = runtime

    def register_backend(self, backend: ModelBackend) -> None:
        if backend.backend_id in self._backends:
            raise ValueError(f"model backend is already registered: {backend.backend_id}")
        self._backends[backend.backend_id] = backend

    def resolve(self, family_id: str, backend_id: str) -> ResolvedModel:
        try:
            family = self._families[family_id]
        except KeyError as exc:
            raise ValueError(f"unknown model family: {family_id}") from exc
        try:
            backend = self._backends[backend_id]
        except KeyError as exc:
            raise ValueError(f"unknown model backend: {backend_id}") from exc
        try:
            family_runtime = self._family_runtimes[family_id]
        except KeyError as exc:
            raise ValueError(f"model family {family_id} has no tokenizer/encoding runtime") from exc
        if family.family_id not in backend.families:
            raise ValueError(f"backend {backend_id} does not implement model family {family.family_id}")
        if family.architecture not in backend.architectures:
            raise ValueError(
                f"backend {backend_id} does not support architecture {family.architecture}"
            )
        missing = family.required_backend_capabilities - backend.capabilities
        if missing:
            raise ValueError(
                "backend " + backend_id + " lacks required capabilities: "
                + ", ".join(sorted(capability.value for capability in missing))
            )
        missing_operations = [name for name in _BACKEND_OPERATIONS
                              if not callable(getattr(backend, name, None))]
        if missing_operations:
            raise ValueError(
                f"backend {backend_id} is incomplete; missing operation(s): "
                + ", ".join(missing_operations)
            )
        return ResolvedModel(family, backend, family_runtime)


DEFAULT_MODEL_REGISTRY = ModelRegistry()


def resolve_model(family_id: str, backend_id: str) -> ResolvedModel:
    """Resolve a registered family/backend pair."""
    return DEFAULT_MODEL_REGISTRY.resolve(family_id, backend_id)
