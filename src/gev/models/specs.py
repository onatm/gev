"""Backend-neutral model-family and encoded-input contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence, TypeAlias


class BackendCapability(StrEnum):
    AUTOGRAD = "autograd"
    CAUSAL_LM = "causal_lm"
    LORA_ADAPTERS = "lora_adapters"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    MLX_GPU = "mlx_gpu"
    BF16_SOURCE_WEIGHTS = "bf16_source_weights"
    TRAIN_PROFILING = "train_profiling"
    PRECISION_DIAGNOSTICS = "precision_diagnostics"
    EXECUTION_DIAGNOSTICS = "execution_diagnostics"


@dataclass(frozen=True)
class ModelFamilySpec:
    """Tokenizer/architecture/training facts owned by a model family."""

    family_id: str
    architecture: str
    marker_roles: tuple[str, ...]
    lora_targets: tuple[str, ...]
    required_backend_capabilities: frozenset[BackendCapability]
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    pointer_width: int = 256
    output_model_id: str = ""


# Encoded rows are deliberately plain Python data. Backends convert them to
# their native tensor representation; no cross-backend tensor facade is used.
EncodedRecord: TypeAlias = Mapping[str, object]
EncodedBatch: TypeAlias = Sequence[EncodedRecord]


GEMMA3_TEXT = ModelFamilySpec(
    family_id="gemma3_text",
    architecture="gemma3_text",
    marker_roles=("state", "question", "option_start", "option_end", "decide"),
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    required_backend_capabilities=frozenset({
        BackendCapability.AUTOGRAD,
        BackendCapability.CAUSAL_LM,
        BackendCapability.LORA_ADAPTERS,
    }),
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    pointer_width=256,
    output_model_id="gev-gemma3-1b",
)

GEMMA4_E2B_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
GEMMA4_E2B = ModelFamilySpec(
    family_id="gemma4_e2b_text",
    architecture="gemma4",
    marker_roles=GEMMA3_TEXT.marker_roles,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    required_backend_capabilities=frozenset({
        BackendCapability.AUTOGRAD,
        BackendCapability.CAUSAL_LM,
        BackendCapability.LORA_ADAPTERS,
    }),
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    pointer_width=256,
    output_model_id="gev-gemma4-e2b",
)
