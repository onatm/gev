"""Model-family runtime policies for tokenizer, markers, and row encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .specs import GEMMA3_TEXT


class ModelFamilyRuntime(Protocol):
    family_id: str

    def load_tokenizer(self, model_name: str, revision: str) -> Any: ...

    def load_markers(self, artifact_path: str | None, tokenizer: Any) -> Any: ...

    def encode_record(self, tokenizer: Any, record: dict, markers: Any, *,
                      state_cap: int, branch_cap: int, packed_cap: int) -> dict: ...


@dataclass(frozen=True)
class Gemma3TextRuntime:
    """Current HF tokenizer and five-marker Gemma text representation."""

    family_id: str = GEMMA3_TEXT.family_id

    def load_tokenizer(self, model_name: str, revision: str) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, revision=revision)

    def load_markers(self, artifact_path: str | None, tokenizer: Any) -> Any:
        from ..domain.tokenization import MarkerMap

        return MarkerMap.load(artifact_path or "runs/reference/model-marker-map.json", tokenizer)

    def encode_record(self, tokenizer: Any, record: dict, markers: Any, *,
                      state_cap: int, branch_cap: int, packed_cap: int) -> dict:
        from ..domain.tokenization import encode

        return encode(tokenizer, record, markers, state_cap=state_cap,
                      branch_cap=branch_cap, packed_cap=packed_cap)
