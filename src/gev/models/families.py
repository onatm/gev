"""Model-family runtime policies for tokenizer, markers, and row encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .specs import GEMMA3_TEXT, GEMMA4_E2B


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


@dataclass(frozen=True)
class Gemma4E2BTextRuntime:
    """Pinned Gemma 4 tokenizer with existing reserved rows registered as tokens."""

    family_id: str = GEMMA4_E2B.family_id

    def load_tokenizer(self, model_name: str, revision: str) -> Any:
        from ..infrastructure.network import use_system_ssl

        use_system_ssl()
        from transformers import AddedToken, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        vocabulary = tokenizer.get_vocab()
        tokens = [f"<unused{i}>" for i in range(5)]
        if any(token not in vocabulary or vocabulary[token] != 6 + index
               for index, token in enumerate(tokens)):
            raise ValueError("Gemma 4 reserved marker rows 6..10 do not match the pinned tokenizer")
        original_size = len(tokenizer)
        tokenizer.add_tokens([AddedToken(token, normalized=False) for token in tokens])
        if len(tokenizer) != original_size or any(tokenizer.get_vocab().get(token) != 6 + index
                                                   for index, token in enumerate(tokens)):
            raise ValueError("registering Gemma 4 reserved marker tokens changed tokenizer rows")
        return tokenizer

    def load_markers(self, artifact_path: str | None, tokenizer: Any) -> Any:
        from ..domain.tokenization import MarkerMap

        return MarkerMap.load(artifact_path or "runs/reference/gemma4-e2b-marker-map.json",
                              tokenizer, roles=GEMMA4_E2B.marker_roles)

    def encode_record(self, tokenizer: Any, record: dict, markers: Any, *,
                      state_cap: int, branch_cap: int, packed_cap: int) -> dict:
        from ..domain.tokenization import encode

        return encode(tokenizer, record, markers, state_cap=state_cap,
                      branch_cap=branch_cap, packed_cap=packed_cap)
