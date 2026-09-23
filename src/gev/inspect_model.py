"""Inspect pinned Gemma config/tokenizer without downloading model weights."""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
from typing import Any

from .network import use_system_ssl

SEMANTIC_ROLES = ("state", "question", "option_start", "option_end", "decide")
UNUSED_TOKEN = re.compile(r"<unused(\d+)>")


def _atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".marker-map-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _architecture(config: Any) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    expected = {
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6912,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "sliding_window": 512,
    }
    actual = {key: getattr(config, key, None) for key in expected}
    layer_types = getattr(config, "layer_types", None) or []
    actual["global_layer_indices"] = [i for i, layer in enumerate(layer_types) if layer == "full_attention"]
    actual["rope_parameters"] = getattr(config, "rope_parameters", None)
    actual["rope_local_theta"] = (actual["rope_parameters"] or {}).get("sliding_attention", {}).get("rope_theta")
    actual["rope_global_theta"] = (actual["rope_parameters"] or {}).get("full_attention", {}).get("rope_theta")
    expected.update({"global_layer_indices": [5, 11, 17, 23], "rope_local_theta": 10000.0, "rope_global_theta": 1000000.0})
    mismatches = {key: (value, actual.get(key)) for key, value in expected.items() if actual.get(key) != value}
    return actual, mismatches


def _special_ids(tokenizer: Any, vocabulary: dict[str, int]) -> set[int]:
    excluded: set[int] = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "sep_token_id", "cls_token_id", "mask_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            excluded.add(value)
    # Some tokenizers expose reserved rows as special tokens. Exclude standard
    # controls, but retain literal <unusedN> rows for discovery.
    for token, value in vocabulary.items():
        if token.startswith("<") and token.endswith(">") and not UNUSED_TOKEN.fullmatch(token):
            excluded.add(value)
    return excluded


def _verify_marker(tokenizer: Any, token: str, token_id: int, vocab_size: int) -> bool:
    if not 0 <= token_id < vocab_size or tokenizer.get_vocab().get(token) != token_id:
        return False
    encoded = tokenizer(token, add_special_tokens=False).input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return bool(encoded) and len(encoded) == 1 and encoded[0] == token_id


def _discover_markers(tokenizer: Any, override: dict[str, int] | None, vocab_size: int) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    vocabulary = tokenizer.get_vocab()
    if override is None:
        candidates = sorted(
            ((int(match.group(1)), token, token_id) for token, token_id in vocabulary.items() if (match := UNUSED_TOKEN.fullmatch(token))),
            key=lambda item: (item[0], item[1]),
        )
        excluded = _special_ids(tokenizer, vocabulary)
        candidates = [(number, token, token_id) for number, token, token_id in candidates if token_id not in excluded]
        selected = [{"token": token, "id": token_id} for _, token, token_id in candidates[: len(SEMANTIC_ROLES)]]
        if len(selected) != len(SEMANTIC_ROLES):
            return {}, {"markers": "fewer than five eligible <unusedN> vocabulary rows"}
        proposed = dict(zip(SEMANTIC_ROLES, selected))
    else:
        proposed = {}
        for role in SEMANTIC_ROLES:
            token_id = override[role]
            tokens = sorted(token for token, value in vocabulary.items() if value == token_id)
            proposed[role] = {"token": tokens[0] if tokens else "", "id": token_id}
    errors: dict[str, str] = {}
    excluded = _special_ids(tokenizer, vocabulary)
    for role, marker in proposed.items():
        if marker["id"] in excluded:
            errors[role] = "marker reuses BOS/EOS/PAD/UNK/control ID"
        elif not marker["token"] or not _verify_marker(tokenizer, marker["token"], marker["id"], vocab_size):
            errors[role] = "not an existing non-empty single-token encoding"
    ids = [marker["id"] for marker in proposed.values()]
    if len(set(ids)) != len(ids):
        errors["distinct"] = "marker IDs are not distinct"
    return proposed, errors


def inspect_model(config: Any, output_root: str = "runs") -> dict[str, Any]:
    """Return verified facts; only write the reference after every check passes."""
    try:
        use_system_ssl()
        from transformers import AutoConfig, AutoTokenizer
        model_config = AutoConfig.from_pretrained(config.model.name, revision=config.model.revision)
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
    except Exception as exc:
        message = str(exc)
        gated = any(word in message.lower() for word in ("gated", "401", "403", "access", "token"))
        return {"status": "blocked" if gated else "failed", "error": f"{type(exc).__name__}: {message}"}

    architecture, mismatches = _architecture(model_config)
    facts: dict[str, Any] = {
        "model": config.model.name,
        "revision": config.model.revision,
        "tokenizer_revision": config.model.revision,
        "model_type": getattr(model_config, "model_type", None),
        "architecture": architecture,
        "bos": False,
        "embedding_inspection": "deferred: weights are intentionally not downloaded by inspect-model",
    }
    if facts["model_type"] != config.model.expected_model_type or mismatches:
        return {"status": "failed", "error": "expected Gemma 3 architecture did not match", "facts": facts, "mismatches": mismatches}
    vocab_size = int(getattr(model_config, "vocab_size", 0))
    markers, errors = _discover_markers(tokenizer, config.model.marker_ids, vocab_size)
    if errors:
        return {"status": "failed", "error": "marker verification failed; no artifact written", "facts": facts, "errors": errors, "markers": markers}
    facts["markers"] = markers
    facts["status"] = "verified"
    _atomic_json(pathlib.Path(output_root) / "reference" / "model-marker-map.json", facts)
    return facts
