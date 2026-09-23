"""Verified marker contracts and non-destructive row/packed encoding.

Encoding contract follows the row metadata shape in:
https://raw.githubusercontent.com/jaredpalmer/kev/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/model.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.specs import GEMMA3_TEXT

ROLES = GEMMA3_TEXT.marker_roles
_SPECIAL = re.compile(r"<\|([^|\n]+)\|>")


class MarkerError(ValueError):
    pass


@dataclass(frozen=True)
class MarkerMap:
    ids: dict[str, int]
    strings: dict[str, str]
    tokenizer_revision: str
    bos: bool = False
    tokenizer_sha256: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], tokenizer: Any | None = None,
                  *, expected_revision: str | None = None) -> "MarkerMap":
        roles = value.get("roles", value.get("markers"))
        revision = value.get("tokenizer_revision", value.get("revision"))
        if roles is None or revision is None or value.get("bos", False) is not False:
            raise MarkerError("marker artifact has unknown or missing fields")
        if not isinstance(roles, dict) or set(roles) != set(ROLES):
            raise MarkerError("marker artifact must contain exactly five semantic roles")
        ids, strings = {}, {}
        for role in ROLES:
            entry = roles[role]
            if not isinstance(entry, dict) or set(entry) not in ({"string", "id"}, {"token", "id"}):
                raise MarkerError(f"invalid marker entry for {role}")
            ident = entry["id"]
            if isinstance(ident, bool) or not isinstance(ident, int) or ident < 0:
                raise MarkerError(f"invalid marker id for {role}")
            ids[role], strings[role] = ident, entry.get("string", entry.get("token"))
        if len(set(ids.values())) != len(ROLES) or not isinstance(revision, str):
            raise MarkerError("marker IDs must be distinct and BOS must be false")
        if expected_revision is not None and revision != expected_revision:
            raise MarkerError("marker artifact tokenizer revision mismatch")
        for field in ("tokenizer_sha256", "artifact_sha256"):
            if field in value and (not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field])):
                raise MarkerError(f"invalid {field}")
        if tokenizer is not None:
            vocab = tokenizer.get_vocab()
            size = max(vocab.values(), default=-1) + 1
            for role in ROLES:
                if strings[role] not in vocab or ids[role] >= size or vocab[strings[role]] != ids[role]:
                    raise MarkerError(f"marker {role} is not the verified tokenizer token")
                encoded = tokenizer(strings[role], add_special_tokens=False).input_ids
                if encoded != [ids[role]]:
                    raise MarkerError(f"marker {role} does not encode as exactly one ID")
            excluded = {getattr(tokenizer, name, None) for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")}
            if any(ident in excluded for ident in ids.values()):
                raise MarkerError("marker cannot reuse BOS/EOS/PAD/UNK")
        return cls(ids, strings, revision, False, value.get("tokenizer_sha256"))

    @classmethod
    def load(cls, path: str | Path, tokenizer: Any | None = None) -> "MarkerMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), tokenizer)


def _user_tokens(tokenizer: Any, text: str, marker_strings: tuple[str, ...] = ()) -> list[int]:
    # Escape Qwen-like controls and Gemma's registered special tokens before the
    # tokenizer sees user data. Structural IDs are inserted only by encode().
    special = set(getattr(tokenizer, "all_special_tokens", ()))
    escaped = _SPECIAL.sub(lambda match: "<¦" + match.group(1) + "¦>", text)
    for token in sorted(special, key=len, reverse=True):
        escaped = escaped.replace(token, token.replace("<", "<¦", 1).replace(">", "¦>", 1))
    for token in sorted(marker_strings, key=len, reverse=True):
        escaped = escaped.replace(token, token.replace("<", "<¦", 1).replace(">", "¦>", 1))
    return list(tokenizer(escaped, add_special_tokens=False).input_ids)


def encode(tokenizer: Any, record: dict, markers: MarkerMap, *, state_cap: int = 384,
           branch_cap: int = 1024, packed_cap: int = 2048) -> dict:
    marker_strings = tuple(markers.strings.values())
    state = _user_tokens(tokenizer, record["state"], marker_strings)
    state_ids = [markers.ids["state"], *state]
    if len(state_ids) > state_cap:
        raise MarkerError(f"state exceeds {state_cap}")
    ids, seg, pos, opt = list(state_ids), [0] * len(state_ids), list(range(len(state_ids))), [-1] * len(state_ids)
    decides, ends, labels = [], [], []
    state_len = len(state_ids)
    for number, question in enumerate(record["questions"], 1):
        instruction = [markers.ids["question"], *_user_tokens(tokenizer, question["instr"], marker_strings)]
        spans = [[markers.ids["option_start"], *_user_tokens(tokenizer, option, marker_strings), markers.ids["option_end"]]
                 for option in question["options"]]
        branch = instruction + [token for span in spans for token in span] + [markers.ids["decide"]]
        if len(branch) + state_len > branch_cap:
            raise MarkerError(f"question {number} exceeds branch cap {branch_cap}")
        start = len(ids)
        ids.extend(branch); seg.extend([number] * len(branch)); pos.extend(range(state_len, state_len + len(branch)))
        cursor = start + len(instruction); question_ends = []
        for index, span in enumerate(spans):
            cursor += len(span); question_ends.append(cursor - 1)
        opt.extend([-1] * len(instruction) + [i for i, span in enumerate(spans) for _ in span] + [-2])
        decides.append(len(ids) - 1); ends.append(question_ends); labels.append(question["label"])
    if len(ids) > packed_cap:
        raise MarkerError(f"packed record exceeds {packed_cap}")
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "decide_idx": decides,
            "opt_idx": ends, "labels": labels, "state_length": state_len,
            "metadata": record.get("metadata", {})}


def rows_of(encoded: dict) -> tuple[list[int], list[int], list[dict]]:
    state_len = encoded["state_length"]
    rows, start = [], state_len
    for question, (decide, ends) in enumerate(zip(encoded["decide_idx"], encoded["opt_idx"]), 1):
        if start >= len(encoded["seg"]) or encoded["seg"][start] != question:
            raise MarkerError("branch layout mismatch")
        stop = decide + 1
        rows.append({"ids": encoded["ids"][start:stop], "pos": encoded["pos"][start:stop],
                     "decide": decide - start, "opts": [end - start for end in ends], "question": question})
        start = stop
    return encoded["ids"][:state_len], encoded["pos"][:state_len], rows


def pad_rows(rows: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad row encodings without changing their logical position IDs."""
    width = max((len(row["ids"]) for row in rows), default=0)
    ids, positions, attention = [], [], []
    for row in rows:
        length = len(row["ids"])
        ids.append(row["ids"] + [pad_id] * (width - length))
        positions.append(row["pos"] + [0] * (width - length))
        attention.append([1] * length + [0] * (width - length))
    return {"ids": ids, "pos": positions, "attention": attention}
