"""Marker tokens and row encoding.

Each question becomes one independent decoder row: ``state + branch``. The
branch is ``<question> instructions (<opt> option </opt>)* <decide>``; the
pointer head reads the ``<decide>`` state and each ``</opt>`` state. Row layout
follows Kev ``model.py`` at 08ab0b87d27cb5577a3b371ad7ed4e4686b0502b.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

ROLES = ("state", "question", "option_start", "option_end", "decide")
_SPECIAL = re.compile(r"<\|([^|\n]+)\|>")


class EncodingError(ValueError):
    pass


@dataclass(frozen=True)
class Markers:
    tokens: dict[str, str]
    ids: dict[str, int]

    @classmethod
    def resolve(cls, tokenizer: Any, tokens: Sequence[str]) -> "Markers":
        """Register reserved vocabulary rows as single tokens and verify them.

        Markers must already exist in the base vocabulary (e.g. ``<unusedN>``):
        embeddings are never resized.
        """
        if len(tokens) != len(ROLES):
            raise EncodingError(f"expected {len(ROLES)} marker tokens, got {len(tokens)}")
        from transformers import AddedToken

        vocab = tokenizer.get_vocab()
        missing = [token for token in tokens if token not in vocab]
        if missing:
            raise EncodingError(f"marker tokens are not in the base vocabulary: {missing}")
        size = len(tokenizer)
        tokenizer.add_tokens([AddedToken(token, normalized=False) for token in tokens])
        if len(tokenizer) != size:
            raise EncodingError("registering marker tokens changed the vocabulary size")
        excluded = {getattr(tokenizer, name, None) for name in
                    ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")}
        ids = {}
        for role, token in zip(ROLES, tokens):
            ident = vocab[token]
            if tokenizer(token, add_special_tokens=False).input_ids != [ident]:
                raise EncodingError(f"marker {token!r} does not encode as one token")
            if ident in excluded:
                raise EncodingError(f"marker {token!r} reuses a BOS/EOS/PAD/UNK id")
            ids[role] = ident
        if len(set(ids.values())) != len(ROLES):
            raise EncodingError("marker ids must be distinct")
        return cls(dict(zip(ROLES, tokens)), ids)

    def to_dict(self) -> dict:
        return {role: {"token": self.tokens[role], "id": self.ids[role]} for role in ROLES}


def _escape(text: str, tokens: Sequence[str]) -> str:
    """Neutralize control/marker tokens in user text; structure is inserted only by id."""
    text = _SPECIAL.sub(lambda match: "<¦" + match.group(1) + "¦>", text)
    for token in sorted(tokens, key=len, reverse=True):
        text = text.replace(token, token.replace("<", "<¦", 1).replace(">", "¦>", 1))
    return text


def _user_tokens(tokenizer: Any, text: str, markers: Markers) -> list[int]:
    protected = set(getattr(tokenizer, "all_special_tokens", ())) | set(markers.tokens.values())
    return list(tokenizer(_escape(text, tuple(protected)), add_special_tokens=False).input_ids)


def encode(tokenizer: Any, record: dict, markers: Markers, *, state_cap: int, branch_cap: int) -> dict:
    """Encode a materialized record as a shared state plus one branch per question."""
    ids = markers.ids
    state = [ids["state"], *_user_tokens(tokenizer, record["state"], markers)]
    if len(state) > state_cap:
        raise EncodingError(f"state has {len(state)} tokens, over state_cap={state_cap}")
    rows = []
    for number, question in enumerate(record["questions"], 1):
        branch = [ids["question"], *_user_tokens(tokenizer, question["instr"], markers)]
        ends = []
        for option in question["options"]:
            branch += [ids["option_start"], *_user_tokens(tokenizer, option, markers), ids["option_end"]]
            ends.append(len(branch) - 1)
        branch.append(ids["decide"])
        if len(state) + len(branch) > branch_cap:
            raise EncodingError(f"question {number} row exceeds branch_cap={branch_cap}")
        rows.append({"ids": branch, "decide": len(branch) - 1, "opts": ends})
    return {"state": state, "rows": rows}


def token_count(encoding: dict) -> int:
    """Tokens processed in row mode: the state is repeated for every question."""
    return sum(len(encoding["state"]) + len(row["ids"]) for row in encoding["rows"])


def flatten_rows(encodings: list[dict]) -> list[tuple[int, list[int], int, list[int]]]:
    """(record index, full row ids, decide index, option-end indices) for every question."""
    flat = []
    for index, encoding in enumerate(encodings):
        state = encoding["state"]
        for row in encoding["rows"]:
            offset = len(state)
            flat.append((index, state + row["ids"], offset + row["decide"],
                         [offset + position for position in row["opts"]]))
    return flat
