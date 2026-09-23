"""Deterministic logical batching and request materialization."""
from __future__ import annotations
import copy, random
from dataclasses import dataclass
from ..data.augmentation import augment, item_rng, none_pair
from ..domain.materialize import materialize
from ..domain.tokenization import encode, rows_of

@dataclass(frozen=True)
class Variant:
    record: dict
    encoding: dict
    original_source_id: str
    source: str
    variant: str = "clean"


def physical_token_count(encoding: dict) -> int:
    """Tokens consumed by row-mode execution (state is repeated per question)."""
    return encoding["state_length"] + sum(len(row["ids"]) for row in rows_of(encoding)[2])

def shuffled_requests(requests: list[dict], seed: int) -> list[dict]:
    result = list(requests)
    random.Random(seed).shuffle(result)
    return result

def variants_for_request(request: dict, *, seed: int, epoch: int, tokenizer, markers,
                         caps: tuple[int, int, int], p_none=.1, p_none_distract=.12,
                         p_distract=.15, p_none_pair=0., encoder=None) -> list[Variant]:
    identifier = request.get("_meta", {}).get("id", request.get("id", ""))
    rng = item_rng(seed, epoch, identifier)
    altered = augment(request, rng, p_none, p_none_distract, p_distract)
    requests = [(altered, "clean")]
    if rng.random() < p_none_pair:
        requests.extend((pair, "none_present" if i == 0 else "none_absent") for i, pair in enumerate(none_pair(request, rng)))
    result = []
    for value, kind in requests:
        rec = materialize(value)
        encode_record = encode if encoder is None else encoder
        enc = encode_record(tokenizer, rec, markers, state_cap=caps[0],
                            branch_cap=caps[1], packed_cap=caps[2])
        result.append(Variant(rec, enc, identifier, request.get("_meta", {}).get("source", "unknown"), kind))
    return result


def variant_count_for_request(request: dict, *, seed: int, epoch: int, p_none=.1,
                              p_none_distract=.12, p_distract=.15, p_none_pair=0.) -> int:
    """Count the exact stream variants without tokenizer/model work."""
    identifier = request.get("_meta", {}).get("id", request.get("id", ""))
    rng = item_rng(seed, epoch, identifier)
    augment(request, rng, p_none, p_none_distract, p_distract)
    return 1 + (2 if rng.random() < p_none_pair and none_pair(request, rng) else 0)

def logical_batches(requests: list[dict], logical_batch: int):
    if logical_batch < 1: raise ValueError("logical_batch must be positive")
    for start in range(0, len(requests), logical_batch):
        yield requests[start:start + logical_batch]
