"""Backend-neutral deterministic logical batch and variant schedule."""

from __future__ import annotations

import hashlib
import json
import random

from ..domain.tokenization import encode
from .batching import logical_batches, variants_for_request


class TrainingSchedule:
    """Own request order, per-record augmentation, and continuation cursor."""

    def __init__(self, requests, config, tokenizer, markers, *, encoder=encode):
        self.requests = list(requests)
        self.ordered = list(requests)
        self.config = config
        self.tokenizer = tokenizer
        self.markers = markers
        self.encoder = encoder
        self.shuffle_rng = random.Random(config.training.seed)
        self.augmentation_digest = hashlib.sha256(b"gev-augmentation-chain-v1").hexdigest()
        self.epoch = 0
        self.batch_cursor = 0
        self._batches = []

    def restore(self, state: dict) -> None:
        progress = state["progress"]
        by_id = {}
        for request in self.requests:
            by_id.setdefault(self._request_id(request), request)
        self.ordered = [by_id[identifier] for identifier in progress["order"]]
        self.epoch = progress["epoch"]
        self.batch_cursor = progress["next_batch"]
        self.shuffle_rng.setstate(progress["shuffle_rng"])
        self.augmentation_digest = progress["augmentation_digest"]

    def batches_for_epoch(self) -> list[list[dict]]:
        if self.batch_cursor == 0:
            self.shuffle_rng.shuffle(self.ordered)
        self._batches = list(logical_batches(self.ordered, self.config.training.logical_batch))
        return self._batches

    def variants_for_batch(self, batch: list[dict], *, epoch: int):
        training = self.config.training
        variants = [variant for request in batch for variant in variants_for_request(
            request, seed=training.seed, epoch=epoch, tokenizer=self.tokenizer,
            markers=self.markers,
            caps=(training.state_cap, training.branch_cap, training.packed_cap),
            p_none=training.p_none, p_none_distract=training.p_none_distract,
            p_distract=training.p_distract, p_none_pair=training.p_none_pair,
            encoder=self.encoder)]
        for order, variant in enumerate(variants):
            payload = json.dumps({"id": variant.original_source_id, "epoch": epoch,
                                  "order": order, "record": variant.record},
                                 sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode()
            self.augmentation_digest = hashlib.sha256(
                bytes.fromhex(self.augmentation_digest) + payload).hexdigest()
        return variants

    def complete_batch(self) -> None:
        self.batch_cursor += 1

    def finish_epoch(self, batch_count: int, *, stopped: bool) -> bool:
        """Reset an exhausted cursor and preserve the exact stop boundary."""
        boundary = self.batch_cursor >= batch_count
        if boundary:
            self.batch_cursor = 0
        if stopped and boundary:
            self.epoch += 1
        return boundary

    def order_ids(self) -> list[str]:
        return [self._request_id(request) for request in self.ordered]

    def _request_id(self, request: dict) -> str:
        return request.get("_meta", {}).get("id", request.get("id", ""))
