"""Explicit masks for independent rows and packed Gemma question streams."""
from __future__ import annotations

import torch


def _format(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    allowed = allowed[:, None]
    if dtype == torch.bool:
        return allowed
    if dtype.is_floating_point:
        return torch.where(allowed, torch.zeros((), dtype=dtype, device=allowed.device),
                           torch.full((), torch.finfo(dtype).min, dtype=dtype, device=allowed.device))
    raise ValueError("mask dtype must be bool or floating point")


def row_attention_mask(positions: torch.Tensor, layer_type: str, *, window: int | None = None,
                       dtype: torch.dtype = torch.bool, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Return [batch, 1, query, key] masks for contiguous physical rows.

    Padding is invalid, full layers are causal, and sliding layers additionally
    restrict logical distance.  Bool is used by SDPA; eager callers may request
    additive fp32 values.
    """
    if positions.ndim != 2: raise ValueError("positions must be [batch, sequence]")
    valid = positions >= 0 if valid is None else valid.to(torch.bool)
    if valid.shape != positions.shape:
        raise ValueError("valid must have the same shape as positions")
    query = positions[:, :, None]
    key = positions[:, None, :]
    allowed = valid[:, :, None] & valid[:, None, :] & (key <= query)
    if layer_type == "sliding_attention":
        if window is None or window < 1: raise ValueError("sliding attention requires a positive window")
        allowed &= (query - key) < window
    elif layer_type != "full_attention":
        raise ValueError(f"unknown attention layer type: {layer_type}")
    # A padded query has no useful output, but giving it a finite diagonal
    # keeps eager softmax and MPS SDPA away from an all-masked row.  The
    # diagonal key remains a padding key and is never visible to a real query.
    padded = ~valid
    allowed |= padded[:, :, None] & torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)[None]
    return _format(allowed, dtype)


def packed_attention_mask(segments: torch.Tensor, positions: torch.Tensor, layer_type: str, *,
                          window: int | None = None, valid: torch.Tensor | None = None,
                          dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """Return a separate mask for one packed stream.

    Segment zero is the shared state; positive segments are questions.  A
    question sees causal state and its own causal branch, never a sibling or
    a future question.  ``positions`` are logical positions, deliberately not
    physical tensor offsets.
    """
    if segments.ndim != 2 or positions.shape != segments.shape:
        raise ValueError("segments and positions must both be [batch, sequence]")
    if layer_type not in {"full_attention", "sliding_attention"}:
        raise ValueError(f"unknown attention layer type: {layer_type}")
    if layer_type == "sliding_attention" and (window is None or window < 1):
        raise ValueError("sliding attention requires a positive window")
    valid = (segments >= 0) if valid is None else valid.to(torch.bool)
    if valid.shape != segments.shape:
        raise ValueError("valid must have the same shape as segments")
    qseg, kseg = segments[:, :, None], segments[:, None, :]
    qpos, kpos = positions[:, :, None], positions[:, None, :]
    same_scope = (kseg == 0) | (kseg == qseg)
    allowed = valid[:, :, None] & valid[:, None, :] & same_scope & (kpos <= qpos)
    if layer_type == "sliding_attention":
        allowed &= (qpos - kpos >= 0) & (qpos - kpos < window)
    padded = ~valid
    allowed |= padded[:, :, None] & torch.eye(segments.shape[1], dtype=torch.bool, device=segments.device)[None]
    return _format(allowed, dtype)


def continuation_attention_mask(query_segments: torch.Tensor, query_positions: torch.Tensor,
                                key_segments: torch.Tensor, key_positions: torch.Tensor,
                                layer_type: str, *, window: int | None = None,
                                dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """Mask a query suffix against the key range exposed by a KV cache."""
    if query_segments.ndim != 2 or key_segments.ndim != 2:
        raise ValueError("segments must be batched")
    if query_segments.shape[0] != key_segments.shape[0]:
        raise ValueError("query and key batches must match")
    qseg, kseg = query_segments[:, :, None], key_segments[:, None, :]
    qpos, kpos = query_positions[:, :, None], key_positions[:, None, :]
    allowed = ((kseg == 0) | (kseg == qseg)) & (kpos <= qpos)
    if layer_type == "sliding_attention":
        if window is None or window < 1:
            raise ValueError("sliding attention requires a positive window")
        allowed &= (qpos - kpos >= 0) & (qpos - kpos < window)
    elif layer_type != "full_attention":
        raise ValueError(f"unknown attention layer type: {layer_type}")
    return _format(allowed, dtype)
