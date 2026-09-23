# Precision and bounded profiling

`check-precision` is a measurement, not a declaration. It loads one trained
checkpoint and evaluates the same development records with eager FP32, SDPA
FP32, and (on MPS) SDPA BF16. It records raw per-question probabilities keyed
by stable record IDs, maximum probability deltas, KL divergence, argmax flips,
loss finiteness, and actual gradient finiteness. Missing measurements are
`null`; they never qualify. FP32 weights remain the master weights in every
profile. Unsupported operators are reported with their exception and are not
silently moved to CPU.

```sh
gev check-precision --run runs/study-v7/seed-0/checkpoint \
  --config configs/gemma3-1b-v7.toml --records 16 \
  --out runs/precision.json
```

Qualification thresholds are `1e-3` for eager-vs-SDPA FP32 and `.02` for
eager-vs-SDPA BF16. A failed threshold or a non-finite gradient makes the
result fail. The development selection is fixed and includes multi-question
records and the longest banking example; it is not the training recipe.
The checked-in fast configuration remains explicit FP32/eager until the BF16
qualification passes; an unqualified BF16/SDPA result is not treated as a
performance recommendation.

On the observed Apple M4 Max (64 GiB), the MPS BF16 maximum-probability delta
was `0.0505`, exceeding the `.02` qualification threshold. FP32 remains the
default; M5 performance has not been measured. This bounded numerical check is
not a full training or throughput claim.

`profile-train` performs only the requested warmup and measurement logical
steps. It expands the full deterministic augmentation/none-pair stream,
honours logical and microbatch sizes, uses gradient clipping at 1 and the
configured AdamW policy, and reports per-step allocation, driver, RSS, IDs,
lengths, attention implementation, checkpointing, and cache settings. Warmup
steps are excluded from timings.
