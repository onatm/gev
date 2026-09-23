# Attention execution modes

The default training and evaluation mode is `rows`. Packed execution is an
explicit ablation (`runtime.execution_mode = "packed"` or `gev evaluate
RUN --execution packed`) and is recorded in run/checkpoint provenance.

Packed records use segment zero for shared state and one positive segment per
question. Full and sliding masks are separate: full attention is logical
causal within state/question scope, while sliding attention additionally uses
the logical window. Padding is never a real key and padded queries receive a
finite diagonal sentinel for eager and SDPA stability.

```bash
mise exec -- uv run gev diagnose execution --run runs/study-v7/seed-0 \
  --config configs/gemma3-1b-v7.toml --records 8 --device mps \
  --out runs/execution-parity.json
```

Prefix caches are inference-only state prefill caches. Each answer gets a
private cache clone; model identity, device, dtype, parameter versions, and
training mode are checked before reuse.

The measured real-weight MPS diagnostic used 8 records and 16 questions,
stratified as 3 AGNews, 2 Yelp, and 3 Banking77 records. Max probability
delta was `3.427e-6` (packed-vs-rows, cached-first-vs-rows, and cached-repeat-
vs-rows all passed) at the MPS `1e-3` threshold. Batch throughput was
`357.67 ms/record` rows and `196.71 ms/record` packed; complete prefix-cache
miss and reused-prefix hit timings were `254.63 ms/record` and `57.97
ms/answer`, respectively. This is a bounded development diagnostic, not a
general performance claim.
