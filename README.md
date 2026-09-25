# Gev

Gev is a reproducible experiment pipeline for decision models. It pairs pinned
Gemma 3 and Gemma 4 E2B text backbones with versioned decision-data suites and
controlled training. Gemma 3 retains the once-only protocol for held-out
evaluation; Gemma 4 is limited to training and development data. An
[early v7 prototype](docs/results/gev-v7.md) trained on the pre-refactor
implementation verified the idea with a three-seed study and one locked test
of its selected checkpoint. Its saved results are historical evidence, not a
new measurement of this runtime. Smoke runs remain diagnostics.

The pinned sources are peers in [`models.lock.json`](models.lock.json); their
Gev output IDs (`gev-gemma3-1b` and `gev-gemma4-e2b`) identify separate produced
models and are not aliases for the Hugging Face base IDs or experiment IDs.

## What you can do

- Validate the pinned model/tokenizer and its marker tokens.
- Fetch and verify approved non-test data, then run a small smoke workflow.
- Plan and run the pinned multi-seed study when data and compute are available.
- Evaluate development data and compare completed study runs.
- Predict on an unlabeled request using a current-format checkpoint.
- Train/evaluate/predict Gemma 4 E2B text adapters on Torch CPU/MPS/CUDA or MLX
  Metal GPU, using the pinned BF16 source with selectable BF16/FP32 frozen-decoder
  compute and FP32 LoRA masters, pointer head, and optimizer state.

## Get started

Install the locked environment:

```bash
mise install
mise exec -- uv sync --locked --extra dev
```

Gemma 3 access is gated. Authenticate using the normal Hugging Face credential
chain; keep credentials out of commands, configs, and artifacts. Verify the
configured tokenizer and five one-token markers (this does not download model
weights):

```bash
mise exec -- uv run hf auth login
mise exec -- uv run gev diagnose model --config configs/gemma3-1b-v7.toml
```

Fetch only the non-test partitions needed for training and development:

```bash
mise exec -- uv run gev data fetch decision-v7 train --data-root data
mise exec -- uv run gev data fetch decision-v7 calibration --data-root data
mise exec -- uv run gev data fetch decision-v7 development --data-root data
mise exec -- uv run gev data fetch transfer-v4 development --data-root data
mise exec -- uv run gev data verify decision-v7 train data/decision-v7/train.jsonl
mise exec -- uv run gev data verify decision-v7 calibration data/decision-v7/calibration.jsonl
mise exec -- uv run gev data verify decision-v7 development data/decision-v7/development.jsonl
mise exec -- uv run gev data verify transfer-v4 development data/transfer-v4/development.jsonl
```

The manifests pin their identities. Normal data and evaluation commands do not
expose held-out test data.

## Smoke workflow

The smoke child is a small structural diagnostic, not the full training recipe
or a comparable score. Training defaults to fp32; choose a device supported by
your environment (the example uses MPS):

```bash
mise exec -- uv run gev data sample --out data/smoke --train-records 128 \
  --dev-records 64 --seed 0
mise exec -- uv run gev train configs/smoke.toml --data data/smoke \
  --out runs/smoke --device mps
mise exec -- uv run gev evaluate runs/smoke --suite decision-v7 \
  --split development --data data/smoke \
  --temperature 1 --out runs/smoke-eval-t1 --device mps
```

## Full study

After fetching and verifying the required non-test partitions, inspect the
three-seed plan. Run only after review, with a fresh, unique output directory:

```bash
mise exec -- uv run gev study plan configs/gemma3-1b-v7.toml \
  --seeds 0,1,2 --data data
mise exec -- uv run gev study run configs/gemma3-1b-v7.toml \
  --seeds 0,1,2 --data data --out runs/study-v7
```

This is a substantial manual run. The checked-in
[prototype results](docs/results/gev-v7.md) came from the old implementation;
its model weights are not included, and its legacy checkpoint cannot be loaded
by the current manifest-based runtime. For inference with a new-format run,
pass one unlabeled Kev-style JSON request from a file or stdin, for example:

```bash
mise exec -- uv run gev predict runs/study-v7/seed-0 --input request.json
```

`predict` uses the checkpoint's saved configuration by default. It returns
per-question probabilities and the winning option at raw T=1 unless a positive
`--temperature` is supplied; it does not read suites or run evaluation.

## Gemma 4 E2B (Torch and MLX)

Gemma 4 uses the pinned BF16 source checkpoint on every backend. Torch supports
CPU, MPS, and CUDA; MLX requires an Apple Silicon Metal GPU. Both backends support
BF16 and FP32 frozen-decoder compute. In BF16 mode the decoder remains BF16; in
FP32 mode it is upcast. LoRA masters, pointer parameters, and optimizer state
remain FP32 in both modes. The existing local M4 Max 64GB example stays MLX
BF16; FP32 is a separate option, not the local default. The five reserved
`<unusedN>` marker rows are registered as single tokenizer tokens without
resizing embeddings:

For Apple Silicon MLX recipes, install the optional pinned dependencies and
inspect the local BF16 configuration:

```bash
mise exec -- uv sync --locked --extra dev --extra mlx
mise exec -- uv run gev diagnose config configs/gemma4-e2b-mlx-bf16.toml
mise exec -- uv run gev diagnose model --config configs/gemma4-e2b-mlx-bf16.toml
```

Four complete examples share the same pinned model, marker IDs, and decision-v7
protocol while retaining distinct experiment IDs:

| Config | Backend | Compute | Default device |
| --- | --- | --- | --- |
| `configs/gemma4-e2b-mlx-bf16.toml` | MLX | BF16 | Metal GPU |
| `configs/gemma4-e2b-mlx-fp32.toml` | MLX | FP32 | Metal GPU |
| `configs/gemma4-e2b-torch-bf16.toml` | Torch | BF16 | `auto` |
| `configs/gemma4-e2b-torch-fp32.toml` | Torch | FP32 | `auto` |

Torch `auto` chooses CUDA, then MPS, then CPU. Override a config's `auto`
selection with `--device cuda`, `--device mps`, or `--device cpu`; for example,
the Torch BF16 config can be selected explicitly on an H100 with `--device cuda`.
That H100 command is an example only: this host has not verified a full H100
training run. MLX never falls back from its GPU to CPU. Example device overrides
for the same Torch BF16 recipe:

```bash
mise exec -- uv run gev train configs/gemma4-e2b-torch-bf16.toml --data data --out runs/g4-cuda --device cuda
mise exec -- uv run gev train configs/gemma4-e2b-torch-bf16.toml --data data --out runs/g4-mps --device mps
mise exec -- uv run gev train configs/gemma4-e2b-torch-bf16.toml --data data --out runs/g4-cpu --device cpu
```

For MLX, `MLX_ENABLE_TF32=0` is enforced before MLX initialization and
`--device gpu` selects Metal explicitly.

Fetch only training and development data. Every model has an independent Gev
output identity (`gev-gemma3-1b` versus `gev-gemma4-e2b`); neither HF base IDs,
experiment IDs, nor Gemma 3 study results serve as model-selection authorities
for the other. Both backends collect precision-specific checks for the pinned
source inventory, selected decoder dtype, FP32 trainable/optimizer state, finite
gradients and updates, row isolation, and complete bounded development coverage.
MLX additionally checks marker/boundary gathers, decoder masks, and shared-KV
structure; Torch checks causal attention behavior. Development accuracy, NLL,
Brier, ECE, and per-source metrics are descriptive, with no Gemma 3 score
threshold. A common trained-checkpoint receipt reports `passed` or `failed`,
records training completeness and checkpoint readiness, and binds backend,
precision, source/code/policy identity, trainable content, development
manifest/selection, the development report digest and checkpoint tensors. It is
created for both BF16 and FP32; no external FP32 receipt is required. Gemma 4
remains locked-test-ineligible. Pre-phase-4 unpublished receipts/checkpoints may
be replaced; they are not a compatibility target.

An optional offline FP32 implementation diagnostic compares MLX and the Torch
CPU oracle on the same pinned source. It is separate from trained-checkpoint
qualification and is not a Gemma 3 golden-model test. It can require unified
memory and downloads the pinned checkpoint:

```bash
mise exec -- uv run gev diagnose model --probe \
  --config configs/gemma4-e2b-mlx-bf16.toml --data data \
  --out runs/reference/gemma4-fp32-implementation-diagnostic.json
```

**Why the old `0.02` limit is not a Gemma 4 gate:** it came from the original
Gemma 3 MPS diagnostic (`docs/precision.md`, `src/gev/precision.py` in the
initial implementation), which compared eager/SDPA BF16 and FP32 profiles of
one trained Gemma 3 model. No scientific derivation for `.02` was recorded. It
is a Gemma 3 engineering diagnostic, not a cross-model quality bar. The old
Gemma 4 BF16-vs-FP32 measurements remain archived evidence only: base random
heads `0.65648`/24 flips; 1-step `0.12697`/1 flip; 12-step `0.24210566`/4 flips.
The separate optional FP32 implementation diagnostic passed at `0.0002383`, but
does not authorize or gate BF16 training.

**Historical host result (2026-09-24):** a native-BF16 12-step run reached 12
logical steps (`complete: false`), updated **205** LoRA modules and the
256-wide pointer, and passed BF16 engineering plus bounded development checks.
That run predates the current common checkpoint-receipt schema; replace its
unpublished receipt/checkpoint before current receipt validation. On the
source-balanced non-paired development
subset, 20 records/24 questions had accuracy `0.30`, NLL `2.04871`, Brier
`0.89170`, ECE `0.41698`; row-isolation checks passed with zero mechanism delta.
One unlabeled raw-T=1 prediction returned `billing` at probability `0.96677`.
Training-only time was `45.85 s`, `0.262` logical steps/s and `461.75`
physical tokens/s. The physical counter now counts the state once per question:
`21,171` row-execution tokens versus `19,893` logical tokens, correcting the
previously over-reported throughput. Training-phase process RSS was
`10,531,536,896` bytes at entry and sampled peak; observed MLX active memory
peaked at `10,058,930,530` bytes, with MLX reporting a `20,404,121,183`-byte
peak. The full command took `59.32 s`; process-wide peak RSS was
`10,559,111,168` bytes. These are bounded local diagnostics, not a full
two-epoch recipe or multi-seed study. Outputs are under
`runs/reference/gemma4-bf16-native-max12-final4-20260924/`.

```bash
mise exec -- uv run gev data fetch decision-v7 train --data-root data
mise exec -- uv run gev data fetch decision-v7 development --data-root data
mise exec -- uv run gev diagnose model --probe --config configs/gemma4-e2b-mlx-bf16.toml
mise exec -- uv run gev data audit tokens --config configs/gemma4-e2b-mlx-bf16.toml --markers runs/reference/gemma4-e2b-marker-map.json --data-root data
mise exec -- uv run gev train configs/gemma4-e2b-mlx-bf16.toml --data data --out runs/gemma4-e2b --device gpu
mise exec -- uv run gev evaluate runs/gemma4-e2b --suite decision-v7 --split development --data data --out runs/gemma4-e2b-development
mise exec -- uv run gev predict runs/gemma4-e2b --config configs/gemma4-e2b-mlx-bf16.toml --input request.json --device gpu
mise exec -- uv run gev study plan configs/gemma4-e2b-mlx-bf16.toml --seeds 0,1,2 --data data
mise exec -- uv run gev study run configs/gemma4-e2b-mlx-bf16.toml --seeds 0,1,2 --data data --out runs/gemma4-study
```

Gemma 4 execution is row-only with BF16 source and BF16/FP32 frozen-decoder
compute: Torch supports CPU/MPS/CUDA and MLX requires a Metal GPU. Packed
execution, prefix caching, quantization, warm-start from Gemma 3, and locked-test
registration are not supported. Gemma 4 studies use training data and decision-v7 development
only (no calibration or transfer splits), with development macro NLL selection.
Do not fetch or evaluate held-out test partitions for this family.

For tests, fetch and verify decision-v7 train, calibration, and development
plus transfer-v4 development; tests do not fetch data and do not require the
held-out test partition:

```bash
mise exec -- uv run pytest -q
```

See [Architecture](docs/ARCHITECTURE.md) for system boundaries and scientific
trust rules and [prototype results](docs/results/gev-v7.md) for the recorded
measurements. The [Kev source revision](https://github.com/jaredpalmer/kev/tree/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b)
is pinned as research reference only, not a Gev runtime dependency.
