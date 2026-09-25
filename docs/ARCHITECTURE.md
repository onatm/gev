# Architecture

Gev is a local, reproducible experiment pipeline. This overview describes its
flow, ownership boundaries, and scientific trust rules; the checked-in configs,
locks, and manifests remain the source of truth for pinned identities.

## Flow and boundaries

```text
config + verified non-test data
              │
              ▼
      resolved configuration ── model-family × backend registry
              │                              │
              ▼                              ▼
 commands → application stages → training/evaluation artifacts
    │                              │
    └── study planning/runs        └── checkpoints + provenance
                                             │
                   candidate registration → reservation → test evaluation
```

- **Configuration and domain:** `configuration/` parses strict TOML and
  resolves model/backend choices. `domain/` holds backend-neutral contracts,
  request projection, representations, and tokenizer-independent encoding.
- **Data and lineage:** `data/` owns pinned suite manifests, acquisition,
  verification, and partition rules. Access verifies bytes and lineage before
  exposing rows. `artifacts/` validates checkpoint manifests, fingerprints, and
  bytes.
- **Model and training:** `models/` defines family contracts, the registry, and
  the family/backend/device/precision policy; `training/` owns backend-neutral
  scheduling. `backends/torch/` owns Gemma 3 and text-only Gemma 4 construction,
  prediction, optimizer/RNG/device behavior, and checkpoint I/O. `backends/mlx/`
  owns pinned Gemma 4 E2B text-only loading, row execution, LoRA/pointer
  optimization, and backend-tagged checkpoints. Family runtimes provide
  tokenizer, markers, and record encoding, not device tensors. The application
  stage evaluates the bounded Gemma 4 development slice and creates the common
  trained-checkpoint receipt; each backend binds it to its tensor artifacts.
- **Orchestration and CLI:** `application/` composes data, configuration, model,
  and backend operations into stages. `study/` plans and orchestrates multi-seed
  runs. `commands/` is the CLI boundary; [`parser.py`](../src/gev/commands/parser.py)
  is its source of truth. `evaluation/locked.py` owns candidate checks,
  reservation, and the only held-out test acquisition/loading route.
- **Supporting code:** `diagnostics/` contains environment, model, precision,
  profiling, and execution diagnostics. `infrastructure/` handles system TLS
  setup and immutable-resource lookup; `resources/` contains packaged manifests.
  The `gev` package root is an executable boundary, not an implementation layer.

## Runtime and execution policy

Gemma 3 text remains on Torch with CPU/MPS/CUDA support for applicable operations;
the pinned v7 and Night 2 recipes remain fp32. MPS BF16 is not qualified: its
measured probability delta of `0.0505` exceeds the `0.02` acceptance threshold.
Gemma 4 E2B is a separate pinned PRETRAINED `gemma4` family with a nested
`gemma4_text` decoder. The pinned source is BF16 on every backend; Torch supports
CPU/MPS/CUDA and MLX requires the Metal GPU, each with BF16 or FP32 decoder
compute. LoRA masters, pointer parameters, and optimizer state remain FP32. Its
independent output identity is `gev-gemma4-e2b`, distinct from both the HF base
ID and study ID; Gemma 3 remains `gev-gemma3-1b`. [`models.lock.json`](../models.lock.json)
lists their pinned sources as peer model entries, not a default model plus an
extension. Gemma 4 supports independent `rows` execution; packed rows, prefix
caches, CPU fallback for MLX, quantized weights, and the locked-test protocol are
not supported.

One application-level trained-checkpoint receipt coordinates backend-specific
precision/source checks, finite training updates, row isolation, and complete
bounded decision-v7 development evaluation. Scores are descriptive and do not
use a Gemma 3 accuracy threshold. The receipt records a simple pass/fail status,
training completeness, checkpoint readiness, and binds the model, backend,
precision, code/policy, trainable tensors, development selection, and checkpoint
tensors. Gemma 4 training and studies use verified train plus decision-v7
development only; calibration, transfer, and locked-test partitions remain out
of scope. Checkpoint loading validates the receipt before loading the pinned
base; cross-backend loading is rejected. An optional FP32 MLX-vs-Torch CPU
implementation diagnostic remains separate from trained-checkpoint qualification.
`MLX_ENABLE_TF32=0` is set before MLX initialization. Precision and execution
mode belong to the scientific recipe; device and operational controls are
recorded separately. Gemma 3 MPS BF16 precision history and Gemma 4 backend
qualification are summarized in the [README](../README.md#gemma-4-e2b-torch-and-mlx).
The backend/precision matrix and receipt contract are detailed in
[`docs/designs/model-policies.md`](designs/model-policies.md).

## Data, provenance, and held-out access

Pinned scientific and data identity lives in
[`configs/gemma3-1b-v7.toml`](../configs/gemma3-1b-v7.toml),
[`configs/gemma3-1b-night2.toml`](../configs/gemma3-1b-night2.toml),
[`kev.lock.json`](../kev.lock.json), [`models.lock.json`](../models.lock.json),
the manifests under [`src/gev/resources/suites/`](../src/gev/resources/suites/),
and [`src/gev/resources/night2/manifest.json`](../src/gev/resources/night2/manifest.json).
Run artifacts record the resolved recipe, data lineage, model/tokenizer identity,
and execution details; hashes and large provenance tables are kept with those
canonical files rather than repeated here.

The [v7 prototype results](results/gev-v7.md) are a historical snapshot from
the pre-refactor runtime. The metric-only reports are retained as evidence that
the approach was tested; they do not establish a result for the current runtime
or supply a current-format inference checkpoint.

Ordinary fetch, verification, and evaluation paths are restricted to train,
calibration, and development partitions. Candidate registration validates a
complete eligible study and its selected checkpoint without reading test rows;
smoke or partial runs are not eligible. For locked evaluation, every requested
suite is reserved in the ledger before the test fetch/load path is invoked.
Failed reservations are recorded and cannot be retried for the same identity.
This once-only reservation is the held-out trust boundary.
Locked registration and evaluation additionally require the original pinned
Gemma 3 text/Torch/base identity, independent of the recorded experiment ID.

## Recovery is not warm-start

Exact `--resume` requires a compatible snapshot and restores optimizer,
scheduler, progress cursor, RNG, and augmentation state. It continues the same
training identity and must write to a fresh output directory. `--init-from`
loads model weights into a new run with a fresh optimizer and schedule; it is a
warm-start, not a continuation of optimizer state.
