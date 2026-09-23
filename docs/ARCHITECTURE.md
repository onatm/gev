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
- **Model and training:** `models/` defines family contracts and the registry;
  `training/` owns backend-neutral policy and scheduling. `backends/torch/`
  owns Torch model construction, tensor prediction, optimizer/RNG/device
  behavior, and checkpoint I/O. Family runtimes provide tokenizer, markers,
  and record encoding, not device tensors.
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

The implemented runtime is Gemma 3 text on Torch, with CPU/MPS support for
applicable operations. The pinned v7 and Night 2 recipes specify fp32. MPS BF16
is not qualified: its measured probability delta of `0.0505` exceeds the `0.02`
acceptance threshold. Training and evaluation default to `rows`; `packed` is an
optional ablation, not the default recipe. Older checkpoints are interpreted
as row-trained. Precision and execution mode belong to the scientific recipe;
device and operational controls are recorded separately in provenance.

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

## Recovery is not warm-start

Exact `--resume` requires a compatible snapshot and restores optimizer,
scheduler, progress cursor, RNG, and augmentation state. It continues the same
training identity and must write to a fresh output directory. `--init-from`
loads model weights into a new run with a fresh optimizer and schedule; it is a
warm-start, not a continuation of optimizer state.
