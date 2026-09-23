# Gev architecture

Gev is a local, reproducible experiment pipeline for decision models. This is
the topology and ownership index; component contracts, scientific rationale,
and workflows are documented in the linked guides. Gev currently has no
comparable full-study result.

## Boundaries and principles

- **Protocol is stable; execution is replaceable.** `protocol`, data lineage,
  and scientific training policy define the experiment. Device, backend,
  checkpoint cadence, and run paths are explicit execution choices and appear
  in provenance without silently changing the scientific recipe.
- **Resolve before loading.** A registered model-family × backend pair and its
  capabilities are validated before tokenizer or weight downloads. Unsupported
  pairs fail early.
- **Keep held-out data behind a trust boundary.** Ordinary data and evaluation
  paths only load train/calibration/development partitions. Locked evaluation
  reserves every requested suite in its ledger before fetching or loading test
  data.

## System topology and ownership

```text
configuration + verified data
             │
             ▼
       resolved config ─── registry ─── family runtime × backend
             │                              │
             ▼                              ▼
 commands → application stages → training/evaluation artifacts
    │                                      │
    └── study planner/runner               └── provenance/checkpoints
                                               │
                           locked registration → reservation → test evaluation
```

## Python package layout

The `gev` package root is an executable boundary only: `__init__.py`,
`__main__.py`, and the thin `cli.py`, alongside `resources/`. Implementation
modules live in their owning packages:

- **`configuration/`** parses strict TOML and resolves model/backend selection;
  **`domain/`** owns transport-neutral contracts, request projection,
  representation/materialization, and tokenizer-independent encoding.
- **`study/`** plans and orchestrates multi-seed runs, generated child configs,
  and isolated child processes; **`artifacts/`** validates checkpoint manifests,
  fingerprints, and bytes. Tensor checkpoint I/O remains in the backend.
- **`diagnostics/`** provides environment, model, precision, profiling, and
  representative-execution diagnostics; **`infrastructure/`** owns system TLS
  setup and packaged immutable-resource lookup. Resources remain under
  `gev/resources/` and are resolved through `importlib.resources`.
- **`data/`** owns pinned suite manifests, acquisition/verification, and
  partition-access rules. Application access verifies bytes and lineage before
  exposing rows.
- **`models/`** defines model-family contracts and registry; **`training/`**
  owns backend-neutral training policy and scheduling.
- **`backends/torch/`** owns Torch model construction, tensor prediction,
  optimizer/RNG/device behavior, checkpoint I/O, and supported diagnostics.
- **`application/`** composes verified data, resolved configuration, family and
  backend operations into train, evaluation, and locked-evaluation stages;
  **`commands/`** is the CLI boundary. `study` is the porcelain workflow;
  `train`, `evaluate`, `data`, `diagnose`, `calibrate`, and `compare` expose
  focused operations. `evaluation/locked` owns candidate checks, once-only
  reservation, and the only route to held-out test acquisition/loading.

Torch-specific implementation imports use `backends/torch/` directly; no
compatibility shims are retained under `models/`, `training/`, or
`evaluation/`. A family runtime supplies tokenizer, markers, and record
encoding; it does not own device tensors.

## Extension model and runtime status

The registry maps family IDs to a family specification/runtime, and backend IDs
to implementations that declare supported families, architectures, and
capabilities. A composition is usable only when the backend satisfies the
family's required capabilities and required operations. To add a family, add a
specification and tokenizer/marker/encoder runtime, then register it; to add a
backend, implement the backend protocol and declare supported pairings and
capabilities. Add the registration and focused contract tests. Neither requires
editing application-stage policy or CLI orchestration. Optional profiling,
precision, and execution diagnostics are capability-gated.

The currently implemented and qualified runtime is Gemma 3 text on Torch, with
CPU/MPS support for the applicable operations. Mac use is supported by that
current Torch/MPS path. **Future qualification roadmap only (not implemented):**
H100/CUDA, then Gemma 4 E4B base, then MLX. These are planned phases, not current
backend/model capabilities or measured results.

## Provenance, resume, and held-out access

Every run records the pinned base/tokenizer identity, family/backend, protocol,
scientific recipe digest, data/manifests, and execution controls. A fresh
warm-start initializes model weights but starts a new optimizer and schedule;
exact resume restores optimizer, scheduler, cursor, RNG, and augmentation state
from a compatible snapshot. The two operations are deliberately distinct.

Before-test reservation is mandatory: candidate registration validates study
selection and completion without reading test rows; locked evaluation atomically
reserves the model/suite identities before invoking its test fetch/load path.
Reservation failures remain recorded and are not retryable under the same key.

## Related documentation

- [Technical design and research record](design.md) — model, protocol, artifact,
  and scientific contracts.
- [Reproduction workflow](reproduction.md) and [installation](install.md) —
  setup, commands, study execution, and recovery.
- [Data provenance](data-provenance.md) and [locked evaluation](locked-evaluation.md)
  — pinned data and the held-out trust boundary.
- [Resume](resume.md) and [continuation](continuation.md) — exact resume versus
  fresh-optimizer warm-start.
- [Runtime execution](attention.md), [precision](precision.md), and
  [evaluation](evaluation.md) — qualification and measurement procedures.
- [Documentation index](README.md).
