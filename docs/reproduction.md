# Reproduction workflow

This guide separates the pinned research recipe from bounded smoke and runtime
diagnostics. There is currently no comparable full Gev three-seed score. Do not
interpret smoke, partial, or one-step pipeline output as a reproduction.

## Prerequisites and non-test data

Install the locked environment and authenticate for gated Gemma access as
described in the [root README](../README.md). `inspect-model` reads the pinned
config/tokenizer and validates marker rows without fetching model weights. The
full v7 run requires verified decision-v7 train, calibration, and development
partitions and transfer-v4 development. Fetch them with `gev data fetch`; verify
each local file with `gev data verify`. Do not fetch or open a test partition
for training, calibration, or development evaluation. Exact hashes and sizes
are in [data provenance](data-provenance.md).

## Smoke diagnostic

Create a group-preserving child from train and development, then train/evaluate
with its explicit smoke config:

```bash
mise exec -- uv run gev data smoke --out data/smoke --train-records 128 \
  --dev-records 64 --seed 0
mise exec -- uv run gev train --config configs/smoke.toml \
  --suite decision-v7 --split train --data data/smoke \
  --out runs/smoke --device mps
mise exec -- uv run gev eval --run runs/smoke --suite decision-v7 \
  --split development --data data/smoke --config configs/smoke.toml \
  --temperature 1 --out runs/smoke-eval-t1 --device mps
```

Use `cpu` or omit `--device` if appropriate. This is a small diagnostic, not
the full 12,576-record training population or a candidate eligible for locked
evaluation. One-step experiments with `--smoke --max-steps 1` are also
diagnostics and cannot be registered as candidates.

Observed smoke evidence: the 128-train/64-development run completed 32 updates
with 290 variants. Its development evaluation covered 64 records and 74
questions (40 clean), with clean accuracy `0.375` and T=1 NLL `1.30848`.
These are smoke measurements, not Kev-table comparisons. A separate one-update
full-data pipeline diagnostic covered calibration 968/1,148, development
1,204/1,468, and transfer 764/764 records/questions, with full coverage and no
rejections/truncations; this is evaluation-path evidence only, not full
training or a comparable Gev score.

## Plan and run the full study

First validate and inspect a dry-run plan. The dry-run checks data identities
and reports the planned work without loading model weights or test data:

```bash
mise exec -- uv run gev validate-config configs/gemma3-1b-v7.toml
mise exec -- uv run gev experiment --config configs/gemma3-1b-v7.toml \
  --seeds 0,1,2 --data data --out runs/study-v7 --dry-run
```

Before loading model weights, audit actual token lengths for all three seeds,
both epochs, and the non-test evaluation splits. The audit checks state plus
each question against the branch cap and includes none-pair variants. Confirm
every partition reports `overflow_records: 0`:

```bash
mise exec -- uv run gev data audit decision-v7 train data/decision-v7/train.jsonl \
  --config configs/gemma3-1b-v7.toml \
  --markers runs/reference/model-marker-map.json --augment \
  --output runs/reference/v7-token-length-audit.json
```

The verified maximum state-plus-question length for seeds 0, 1, and 2 is
1,037 tokens. The branch cap is 2,048; row tensors are padded to the actual
batch maximum, not to this limit.

Once the plan, token audit, and resources are reviewed, run it using a fresh,
unique output directory (omit `--dry-run`):

```bash
mise exec -- uv run gev experiment --config configs/gemma3-1b-v7.toml \
  --seeds 0,1,2 --data data --out runs/study-v7
```

Full study execution defaults to atomic resumable snapshots every 100 optimizer
steps. The runner writes seed-specific configs under `configs/`, per-stage
combined output under `logs/seed-<n>/`, and atomically refreshes `status.json`
with the active seed/stage, latest training step/loss, elapsed time, log path,
and outcome. `result.json` records stage outcomes and error excerpts; the full
traceback/output is in each stage log. A failed child is not automatically
retried or resumed. Inspect status, result, logs, and the trial's last good
snapshot before choosing to continue.

### Manual resume

Resume from the interrupted seed's saved config and `last_good.resume.pt`,
using the original data and a new training output directory. For example:

```bash
mise exec -- uv run gev train \
  --config runs/study-v7/configs/seed-0.toml \
  --suite decision-v7 --split train --data data \
  --resume runs/study-v7/seed-0/last_good.resume.pt \
  --out runs/study-v7-resumed/seed-0
```

Only operational limits such as `max_steps` and snapshot interval may change;
the recipe, source data, and manifest must match the snapshot. A failed run may
not have reached its first saved optimizer boundary. A manually resumed run is
a distinct run: evaluate it and retain its lineage rather than treating the
original failed trial as completed.

## Evaluation, calibration, and reference comparison

The experiment runner evaluates each trial on decision-v7 calibration and
development and transfer-v4 development at raw temperature one. Calibration
uses saved evaluation rows; `--run` names the directory containing
`rows.json` and `report.json` (for example the calibration or development
subdirectory):

```bash
mise exec -- uv run gev calibrate --run runs/study-v7/seed-0/calibration \
  --protocol kev-screening \
  --out runs/study-v7/seed-0/calibration/temperature.json
```

The screening protocol uses clean ID calibration and its screening grid. The
release protocol fits from clean ID development with grouped out-of-fold
evaluation and bootstrap; do not fit temperature on transfer or test data.
Apply the selected temperature only once. Compare saved evaluation rows using
the paired grouped bootstrap, and compare only matching suites/splits:

```bash
mise exec -- uv run gev compare \
  --candidate runs/candidate/development \
  --reference runs/reference/development --aggregation micro \
  --samples 1000 --seed 0 --out runs/compare.json
```

Kev's published scores are reference baselines, not Gev measurements. The
correctly labeled reference table and evidence are in [design](design.md).
No full Gev study score is claimed here.

## Optional Night 2 continuation

Night 2 is a warm-start continuation, not a replacement for v7 training. It
requires a verified complete v7 checkpoint as initializer; smoke or partial
initializers are diagnostic only. Fetch the pinned continuation data, then
inspect the plan before starting any model run:

```bash
mise exec -- uv run gev data continuation-fetch --data-root data
mise exec -- uv run gev continue-training \
  --config configs/gemma3-1b-night2.toml \
  --init-from runs/study-v7/seed-0 --data data \
  --out runs/gev-night2 --dry-run
```

If the initializer lineage and plan are verified, omit `--dry-run` and use a
new output path to execute. The operation loads weights and starts a fresh
optimizer; it does not resume the v7 optimizer state. The intended assembly is
1,425 pinned Night 2 records followed by 2,000 deterministic train-replay
records (3,425 total, 429 logical steps). See [continuation](continuation.md)
and [data provenance](data-provenance.md).

## Candidate registration and locked test

Locked test access is a separate manual, once-only operation, and must happen
only after a complete eligible study, its promotion choice, and candidate
weights have been reviewed. Candidate registration checks the study's selected
trial and full training lineage without loading test examples. A one-step or
smoke result cannot pass those checks:

```bash
mise exec -- uv run gev register-candidate \
  --run runs/study-v7/seed-0 --study runs/study-v7/result.json \
  --out runs/selection.json
```

Use the actual selected trial path from `result.json` (the example seed is not
an instruction to select seed 0). Confirm the generated selection and data
identities before explicitly performing the once-only evaluation:

```bash
mise exec -- uv run gev eval-locked --run runs/study-v7/seed-0 \
  --selection runs/selection.json --suites decision-v7,transfer-v4 \
  --data data --out runs/locked \
  --ledger runs/locked-ledger.jsonl
```

The locked command reserves each requested suite in the ledger before loading
test examples. Failed or interrupted reservations cannot be retried for the
same model/suite identity. See [locked evaluation](locked-evaluation.md) for
the selection and ledger contract. Never run this command for a diagnostic or
to explore alternative candidates.
