# Gev

Gev is a reproducible experiment pipeline for decision models. It pairs a
pinned Gemma 3 text backbone with versioned decision-data suites, controlled
training, and a once-only protocol for held-out evaluation. Gev has no
comparable full-study result; smoke runs are diagnostics, not comparable scores.

## What you can do

- Validate the pinned model/tokenizer and its marker tokens.
- Fetch and verify approved non-test data, then run a small smoke workflow.
- Plan and run the pinned multi-seed study when data and compute are available.
- Evaluate development data and compare completed study runs.

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

This is a substantial manual run, and this repository does not report a
comparable full Gev study result. For tests, fetch and verify decision-v7 train,
calibration, and development plus transfer-v4 development; tests do not fetch
data and do not require the held-out test partition:

```bash
mise exec -- uv run pytest -q
```

See [Architecture](docs/ARCHITECTURE.md) for system boundaries and scientific
trust rules. The [Kev source revision](https://github.com/jaredpalmer/kev/tree/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b)
is pinned as research reference only, not a Gev result or runtime dependency.
