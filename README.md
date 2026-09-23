# Gev

Gev is a reproducible decision-model experiment using a pointer head over a
pinned Gemma 3 text backbone. It includes frozen Kev-style data suites,
controlled augmentation, evaluation/calibration, and an explicit once-only
locked-test protocol. Gev has no comparable full-study accuracy result; the
bounded diagnostics documented here are not full-study results.

## Setup and data

```bash
mise install
mise exec -- uv sync --locked --extra dev
```

Gemma 3 is gated. Obtain access and authenticate through Hugging Face's normal
credential chain when needed; never put a token in a command, config, or
artifact:

```bash
mise exec -- uv run hf auth login
mise exec -- uv run gev inspect-model --config configs/gemma3-1b-v7.toml
```

`inspect-model` verifies the pinned tokenizer and five one-token markers; it
does not download model weights. Fetch only the non-test partitions required by
the workflow:

```bash
mise exec -- uv run gev data fetch decision-v7 train --output data
mise exec -- uv run gev data fetch decision-v7 calibration --output data
mise exec -- uv run gev data fetch decision-v7 development --output data
mise exec -- uv run gev data fetch transfer-v4 development --output data
mise exec -- uv run gev data verify decision-v7 train data/decision-v7/train.jsonl
```

See [`docs/data-provenance.md`](docs/data-provenance.md) for pinned identities,
hashes, and exact partition sizes. Normal data/eval commands cannot load test
examples.

## Smoke run

The smoke child is a small structural diagnostic, not the full training recipe.
The default training dtype is fp32; use a device supported by your environment.

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

## Full study and recovery

After fetching/verifying train, calibration, development, and transfer
development, inspect the plan without loading model weights:

```bash
mise exec -- uv run gev experiment --config configs/gemma3-1b-v7.toml \
  --seeds 0,1,2 --data data --out runs/study-v7 --dry-run
```

If the plan is correct and resources are available, run into a new, unique
output directory by omitting `--dry-run`. The full three-seed study is a
substantial manual run and has not produced a comparable Gev score in this
repository. Monitor `runs/<study>/status.json` and per-stage logs under
`runs/<study>/logs/`. For recovery, use the saved seed config and
`last_good.resume.pt` in a fresh training output directory; never overwrite or
reuse the interrupted trial. See [`docs/reproduction.md`](docs/reproduction.md)
for end-to-end commands and evaluation/selection rules.

## Checks and further reading

```bash
mise exec -- uv run pytest -q
```

The [documentation index](docs/README.md) links the technical design, research
evidence, data provenance, reproduction, evaluation, runtime, and installation
guides. The Kev reference table is research baseline only, not a Gev result.
