# Training and evaluation

How to train, select, calibrate, and test a new Gev run. It continues the
published [`g4-s0` study](results/gemma4-e2b-s0.md) with seed 1. Commands run
from the repository root after the [install steps](../README.md#install).
To release the selected model, see [Publishing](publishing.md).

## Data

The suites are pinned by the Hugging Face dataset revision and verified against
the SHA-256 hashes in the packaged manifests every time they are read.

```bash
uv run gev data fetch decision-v7 train calibration development
uv run gev data fetch transfer-v4 development
```

Leave the test splits until after choosing a checkpoint on development data.
Data fetch verifies each split against its pinned SHA-256 and record counts.
Gemma 4 is public; for gated Gemma 3, run `uv run hf auth login`.

## Train the next seed

The reported run is `runs/g4-s0`. Train **seed 1** with the same MLX/BF16
recipe in a fresh directory to compare with it:

```bash
uv run gev train configs/gemma4-e2b-mlx-bf16.toml --seed 1 --out runs/g4-s1
```

`gev train` saves the LoRA adapter and pointer head in `runs/g4-s1/checkpoint/`;
the frozen Gemma base weights are loaded separately at inference. The run's
`config.json` records the *effective* config, including its seed and backend.
For later seeds, change **both** `--seed` and `--out` (for example, seed 2 goes
to `runs/g4-s2`). Use a new directory rather than overwriting `runs/g4-s0`.

For an interrupted seed-1 run, resume it in place with the **same seed and
config** so the data order and augmentations stay the same. Alternatively,
train with the PyTorch config on CUDA/MPS/CPU; this is a different backend from
the reported seed-0 MLX run:

```bash
uv run gev train configs/gemma4-e2b-mlx-bf16.toml --seed 1 --out runs/g4-s1 --resume
uv run gev train configs/gemma4-e2b-torch-bf16.toml --seed 1 --out runs/g4-torch-s1
```

A small structural smoke run uses an independently sampled subset:

```bash
uv run gev data sample --out data/smoke
uv run gev train configs/smoke.toml --data data/smoke --out runs/smoke --max-steps 20
```

A run directory contains `config.json`, `log.jsonl` (loss, learning rate, tokens,
and step duration), `state/` (the last resumable state, written every
`save_every` steps), and the final `checkpoint/`. The epoch order and the
augmentation are derived from the seed, so a resumed run follows the same data
order as an uninterrupted one.

## Hardware

The reported `g4-s0` run took 3 h 37 min on an Apple M4 Max (64 GB) with
MLX/BF16 and `microbatch = 1`; see its [results](results/gemma4-e2b-s0.md#reproducibility-and-weights)
for the exact token count and throughput. Earlier short-run measurements on an
M1 Max (64 GB) were about 350 tokens/s with MLX and 150 tokens/s with Torch on
MPS; those are not full-run timings. Throughput also depends on the seed's
token mix and hardware.

On a rented CUDA GPU: Gemma 4 E2B's text decoder is about 10 GB in BF16.
Activations dominate memory: each row is backpropagated through all 35 layers,
and rows reach about 1,000 tokens. On Apple MPS, `microbatch = 1` used about
31 GB, and a whole 8-record batch with `microbatch = 8` ran out of memory at
88 GB. Start on a 48–80 GB GPU (L40S, A100, or H100) with the checked-in
`microbatch = 1`. Watch `nvidia-smi` for 20 steps, then raise `microbatch` if
there is headroom. If memory is short, set `gradient_checkpointing = true`: it
trades roughly a third more compute for much less memory. Confirm throughput
with `--max-steps 20` before a full run:

```bash
git clone <this repo> && cd gev
curl https://mise.run | sh && mise install && uv sync --locked
uv run gev data fetch decision-v7 train development
uv run gev train configs/gemma4-e2b-torch-bf16.toml --out runs/profile-torch --max-steps 20
```

`microbatch` in the config is how many augmented records share one forward pass;
the loss and gradients are the same for any value.

## Evaluate and select on development

Evaluate seed 1 on both development suites at raw temperature 1:

```bash
uv run gev evaluate runs/g4-s1 --suite decision-v7 --split development --out runs/g4-s1/eval-dev
uv run gev evaluate runs/g4-s1 --suite transfer-v4 --split development --out runs/g4-s1/eval-transfer-dev
```

Compare `clean.acc` in `runs/g4-s1/eval-transfer-dev/report.json` and
`runs/g4-s0/eval-transfer-dev/report.json`; if tied, compare lower
`clean.brier`. Inspect decision-v7 development as well. For a paired bootstrap
**when both runs' local `rows.jsonl` files are available**:

```bash
uv run gev compare runs/g4-s1/eval-transfer-dev runs/g4-s0/eval-transfer-dev
```

Choose a checkpoint from **development** results, not test. The next section
uses seed 1 *only if it was selected*; if seed 0 still wins, use its existing
calibration and test reports rather than repeating them.

## Calibrate and test the selected model

Evaluate the selected checkpoint on decision-v7 calibration. Fit the
temperature on those saved rows, save the fit, and update its checkpoint:

```bash
uv run gev evaluate runs/g4-s1 --suite decision-v7 --split calibration --out runs/g4-s1/eval-cal
uv run gev calibrate runs/g4-s1/eval-cal --update > runs/g4-s1/eval-cal/calibration.json
```

Check that the temperature in `eval-cal/calibration.json` matches the one in
`checkpoint/gev.json`. Calibration changes metadata, not model weights. Fetch
the test splits if they are not already present, then evaluate this selected
checkpoint once on each:

```bash
uv run gev data fetch decision-v7 test
uv run gev data fetch transfer-v4 test
uv run gev evaluate runs/g4-s1 --suite decision-v7 --split test --out runs/g4-s1/eval-test
uv run gev evaluate runs/g4-s1 --suite transfer-v4 --split test --out runs/g4-s1/eval-transfer-test
```

Reports include accuracy, NLL, Brier, ECE, coverage at 5%/1% error, and AURC
on clean questions, broken down by task, source, and variant. They also include
contrastive-pair flip rates and unknowable-question confidence. Rows store raw
logits: `clean` metrics are always raw T=1; after fitting a temperature, both
test reports also contain `clean_calibrated`. The `g4-s0` run fitted its
temperature between its two test reports; its [results](results/gemma4-e2b-s0.md)
explain how its calibrated decision-test metrics were derived. Following the
order above avoids that.
