# Gev

Gev trains small, calibrated, **Jev-like decision models** on Gemma backbones.
Given a state and a set of questions (choice, yes/no, or score), a model returns a
probability for every option. The architecture follows
[Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked).
Gev is inspired by [Kev](https://github.com/jaredpalmer/kev), another open
Jev-like implementation built on Qwen. It reuses parts of Kev: the request
representation, augmentation, metrics, and the pinned `decision-v7` /
`transfer-v4` evaluation suites.

The completed [Gemma 4 E2B seed-0 result](docs/results/gemma4-e2b-s0.md)
includes development and test measurements on both suites. It is a single-seed
result, not a three-seed study.

A Gev model is:

- a frozen Gemma text decoder with a rank-16 **LoRA** adapter,
- five marker tokens (`<unused0>`..`<unused4>`) that frame the state, question,
  and options,
- a small **pointer head** that scores each option's `</opt>` state against the
  question's `<decide>` state,
- trained with Kev's augmentation (none-of-the-above and distractor options,
  contrastive pairs) on Kev's pinned data.

Each question runs as its own row (`state + question`), so questions never see
each other. Rows are right-padded and batched, which is exact for causal decoders.

| Config | Model | Backend | Notes |
| --- | --- | --- | --- |
| `configs/gemma4-e2b-torch-bf16.toml` | Gemma 4 E2B (text decoder) | PyTorch: CUDA / MPS / CPU | Recommended for training |
| `configs/gemma4-e2b-mlx-bf16.toml` | Gemma 4 E2B (text decoder) | MLX: Apple Silicon GPU | Local training and serving |
| `configs/gemma3-1b-torch-fp32.toml` | Gemma 3 1B | PyTorch | FP32 example; not the published Gemma 4 run |
| `configs/smoke.toml` | Gemma 3 1B | PyTorch | Tiny structural run on a data sample |

Checkpoints use one format on both backends, so a model trained with CUDA can be
served with MLX on a Mac, and the other way round.

## Install

```bash
mise install
mise exec -- uv sync --locked --extra dev --extra mlx  # omit --extra mlx off Apple Silicon
mise exec -- uv run hf auth login                     # Gemma 3 is gated; Gemma 4 is public
```

The examples below write `gev ...`. Run them as `mise exec -- uv run gev ...`, or activate `.venv` first.

## Data

The suites are pinned by the Hugging Face dataset revision and verified against
the SHA-256 hashes in the packaged manifests every time they are read.

```bash
gev data fetch decision-v7 train calibration development
gev data fetch transfer-v4 development
gev data sample --out data/smoke            # 128 train / 64 development records for smoke runs
```

The `test` splits can be fetched and evaluated too. Keep them for the final
measurement of a model you have already selected on development data.

## Train

To produce a Gemma 4 E2B model like the reported seed-0 run on Apple Silicon,
fetch the training and development splits above, then use the **MLX** config and
a new output directory. `gev train` writes the adapter and pointer head to
`<run>/checkpoint/`; the Gemma base model is loaded separately when predicting.

```bash
gev train configs/gemma4-e2b-mlx-bf16.toml --out runs/my-g4-s0
```

For any additional seed, choose an unused integer `n`:

```bash
n=2
gev train configs/gemma4-e2b-mlx-bf16.toml --out "runs/my-g4-s${n}" --seed "$n"
```

Resume an interrupted run, or train with the CUDA/MPS/CPU backend:

```bash
gev train configs/gemma4-e2b-mlx-bf16.toml --out runs/my-g4-s0 --resume
gev train configs/gemma4-e2b-torch-bf16.toml --out runs/my-g4-torch-s0
```

A small smoke run uses the separate sample created above:

```bash
gev train configs/smoke.toml --data data/smoke --out runs/smoke --max-steps 20
```

The reported `runs/g4-s0` checkpoint was trained with MLX/BF16, seed 0, two
epochs, and all 3,144 steps; its saved `config.json` records the exact settings.
Use a fresh directory for a new run rather than overwriting it.

A run directory contains `config.json`, `log.jsonl` (loss, learning rate, tokens,
and step duration), `state/` (the last resumable state, written every
`save_every` steps), and the final `checkpoint/`. The epoch order and the
augmentation are derived from the seed, so a resumed run follows the same data
order as an uninterrupted one.

### On a rented CUDA GPU

Gemma 4 E2B's text decoder is about 10 GB in BF16. Activations dominate memory:
each row is backpropagated through all 35 layers, and rows reach about 1,000
tokens. On Apple MPS, `microbatch = 1` used about 31 GB, and a whole 8-record
batch with `microbatch = 8` ran out of memory at 88 GB. Start on a 48–80 GB GPU
(L40S, A100, or H100) with the checked-in `microbatch = 1`. Watch `nvidia-smi`
for 20 steps, then raise `microbatch` if there is headroom. If memory is short,
set `gradient_checkpointing = true`: it trades roughly a third more compute for
much less memory.

Measured on an M1 Max (64 GB), real data, BF16: MLX trains at about 350 tokens/s
(about 10 s per 8-record step) and Torch on MPS at about 150 tokens/s. The full
two-epoch recipe is about 3,144 steps and 10M tokens, so it takes roughly 8 hours
with MLX on that machine. A datacenter GPU should be one to two orders of
magnitude faster; confirm with `--max-steps 20` before a full run.

```bash
git clone <this repo> && cd gev
curl https://mise.run | sh && mise install && mise exec -- uv sync --locked
mise exec -- uv run gev data fetch decision-v7 train development
mise exec -- uv run gev train configs/gemma4-e2b-torch-bf16.toml --out runs/profile-torch --max-steps 20
```

`microbatch` in the config is how many augmented records share one forward pass;
the loss and gradients are the same for any value.

## Evaluate, calibrate, compare

Evaluate both development suites before choosing a checkpoint. If serving
calibrated probabilities, fit a temperature on the decision calibration split
**after selection and before test**; this changes `checkpoint/gev.json`, not the
weights. Reports always retain raw T=1 metrics in `clean`, and reports made
after fitting also include `clean_calibrated`. Test is the final measurement of
the chosen checkpoint, not a seed-selection tool.

```bash
gev evaluate runs/my-g4-s0 --suite decision-v7 --split development --out runs/my-g4-s0/eval-dev
gev evaluate runs/my-g4-s0 --suite transfer-v4 --split development --out runs/my-g4-s0/eval-transfer-dev
gev evaluate runs/my-g4-s0 --suite decision-v7 --split calibration --out runs/my-g4-s0/eval-cal
```

If you trained another seed, reuse its `n` to compare transfer development
results before selecting a checkpoint:

```bash
n=2  # same value used for the additional run
gev evaluate "runs/my-g4-s${n}" --suite transfer-v4 --split development --out "runs/my-g4-s${n}/eval-transfer-dev"
gev compare "runs/my-g4-s${n}/eval-transfer-dev" runs/my-g4-s0/eval-transfer-dev
```

Optionally calibrate the chosen run, then evaluate its test splits once:

```bash
gev calibrate runs/my-g4-s0/eval-cal --update
gev data fetch decision-v7 test
gev data fetch transfer-v4 test
gev evaluate runs/my-g4-s0 --suite decision-v7 --split test --out runs/my-g4-s0/eval-test
gev evaluate runs/my-g4-s0 --suite transfer-v4 --split test --out runs/my-g4-s0/eval-transfer-test
```

Reports include accuracy, NLL, Brier, ECE, coverage at 5%/1% error, and AURC on
clean questions, broken down by task, source, and variant. They also include
contrastive-pair flip rates and unknowable-question confidence. Rows store raw
logits; the temperature is applied only when scoring. At inference every question
row runs alone: in BF16, batching rows of different lengths shifts Gemma 4 logits by
up to about 0.5, so a score would otherwise depend on its batch neighbours.

The published `g4-s0` run fitted temperature after its decision-test report and
before its transfer-test report. Its [results page](docs/results/gemma4-e2b-s0.md)
compares raw scores across both tests and explains the additional calibrated
transfer metrics.

## Predict

```bash
echo '{"state": "Order #1 arrived damaged.", "questions": {"route": {"type": "choice",
  "instructions": "Which team handles this?", "criteria": {"billing": "Payments", "support": "Product issues"}}}}' \
  | gev predict runs/my-g4-s0
gev predict runs/my-g4-torch-s0 --backend mlx --input request.json  # serve a Torch-trained model with MLX
```

## Publish

```bash
gev push runs/my-g4-s0 --repo <user>/gev-gemma4-e2b \
  --report runs/my-g4-s0/eval-dev/report.json \
  --report runs/my-g4-s0/eval-transfer-dev/report.json \
  --report runs/my-g4-s0/eval-test/report.json \
  --report runs/my-g4-s0/eval-transfer-test/report.json
gev predict <user>/gev-gemma4-e2b --input request.json     # load straight from the Hub
```

`push` uploads the checkpoint (PEFT `adapter_config.json` +
`adapter_model.safetensors`, `pointer.safetensors`, `gev.json`) and generates a
model card with the base model, license, and the reported metrics. Repositories
are private unless you pass `--public`. Gemma 4 derivatives are Apache-2.0;
Gemma 3 derivatives fall under the Gemma terms.

### Evidence in Git

`.gitignore` keeps generated data, weights, and resumable state local. For the
selected `g4-s0` run it allowlists `config.json`, `log.jsonl`,
`checkpoint/{gev,adapter_config}.json`, and the five `eval-*/report.json`
files. They document the exact MLX recipe, training trace, and aggregate
development/test scores. Per-question `rows.jsonl` and all `.safetensors` stay
ignored. To publish a different run's reports, review it first and add an
equally narrow allowlist; do not unignore all of `runs/`.

## Tests

```bash
mise exec -- uv run pytest -q
```

The tests are offline. They use tiny random Gemma 3/Gemma 4 decoders and a
word-level tokenizer to check augmentation and metric parity with pinned Kev
code, row isolation, training, stateless resume, checkpoint round trips, and
Torch↔MLX parity. MLX tests run only on Apple Silicon.

See [Architecture](docs/ARCHITECTURE.md) for the module layout and
[Gemma 4 E2B results](docs/results/gemma4-e2b-s0.md) for the measured run.
