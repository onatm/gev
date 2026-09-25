# Gev

Gev trains small, calibrated, **Jev-like decision models** on Gemma backbones.
Given a state and a set of questions (choice, yes/no, or score), a model returns a
probability for every option. The architecture follows
[Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked).
Gev is inspired by [Kev](https://github.com/jaredpalmer/kev), another open
Jev-like implementation built on Qwen. It reuses parts of Kev: the request
representation, augmentation, metrics, and the pinned `decision-v7` /
`transfer-v4` evaluation suites.

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
| `configs/gemma3-1b-v7.toml` | Gemma 3 1B | PyTorch | FP32 recipe of the [v7 prototype](docs/results/gev-v7.md) |
| `configs/smoke.toml` | Gemma 3 1B | PyTorch | Tiny structural run on a data sample |

Checkpoints use one format on both backends, so a model trained with CUDA can be
served with MLX on a Mac, and the other way round.

## Install

```bash
mise install
mise exec -- uv sync --locked --extra dev            # add --extra mlx on Apple Silicon
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

```bash
gev train configs/gemma4-e2b-torch-bf16.toml --out runs/g4-s0
gev train configs/gemma4-e2b-torch-bf16.toml --out runs/g4-s1 --seed 1   # more seeds: a shell loop
gev train configs/gemma4-e2b-torch-bf16.toml --out runs/g4-s0 --resume   # after an interruption
gev train configs/smoke.toml --data data/smoke --out runs/smoke --max-steps 20
```

A run directory contains `config.json`, `log.jsonl` (loss, learning rate, and
tokens/s per step), `state/` (the last resumable state, written every
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
mise exec -- uv run gev train configs/gemma4-e2b-torch-bf16.toml --out runs/g4-s0 --max-steps 20
```

`microbatch` in the config is how many augmented records share one forward pass;
the loss and gradients are the same for any value.

## Evaluate, calibrate, compare

```bash
gev evaluate runs/g4-s0 --suite decision-v7 --split development --out runs/g4-s0/eval-dev
gev evaluate runs/g4-s0 --suite decision-v7 --split calibration --out runs/g4-s0/eval-cal
gev calibrate runs/g4-s0/eval-cal --update      # fit a serving temperature into the checkpoint
gev compare runs/g4-s1/eval-dev runs/g4-s0/eval-dev
```

Reports include accuracy, NLL, Brier, ECE, coverage at 5%/1% error, and AURC on
clean questions, broken down by task, source, and variant. They also include
contrastive-pair flip rates and unknowable-question confidence. Rows store raw
logits; the temperature is applied only when scoring. At inference every question
row runs alone: in BF16, batching rows of different lengths shifts Gemma 4 logits by
up to about 0.5, so a score would otherwise depend on its batch neighbours.

## Predict

```bash
echo '{"state": "Order #1 arrived damaged.", "questions": {"route": {"type": "choice",
  "instructions": "Which team handles this?", "criteria": {"billing": "Payments", "support": "Product issues"}}}}' \
  | gev predict runs/g4-s0
gev predict runs/g4-s0 --backend mlx --input request.json   # serve a Torch-trained model with MLX
```

## Publish

```bash
gev push runs/g4-s0 --repo <user>/gev-gemma4-e2b --report runs/g4-s0/eval-dev/report.json
gev predict <user>/gev-gemma4-e2b --input request.json     # load straight from the Hub
```

`push` uploads the checkpoint (PEFT `adapter_config.json` +
`adapter_model.safetensors`, `pointer.safetensors`, `gev.json`) and generates a
model card with the base model, license, and the reported metrics. Repositories
are private unless you pass `--public`. Gemma 4 derivatives are Apache-2.0;
Gemma 3 derivatives fall under the Gemma terms.

## Tests

```bash
mise exec -- uv run pytest -q
```

The tests are offline. They use tiny random Gemma 3/Gemma 4 decoders and a
word-level tokenizer to check augmentation and metric parity with pinned Kev
code, row isolation, training, stateless resume, checkpoint round trips, and
Torch↔MLX parity. MLX tests run only on Apple Silicon.

See [Architecture](docs/ARCHITECTURE.md) for the module layout, and
[the v7 prototype](docs/results/gev-v7.md) for the historical Gemma 3 result.
