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

## Highlights

- Ask yes/no (`noul`), multiple-choice (`choice`), and rating (`score`) questions
  about one state. Each question is scored independently and returns option
  probabilities rather than generated text.
- The `gev-e2b` seed-0 run scores **0.829** on trained-source test questions and
  **0.625** on new-source test questions; its raw new-source Brier is **0.468**
  (lower is better). See the [full results](docs/results/gemma4-e2b-s0.md).
- A temperature fitted on the separate decision-v7 calibration split is saved
  with the checkpoint. On new-source test questions it reduces ECE from 0.114
  to 0.057 without changing accuracy.
- Train on Apple Silicon with MLX or on CUDA/MPS/CPU with PyTorch. Both backends
  share the same LoRA-adapter and pointer-head checkpoint format.

## Models

Currently the only reported model is **gev-e2b**, a single Gemma 4 E2B seed-0
run. Its [model card and weights](https://huggingface.co/onatm/gev-e2b) are on
Hugging Face; the results and checkpoint provenance are linked below.

| Model | Base | Accuracy: New Sources | Accuracy: Trained Sources | Brier: New Sources ↓ | Backends | Results |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [gev-e2b](https://huggingface.co/onatm/gev-e2b) | Gemma 4 E2B | 0.611 / 0.625 | 0.797 / 0.829 | 0.499 / 0.468 | MLX (Apple Silicon), PyTorch (CUDA/MPS/CPU) | [Details](docs/results/gemma4-e2b-s0.md) |

Each number is **development / test** at raw temperature 1. “New sources” are
`transfer-v4` questions from held-out datasets and policy structures; “trained
sources” are held-out `decision-v7` questions from the training families.
These are seed-0 measurements only; select future checkpoints using development
data. Temperature changes confidence scores, not the winning answer.

## How it works

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
uv sync --locked --extra dev --extra mlx  # omit --extra mlx off Apple Silicon
```

Use `uv run gev` from the repository root as shown, or activate `.venv` and
substitute `gev`.

## Predict

Ask the published model a question. The base model and checkpoint download from
Hugging Face on first use:

```bash
echo '{"state": "Order #1 arrived damaged.", "questions": {"route": {"type": "choice",
  "instructions": "Which team handles this?", "criteria": {"billing": "Payments", "support": "Product issues"}}}}' \
  | uv run gev predict onatm/gev-e2b
```

The response has an `answer` and one probability per option for each question.
`gev predict` also accepts a local run or checkpoint directory, and
`--backend mlx` serves Torch-trained weights on a Mac.

## Tests

```bash
uv run pytest -q
```

The tests are offline. They use tiny random Gemma 3/Gemma 4 decoders and a
word-level tokenizer to check augmentation and metric parity with pinned Kev
code, row isolation, training, stateless resume, checkpoint round trips, and
Torch↔MLX parity. MLX tests run only on Apple Silicon.

## Documentation

- [Training and evaluation](docs/training.md): fetch data, train a new seed,
  select on development, calibrate, and test.
- [Publishing](docs/publishing.md): model cards, pushing to the Hub, and
  recording a run in Git.
- [Architecture](docs/ARCHITECTURE.md): module layout and design choices.
- [Gemma 4 E2B results](docs/results/gemma4-e2b-s0.md): the measured seed-0 run.
- [Model cards](docs/models/cards): the cards published on Hugging Face.
