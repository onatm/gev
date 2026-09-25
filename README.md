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
run. The model name links to its results and local checkpoint provenance; a
public weights URL is not listed here.

| Model | Base | Accuracy: New Sources | Accuracy: Trained Sources | Brier: New Sources ↓ | Backends | Results |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [gev-e2b](docs/results/gemma4-e2b-s0.md) | Gemma 4 E2B | 0.611 / 0.625 | 0.797 / 0.829 | 0.499 / 0.468 | MLX (Apple Silicon), PyTorch (CUDA/MPS/CPU) | [Details](docs/results/gemma4-e2b-s0.md) |

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

## 1. Install

```bash
mise install
uv sync --locked --extra dev --extra mlx  # omit --extra mlx off Apple Silicon
```

Gemma 4 is public; for gated Gemma 3, run `uv run hf auth login`.
The commands below run from the repository root. Use `uv run gev`
as shown, or activate `.venv` and substitute `gev`.

## 2. Prepare data

The suites are pinned by the Hugging Face dataset revision and verified against
the SHA-256 hashes in the packaged manifests every time they are read.

```bash
uv run gev data fetch decision-v7 train calibration development
uv run gev data fetch transfer-v4 development
```

Leave the test splits until after choosing a checkpoint on development data.
Data fetch verifies each split against its pinned SHA-256 and record counts.

## 3. Train the next seed

The reported run is `runs/g4-s0`. Train **seed 1** with the same MLX/BF16
recipe in a fresh directory to compare with it:

```bash
uv run gev train configs/gemma4-e2b-mlx-bf16.toml --seed 1 --out runs/g4-s1
```

`gev train` saves the LoRA adapter and pointer head in `runs/g4-s1/checkpoint/`;
the frozen Gemma base weights are loaded separately at inference. The run's
`config.json` records the *effective* config, including its seed and backend.
For later seeds, change **both** `--seed` and `--out` (for example, seed 2 goes
to `runs/g4-s2`).

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

The reported `runs/g4-s0` checkpoint completed two epochs and all 3,144 steps.
Use a new directory rather than overwriting that run.

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
curl https://mise.run | sh && mise install && uv sync --locked
uv run gev data fetch decision-v7 train development
uv run gev train configs/gemma4-e2b-torch-bf16.toml --out runs/profile-torch --max-steps 20
```

`microbatch` in the config is how many augmented records share one forward pass;
the loss and gradients are the same for any value.

## 4. Evaluate and select on development

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

## 5. Calibrate and test the selected model

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
test reports also contain `clean_calibrated`. At inference every question row
runs alone: in BF16, batching rows of different lengths shifts Gemma 4 logits
by up to about 0.5, so a score would otherwise depend on its batch neighbours.

The published `g4-s0` run fitted temperature *between* its two test reports.
Its [results page](docs/results/gemma4-e2b-s0.md) preserves the raw decision
test report and links calibrated decision-test metrics computed from its saved
raw logits. Following the order above gives a new selected run both raw and
calibrated scores directly in each test report.

## Predict

```bash
echo '{"state": "Order #1 arrived damaged.", "questions": {"route": {"type": "choice",
  "instructions": "Which team handles this?", "criteria": {"billing": "Payments", "support": "Product issues"}}}}' \
  | uv run gev predict runs/g4-s1
uv run gev predict runs/g4-torch-s1 --backend mlx --input request.json  # Torch-trained weights on MLX
```

## Publish

If seed 1 was selected, publish its checkpoint and its saved evaluation
reports to a Hub repository you control (replace `your-hf-user`):

```bash
uv run gev push runs/g4-s1 --repo your-hf-user/gev-e2b \
  --report runs/g4-s1/eval-dev/report.json \
  --report runs/g4-s1/eval-transfer-dev/report.json \
  --report runs/g4-s1/eval-test/report.json \
  --report runs/g4-s1/eval-transfer-test/report.json
uv run gev predict your-hf-user/gev-e2b --input request.json
```

`push` uploads the checkpoint (PEFT `adapter_config.json` +
`adapter_model.safetensors`, `pointer.safetensors`, `gev.json`) and generates a
model card with the base model, license, and the reported metrics. Repositories
are private unless you pass `--public`. Gemma 4 derivatives are Apache-2.0;
Gemma 3 derivatives fall under the Gemma terms.

### Evidence in Git

`.gitignore` keeps generated data, weights, and resumable state local. For the
selected `g4-s0` run it allowlists `config.json`, `log.jsonl`,
`checkpoint/{gev,adapter_config}.json`, five `eval-*/report.json` files, and
the calibration fit and derived decision-test calibration JSON. They document
the MLX recipe, training trace, raw results, and fitted confidence. Per-question
`rows.jsonl` and all `.safetensors` stay ignored. If a later seed is selected,
review its results before adding an equally narrow allowlist for that run.

## Tests

```bash
uv run pytest -q
```

The tests are offline. They use tiny random Gemma 3/Gemma 4 decoders and a
word-level tokenizer to check augmentation and metric parity with pinned Kev
code, row isolation, training, stateless resume, checkpoint round trips, and
Torch↔MLX parity. MLX tests run only on Apple Silicon.

See [Architecture](docs/ARCHITECTURE.md) for the module layout and
[Gemma 4 E2B results](docs/results/gemma4-e2b-s0.md) for the measured run.
