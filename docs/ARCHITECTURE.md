# Architecture

Gev is a small library and CLI. Every module is a flat file under `src/gev/`.

```text
data.py ──► records.py ──► encoding.py ──► {torch,mlx}_backend.py ──► checkpoint.py
 fetch/verify   render,        markers,         Runner: forward rows,       PEFT adapter +
 augmentation   labels         row encoding     train_step, save/load       pointer + gev.json
      │                                               ▲
      └────────────► train.py (loop, schedule, resume)┘
                     evaluate.py (Model, evaluate, calibrate, compare) ──► metrics.py
cli.py: data | train | evaluate | calibrate | compare | predict | card | push
```

| Module | Responsibility |
| --- | --- |
| `config.py` | One frozen `Config` (`[model]` and `[training]` tables plus runtime keys) loaded from TOML; unknown keys are rejected. |
| `data.py` | Pinned Kev suites: fetch from a fixed HF dataset revision, verify SHA-256 and counts on every load, `sample` smoke subsets, Kev augmentation, and seed-derived epoch order. |
| `records.py` | Kev request representation: rendering, option keys, label/target validation, `materialize`, and `public_request` (labels removed). |
| `encoding.py` | Resolves the five marker tokens against the tokenizer; escapes control and marker text in user input; encodes a record as a shared state plus one branch per question. |
| `torch_backend.py` | `TorchRunner`: loads the Gemma 3 decoder via Transformers or streams only the Gemma 4 text decoder out of the multimodal checkpoint; PEFT LoRA; FP32 pointer head; AdamW training step; PEFT-format save and load. |
| `mlx_backend.py` | `MlxRunner`: the same contract on mlx-lm's Gemma 4 decoder. LoRA tensors are transposed to and from PEFT layout, so checkpoints are shared with Torch. |
| `train.py` | The backend-neutral loop: logical batches, one-cycle cosine schedule, a JSONL log, `state/` snapshots, and resume. |
| `evaluate.py` | `Model` (a checkpoint, its runner, tokenizer, and temperature), suite evaluation, temperature calibration, and paired comparison. |
| `metrics.py` | Kev metrics: accuracy, NLL, Brier, ECE, selective coverage, AURC, contrastive flips, temperature fit, and a paired cluster bootstrap. |
| `checkpoint.py` | The checkpoint layout, `gev.json` metadata, refreshing the generated regions of hand-written model cards, and Hub `push`/`download`. |

## Design choices

- **Rows, not packing.** Each question is decoded as `state + question` in its
  own row. Training right-pads and batches rows, which is exact under causal
  attention. Inference runs each row alone, because BF16 kernels give slightly
  different results for different batch shapes. This works for every decoder, including KV-shared Gemma 4 and hybrid
  recurrent models such as Qwen3.5. A prefix cache or packed masks would only help
  some architectures.
- **Precision.** The frozen base runs in the config's `dtype` (BF16 by default;
  FP32 for the Gemma 3 v7 recipe). LoRA weights, the pointer head, the loss, and
  optimizer state are FP32. MLX AdamW uses bias correction so both backends
  optimize identically; the parity test checks this.
- **Reproducibility.** The model revision, dataset revision, and suite manifests
  are pinned. The epoch order and augmentation are pure functions of
  `(seed, epoch, record id)`, so resume needs only a step counter. The training
  data hash is recorded in the run and the checkpoint.
- **Held-out data.** Test splits load like any other split. Select models on
  development data and evaluate the chosen model on test once; the process is
  the protection, not the code.
- **Publishing.** A checkpoint is a standard PEFT adapter plus
  `pointer.safetensors` and `gev.json`. `gev predict <hub repo>` loads it directly.
