---
language: en
base_model: google/gemma-4-E2B
base_model_relation: adapter
library_name: peft
license: apache-2.0
tags: [gev, decision-model, lora, pointer-head, multiple-choice, calibration]
datasets: [jaredpalmer/kev-suites]
metrics: [accuracy, brier_score, expected_calibration_error]
model-index:
  - name: gev-e2b
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: jaredpalmer/kev-suites, name: "decision-v7/test (clean questions)" }
        metrics:
          - { type: accuracy, value: 0.8292 }
          - { type: brier_score, value: 0.2336, name: "Brier (as served)" }
          - { type: expected_calibration_error, value: 0.0170, name: "ECE (as served)" }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: jaredpalmer/kev-suites, name: "transfer-v4/test (clean questions)" }
        metrics:
          - { type: accuracy, value: 0.6250 }
          - { type: brier_score, value: 0.4491, name: "Brier (as served)" }
          - { type: expected_calibration_error, value: 0.0571, name: "ECE (as served)" }
---

# gev-e2b — Gemma 4 E2B decision model

**State and typed questions in; one answer and a probability distribution per question out.** Gev scores the supplied options rather than generating text. It combines a LoRA adapter on [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) with a separately trained pointer head.

The pointer head (`pointer.safetensors`) is required: loading the PEFT adapter alone does not produce Gev decisions. The Gev package loads the base model, adapter, pointer head, tokenizer markers, and serving temperature. Its [source repository](https://github.com/onatm/gev) is currently private; access to it is needed to run inference.

- **Trained-source test:** 82.9% accuracy on 1,200 clean questions.
- **New-source test:** 62.5% accuracy on 656 clean questions.

## Use

With access to the Gev source, run `uv sync --locked` in its checkout (add `--extra mlx` on Apple Silicon). Save this as `request.json`:

```json
{"state":"Order #1 arrived damaged.","questions":{"route":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":"Payments","support":"Product issues"}}}}
```

```bash
uv run gev predict onatm/gev-e2b --input request.json
```

The response includes `questions.route.answer`, `questions.route.probabilities` (one per option), and the serving temperature. Choice, yes/no (`noul`), and ordinal (`score`) questions are supported. A Hugging Face text-generation or PEFT-only pipeline cannot serve this model.

## Evaluation

Clean-question scores from the saved reports. Accuracy is at raw T=1 (temperature scaling does not change the winning answer); Brier and ECE are lower-is-better. The served columns use the saved temperature **T=1.6245**. A dash means calibrated metrics were not recorded for that split.

| Suite / split | Clean n | Accuracy | Brier raw | Brier served | ECE raw | ECE served |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decision-v7/development | 1,264 | 0.7975 | 0.2847 | - | 0.0628 | - |
| transfer-v4/development | 656 | 0.6113 | 0.4990 | 0.4735 | 0.1163 | 0.0522 |
| decision-v7/test | 1,200 | 0.8292 | 0.2457 | 0.2336 | 0.0661 | 0.0170 |
| transfer-v4/test | 656 | 0.6250 | 0.4680 | 0.4491 | 0.1136 | 0.0571 |

`decision-v7` contains held-out questions from the training source families; `transfer-v4` contains new sources and held-out policy structures. Development was for model selection; these reports describe seed 0. Temperature was fitted on the separate `decision-v7/calibration` split, not on either test split.

The decision-v7 test report predates the temperature fit. Its served metrics were computed afterward from saved raw logits, without rerunning inference or fitting on test.

## Known limits

This is one seed, not a multi-seed study. The new-source test is substantially harder than the trained-source test; evaluate on your own decisions before use.

- For changed-answer contrastive pairs, both answers were correct in 12/64 pairs.
- With a none-of-the-above option present, 14/36 questions were correct.

## Model and provenance

- Base: `google/gemma-4-E2B` at revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`; frozen base weights are not in this repo.
- Architecture: LoRA rank 16, alpha 32 on the text decoder and a 256-wide pointer head. Each question is scored independently.
- Recipe: seed 0, 2 epochs, 12,576 training records, 3,144 steps, MLX/BF16. The checkpoint stores the fitted temperature in `gev.json`.
- Training data SHA-256: `7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`; each saved evaluation report also records its split's SHA-256.
- Training suite: [jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) (`decision-v7`, pinned and verified by hash).

Training uses option permutation, none-of-the-above and distractor augmentation, and contrastive pairs. The architecture follows [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked); the data and evaluation protocol are adapted from [Kev](https://github.com/jaredpalmer/kev). The detailed run reports and code are in the private Gev repository.

## License

The adapter and pointer head are apache-2.0; check the separately loaded base model and dataset licenses as well.
