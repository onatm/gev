# Gemma 4 E2B: seed-0 training and evaluation

This is one completed Gev run on the pinned `google/gemma-4-E2B` text decoder,
trained with MLX in BF16 on seed 0. It is a **single-seed result**, not a
three-seed study. The model completed 3,144/3,144 training steps on 12,576
decision-v7 source records (two epochs). Its saved [effective config](../../runs/g4-s0/config.json)
and [checkpoint metadata](../../runs/g4-s0/checkpoint/gev.json) identify the
training backend, base revision, recipe, and fitted serving temperature.

The selected Git evidence consists of the [training log](../../runs/g4-s0/log.jsonl)
and metric-only reports for
[decision development](../../runs/g4-s0/eval-dev/report.json),
[decision calibration](../../runs/g4-s0/eval-cal/report.json),
[transfer development](../../runs/g4-s0/eval-transfer-dev/report.json),
[decision test](../../runs/g4-s0/eval-test/report.json), and
[transfer test](../../runs/g4-s0/eval-transfer-test/report.json).
The [calibration fit](../../runs/g4-s0/eval-cal/calibration.json) and
[derived decision-test calibrated metrics](../../runs/g4-s0/eval-test/calibration.json)
are also checked in.
These are the run's actual report files at their original paths. Per-example
scoring rows, source data, optimizer state, and weights remain ignored by Git.

## Clean-question results

These are the `clean` metrics at **raw T=1**, including in reports generated
after the serving temperature was fitted. Brier, NLL, and ECE are lower-is-better.
The full record/question populations include non-clean variants; the headline
metrics below use only the indicated clean questions.

| Suite / split | Evaluated records / questions | Clean questions | Accuracy | Brier ↓ | NLL ↓ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decision-v7 development | 1,204 / 1,468 | 1,264 | 0.7975 | 0.2847 | 0.5446 | 0.0628 |
| transfer-v4 development | 764 / 764 | 656 | 0.6113 | 0.4990 | 0.8726 | 0.1163 |
| decision-v7 test | 1,176 / 1,440 | 1,200 | **0.8292** | 0.2457 | 0.4709 | 0.0661 |
| transfer-v4 test | 764 / 764 | 656 | **0.6250** | 0.4680 | 0.8340 | 0.1136 |

On the transfer test, the model predicted both answers correctly for 12/64
contrastive pairs whose correct answer changed (`both_correct_rate=0.1875`).
The `none_present` variant scored 14/36 (`acc=0.3889`). These are useful
limitations alongside the overall clean transfer accuracy.

### Temperature and report chronology

The calibration split scored 0.7909 clean accuracy (1,148 questions) at raw
T=1. A temperature of **1.6245047927124707** was fitted on that split and
written to `checkpoint/gev.json` after the decision-test report had been
generated, but before the transfer-development and transfer-test reports. The
decision-test report therefore records only raw metrics. Its separate
`calibration.json` applies the **already fitted** temperature to its saved raw
logits, without refitting on test data or rerunning inference. It records the
source report and row hashes so the derived values can be traced back to the
original evaluation. The transfer reports also include `clean_calibrated`:

| Clean test metric | decision-v7 raw / calibrated | transfer-v4 raw / calibrated |
| --- | ---: | ---: |
| Accuracy | 0.8292 / 0.8292 | 0.6250 / 0.6250 |
| Brier ↓ | 0.2457 / 0.2336 | 0.4680 / 0.4491 |
| NLL ↓ | 0.4709 / 0.4198 | 0.8340 / 0.7515 |
| ECE ↓ | 0.0661 / 0.0170 | 0.1136 / 0.0571 |

All four rows of the headline table above compare the saved *raw* scores; the
decision test was not rerun.

## Published Kev reference

Kev's pinned [0.6B model card](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/docs/model-cards/kev-0.6b-qwen3.md)
and [test summary](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/runs/locked/kev-06b-v7-ungated/summary.json)
give reference results for the same frozen test suites:

| Clean test metric | Gev / Gemma 4 E2B | Kev / Qwen3 0.6B | Gev − Kev |
| --- | ---: | ---: | ---: |
| decision-v7 accuracy | 0.8292 | 0.808 | +2.1 percentage points |
| transfer-v4 accuracy | 0.6250 | 0.642 | −1.7 percentage points |
| transfer-v4 Brier ↓ | 0.4680 | 0.483 | −0.0150 (better) |

These are published Kev reference scores, not another Gev run. The backbones,
tokenizers, and execution details differ, and these point estimates are not a
paired significance test. This Gemma 4 result covers only seed 0; neither
multiple-seed uncertainty nor a matched-backbone ablation is claimed.

## Reproducibility and weights

- Base model: `google/gemma-4-E2B` revision
  `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`; LoRA rank 16, alpha 32,
  dropout 0.05, with a separate 256-wide pointer head. The exact effective
  settings are in `runs/g4-s0/config.json`, rather than inferred from the
  example PyTorch config in the README.
- Train data SHA-256:
  `7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`.
  Each evaluation report records its own split's SHA-256 and question count.
- Training throughput: the completed MLX/BF16 run on an Apple M4 Max (64 GB),
  with `microbatch = 1`, recorded 6,800,870 tokens across 3,144 steps in
  13,010 seconds (3 h 37 min), or about 523 tokens/s over the training loop.
  Token count comes from `runs/g4-s0/log.jsonl`; elapsed time is recorded in
  `checkpoint/gev.json`.
- Checkpoint weight SHA-256 (weights kept out of Git): LoRA adapter
  `a7cab6151e48b5189c47d742f7f30cba64286ddd47a047dfcfa5435d59cb16fc`;
  pointer head
  `442be99c882e4cbf7aa21c3d55e00ce5418b19c8f3be78d8323b7599a5b8b4b9`.
  The [adapter configuration](../../runs/g4-s0/checkpoint/adapter_config.json)
  and checkpoint metadata are retained with the reports.
- The checkpoint and resumable optimizer state remain in the ignored local run.
  To make the model available for inference elsewhere, publish the weights via
  `gev push` as described in the [README](../../README.md#publish).
