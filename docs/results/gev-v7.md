# Gev v7 early prototype: three-seed study and locked test

This is a historical result from the **pre-refactor implementation**, run on
2026-09-23 as an early prototype to verify the idea. It is not a measurement
produced by the current modular runtime. The completed Gev/Gemma 3 1B v7
experiment used seeds 0, 1, and 2. The study selected seed 0 on **transfer-v4
development clean accuracy**, with transfer Brier and decision-v7 development
task-average NLL as tie-breakers. Only that selected checkpoint was evaluated
on the locked test partitions. All **Gev** scores below are from the saved
reports at raw temperature **T=1**; the screening temperature fitted on
calibration data was not applied to the checkpoint or the locked test.

The checked-in, metric-only evidence is the
[study result](../../reports/gev-v7/study-result.json),
[decision-v7 locked-test report](../../reports/gev-v7/decision-v7-test-report.json),
and [transfer-v4 locked-test report](../../reports/gev-v7/transfer-v4-test-report.json).
These are copies of the completed local reports, not new evaluations. The
per-example rows, predictions, datasets, logs, and model weights are not part
of this results snapshot. The original run used the old checkpoint format;
the current manifest-based runtime cannot load that saved model directly.
These reports are not a reason to re-run the once-only locked test.

## Development: model selection

All three trials completed the two-epoch, 3,144-step recipe and were eligible
for selection. Accuracy is question-micro accuracy over clean questions; Brier
is mean-question squared probability error (lower is better).

| Seed | decision-v7 dev accuracy (1,264 clean) | transfer-v4 dev accuracy (656 clean) | transfer-v4 dev Brier |
| --- | ---: | ---: | ---: |
| **0 (selected)** | 0.7611 | **0.5762** | 0.5718 |
| 1 | **0.7722** | 0.5595 | 0.5800 |
| 2 | 0.7381 | 0.5488 | **0.5389** |
| Mean ± sample standard deviation | 0.7571 ± 0.0174 | 0.5615 ± 0.0138 | 0.5636 ± 0.0218 |

Seed 1 led on in-distribution development accuracy, and seed 2 had the lowest
transfer Brier. Seed 0 won because transfer accuracy is the first selection
criterion. All three trials evaluated every requested development question with
zero rejected or truncated records and passed their mechanism checks.

## Locked test: selected seed 0 only

| Suite | Test coverage | Clean questions | Accuracy | Brier ↓ | NLL ↓ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decision-v7 | 1,176/1,176 records; 1,440/1,440 questions | 1,200 | **0.8058** | 0.2722 | 0.4978 | 0.0619 |
| transfer-v4 | 764/764 records; 764/764 questions | 656 | **0.5716** | 0.5592 | 0.9847 | 0.1311 |

There were no rejected or truncated test records. The locked ledger recorded
both suites as complete for the same seed-0 model fingerprint. Transfer
performance is the main limitation: on the 64 test contrastive pairs whose
correct answer changes, the prediction flipped on 2/64 and both answers were
correct on **0/64**. On transfer clean questions answered with at least 0.9
confidence, 36/149 were wrong (24.2% error). These diagnostics describe this
selected checkpoint, not an additional selection on test data.

## Published Kev reference

The pinned [Kev-0.6B model card](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/docs/model-cards/kev-0.6b-qwen3.md)
and [locked-test summary](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/runs/locked/kev-06b-v7-ungated/summary.json)
report the following results on the same frozen test partitions:

| Clean locked-test metric | Gev/Gemma 3 1B (seed 0) | Kev/Qwen3 0.6B | Gev − Kev |
| --- | ---: | ---: | ---: |
| decision-v7 accuracy | 0.8058 | 0.808 | −0.2 percentage points |
| transfer-v4 accuracy | 0.5716 | 0.642 | −7.0 percentage points |
| transfer-v4 Brier ↓ | 0.5592 | 0.483 | +0.0762 (worse) |

Kev's numbers are published reference results, not Gev measurements. The
backbones, tokenizers, and execution details differ; the point estimates alone
are not a paired significance test. This comparison is limited to the early
prototype, not a current-runtime benchmark.

## Provenance and scope

- Backbone: `google/gemma-3-1b-pt` at revision
  `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; training was fp32 on MPS,
  with a LoRA adapter and a separately trained pointer head.
- Training: decision-v7 train SHA-256
  `7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`;
  12,576 source records, 25,152 processed exposures, and 3,144 logical steps
  per seed. The [study result](../../reports/gev-v7/study-result.json) records
  the original recipe hash, split hashes, and each trial's coverage and
  mechanism checks. The current [v7 config](../../configs/gemma3-1b-v7.toml)
  includes post-refactor runtime metadata and is not the saved run config.
- Selected weights: seed 0, stable model fingerprint
  `aaba70be38e70e0af2826d6a0cd7b44e408c867d81eaed576e3e72bb12e7acce`.
  The selection registered `temperature: 1.0` and verified full-v7 lineage.
- Locked protocol: one evaluation of seed 0 against decision-v7 and transfer-v4
  test, with complete ledger entries for both suites. The checked-in test
  reports include the suite manifest, test-source, model, and result-row hashes.
  Their `raw_clean` and `calibrated_clean` values coincide because T=1.

The stored reports document the completed study and locked evaluation. Test
results were reported once for the selected checkpoint; seeds 1 and 2 have
development results only. For the current code's boundaries and held-out
protocol, see the [architecture overview](../ARCHITECTURE.md).
