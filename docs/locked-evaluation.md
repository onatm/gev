# Locked evaluation

`gev register-candidate` consumes the study result interface: `promotion`
selects a seed from the matching `trials[]` entry, whose `path`, eligibility,
completion, and calibration/development/transfer reports are verified. The
`--run` checkpoint must be that selected trial, not an unrelated run. The
checkpoint must prove either the complete v7 source (12,576 source records,
25,152 processed records, 3,144 logical steps) or the complete 1,425-record
Night 2 plus 2,000-record replay continuation (3,425 records, 429 steps) from
a verified full-v7 ancestor; `complete` alone and diagnostic smoke
initializers are insufficient. The record stores study and checkpoint
provenance, stable weights identity, any development temperature fit, and
pinned suite/test-file identities; it does not load test examples.

`gev eval-locked` validates that record against the actual checkpoint, checks
all requested suites and the output path, then reserves one ledger key per
`(stable weights fingerprint, suite manifest)` while holding one ledger lock.
Every suite is reserved before any test loader is called.  A failed attempt is
permanently recorded and cannot be retried under the same key.

Locked output is kept separate by suite:

```
out/decision-v7/{rows.json,predictions.jsonl,report.json}
out/transfer-v4/{rows.json,predictions.jsonl,report.json}
```

The predictor is configured and checked at raw temperature one and runs once
per record. Reports derive raw and selected-temperature metrics from those
saved rows; no second test inference or additional calibration inference is
performed. Each suite writes predictions, rows, and its report atomically,
with source, suite, model, and row fingerprints plus complete question
coverage. Test data is inaccessible through the normal evaluation split loader
and may only be fetched by the reserved locked path.
