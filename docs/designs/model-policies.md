# Gemma model precision and checkpoint policy

**Status:** Implemented

**Location:** `docs/designs/model-policies.md`

---

## Overview

`models/policy.py` owns Gemma family/backend/device/compute support and required
qualification checks. The registry and resolved configuration reject unsupported
pairs before model loading. Application training creates one backend-neutral
Gemma 4 receipt after training and bounded development evaluation; each backend
binds that receipt to its saved adapter and pointer tensors.

## Supported Gemma 4 matrix

| Backend | Device | Compute | Source decoder | Trainable state |
| --- | --- | --- | --- | --- |
| Torch | CPU, MPS, CUDA | BF16 | pinned BF16, kept BF16 | FP32 LoRA, pointer, optimizer |
| Torch | CPU, MPS, CUDA | FP32 | pinned BF16, upcast to FP32 | FP32 LoRA, pointer, optimizer |
| MLX | GPU only | BF16 | pinned BF16, kept BF16 | FP32 LoRA, pointer, optimizer |
| MLX | GPU only | FP32 | pinned BF16, upcast to FP32 | FP32 LoRA, pointer, optimizer |

Gemma 4 uses independent `rows` execution. MLX requires microbatch one and does
not enable gradient checkpointing or CPU fallback; Torch may use its implemented
microbatch and gradient-checkpointing behavior. FP32 does not change the source
revision or source dtype. Gemma 3 remains governed by its existing Torch policy
and locked-test boundary.

The Torch source loader constructs only `Gemma4TextModel`, streams BF16 text
tensors from the pinned snapshot, and enforces the text inventory: 600 tensors,
4,647,449,891 serialized text parameters, and 4,628,569,344 effective text
parameters. When the safetensors index is present, it also validates the full
2,011-entry index and downloads only shards referenced by text-decoder keys.
Small local fixtures use a private snapshot seam; production loading always
enforces the pinned dimensions and inventory. MLX retains its strict source
inventory loader. Both backends fail closed on a source/compute dtype mismatch.

## Qualification checks

`ModelPolicy.required_qualification_checks(backend, dtype)` supplies the required
check names to the receipt builder and validator. Both backends check source
identity/inventory, selected decoder dtype, FP32 trainable state and optimizer
state, finite loss/gradients, nonzero updates, row isolation, complete
development coverage, and development mechanisms (`passed` with zero failures).
Torch additionally checks causal attention behavior. MLX additionally checks
boundary gathers, decoder masks, and shared-KV mapping. Development scores are
recorded descriptively; no Gemma 3 accuracy threshold is applied.

Gemma 4 training and study evaluation are limited to `decision-v7/development`;
calibration, transfer, and test partitions are not used. The once-only locked
evaluator accepts only the pinned Gemma 3 Torch candidate.

## Receipt and checkpoint contract

The receipt schema is `gev.trained-checkpoint-qualification` version 1. It has a
single `passed`/`failed` status, model/family/backend identity, pinned BF16 source
and selected compute dtype, policy and code hashes, required checks, training
completeness/step count, development manifest and selected-ID digest, coverage,
mechanism summary, report-file SHA-256, and trainable-content SHA-256. It does
not encode separate development/full-recipe/held-out eligibility flags.

Before save, the receipt is passed but not checkpoint-ready. The backend writes
`development_report.json` and `qualification.json` beside the adapter-only
checkpoint, hashes the adapter and pointer files, then seals a checkpoint-ready
receipt into the manifest and sidecar. The application copies the final receipt
back to the run directory and training metrics. Load validates the receipt hash,
policy/code identity, report digest, checkpoint tensor hashes, backend, precision,
and sidecar equality before loading the pinned base. A changed receipt, report,
adapter, pointer, backend, or precision is rejected before base loading.

Receipts bind the current implementation code hash. After code or policy changes,
old unpublished receipts and checkpoints may be replaced; cross-version receipt
compatibility is not promised.

## Optional cross-backend diagnostic

`gev diagnose model --probe` retains a separate bounded MLX/Torch-CPU FP32
implementation diagnostic. It is not a training prerequisite, training receipt,
or golden-model quality gate. MLX itself still requires Metal GPU execution, and
`MLX_ENABLE_TF32=0` must be set before MLX initialization. No full Gemma 4 model
download or training is part of the test suite. CUDA/MPS smoke tests run only on
hosts where those devices are available; the current H100 example is not a claim
of local H100 verification.
