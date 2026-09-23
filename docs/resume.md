# Resumable training

The training stage resumes only from `last_good.resume.pt`, which is written
atomically after a completed logical optimizer step. A process stopped
inside a logical step therefore repeats that step rather than skipping it.
Snapshots use the independently versioned `gev.logical-resume` v1 contract,
separate from the independently versioned `gev.inference-checkpoint` manifest
v1 contract. The resume format has distinct identity, progress, and Torch state
sections. It contains LoRA and pointer-head tensors only; the pinned frozen
base is loaded again. AdamW state is serialized as CPU fp32 state and moved to
the selected device after `load_state_dict`. CPU, MPS, Python, and NumPy RNG
state, shuffle state/order, cursor, OneCycle scheduler state, metrics, and the
chained augmentation digest are restored.

The identity contract groups the scientific recipe hash, family/backend, pinned
base and tokenizer, protocol, marker IDs/strings/BOS, source/manifest lineage,
replicate seed, and runtime identity. `max_steps` and `save_every` are excluded
operational controls and may change for a continuation; all identity fields
must remain identical. Older resume snapshots are intentionally rejected.

The application training stage builds a deterministic `TrainingSchedule` and
passes it to the selected backend's trainer. The CLI below exposes exact
continuation without relying on backend-specific Python calls.

The CLI `train --resume` requires a fresh `--out` directory. It records the
absolute snapshot path and SHA-256 in both `resume_input.json` and final
checkpoint manifest lineage, providing an auditable input chain. It rejects an
output containing `checkpoint` before loading data or model weights; never
resume into a completed run or overwrite an older partial snapshot.

Typical continuation uses separate directories:

```text
gev train configs/gemma3-1b-v7.toml --max-steps 8 --out runs/resume-start
gev train configs/gemma3-1b-v7.toml --resume runs/resume-start/last_good.resume.pt \
  --out runs/resume-finished
```

This is exact continuation, not warm-start training: optimizer, scheduler,
cursor, RNGs, and the augmentation chain are restored. A fresh run omits
`--resume`. By contrast, `gev train CONFIG --init-from RUN` loads model weights
into a new training run with a fresh optimizer and schedule; see
[continuation](continuation.md).
