# Resumable training

`gev.training.loop.train` resumes only from `last_good.resume.pt`, which is
written atomically after a completed logical optimizer step. A process stopped
inside a logical step therefore repeats that step rather than skipping it.
Snapshots contain LoRA and pointer-head tensors only; the pinned frozen base is
loaded again. AdamW state is serialized as CPU fp32 state and moved to the
selected device after `load_state_dict`. CPU, MPS, Python, and NumPy RNG state,
shuffle state/order, cursor, OneCycle scheduler state, metrics, and the chained
augmentation digest are restored.

The resume recipe fingerprint includes the complete dataclass configuration
(including runtime fields), source/manifest hashes, model and tokenizer
revisions, and the marker IDs/strings/BOS contract. `max_steps` and `save_every`
are operational controls and may change for a continuation; all recipe fields
must remain identical.

The direct API is:

```python
train(model, rows, tokenizer, markers, config, output,
      source_hash=source_hash, manifest=manifest,
      resume=output / "last_good.resume.pt")
```

The CLI `train --resume` requires a fresh `--out` directory. It records the
absolute snapshot path and SHA-256 in both `resume_input.json` and final
checkpoint metadata, providing an auditable input chain. It rejects an output
containing `checkpoint` before loading data or model weights; never resume into
a completed run or overwrite an older partial snapshot.

Typical continuation uses separate directories:

```text
gev train ... --max-steps 8 --out runs/resume-start
gev train ... --resume runs/resume-start/last_good.resume.pt \
  --out runs/resume-finished
```

This is continuation, not warm-start training: optimizer, scheduler, cursor,
RNGs, and the augmentation chain are restored. A fresh run omits `--resume`.
