# Research baseline and decisions

This source register complements [design.md](design.md), keeping published
research facts distinct from Gev observations.

## Source register

* Kev source: [`08ab0b87d27cb5577a3b371ad7ed4e4686b0502b`](https://github.com/jaredpalmer/kev/tree/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b).
  The API/rendering and metric semantics are the compatibility baseline.
* Kev model cards/reports are the source for the historical reference table;
  the pinned adapter/base identities and source links are recorded in
  [design.md](design.md), together with the corrected SHA
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (40 hex characters).
* Gemma model: [`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt),
  revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; Google config revision
  `014acb7ac4563a5f77c76d7ff98f31b568c16508`.
* Gemma implementation: [Transformers v5.17 source](https://github.com/huggingface/transformers/tree/v5.17.0/src/transformers),
  including Gemma modeling/config/tokenization, masking, cache, and SDPA.
* Public inputs: the ten immutable dataset identities and historical builder
  commits are listed in [data-provenance.md](data-provenance.md).

## Decisions retained from research

1. Use Gemma’s actual reserved rows for five semantic markers; no vocabulary
   resize or automatic BOS insertion. Escape control-token-like user content
   before tokenization. The verified map is IDs 6–10.
2. Use 384 context including the state marker, 2,048 state+question, and 2,048
   whole-packed caps. The observed maximum is 1,037 for the exact three-seed
   training stream. Every branch starts at position `S`.
3. Keep full and sliding packed masks separate. Full attention permits causal
   state/same-question access; native local attention additionally checks the
   logical 512-token distance. Physical packed indices are not local positions.
4. Prefer row-primary stochastic LoRA training. Native recurrent Qwen
   DeltaNet rows cannot justify arbitrary masks; Gemma cache prefill is
   inference-only and copied per question.
5. Use the 1,152-to-256 scaled pointer head and LoRA r16/alpha32/dropout.05 on
   all seven dense target families. Freeze embeddings and the rest of the
   backbone. A tokenizer/embedding-ID substitution is a controlled-backbone
   substitution, not a pure matched-pretraining claim.
6. Assemble all logical variants before microchunking. Average question losses
   within each variant, then average those variant losses across the complete
   logical batch. Keep the exact augmentation probabilities and per-item RNG
   derivation, including pairs.
7. Calibrate only under the locked protocol: screening ID calibration 81-point
   grid; release ID development 121-point grouped OOF; 1,000 clustered
   bootstrap; raw fp32 T=1 inputs and no double temperature.
8. Treat the historical Kev table as baseline only. Gev smoke numbers and
   packed checks are observed diagnostics, never full-study results. The full
   v7 study and locked test are reported in [Gev v7 results](results/gev-v7.md).

For the reproduction workflow and CLI entry points, see
[reproduction.md](reproduction.md). Acquisition, hashes, counts, and
continuation identities are in [data-provenance.md](data-provenance.md).
