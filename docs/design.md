# Gev technical design and research record

This document records the researched design and separates published reference
results from observations made in Gev smoke/diagnostic runs. Gev has no
comparable full-study result; the bounded local measurements are not full
three-seed v7 or full Night 2 results.

## Source pins and evidence boundary

The primary Kev source is pinned at
[`08ab0b87`](https://github.com/jaredpalmer/kev/tree/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b).
Its `api.py` preserves structured-field insertion order and labels; options may
be `name` or `name: description`, binary choices render `no,yes` and
`label0,label1`, and qids/labels/metadata are not model text.

The backbone is [`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt)
at revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; the official API reports
999,885,952 parameters and `gemma3_text`. The pinned Google config source is
[`014acb7`](https://huggingface.co/google/gemma-3-1b-pt/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/config.json).
Transformers research uses the [v5.17.0 source tree](https://github.com/huggingface/transformers/tree/v5.17.0/src/transformers),
especially `modeling_gemma3.py`, `configuration_gemma3.py`, `masking_utils.py`,
`cache_utils.py`, `integrations/sdpa_attention.py`, and
`models/gemma/tokenization_gemma.py`. The resolved dependency set is the uv
lock, not an inference from source links.

The evidence boundary is strict: published Kev values below are research
baseline; Gev smoke values are observed local diagnostics. No full Gev test
result is claimed.

## Configuration, rendering, and markers

TOML has separate `[model]`, `[training]`, and `[runtime]` tables plus an
experiment ID. Unknown/missing keys, invalid types/ranges/modes, and unpinned
model revisions fail before output creation. Artifacts record model/tokenizer
revisions, delimiter/BOS contract, context/execution/dtype, LoRA targets,
suite/augmentation hashes, and temperature policy. Credentials use the
Hugging Face chain and are never copied to config or output.

Kev rendering is `[state]`, followed by each question `[q] instruction [opt]
option [/opt] ... [decide]`. Gev uses five existing Gemma rows, not Qwen FIM
strings. The verified current mapping is `6 state <unused0>`, `7 question
<unused1>`, `8 option_start <unused2>`, `9 option_end <unused3>`, and `10 decide
<unused4>`. There is no vocabulary resize or automatic BOS insertion.
Control-token-like user content is escaped before tokenization. Marker rows must
be distinct, in range, present in the vocabulary, and exactly one non-empty ID
with `add_special_tokens=False`; the artifact records `BOS=false`.

Kev's corresponding Qwen markers are `<|fim_prefix|>`, `<|fim_middle|>`,
`<|box_start|>`, `<|box_end|>`, and `<|fim_suffix|>`. Its encoder escapes
`<|name|>` strings to `<¦name¦>`; Gev also escapes the selected Gemma markers
and Gemma control tokens. Structural IDs are inserted by the encoder rather
than obtained from user text. See the pinned [Kev encoder](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/model.py).

The question representation follows [Kev's API conversion](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/kev/api.py):

| Type | Model options | Label and interpretation |
| --- | --- | --- |
| `noul` | `no`, `yes`, optionally followed by descriptions | Boolean label becomes index 0/1; answer is `p(yes)` |
| `choice` | Option key or `key: description`, in supplied order | Correct key becomes an index; answer is argmax plus probabilities |
| `score` | Ordered level descriptions | Integer level label; returned score is the expected level index |

Structured state/instructions/descriptions are flattened into labeled text
with insertion order retained. Question IDs, labels, targets, and provenance
are not model input. All three types use a categorical pointer distribution;
the reference recipe has no additional ordinal loss. Options within one
question remain causal and order-sensitive. Isolation applies between
questions, not between options.

## Model-independent versus Qwen-specific decisions

| Concern | Model-independent contract | Qwen-specific research baseline | Gev/Gemma decision |
| --- | --- | --- | --- |
| State/question layout | State is shared; each question is isolated and begins at the same logical position | Qwen DeltaNet recurrent state cannot accept arbitrary row masks | Row-primary Gemma: every branch starts at position `S`; never `S+iB` |
| Packed attention | Full mask is causal within state and same-question blocks; local mask adds a logical distance bound | Native DeltaNet recurrence cannot be masked into arbitrary packed rows | Separate `full_attention` and `sliding_attention` dictionaries; full `j<=i` plus state/same-question, local `0 <= logical_i-logical_j < 512` |
| Cache | The implemented prefix cache is inference-only | Qwen3.5 recurrent state cannot be cropped back like a full-attention KV cache | Gemma's rolling local cache also loses history; prefill state separately and copy it per question |
| Backbone identity | Tokenizer IDs and embedding parameters are part of the controlled experiment | A Qwen tokenizer/backbone substitution cannot support a pure matched-pretraining claim | Gemma-native tokenizer/embeddings and native architecture are the controlled substitution |
| Training stochasticity | Logical records are assembled before microchunking; variant weighting is explicit | — | Row-primary default; stochastic LoRA dropout is shared-state packing-difference training, not a parity guarantee |

The model-independent contract is the experiment invariant. Qwen3.5 Kev
already uses independent state-plus-question rows because its recurrent
layers do not honor arbitrary attention masks; Gemma does not have those
recurrent layers.

## Gemma architecture and masks

The pinned Gemma text model has 26 layers, hidden size 1,152, four query heads,
one KV head, head dimension 256, q width 1,024, MLP width 6,912, vocabulary
262,144, context 32,768, and native local window 512. Layers 5, 11, 17, and 23
are global: 22 local and four global. Preserve native embedding scaling
`sqrt(hidden)`, q/k RMSNorm before RoPE, residual ordering, four norms, gated
GELU, and weight normalization. Native RoPE uses base 10,000 locally and 1M
globally. These details come from the pinned [Google Gemma documentation](https://ai.google.dev/gemma/docs)
and the [HF Gemma 3 implementation](https://github.com/huggingface/transformers/tree/v5.17.0/src/transformers/models/gemma3).

Context is 384 including the state marker. State plus one question is capped at
2,048 and a whole packed example at 2,048. The observed maximum state-plus-branch
length over the pinned non-test splits and the exact three-seed, two-epoch
training variants (including none pairs) is 1,037.
The encoder pads to the actual batch maximum, not the cap. Every branch starts at
the same state position `S`, not `S+iB`. Gev's custom local mask compares logical
positions; HF's ordinary sliding-mask builder uses physical attention indices.
A single prepared 4D mask can bypass that builder for both layer types, so
separate prepared masks are required. A packed prefix must not be cropped
or rolled back: the dynamic sliding cache retains 511 prior tokens. The only
implemented cache shape is immutable state prefill copied per question.

## Pointer head, LoRA, and training

The decision head is not an LM head. Two biased linear projections map
`h_decide` and `h_end_option` to 256 dimensions, use a scaled dot product divided by
`sqrt(256)`, and applies a categorical softmax. Temperature is evaluation-only;
training uses T=1. LoRA is r=16, alpha=32, dropout .05 on q/k/v/o and
gate/up/down projections in every layer. Embeddings remain frozen. Trainable
counts are 13,045,760 LoRA plus 590,336 head = 13,636,096.

The recipe is two epochs, seed 0/1/2, learning rate 1e-4, AdamW weight decay
.01, OneCycle with `pct_start=.1`, cycling beta1, clip 1, and 3,144 updates
(1,572 per epoch). Each variant's loss is the mean of its question losses;
the logical-batch loss is the mean of these variant losses.
Build the logical eight-record batch before microchunking and divide by all
variants. Exact augmentation probabilities are `p_none=.10`,
`p_none_distract=.12`, `p_distract=.15`, `p_none_pair=.25`; `p_none_pair`
creates two variants. The reference seeds produced 30,428 / 30,530 / 30,370
variants for seeds 0/1/2 and 25,152 original exposures.

## Frozen v7 data and counts

Acquisition is `jaredpalmer/kev-suites@a88f56db5341397299137cb68775c2ea6e3f68cb`,
path `v7/decision-v7/train.jsonl`, SHA-256
`7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`, size
18,679,889 bytes. The authoritative v7 manifest SHA is
`a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2`.

| partition | records | questions |
| --- | ---: | ---: |
| decision train | 12,576 | 15,576 |
| decision calibration | 968 | 1,148 |
| decision development | 1,204 | 1,468 |
| decision test | 1,176 | 1,440 |
| clean scored decision development | — | 1,264 |
| clean scored decision test | — | 1,200 |
| transfer development | 764 | 764 (656 clean + 108 variants) |
| transfer test | 764 | 764 (656 clean + 108 variants) |

The decision clean scored counts differ from full manifest counts; they are not
interchangeable. Published metadata was not treated as test contents in this
session.

The public pool is v4, not v6: 1,000 samples per source, 3,180 train and 600
eval candidates, pool seed `20260919`, row-hash/dedup ordering, 64 headroom,
and strict cutoff under both Qwen tokenizers in the historical builder. The
eight legacy families provide 896 train/128 calibration; 60 compositional
structures provide 1,680 train/240 calibration; 60 public calibration samples
per source provide 600. Raw converters use reviewed 220-word caps. There are no
Jev training labels.

Historical pins are public pool commit
`6cfa03ab74728c18913c3c1008632124cd09ad78`, v7 generator
`79d95cf683258068684f8f7b3de4edb5da44615a`, and suite parent
`e07ef435a5db687e2078a94398c37e528b0ff6ea`; recorded hashes matched.
Rebuilding requires `python -m kev.study_v3` with the flags recorded in
`docs/data-provenance.md`; current Hub heads must inject those pinned revisions.
Source-regeneration is not implemented; primary-artifact fetching is.

## Published research baseline

Pinned Kev model cards/reports are the source for this table: the [Kev v7
seed-2 result](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/runs/q35-08b/02-trial-2/result.json),
[current 0.8B model card](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/docs/model-cards/kev-0.8b.md),
and [previous 0.6B model card](https://github.com/jaredpalmer/kev/blob/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b/docs/model-cards/kev-0.6b-qwen3.md).
The column labels are important: these are accuracy and Brier fields, not a
generic “Brier/Brier/NLL” shorthand.

| checkpoint | ID dev/test **accuracy** | OOD dev/test **accuracy** | OOD dev/test **Brier** |
| --- | --- | --- | --- |
| Kev 0.6 | .801/.808 | .620/.642 | .536/.483 |
| Kev 0.8 v7 | .829/.827 | .643/.668 | .513/.473 |
| current night2 | .825/.834 | .652/.684 | .499/.460 |

Kev references: base adapter `c917edefdfd72b3e9ba71455584700acc70595f6`
over base `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, current adapter
`54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`, old adapter
`dece6dba8d43f0f7ded45e9f5b9df12474d90843`, and v7 provenance
`5f78968927069eaacc3b2bdb688586989b3933ac`. These are not Gev scores.

## Evaluation and calibration protocol

Use raw fp32 logits at T=1, no date facts, no rotations above one, and no
quantization. Brier sums squared class errors then averages by question; it is
not mean-by-class. ECE uses 10 bins. Confidence is `max(p)`, not an API choice
confidence. Report question-micro accuracy, mean-question Brier, exact-logit
NLL, ECE, confident error, coverage at 5/1%, AURC, score MAE/RPS, per
type/task/variant, none/permutation flips, and paired-rule correctness.

Calibration screening uses clean ID calibration, micro NLL, and an 81-point
log grid .25–4. Release calibration uses clean ID development, 121 points,
five-fold grouped OOF, and 1,000 grouped bootstrap samples. Never fit transfer
or test and never double-apply T. Confidence intervals cluster by source/group
with separate micro and macro results. The ledger is per model-weight
fingerprint and per suite, preventing repeated sets or T changes. Kev's
`--allow-test` selects test; it has no `--split` flag. Gev's CLI has separate
eval-locked handling.

## Night 2 and continuation

Night 2 is 1,425 pinned records (900 dates + 525 unknowable/control)
plus 2,000 replay records sampled from verified train with
`random.Random('replay:1')`; replay IDs SHA-256 is
`16f794dfed43ab65b19eb62e006393e49abd43be5add87e1f0ba0c0065a8bbe9`. Assembly
is night2 first, then replay, before epoch shuffle: 3,425 source records and
429 source/batch steps. It is
one epoch, seed 1, lr 4e-5, and 1,425 pinned + 2,000 replay. `night2-20260920`
is the generator seed, not the replay seed. Warm-start loads Gev weights with a
fresh optimizer; it does not resume optimizer state. The current baseline is
evaluated first and no assertion is made for 800 samples.

## Runtime qualification and failure policy

FP32 eager is the default. bf16 MPS must pass full-model forward/backward and
the probability-delta gate; current measured delta 0.0505 fails the .02 gate.
CPU/MPS tiny precision checks are valid diagnostics, not throughput claims.
Native RoPE's CPU-autocast-off context is not evidence of CPU execution.
Avoid float64 MPS metric kernels, CUDA-only kernels, custom embeddings,
all-layer outputs, and `torch.compile` initially. Artifacts use atomic writes;
failed marker verification leaves no mapping. Checkpoints include all model,
data, RNG, optimizer, scheduler, cursor, and execution metadata needed for
exact resume.
