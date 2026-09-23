# Data and model provenance

This ledger describes the data workflow and immutable identities. The declared
non-test partitions have been fetched and verified; test contents were used
only for the selected checkpoint's [locked evaluation](results/gev-v7.md),
never for training or development evaluation.

## Model

Target: pretrained [`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt),
revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, 999,885,952 parameters,
`model_type=gemma3_text`. `models.lock.json` and the v7 config are the identity
contract. Gemma's license is separate from this repository's Apache-2.0 source
license. `inspect-model` fetches config/tokenizer through the normal HF auth
chain and verifies the actual five marker rows; it does not infer IDs from
documentation or resize embeddings.

## Kev and v7 acquisition

Kev is pinned in `kev.lock.json` to
[`08ab0b87`](https://github.com/jaredpalmer/kev/tree/08ab0b87d27cb5577a3b371ad7ed4e4686b0502b).
The authoritative acquisition is
`jaredpalmer/kev-suites@a88f56db5341397299137cb68775c2ea6e3f68cb`,
`v7/decision-v7/train.jsonl`, size 18,679,889 bytes, SHA-256
`7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`.
The authoritative v7 manifest SHA-256 is
`a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2`.

| partition | records | questions | status |
| --- | ---: | ---: | --- |
| decision train | 12,576 | 15,576 | fetched/verified workflow |
| decision calibration | 968 | 1,148 | fetched/verified workflow |
| decision development | 1,204 | 1,468 | fetched/verified workflow |
| decision test | 1,176 | 1,440 | fetched/verified; selected checkpoint evaluated once |
| transfer development | 764 | 764 (656 clean + 108 variants) | fetched/verified |
| transfer test | 764 | 764 (656 clean + 108 variants) | fetched/verified; selected checkpoint evaluated once |

Clean scored decision development/test contain 1,264/1,200 questions;
they are clean views, not full manifest populations. Do not describe v7 as
“decision and transfer each 764 records”:
the 764 number is for transfer dev/test only.

The builder’s 10 public source IDs and links are:

| source | dataset identity |
| --- | --- |
| banking77 | [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) |
| BoolQ | [`google/boolq`](https://huggingface.co/datasets/google/boolq) |
| AG News | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) |
| MultiNLI | [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| SST-5 | [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5) |
| Yelp | [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) |
| TREC | [`CogComp/trec`](https://huggingface.co/datasets/CogComp/trec) |
| DBPedia | [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) |
| Amazon Reviews Multi EN | [`SetFit/amazon_reviews_multi_en`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) |
| IMDB | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) |

The historical source-builder used 1,000 samples per source, public pool v4
(not v6), 3,180 train and 600 eval candidates, pool seed `20260919`, row-hash
sort/dedup, 64 headroom, and a strict cutoff under both Qwen tokenizers.
Public calibration contributes 60 samples/source (600). The eight legacy
families contribute 896 train/128 calibration, and 60 compositional structures
contribute 1,680 train/240 calibration. Raw converters use reviewed 220-word
caps. No Jev training labels are used. Each actual fetch must retain its
source revision and license metadata; source pages alone are not locks.

## Historical regeneration pins

Public-pool commit is `6cfa03ab74728c18913c3c1008632124cd09ad78`, v7 generator
is `79d95cf683258068684f8f7b3de4edb5da44615a`, and the suite parent is
`e07ef435a5db687e2078a94398c37e528b0ff6ea`. Source code involved is
`kev/data.py`, `kev/suite.py`, and `kev/study_v3.py`. The recorded hashes matched.

Historical rebuild command (spaces in the actual flags are intentional):

```text
python -m kev.study_v3 --out evals/v7 --public-train evals/public-pool-v4 \
  --inherit-eval evals/v4/decision-v4 --random-structures 60 \
  --groups-per-structure 8 --train-styles 0,1,3,4 --unmatched-arms \
  --legacy-families all
```

Output basename affects the seed. A byte-identical rebuild must inject pinned
Hub revisions; source regeneration is not implemented in Gev. Primary-artifact
fetching, byte/hash/count verification, atomic publication, and provenance
recording are implemented.

## Augmentation and continuation

Augmentation is deterministic: `p_none=.10`, `p_none_distract=.12`,
`p_distract=.15`, `p_none_pair=.25`. The per-record RNG is seeded with
`source_seed(run_seed, f"{epoch}:{record_id}")`, where `source_seed` takes
the first eight SHA-256 bytes of `f"{seed}:{source}"` as a big-endian integer.
Assemble all logical variants before chunking. Average questions within each
variant, then average variant losses across the logical batch.

Night 2 is pinned at 1,425 records (900 dates and 525 unknowable/control), plus
2,000 verified-train replay records sampled with `random.Random('replay:1')`.
Replay-ID SHA-256 is
`16f794dfed43ab65b19eb62e006393e49abd43be5add87e1f0ba0c0065a8bbe9`. The
Night 2 source artifact SHA-256 is
`afd8502d162163605ac446439e32c7b9083302dd78a6bfdc5e30075619e98437`. The
assembly is night2 first, replay second, then epoch shuffle. Generator seed
`night2-20260920` is not the replay seed. Warm-start loads Gev weights with a
fresh optimizer; it is not an optimizer resume.
