# Publishing

How to release a selected run to the Hugging Face Hub and record it in Git.
Select and test the run first, following [Training and evaluation](training.md).

## Model cards

Model cards are written by hand, one per published model, under
[`models/cards/`](models/cards). Only two parts are generated from the run: the
`model-index` front matter (test-split scores) and the evaluation table between
`<!-- gev:eval -->` and `<!-- /gev:eval -->`. Everything else, including
headline numbers quoted in prose, is yours to keep accurate. For a new model,
copy an existing card and rewrite its prose before refreshing it.

## Push

If seed 1 was selected, refresh the card from its saved reports and publish it
to a Hub repository you control (replace `your-hf-user`):

```bash
uv run gev card runs/g4-s1 --card docs/models/cards/gev-e2b.md
git diff docs/models/cards/gev-e2b.md  # review, then fix any prose the new numbers contradict
uv run gev push runs/g4-s1 --repo your-hf-user/gev-e2b --card docs/models/cards/gev-e2b.md
uv run gev predict your-hf-user/gev-e2b --input request.json
```

`card` reads the run's `eval-*/report.json` files (calibration excluded; pass
`--report` to choose them) and rewrites only the generated parts; `--check`
fails instead of writing when the card is stale. `push` uploads the checkpoint
(PEFT `adapter_config.json` + `adapter_model.safetensors`, `pointer.safetensors`,
`gev.json`) and the card as `README.md` in one commit. For wording fixes, add
`--card-only` to upload just the card to an existing repository. Pushing needs
a Hugging Face write token.

Before replacing a published model's weights, tag the current revision on the
Hub (for example `uv run hf repos tag create your-hf-user/gev-e2b v1`) so it
stays loadable. Repositories are private unless you pass `--public`. Gemma 4
derivatives are Apache-2.0; Gemma 3 derivatives fall under the Gemma terms.

## Evidence in Git

`.gitignore` keeps generated data, weights, and resumable state local. For the
selected `g4-s0` run it allowlists `config.json`, `log.jsonl`,
`checkpoint/{gev,adapter_config}.json`, five `eval-*/report.json` files, and
the calibration fit and derived decision-test calibration JSON. They document
the MLX recipe, training trace, raw results, and fitted confidence. Per-question
`rows.jsonl` and all `.safetensors` stay ignored. If a later seed is selected,
review its results before adding an equally narrow allowlist for that run.
