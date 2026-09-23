# Installing Gev

Build and install the wheel from a clean directory:

```bash
uv build
uv pip install dist/gev-*.whl
gev --help
gev data verify decision-v7 development /path/to/data/decision-v7/development.jsonl
```

The wheel contains the immutable suite manifests used for provenance checks;
they are loaded with package resources and do not depend on the source tree or
the current working directory.  The downloaded JSONL data is intentionally
not part of the wheel.  Pass its location to data and evaluation commands.

The repository's `configs/` files are experiment templates rather than package
defaults.  Installed commands that run or inspect a model require an explicit
`--config /path/to/config.toml`; `gev --help`, `gev doctor`, and manifest
loading do not require a checkout or a model download.
