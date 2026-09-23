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
defaults. `gev train CONFIG` and `gev study plan|run CONFIG` take an explicit
config path; `gev evaluate RUN` can use the resolved config recorded with that
checkpoint. `gev diagnose config CONFIG` validates a config and its registered
family/backend without downloading a model. Diagnostic commands accept a
config when they need one. `gev --help`, `gev diagnose environment`, and
manifest loading do not require a checkout or model download. See the
[architecture index](ARCHITECTURE.md) for current runtime support and the
future qualification roadmap; planned backends are not currently available.
