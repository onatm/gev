"""The checkpoint format shared by both backends, and Hugging Face Hub publishing.

A checkpoint directory is self-describing and backend-neutral::

    adapter_config.json        PEFT LoraConfig
    adapter_model.safetensors  PEFT LoRA weights (base_model.model.layers.N.<module>.lora_{A,B}.weight)
    pointer.safetensors        pointer head (query/key weight and bias)
    gev.json                   base model + revision, markers, caps, temperature, config

Torch and MLX both read and write this layout, so a model trained on CUDA can be
served with MLX and vice versa. The frozen base weights are never stored. Model cards
are written by hand under ``docs/models/cards/``; ``gev card`` regenerates only their
``model-index`` metadata and the region between the ``EVAL_MARKERS``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import __version__
from . import hub
from .config import Config, from_dict
from .data import HF_DATASET, SPLITS

FORMAT = "gev-checkpoint"
FORMAT_VERSION = 1
ADAPTER_PREFIX = "base_model.model."
EVAL_MARKERS = ("<!-- gev:eval -->", "<!-- /gev:eval -->")


def resolve(path: str | Path) -> Path:
    """Accept a run directory (``<run>/checkpoint``) or a checkpoint directory."""
    path = Path(path)
    return path / "checkpoint" if (path / "checkpoint" / "gev.json").exists() else path


def lora_config(config: Config, *, with_base: bool = False):
    """PEFT LoraConfig; ``with_base`` records the base model and revision for adapter_config.json."""
    from peft import LoraConfig

    model = config.model
    base = {"base_model_name_or_path": model.name, "revision": model.revision} if with_base else {}
    return LoraConfig(r=model.lora_rank, lora_alpha=model.lora_alpha, lora_dropout=model.lora_dropout,
                      target_modules=list(model.lora_targets), bias="none", **base)


def write_metadata(directory: Path, config: Config, markers, *, temperature: float = 1.0,
                   training: dict | None = None) -> None:
    metadata = {"format": FORMAT, "format_version": FORMAT_VERSION, "gev_version": __version__,
                "base_model": config.model.name, "revision": config.model.revision,
                "family": config.model.family, "markers": markers.to_dict(),
                "state_cap": config.training.state_cap, "branch_cap": config.training.branch_cap,
                "temperature": temperature, "trained_with": {"backend": config.backend, "dtype": config.dtype},
                "config": config.to_dict(), "training": training or {}}
    (directory / "gev.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def read_metadata(directory: str | Path) -> dict:
    path = resolve(directory) / "gev.json"
    if not path.exists():
        raise FileNotFoundError(f"no gev checkpoint at {path.parent}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format in {path}")
    return metadata


def config_of(directory: str | Path) -> Config:
    return from_dict(read_metadata(directory)["config"])


def set_temperature(directory: str | Path, temperature: float) -> None:
    path = resolve(directory) / "gev.json"
    metadata = read_metadata(path.parent)
    metadata["temperature"] = temperature
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def check_markers(metadata: dict, markers) -> None:
    if metadata["markers"] != markers.to_dict():
        raise ValueError("tokenizer marker ids differ from the ones this checkpoint was trained with")


def _card_reports(paths: list[str | Path], temperature: float) -> dict[str, dict]:
    """Include a saved calibrated test read only when it matches the original report."""
    loaded = {}
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        derived_path = path.with_name("calibration.json")
        if "clean_calibrated" not in report and derived_path.exists():
            derived = json.loads(derived_path.read_text(encoding="utf-8"))
            if (derived.get("source_report_sha256") == hashlib.sha256(raw).hexdigest()
                    and Path(derived.get("source_report", "")).resolve() == path.resolve()
                    and derived.get("temperature") == temperature):
                report["clean_calibrated"] = derived["clean_calibrated"]
                report["_derived_calibration"] = True
        loaded[f"{report['suite']}/{report['split']}"] = report
    return loaded


def discover_reports(run: str | Path) -> list[Path]:
    """The run's saved ``eval-*/report.json`` files, development before test, calibration excluded."""
    found = []
    for path in resolve(run).parent.glob("eval-*/report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["split"] != "calibration":
            found.append((SPLITS.index(report["split"]), report["suite"], path))
    return [path for *_, path in sorted(found)]


def _model_index(name: str, reports: dict[str, dict]) -> list[str]:
    tests = {key: report for key, report in reports.items() if report["split"] == "test"}
    if not tests:
        return []
    lines = ["model-index:", f"  - name: {name}", "    results:"]
    for key, report in tests.items():
        scores = report.get("clean_calibrated", report["clean"])
        calibration = "as served" if "clean_calibrated" in report else "raw T=1"
        lines += ["      - task: { type: text-classification, name: typed decision (choice / noul / score) }",
                  f'        dataset: {{ type: {HF_DATASET}, name: "{key} (clean questions)" }}',
                  "        metrics:",
                  f"          - {{ type: accuracy, value: {report['clean']['acc']:.4f} }}",
                  f'          - {{ type: brier_score, value: {scores["brier"]:.4f}, name: "Brier ({calibration})" }}',
                  f'          - {{ type: expected_calibration_error, value: {scores["ece"]:.4f}, '
                  f'name: "ECE ({calibration})" }}']
    return lines


def _eval_table(reports: dict[str, dict], temperature: float) -> list[str]:
    lines = [f"The served columns use the saved temperature **T={temperature:.4f}**.", "",
             "| Suite / split | Clean n | Accuracy | Brier raw | Brier served | ECE raw | ECE served |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, report in reports.items():
        raw, served = report["clean"], report.get("clean_calibrated")
        served_brier = f"{served['brier']:.4f}" if served else "-"
        served_ece = f"{served['ece']:.4f}" if served else "-"
        lines.append(f"| {key} | {raw['n']:,} | {raw['acc']:.4f} | {raw['brier']:.4f} | "
                     f"{served_brier} | {raw['ece']:.4f} | {served_ece} |")
    return lines


def render_card(card: str, metadata: dict, reports: dict[str, dict]) -> str:
    """Regenerate a hand-written card's ``model-index`` and eval table; leave everything else as written."""
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        raise ValueError("model card has no YAML front matter")
    front, body = card[4:].split("\n---\n", 1)
    lines = front.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("model-index:")), len(lines))
    end = next((i for i in range(start + 1, len(lines)) if lines[i][:1] not in (" ", "-")), len(lines))
    lines[start:end] = _model_index(metadata["config"]["name"], reports)
    open_marker, close_marker = EVAL_MARKERS
    if body.count(open_marker) != 1 or body.count(close_marker) != 1:
        raise ValueError(f"model card needs exactly one {open_marker} ... {close_marker} region")
    before, rest = body.split(open_marker)
    after = rest.split(close_marker)[1]
    table = "\n".join(_eval_table(reports, metadata["temperature"]))
    return f"---\n{chr(10).join(lines)}\n---\n{before}{open_marker}\n\n{table}\n\n{close_marker}{after}"


def update_card(run: str | Path, card: str | Path, *, reports: list[str | Path] = (),
                check: bool = False) -> bool:
    """Refresh ``card`` from the run's metadata and reports; returns whether it changed.

    With ``check``, a stale card is an error and the file is left untouched.
    """
    metadata = read_metadata(run)
    card = Path(card)
    text = card.read_text(encoding="utf-8")
    loaded = _card_reports(list(reports) or discover_reports(run), metadata["temperature"])
    rendered = render_card(text, metadata, loaded)
    if rendered == text:
        return False
    if check:
        raise ValueError(f"{card} is out of date; run `gev card {run} --card {card}`")
    card.write_text(rendered, encoding="utf-8")
    return True


def _prepare_adapter_config(directory: Path) -> None:
    """Omit PEFT's optional null task type; Hub metadata expects a string if present."""
    path = directory / "adapter_config.json"
    if path.exists():
        adapter = json.loads(path.read_text(encoding="utf-8"))
        if "task_type" in adapter and adapter["task_type"] is None:
            del adapter["task_type"]
            path.write_text(json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def push(checkpoint: str | Path, repo_id: str, card: str | Path, *, private: bool = True,
         card_only: bool = False) -> str:
    """Upload a checkpoint and its model card to the Hub in one commit, or only the card."""
    from huggingface_hub import CommitOperationAdd, HfApi, ModelCard

    directory = resolve(checkpoint)
    read_metadata(directory)
    card = Path(card)
    ModelCard(card.read_text(encoding="utf-8"))  # rejects invalid metadata before anything is uploaded
    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card))]
    if not card_only:
        _prepare_adapter_config(directory)
        operations += [CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=str(path))
                       for path in sorted(directory.iterdir()) if path.suffix in (".json", ".safetensors")]
    hub.configure_hub()
    api = HfApi()
    if not card_only:
        api.create_repo(repo_id, private=private, exist_ok=True)
    message = "Update model card" if card_only else "Upload gev checkpoint"
    return api.create_commit(repo_id, operations=operations, commit_message=message).commit_url


def download(repo_id: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    hub.configure_hub()
    return Path(snapshot_download(repo_id, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "README.md"]))
