"""The checkpoint format shared by both backends, and Hugging Face Hub publishing.

A checkpoint directory is self-describing and backend-neutral::

    adapter_config.json        PEFT LoraConfig
    adapter_model.safetensors  PEFT LoRA weights (base_model.model.layers.N.<module>.lora_{A,B}.weight)
    pointer.safetensors        pointer head (query/key weight and bias)
    gev.json                   base model + revision, markers, caps, temperature, config
    README.md                  model card (written by ``gev push`` when absent)

Torch and MLX both read and write this layout, so a model trained on CUDA can be
served with MLX and vice versa. The frozen base weights are never stored.
"""

from __future__ import annotations

import hashlib
import json
import ssl
from pathlib import Path

import httpx
import truststore

from . import __version__
from .config import Config, from_dict
from .data import HF_DATASET

FORMAT = "gev-checkpoint"
FORMAT_VERSION = 1
ADAPTER_PREFIX = "base_model.model."
LICENSES = {"gemma3": "gemma", "gemma4": "apache-2.0"}


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


def model_card(metadata: dict, reports: dict[str, dict], *, repo_id: str | None = None) -> str:
    base, family = metadata["base_model"], metadata["family"]
    config = metadata["config"]
    name = config["name"]
    base_label = "Gemma 4 E2B" if family == "gemma4" else "Gemma 3"
    repo_id = repo_id or f"YOUR_USERNAME/{name}"
    temperature = metadata["temperature"]
    test_reports = {key: report for key, report in reports.items() if report["split"] == "test"}
    lines = ["---", "language: en", f"base_model: {base}", "base_model_relation: adapter",
             "library_name: peft", f"license: {LICENSES[family]}",
             "tags: [gev, decision-model, lora, pointer-head, multiple-choice, calibration]",
             f"datasets: [{HF_DATASET}]", "metrics: [accuracy, brier_score, expected_calibration_error]"]
    if test_reports:
        lines += ["model-index:", f"  - name: {name}", "    results:"]
        for key, report in test_reports.items():
            scores = report.get("clean_calibrated", report["clean"])
            calibration = "as served" if "clean_calibrated" in report else "raw T=1"
            lines += ["      - task: { type: text-classification, name: typed decision (choice / noul / score) }",
                      f'        dataset: {{ type: {HF_DATASET}, name: "{key} (clean questions)" }}',
                      "        metrics:",
                      f"          - {{ type: accuracy, value: {report['clean']['acc']:.4f} }}",
                      f'          - {{ type: brier_score, value: {scores["brier"]:.4f}, name: "Brier ({calibration})" }}',
                      f'          - {{ type: expected_calibration_error, value: {scores["ece"]:.4f}, '
                      f'name: "ECE ({calibration})" }}']
    lines += ["---", "", f"# {name} — {base_label} decision model", "",
              "**State and typed questions in; one answer and a probability distribution per question out.** "
              "Gev scores the supplied options rather than generating text. It combines a LoRA adapter on "
              f"[`{base}`](https://huggingface.co/{base}) with a separately trained pointer head.", "",
              "The pointer head (`pointer.safetensors`) is required: loading the PEFT adapter alone does not "
              "produce Gev decisions. The Gev package loads the base model, adapter, pointer head, tokenizer "
              "markers, and serving temperature. Its [source repository](https://github.com/onatm/gev) is "
              "currently private; access to it is needed to run inference.", ""]
    if test_reports:
        for key, label in (("decision-v7/test", "trained-source"), ("transfer-v4/test", "new-source")):
            if key in test_reports:
                m = test_reports[key]["clean"]
                lines.append(f"- **{label.capitalize()} test:** {m['acc']:.1%} accuracy on {m['n']:,} clean questions.")
        lines.append("")
    lines += ["## Use", "", "With access to the Gev source, run `uv sync --locked` in its checkout "
              "(add `--extra mlx` on Apple Silicon). Save this as `request.json`:", "", "```json",
              '{"state":"Order #1 arrived damaged.","questions":{"route":{"type":"choice",'
              '"instructions":"Which team handles this?","criteria":{"billing":"Payments",'
              '"support":"Product issues"}}}}', "```", "", "```bash",
              f"uv run gev predict {repo_id} --input request.json", "```", "",
              "The response includes `questions.route.answer`, `questions.route.probabilities` (one per option), "
              "and the serving temperature. Choice, yes/no (`noul`), and ordinal (`score`) questions are "
              "supported. A Hugging Face text-generation or PEFT-only pipeline cannot serve this model.", ""]
    if reports:
        lines += ["## Evaluation", "", "Clean-question scores from the saved reports. Accuracy is at raw "
                  "T=1 (temperature scaling does not change the winning answer); Brier and ECE are lower-is-better. "
                  f"The served columns use the saved temperature **T={temperature:.4f}**. "
                  "A dash means calibrated metrics were not recorded for that split.", "",
                  "| Suite / split | Clean n | Accuracy | Brier raw | Brier served | ECE raw | ECE served |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, report in reports.items():
            raw = report["clean"]
            served = report.get("clean_calibrated")
            served_brier = f"{served['brier']:.4f}" if served else "-"
            served_ece = f"{served['ece']:.4f}" if served else "-"
            lines.append(f"| {key} | {raw['n']:,} | {raw['acc']:.4f} | {raw['brier']:.4f} | "
                         f"{served_brier} | {raw['ece']:.4f} | {served_ece} |")
        lines += ["", "`decision-v7` contains held-out questions from the training source families; "
                  "`transfer-v4` contains new sources and held-out policy structures. Development was for model "
                  f"selection; these reports describe seed {config['training']['seed']}. Temperature was fitted on "
                  "the separate `decision-v7/calibration` split, not on either test split.", ""]
        if reports.get("decision-v7/test", {}).get("_derived_calibration"):
            lines += ["The decision-v7 test report predates the temperature fit. Its served metrics were "
                      "computed afterward from saved raw logits, without rerunning inference or fitting on test.", ""]
    transfer = reports.get("transfer-v4/test", {})
    if transfer.get("paired_flip") and transfer.get("variants", {}).get("none_present"):
        flip = transfer["paired_flip"]
        none = transfer["variants"]["none_present"]
        lines += ["## Known limits", "", "This is one seed, not a multi-seed study. The new-source test is "
                  "substantially harder than the trained-source test; evaluate on your own decisions before use.", "",
                  f"- For changed-answer contrastive pairs, both answers were correct in "
                  f"{round(flip['both_correct_rate'] * flip['pairs'])}/{flip['pairs']} pairs.",
                  f"- With a none-of-the-above option present, {round(none['acc'] * none['n'])}/{none['n']} "
                  "questions were correct.", ""]
    training = metadata.get("training", {})
    recipe = config["training"]
    model = config["model"]
    lines += ["## Model and provenance", "",
              f"- Base: `{base}` at revision `{metadata['revision']}`; frozen base weights are not in this repo.",
              f"- Architecture: LoRA rank {model['lora_rank']}, alpha {model['lora_alpha']} on the text decoder "
              f"and a {model['pointer_width']}-wide pointer head. Each question is scored independently.",
              f"- Recipe: seed {recipe['seed']}, {recipe['epochs']} epochs, "
              f"{training['train_records']:,} training records, "
              f"{training['steps']:,} steps, {metadata['trained_with']['backend'].upper()}/"
              f"{metadata['trained_with']['dtype'].upper()}. The checkpoint stores the fitted temperature in `gev.json`.",
              f"- Training data SHA-256: `{training['train_sha256']}`; each saved evaluation report "
              "also records its split's SHA-256.",
              f"- Training suite: [{HF_DATASET}](https://huggingface.co/datasets/{HF_DATASET}) "
              "(`decision-v7`, pinned and verified by hash).", "",
              "Training uses option permutation, none-of-the-above and distractor augmentation, and "
              "contrastive pairs. The architecture follows [Jev's Architecture Unmasked]"
              "(https://archerhume.com/posts/jevs-architecture-unmasked); the data and evaluation "
              "protocol are adapted from [Kev](https://github.com/jaredpalmer/kev). "
              "The detailed run reports and code are in the private Gev repository.", ""]
    lines += ["## License", "", f"The adapter and pointer head are {LICENSES[family]}; "
              "check the separately loaded base model and dataset licenses as well.", ""]
    return "\n".join(lines)


def _prepare_adapter_config(directory: Path) -> None:
    """Omit PEFT's optional null task type; Hub metadata expects a string if present."""
    path = directory / "adapter_config.json"
    if path.exists():
        adapter = json.loads(path.read_text(encoding="utf-8"))
        if "task_type" in adapter and adapter["task_type"] is None:
            del adapter["task_type"]
            path.write_text(json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def push(checkpoint: str | Path, repo_id: str, *, private: bool = True,
         reports: list[str | Path] = ()) -> str:
    """Upload a checkpoint directory to the Hub, generating a model card if none exists."""
    from huggingface_hub import HfApi, set_client_factory
    from huggingface_hub.utils._http import hf_request_event_hook

    directory = resolve(checkpoint)
    metadata = read_metadata(directory)
    card = directory / "README.md"
    if not card.exists():
        loaded = _card_reports(reports, metadata["temperature"])
        card.write_text(model_card(metadata, loaded, repo_id=repo_id), encoding="utf-8")
    _prepare_adapter_config(directory)
    set_client_factory(lambda: httpx.Client(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    ))
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, folder_path=str(directory),
                             allow_patterns=["*.json", "*.safetensors", "README.md"]).commit_url


def download(repo_id: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "README.md"]))
