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

import json
import ssl
from pathlib import Path

import httpx
import truststore

from . import __version__
from .config import Config, from_dict

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


def model_card(metadata: dict, reports: dict[str, dict]) -> str:
    base, family = metadata["base_model"], metadata["family"]
    lines = ["---", f"base_model: {base}", "library_name: peft", f"license: {LICENSES[family]}",
             "tags: [gev, jev, lora, pointer-head, decision]", "---", "",
             f"# {metadata['config']['name']}", "",
             f"A Jev-like Gev decision model: a LoRA adapter on [`{base}`](https://huggingface.co/{base}) "
             f"(revision `{metadata['revision']}`) plus a pointer head that scores each option "
             "against the question. Load it with the `gev` package: `gev predict <this repo> --input request.json`.",
             "", "The architecture follows [Jev's Architecture Unmasked]"
             "(https://archerhume.com/posts/jevs-architecture-unmasked); the data, augmentation, and metrics "
             "come from [Kev](https://github.com/jaredpalmer/kev).",
             "", f"Serving temperature: `{metadata['temperature']}`.", ""]
    if reports:
        lines += ["## Evaluation (clean questions, raw T=1)", "",
                  "| Suite/split | n | Accuracy | NLL | Brier | ECE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, report in reports.items():
            m = report["clean"]
            lines.append(f"| {name} | {m['n']} | {m['acc']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def push(checkpoint: str | Path, repo_id: str, *, private: bool = True,
         reports: list[str | Path] = ()) -> str:
    """Upload a checkpoint directory to the Hub, generating a model card if none exists."""
    from huggingface_hub import HfApi, set_client_factory
    from huggingface_hub.utils._http import hf_request_event_hook

    directory = resolve(checkpoint)
    metadata = read_metadata(directory)
    card = directory / "README.md"
    if not card.exists():
        loaded = {}
        for path in reports:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            loaded[f"{report['suite']}/{report['split']}"] = report
        card.write_text(model_card(metadata, loaded), encoding="utf-8")
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
