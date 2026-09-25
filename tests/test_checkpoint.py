import json
from pathlib import Path

import huggingface_hub
import truststore
import yaml

from gev import checkpoint


def test_published_card_matches_seed_zero_reports_and_metadata():
    run = Path(__file__).parents[1] / "runs/g4-s0"
    metadata = checkpoint.read_metadata(run)
    reports = checkpoint._card_reports(
        [run / name / "report.json" for name in
         ("eval-dev", "eval-transfer-dev", "eval-test", "eval-transfer-test")],
        metadata["temperature"],
    )
    card = checkpoint.model_card(metadata, reports, repo_id="onatm/gev-e2b")
    assert (run / "checkpoint/README.md").read_text() == card

    frontmatter = yaml.safe_load(card.split("---", 2)[1])
    assert frontmatter["base_model"] == metadata["base_model"]
    assert [result["metrics"][0]["value"] for result in frontmatter["model-index"][0]["results"]] == [
        round(reports[key]["clean"]["acc"], 4) for key in ("decision-v7/test", "transfer-v4/test")]
    assert "0.2336 | 0.0661 | 0.0170" in card  # calibrated decision-test read
    assert "12/64 pairs" in card and "14/36 questions" in card
    assert "uv run gev predict onatm/gev-e2b --input request.json" in card
    example = json.loads(card.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert example["questions"]["route"]["type"] == "choice"


def test_optional_peft_task_type_can_be_omitted(tmp_path):
    from peft import PeftConfig

    adapter = json.loads((Path(__file__).parents[1] /
                          "runs/g4-s0/checkpoint/adapter_config.json").read_text())
    assert "task_type" not in adapter
    (tmp_path / "adapter_config.json").write_text(json.dumps({**adapter, "task_type": None}))
    checkpoint._prepare_adapter_config(tmp_path)
    assert "task_type" not in json.loads((tmp_path / "adapter_config.json").read_text())
    assert PeftConfig.from_pretrained(tmp_path).task_type is None


def test_push_uses_system_trust_store_for_hub_requests(tmp_path, monkeypatch):
    (tmp_path / "gev.json").write_text(json.dumps({"format": checkpoint.FORMAT,
                                                   "format_version": checkpoint.FORMAT_VERSION}))
    (tmp_path / "README.md").write_text("Model card")
    factories = []
    clients = []
    monkeypatch.setattr(huggingface_hub, "set_client_factory", factories.append)
    monkeypatch.setattr(checkpoint.httpx, "Client", lambda **kwargs: clients.append(kwargs))

    class FakeApi:
        def create_repo(self, repo_id, *, private, exist_ok):
            assert factories and (repo_id, private, exist_ok) == ("user/gev-e2b", False, True)

        def upload_folder(self, *, repo_id, folder_path, allow_patterns):
            assert repo_id == "user/gev-e2b" and folder_path == str(tmp_path)
            assert "*.safetensors" in allow_patterns
            return type("Commit", (), {"commit_url": "https://huggingface.co/user/gev-e2b/commit/123"})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assert checkpoint.push(tmp_path, "user/gev-e2b", private=False).endswith("/commit/123")
    factories[0]()
    assert isinstance(clients[0]["verify"], truststore.SSLContext)
    assert clients[0]["follow_redirects"] is True and clients[0]["timeout"] is None
    assert len(clients[0]["event_hooks"]["request"]) == 1
