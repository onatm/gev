import json
from pathlib import Path

import huggingface_hub
import pytest
import truststore

from gev import checkpoint, hub


ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/g4-s0"
CARD = ROOT / "docs/models/cards/gev-e2b.md"


def test_published_card_matches_seed_zero_reports_and_metadata():
    assert [path.parent.name for path in checkpoint.discover_reports(RUN)] == [
        "eval-dev", "eval-transfer-dev", "eval-test", "eval-transfer-test"]
    assert checkpoint.update_card(RUN, CARD, check=True) is False

    metadata = checkpoint.read_metadata(RUN)
    reports = checkpoint._card_reports(checkpoint.discover_reports(RUN), metadata["temperature"])
    card = CARD.read_text()
    data = huggingface_hub.ModelCard(card).data
    assert data.base_model == metadata["base_model"]
    assert [result.metric_value for result in data.eval_results if result.metric_type == "accuracy"] == [
        round(reports[key]["clean"]["acc"], 4) for key in ("decision-v7/test", "transfer-v4/test")]
    assert "0.2336 | 0.0661 | 0.0170" in card  # calibrated decision-test read
    assert "**T=1.6245**" in card
    example = json.loads(card.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert example["questions"]["route"]["type"] == "choice"


def test_card_regenerates_only_marked_regions(tmp_path):
    card = tmp_path / "card.md"
    original = CARD.read_text()
    stale = (original.replace("value: 0.8292", "value: 0.1")
             .replace("| decision-v7/test | 1,200 | 0.8292", "| decision-v7/test | 1,200 | 0.1")
             .replace("# gev-e2b", "# edited by hand"))
    card.write_text(stale)
    with pytest.raises(ValueError, match="out of date"):
        checkpoint.update_card(RUN, card, check=True)
    assert card.read_text() == stale
    assert checkpoint.update_card(RUN, card) is True
    assert card.read_text() == original.replace("# gev-e2b", "# edited by hand")

    card.write_text(original.replace("<!-- /gev:eval -->", ""))
    with pytest.raises(ValueError, match="region"):
        checkpoint.update_card(RUN, card)


def test_optional_peft_task_type_can_be_omitted(tmp_path):
    from peft import PeftConfig

    adapter = json.loads((Path(__file__).parents[1] /
                          "runs/g4-s0/checkpoint/adapter_config.json").read_text())
    assert "task_type" not in adapter
    (tmp_path / "adapter_config.json").write_text(json.dumps({**adapter, "task_type": None}))
    checkpoint._prepare_adapter_config(tmp_path)
    assert "task_type" not in json.loads((tmp_path / "adapter_config.json").read_text())
    assert PeftConfig.from_pretrained(tmp_path).task_type is None


def test_push_uploads_checkpoint_and_card_in_one_commit(tmp_path, monkeypatch):
    (tmp_path / "gev.json").write_text(json.dumps({"format": checkpoint.FORMAT,
                                                   "format_version": checkpoint.FORMAT_VERSION}))
    (tmp_path / "pointer.safetensors").write_bytes(b"weights")
    (tmp_path / "README.md").write_text("stale card that must not be uploaded")
    (tmp_path / "notes.txt").write_text("not uploaded")
    factories = []
    clients = []
    commits = []
    monkeypatch.setattr(huggingface_hub, "set_client_factory", factories.append)
    monkeypatch.setattr(hub.httpx, "Client", lambda **kwargs: clients.append(kwargs))

    class FakeApi:
        def create_repo(self, repo_id, *, private, exist_ok):
            assert factories and (repo_id, private, exist_ok) == ("user/gev-e2b", False, True)
            commits.append("create")

        def create_commit(self, repo_id, *, operations, commit_message):
            assert repo_id == "user/gev-e2b"
            commits.append({op.path_in_repo: op.path_or_fileobj for op in operations})
            return type("Commit", (), {"commit_url": "https://huggingface.co/user/gev-e2b/commit/123"})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assert checkpoint.push(tmp_path, "user/gev-e2b", CARD, private=False).endswith("/commit/123")
    assert commits == ["create", {"README.md": str(CARD), "gev.json": str(tmp_path / "gev.json"),
                                  "pointer.safetensors": str(tmp_path / "pointer.safetensors")}]
    factories[0]()
    assert isinstance(clients[0]["verify"], truststore.SSLContext)
    assert clients[0]["follow_redirects"] is True and clients[0]["timeout"] is None
    assert len(clients[0]["event_hooks"]["request"]) == 1

    commits.clear()
    checkpoint.push(tmp_path, "user/gev-e2b", CARD, card_only=True)
    assert commits == [{"README.md": str(CARD)}]

    bad = tmp_path / "bad.md"
    bad.write_text("---\nmodel-index: [{name: x, results: [{task: {}}]}]\n---\n")
    with pytest.raises(ValueError):
        checkpoint.push(tmp_path, "user/gev-e2b", bad, card_only=True)
