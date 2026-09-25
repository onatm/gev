import json

import huggingface_hub
import truststore

from gev import checkpoint


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
