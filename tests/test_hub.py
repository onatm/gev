import huggingface_hub
import truststore
from transformers import AutoTokenizer

from gev import checkpoint, hub
from gev.train import load_tokenizer
from tests.conftest import tiny_config


def test_tokenizer_uses_system_trust_before_hub_request(monkeypatch, tokenizer):
    factories = []
    clients = []
    monkeypatch.setattr(huggingface_hub, "set_client_factory", factories.append)
    monkeypatch.setattr(hub.httpx, "Client", lambda **kwargs: clients.append(kwargs))

    config = tiny_config()

    def from_pretrained(name, *, revision):
        assert (name, revision) == (config.model.name, config.model.revision)
        factories[0]()
        assert isinstance(clients[0]["verify"], truststore.SSLContext)
        assert clients[0]["follow_redirects"] is True and clients[0]["timeout"] is None
        assert len(clients[0]["event_hooks"]["request"]) == 1
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    loaded, markers = load_tokenizer(config)
    assert loaded is tokenizer
    assert markers is not None


def test_checkpoint_download_configures_hub_first(monkeypatch, tmp_path):
    def snapshot_download(repo_id, *, revision, allow_patterns):
        assert configured == [True]
        assert (repo_id, revision) == ("user/gev-e2b", "revision")
        assert allow_patterns == ["*.json", "*.safetensors", "README.md"]
        return str(tmp_path)

    configured = []
    monkeypatch.setattr(hub, "configure_hub", lambda: configured.append(True))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert checkpoint.download("user/gev-e2b", "revision") == tmp_path
