"""Configure Hugging Face Hub requests to use the operating system's trust store."""

from __future__ import annotations

import ssl

import httpx
import truststore


def configure_hub() -> None:
    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import hf_request_event_hook

    set_client_factory(lambda: httpx.Client(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    ))
