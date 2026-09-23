"""Network setup shared by public artifact fetches and Hub clients."""

from __future__ import annotations


def use_system_ssl() -> None:
    """Install the platform trust store without weakening certificate checks."""
    import truststore

    truststore.inject_into_ssl()
