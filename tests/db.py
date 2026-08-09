"""Test database helpers.

Tests run against a real Postgres, not a fake, because the frontier's claim
depends on SELECT ... FOR UPDATE SKIP LOCKED and a fake would not have it.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from rag.config.settings import PostgresSettings, get_settings

TEST_DATABASE = "agentic_rag_test"

TABLES = ("dead_letter", "frontier", "source_state", "source")


def test_postgres_settings() -> PostgresSettings:
    """Same server and credentials, different database."""
    configured = get_settings().postgres
    parts = urlsplit(configured.dsn)
    dsn = urlunsplit(
        (parts.scheme, parts.netloc, f"/{TEST_DATABASE}", parts.query, parts.fragment)
    )
    return configured.model_copy(update={"dsn": dsn})
