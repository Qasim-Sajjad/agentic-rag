"""Protocols for the parts of fetching that have more than one implementation."""

from __future__ import annotations

from typing import Protocol

from rag.fetch.types import FetchResult, FetchTier


class Fetcher(Protocol):
    """One rung of the ladder.

    Returns a `FetchResult` for any HTTP response, including 4xx and 5xx.
    Deciding what a status means is the orchestrator's job, not the fetcher's,
    which is what keeps escalation policy in one place. Raises only for
    transport failures: timeout, DNS, connection reset.
    """

    tier: FetchTier

    async def fetch(self, url: str, timeout: float) -> FetchResult: ...

    async def close(self) -> None: ...


class FetchTransportError(Exception):
    """Transport failed before a response existed. Carries no HTTP status."""


class FetchTimeoutError(FetchTransportError):
    """The tier's timeout elapsed."""


class UnlockerNotConfiguredError(FetchTransportError):
    """Tier 4 is an interface with no paid service wired up."""
