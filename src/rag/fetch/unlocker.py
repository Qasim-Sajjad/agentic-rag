"""Tier 4. Interface only, no paid service wired up.

Swapping in Zyte or Bright Data is this class plus a config entry. It is left
as a stub deliberately: the ladder's shape is the design decision, and buying
the hostile few percent is a purchasing decision, not an engineering one.
"""

from __future__ import annotations

from rag.config.settings import FetchSettings
from rag.fetch.protocols import UnlockerNotConfiguredError
from rag.fetch.types import FetchResult, FetchTier


class UnlockerFetcher:
    tier = FetchTier.UNLOCKER

    def __init__(self, settings: FetchSettings) -> None:
        self._settings = settings

    async def fetch(self, url: str, timeout: float) -> FetchResult:
        raise UnlockerNotConfiguredError(
            "tier 4 unlocker is a stub. Set a provider in config and implement "
            "UnlockerFetcher.fetch to enable it"
        )

    async def close(self) -> None:
        return None
