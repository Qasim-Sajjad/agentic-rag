"""Wiring. One place that knows how the fetch pieces fit together."""

from __future__ import annotations

import random

from rag.clock import Clock, SystemClock
from rag.config.settings import FetchSettings, get_settings
from rag.db.pool import Database
from rag.fetch.browser import BrowserFetcher, StealthFetcher
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.protocols import Fetcher
from rag.fetch.ratelimit import DomainLimiter
from rag.fetch.registry import SourceRegistry
from rag.fetch.robots import RobotsChecker
from rag.fetch.service import FetchDependencies, FetchService
from rag.fetch.static import StaticFetcher
from rag.fetch.types import FetchTier
from rag.fetch.unlocker import UnlockerFetcher


def build_fetchers(settings: FetchSettings) -> dict[FetchTier, Fetcher]:
    """Browsers launch lazily, so building all four tiers costs nothing."""
    return {
        FetchTier.STATIC: StaticFetcher(settings),
        FetchTier.BROWSER: BrowserFetcher(settings),
        FetchTier.STEALTH: StealthFetcher(settings),
        FetchTier.UNLOCKER: UnlockerFetcher(settings),
    }


def build_service(
    db: Database,
    clock: Clock | None = None,
    settings: FetchSettings | None = None,
    fetchers: dict[FetchTier, Fetcher] | None = None,
) -> FetchService:
    resolved_settings = settings if settings is not None else get_settings().fetch
    resolved_clock = clock if clock is not None else SystemClock()
    resolved_fetchers = (
        fetchers if fetchers is not None else build_fetchers(resolved_settings)
    )
    static = resolved_fetchers[FetchTier.STATIC]
    deps = FetchDependencies(
        registry=SourceRegistry(db),
        dead_letter=DeadLetterStore(db),
        robots=RobotsChecker(static, resolved_clock, resolved_settings),
        limiter=DomainLimiter(
            resolved_clock, resolved_settings.default_requests_per_second
        ),
        fetchers=resolved_fetchers,
        clock=resolved_clock,
        settings=resolved_settings,
        rng=random.Random(),
    )
    return FetchService(deps)


async def close_fetchers(fetchers: dict[FetchTier, Fetcher]) -> None:
    for fetcher in fetchers.values():
        await fetcher.close()
