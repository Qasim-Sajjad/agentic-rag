"""robots.txt fetching, caching and evaluation.

Cached on the `source` row rather than in memory, so a restart does not mean
re-fetching robots.txt for every domain in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from protego import Protego

from rag.clock import Clock
from rag.config.settings import FetchSettings
from rag.fetch.protocols import Fetcher, FetchTransportError
from rag.fetch.types import Source


@dataclass(frozen=True)
class RobotsDecision:
    allowed: bool
    crawl_delay: float | None
    fetched_text: str | None  # set only on a fresh fetch, for the caller to persist
    detail: str


def robots_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


class RobotsChecker:
    def __init__(self, fetcher: Fetcher, clock: Clock, settings: FetchSettings) -> None:
        self._fetcher = fetcher
        self._clock = clock
        self._settings = settings

    async def check(self, source: Source, url: str) -> RobotsDecision:
        cached = self._cached_text(source)
        if cached is not None:
            return self._evaluate(cached, url, fetched_text=None)
        text, detail = await self._fetch_text(url)
        if text is None:
            return RobotsDecision(False, None, None, detail)
        return self._evaluate(text, url, fetched_text=text)

    def _cached_text(self, source: Source) -> str | None:
        if source.robots_txt is None or source.robots_fetched_at is None:
            return None
        age = self._clock.now() - source.robots_fetched_at
        ttl = timedelta(hours=self._settings.robots_cache_ttl_hours)
        return source.robots_txt if age < ttl else None

    async def _fetch_text(self, url: str) -> tuple[str | None, str]:
        """RFC 9309: 4xx means allow all, 5xx means disallow all."""
        try:
            result = await self._fetcher.fetch(
                robots_url(url), self._settings.timeouts.static
            )
        except FetchTransportError as exc:
            return None, f"robots.txt unreachable: {exc}"
        if result.status >= 500:
            return None, f"robots.txt returned {result.status}, treating as disallow"
        if result.status >= 400:
            return "", f"robots.txt returned {result.status}, treating as allow all"
        return result.content.decode("utf-8", errors="replace"), "robots.txt fetched"

    def _evaluate(
        self, text: str, url: str, fetched_text: str | None
    ) -> RobotsDecision:
        parser = Protego.parse(text)
        agent = self._settings.user_agent
        allowed = bool(parser.can_fetch(url, agent))
        delay = parser.crawl_delay(agent)
        detail = "allowed by robots.txt" if allowed else "disallowed by robots.txt"
        return RobotsDecision(
            allowed=allowed,
            crawl_delay=float(delay) if delay is not None else None,
            fetched_text=fetched_text,
            detail=detail,
        )


def robots_fetched_now(clock: Clock) -> datetime:
    return clock.now()
