"""Frontier worker. Claims URLs, fetches them, and applies the give up rule.

The give up rule lives here rather than in `FetchService` because it spans
scheduling passes: a URL is permanently unreachable only after the highest
allowed tier has failed twice, in two passes at least an hour apart.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag.clock import Clock
from rag.config.settings import FetchSettings
from rag.fetch.deadletter import DeadLetterEntry, DeadLetterStore
from rag.fetch.frontier import Frontier
from rag.fetch.service import FetchService
from rag.fetch.types import FailureReason, FetchFailure, FetchResult, FrontierEntry
from rag.log import get_logger

log = get_logger(__name__)

Handler = Callable[[FrontierEntry, FetchResult], Awaitable[None]]


@dataclass
class WorkerDeps:
    service: FetchService
    frontier: Frontier
    dead_letter: DeadLetterStore
    clock: Clock
    settings: FetchSettings


class FetchWorker:
    def __init__(
        self, deps: WorkerDeps, name: str = "worker-1", source_id: str | None = None
    ) -> None:
        self._deps = deps
        self._name = name
        self._source_id = source_id

    async def run_once(self, limit: int, handler: Handler | None = None) -> int:
        """Claims up to `limit` URLs and processes each. Returns how many ran."""
        claimed = await self._deps.frontier.claim(
            self._name, limit, self._deps.settings.lease_minutes, self._source_id
        )
        for entry in claimed:
            outcome = await self._deps.service.fetch(entry.url)
            await self._apply(entry, outcome, handler)
        return len(claimed)

    async def _apply(
        self,
        entry: FrontierEntry,
        outcome: FetchResult | FetchFailure,
        handler: Handler | None,
    ) -> None:
        if isinstance(outcome, FetchResult):
            if handler is not None:
                await handler(entry, outcome)
            await self._deps.frontier.complete(entry.url_hash)
            return
        await self._handle_failure(entry, outcome)

    async def _handle_failure(
        self, entry: FrontierEntry, failure: FetchFailure
    ) -> None:
        if failure.reason is FailureReason.RATE_LIMITED:
            await self._requeue(entry, failure)
            return
        passes = await self._deps.frontier.record_pass(entry.url_hash)
        if passes >= self._deps.settings.give_up_passes:
            await self._give_up(entry, failure)
            return
        delay = self._deps.settings.give_up_pass_gap_hours * 3600
        await self._deps.frontier.requeue(entry.url_hash, delay)

    async def _requeue(self, entry: FrontierEntry, failure: FetchFailure) -> None:
        """Rate limited is temporary. It never produces a dead letter entry."""
        delay = failure.retry_after_seconds or self._deps.settings.backoff_cap_seconds
        visible_at = await self._deps.frontier.requeue(entry.url_hash, delay)
        log.info(
            "requeued", url=entry.url, delay_seconds=delay, visible_at=str(visible_at)
        )

    async def _give_up(self, entry: FrontierEntry, failure: FetchFailure) -> None:
        await self._deps.dead_letter.record(
            DeadLetterEntry(
                url=entry.url,
                source_id=entry.source_id,
                reason=failure.reason,
                stage="fetch",
                attempts=failure.attempts,
                last_tier=failure.last_tier,
                detail=failure.detail,
            )
        )
        await self._deps.frontier.kill(entry.url_hash)
        log.warning("gave up", url=entry.url, reason=str(failure.reason))
