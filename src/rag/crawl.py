"""The crawl loop: claim, fetch, extract, index, discover, repeat.

Ties the four built modules together behind one entry point. Each stage already
has its own failure handling, so this file is sequencing and budgets only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.extract.protocols import (
    EmptyExtractionError,
    ParserUnavailableError,
    UnsupportedTypeError,
)
from rag.extract.service import ExtractService
from rag.fetch.deadletter import DeadLetterEntry, DeadLetterStore
from rag.fetch.discover import extract_links
from rag.fetch.frontier import Frontier
from rag.fetch.types import FailureReason, FetchResult, FrontierEntry
from rag.fetch.worker import FetchWorker
from rag.index.pipeline import IngestPipeline
from rag.log import get_logger

log = get_logger(__name__)

BATCH_SIZE = 10


@dataclass
class CrawlStats:
    fetched: int = 0
    indexed: int = 0
    chunks: int = 0
    skipped: int = 0
    discovered: int = 0
    unsupported: int = 0
    empty: int = 0
    tiers: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "fetched": self.fetched,
            "indexed": self.indexed,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "discovered": self.discovered,
            "unsupported": self.unsupported,
            "empty": self.empty,
            "tiers": {str(tier): count for tier, count in sorted(self.tiers.items())},
        }


@dataclass
class CrawlDependencies:
    worker: FetchWorker
    extract: ExtractService
    ingest: IngestPipeline
    frontier: Frontier
    dead_letter: DeadLetterStore


class Crawler:
    def __init__(self, deps: CrawlDependencies, domain: str, max_pages: int) -> None:
        self._deps = deps
        self._domain = domain
        self._max_pages = max_pages
        self.stats = CrawlStats()

    async def run(self) -> CrawlStats:
        """Runs until the page budget or the frontier is exhausted.

        Sweeping first is what recovers work from a crashed worker. Without it
        a process that died mid fetch leaves its rows `leased` forever, and the
        next run cannot see them because it only claims `pending`.
        """
        reclaimed = await self._deps.frontier.sweep_expired_leases()
        if reclaimed:
            log.info("reclaimed expired leases", urls=reclaimed)
        while self.stats.fetched < self._max_pages:
            claimed = await self._deps.worker.run_once(
                min(BATCH_SIZE, self._max_pages - self.stats.fetched), self._handle
            )
            if claimed == 0:
                break
        log.info("crawl finished", domain=self._domain, **self.stats.as_dict())
        return self.stats

    async def _handle(self, entry: FrontierEntry, result: FetchResult) -> None:
        self.stats.fetched += 1
        tier = int(result.tier_used)
        self.stats.tiers[tier] = self.stats.tiers.get(tier, 0) + 1
        await self._discover(result, entry)
        await self._index(entry, result)

    async def _discover(self, result: FetchResult, entry: FrontierEntry) -> None:
        if "html" not in result.content_type.lower():
            return
        if self.stats.discovered >= self._max_pages * 3:
            return
        for url in extract_links(result.content, result.final_url, self._domain):
            await self._deps.frontier.add(url, entry.source_id)
            self.stats.discovered += 1

    async def _index(self, entry: FrontierEntry, result: FetchResult) -> None:
        try:
            doc = await self._deps.extract.extract(
                result.content, result.final_url, result.content_type
            )
        except UnsupportedTypeError as exc:
            await self._dead_letter(entry, FailureReason.UNSUPPORTED_TYPE, str(exc))
            self.stats.unsupported += 1
            return
        except (EmptyExtractionError, ParserUnavailableError) as exc:
            await self._dead_letter(entry, FailureReason.UNSUPPORTED_TYPE, str(exc))
            self.stats.empty += 1
            return
        outcome = await self._deps.ingest.ingest(
            doc, entry.source_id, int(result.tier_used)
        )
        if outcome.skipped:
            self.stats.skipped += 1
            return
        self.stats.indexed += 1
        self.stats.chunks += outcome.chunks_written

    async def _dead_letter(
        self, entry: FrontierEntry, reason: FailureReason, detail: str
    ) -> None:
        """Extraction failures are dead lettered with stage=extract, so ingest
        status can say "89 unsupported type" rather than one failure count."""
        await self._deps.dead_letter.record(
            DeadLetterEntry(
                url=entry.url,
                source_id=entry.source_id,
                reason=reason,
                stage="extract",
                attempts=entry.attempts,
                detail=detail[:500],
            )
        )
