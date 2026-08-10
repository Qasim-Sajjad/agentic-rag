"""Tool implementations, independent of the transport.

Kept out of `server.py` so both tools are testable without starting a server,
and so the FastMCP wrapper stays a thin registration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.clock import Clock
from rag.config.settings import McpSettings
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import CircuitState, Source, SourceState, SourceStatus
from rag.log import get_logger
from rag.mcp.schemas import (
    CallBudgetExceededError,
    IngestStatusInput,
    IngestStatusOutput,
    SearchCorpusInput,
    SearchCorpusOutput,
    SourceHealth,
)
from rag.retrieve.service import SearchService
from rag.retrieve.types import SearchFilters

log = get_logger(__name__)

NEVER_INGESTED = IngestStatusOutput(
    source_id="unknown",
    status="never_ingested",
    circuit_state="closed",
    coverage_note="no such source is registered, so the corpus has never covered it",
)

QUEUE_KEYS = ("pending", "in_flight", "requeued")


@dataclass
class SessionBudget:
    """Counts calls per session. Server side, so the agent cannot raise it."""

    limit: int
    used: int = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise CallBudgetExceededError(
                f"session call budget of {self.limit} exhausted"
            )
        self.used += 1


@dataclass(frozen=True)
class _Counts:
    """Collected so `_to_status` stays inside the argument limit."""

    docs_failed: int
    docs_indexed: int
    queue: dict[str, int]


@dataclass
class ToolDependencies:
    search: SearchService
    registry: SourceRegistry
    dead_letter: DeadLetterStore
    settings: McpSettings
    clock: Clock
    budgets: dict[str, SessionBudget] = field(default_factory=dict)


class ToolService:
    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    def budget(self, session_id: str) -> SessionBudget:
        existing = self._deps.budgets.get(session_id)
        if existing is None:
            existing = SessionBudget(self._deps.settings.session_call_budget)
            self._deps.budgets[session_id] = existing
        return existing

    async def search_corpus(
        self, request: SearchCorpusInput, tenant_id: str, session_id: str = "default"
    ) -> SearchCorpusOutput:
        """Returns chunks, never a generated answer.

        `tenant_id` is a parameter of this method and not of the input model on
        purpose: it comes from session context, not from the caller.
        """
        self.budget(session_id).spend()
        filters = SearchFilters(
            doc_type=request.doc_type,
            source_id=request.source_id,
            date_from=request.date_from,
            tenant_id=tenant_id,
        )
        result = await self._deps.search.search(request.query, filters, request.top_k)
        return SearchCorpusOutput(
            chunks=result.chunks,
            confidence=result.confidence,
            k_used=result.k_used,
            reason=result.reason,
        )

    async def get_ingest_status(
        self, request: IngestStatusInput, session_id: str = "default"
    ) -> IngestStatusOutput:
        """Turns "I do not know" into "that source has been blocked since the 3rd"."""
        self.budget(session_id).spend()
        source = await self._resolve(request)
        if source is None:
            return NEVER_INGESTED
        state = await self._deps.registry.state(source.source_id)
        failures = await self._deps.dead_letter.counts_by_reason(source.source_id)
        counts = _Counts(
            docs_failed=sum(failures.values()),
            docs_indexed=await self._deps.registry.docs_indexed(source.source_id),
            queue=await self._deps.registry.queue_counts(source.source_id),
        )
        return self._to_status(source, state, counts)

    async def _resolve(self, request: IngestStatusInput) -> Source | None:
        if request.source_id is not None:
            return await self._deps.registry.get(request.source_id)
        if request.domain is not None:
            return await self._deps.registry.by_domain(request.domain)
        sources = await self._deps.registry.list_all()
        return sources[0] if sources else None

    def _to_status(
        self, source: Source, state: SourceState, counts: _Counts
    ) -> IngestStatusOutput:
        health = _health(source, state, counts.docs_indexed)
        return IngestStatusOutput(
            source_id=source.source_id,
            status=health,
            circuit_state=state.circuit_state.value,
            last_success_at=state.last_success_at,
            last_failure_reason=state.last_failure_reason,
            docs_indexed=counts.docs_indexed,
            docs_failed=counts.docs_failed,
            pending=counts.queue.get("pending", 0),
            in_flight=counts.queue.get("in_flight", 0),
            requeued=counts.queue.get("requeued", 0),
            coverage_note=_coverage_note(health, state, counts),
        )


def _health(source: Source, state: SourceState, docs_indexed: int) -> SourceHealth:
    unreachable = (
        source.status is SourceStatus.UNREACHABLE
        or state.circuit_state is CircuitState.OPEN
    )
    if unreachable:
        return "unreachable"
    if state.last_success_at is None and docs_indexed == 0:
        return "never_ingested"
    degraded = (
        state.consecutive_failures > 0 or state.circuit_state is CircuitState.HALF_OPEN
    )
    return "degraded" if degraded else "healthy"


HEALTH_NOTES: dict[str, str] = {
    "degraded": "this source is failing intermittently, so coverage may be incomplete",
    "never_ingested": (
        "this source has never been ingested, so the corpus does not cover it"
    ),
    "healthy": "this source is up to date",
}


def _coverage_note(health: SourceHealth, state: SourceState, counts: _Counts) -> str:
    """Written for the responder, which turns it into a sentence for a user."""
    if health == "unreachable":
        return _unreachable_note(state)
    return _queue_note(counts) or HEALTH_NOTES[health]


def _unreachable_note(state: SourceState) -> str:
    since = state.circuit_opened_at or state.last_failure_at
    stamp = since.date().isoformat() if since else "an unknown date"
    return (
        f"this source has been unreachable since {stamp}, so recent content is missing"
    )


def _queue_note(counts: _Counts) -> str | None:
    """Queue depth is a coverage caveat, not a health problem.

    A backlog and a running crawl are different claims and must not share a
    sentence: `in_flight` means a worker holds a lease right now, `pending`
    means urls are known and unfetched with nothing working on them.
    """
    pending = counts.queue.get("pending", 0)
    in_flight = counts.queue.get("in_flight", 0)
    requeued = counts.queue.get("requeued", 0)
    if in_flight:
        return (
            f"an ingest is running for this source, {in_flight} urls in flight and "
            f"{pending} still queued, so coverage is changing as you read this"
        )
    if pending:
        return (
            f"{pending} discovered urls are queued and not yet fetched, so the "
            "corpus does not cover this source completely. No crawl is running"
        )
    if requeued:
        return (
            f"{requeued} urls are deferred and waiting to retry, usually after "
            "rate limiting"
        )
    return None
