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
        return self._to_status(source, state, sum(failures.values()))

    async def _resolve(self, request: IngestStatusInput) -> Source | None:
        if request.source_id is not None:
            return await self._deps.registry.get(request.source_id)
        if request.domain is not None:
            return await self._deps.registry.by_domain(request.domain)
        sources = await self._deps.registry.list_all()
        return sources[0] if sources else None

    def _to_status(
        self, source: Source, state: SourceState, docs_failed: int
    ) -> IngestStatusOutput:
        health = _health(source, state)
        return IngestStatusOutput(
            source_id=source.source_id,
            status=health,
            circuit_state=state.circuit_state.value,
            last_success_at=state.last_success_at,
            last_failure_reason=state.last_failure_reason,
            docs_indexed=state.docs_indexed,
            docs_failed=docs_failed,
            coverage_note=_coverage_note(health, state),
        )


def _health(source: Source, state: SourceState) -> SourceHealth:
    unreachable = (
        source.status is SourceStatus.UNREACHABLE
        or state.circuit_state is CircuitState.OPEN
    )
    if unreachable:
        return "unreachable"
    if state.last_success_at is None and state.docs_indexed == 0:
        return "never_ingested"
    degraded = (
        state.consecutive_failures > 0 or state.circuit_state is CircuitState.HALF_OPEN
    )
    return "degraded" if degraded else "healthy"


def _coverage_note(health: SourceHealth, state: SourceState) -> str:
    """Written for the responder, which turns it into a sentence for a user."""
    if health == "unreachable":
        since = state.circuit_opened_at or state.last_failure_at
        stamp = since.date().isoformat() if since else "an unknown date"
        return (
            f"this source has been unreachable since {stamp}, "
            "so recent content is missing"
        )
    if health == "degraded":
        return "this source is failing intermittently, so coverage may be incomplete"
    if health == "never_ingested":
        return "this source has never been ingested, so the corpus does not cover it"
    return "this source is up to date"
