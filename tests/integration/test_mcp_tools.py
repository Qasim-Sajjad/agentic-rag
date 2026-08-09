"""MCP boundary: schema caps, tenant scoping, budget, and status reporting."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.clock import FakeClock
from rag.config.settings import McpSettings, QdrantSettings, RetrieveSettings
from rag.db.pool import Database
from rag.fetch.deadletter import DeadLetterEntry, DeadLetterStore
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import (
    CircuitState,
    FailureReason,
    FetchTier,
    Source,
    SourceStatus,
)
from rag.index.embed import FakeEmbedder
from rag.index.store import QdrantStore
from rag.mcp.schemas import (
    CallBudgetExceededError,
    IngestStatusInput,
    SearchCorpusInput,
    SearchCorpusOutput,
)
from rag.mcp.server import build_server
from rag.mcp.tools import ToolDependencies, ToolService
from rag.retrieve.rerank import IdentityReranker
from rag.retrieve.service import RetrieveDependencies, SearchService

pytestmark = pytest.mark.integration

SOURCE_ID = "fixture"


@pytest.fixture
async def service(db: Database, tmp_path: Path):
    registry = SourceRegistry(db)
    await registry.upsert(
        Source(source_id=SOURCE_ID, domain="example.test", seed_urls=[])
    )
    qdrant = QdrantSettings(path=str(tmp_path / "qdrant"), collection="test")
    store = QdrantStore(qdrant)
    embedder = FakeEmbedder()
    await store.bootstrap(embedder.dims)
    search = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=IdentityReranker(),
            settings=RetrieveSettings(),
            qdrant=qdrant,
        )
    )
    tools = ToolService(
        ToolDependencies(
            search=search,
            registry=registry,
            dead_letter=DeadLetterStore(db),
            settings=McpSettings(session_call_budget=3),
            clock=FakeClock(),
        )
    )
    try:
        yield tools, registry, DeadLetterStore(db)
    finally:
        await store.close()


def test_top_k_over_the_cap_is_rejected_by_the_schema():
    """The cap lives in the schema, so it is discoverable by the calling model."""
    with pytest.raises(ValidationError):
        SearchCorpusInput(query="anything", top_k=50)


def test_top_k_at_the_cap_is_accepted():
    assert SearchCorpusInput(query="anything", top_k=20).top_k == 20


def test_tenant_is_not_settable_from_the_client():
    with pytest.raises(ValidationError):
        SearchCorpusInput(query="anything", tenant_id="other-tenant")


def test_arbitrary_filters_are_not_settable():
    with pytest.raises(ValidationError):
        SearchCorpusInput(query="anything", filter="doc_type == 'pdf' or 1=1")


async def test_search_output_validates_against_the_schema(service):
    tools, _, _ = service
    result = await tools.search_corpus(SearchCorpusInput(query="risk"), "default")
    assert isinstance(result, SearchCorpusOutput)


async def test_search_returns_chunks_not_an_answer(service):
    tools, _, _ = service
    result = await tools.search_corpus(SearchCorpusInput(query="risk"), "default")
    assert not hasattr(result, "answer")


async def test_the_session_budget_rejects_calls_after_the_cap(service):
    tools, _, _ = service
    for _ in range(3):
        await tools.get_ingest_status(IngestStatusInput(), session_id="s1")
    with pytest.raises(CallBudgetExceededError):
        await tools.get_ingest_status(IngestStatusInput(), session_id="s1")


async def test_the_budget_is_per_session(service):
    tools, _, _ = service
    for _ in range(3):
        await tools.get_ingest_status(IngestStatusInput(), session_id="s1")
    assert await tools.get_ingest_status(IngestStatusInput(), session_id="s2")


async def test_an_unknown_source_reports_never_ingested_rather_than_erroring(service):
    tools, _, _ = service
    status = await tools.get_ingest_status(IngestStatusInput(source_id="nope"))
    assert status.status == "never_ingested"


async def test_an_open_circuit_reports_unreachable(service):
    tools, registry, _ = service
    state = await registry.state(SOURCE_ID)
    await registry.save_state(
        state.model_copy(
            update={
                "circuit_state": CircuitState.OPEN,
                "last_failure_reason": FailureReason.BLOCKED_PERSISTENT,
            }
        )
    )
    status = await tools.get_ingest_status(IngestStatusInput(source_id=SOURCE_ID))
    assert status.status == "unreachable"


async def test_an_unreachable_source_explains_the_gap_in_coverage(service):
    """The whole reason this tool exists: a dead end becomes an explanation."""
    tools, registry, _ = service
    state = await registry.state(SOURCE_ID)
    await registry.save_state(
        state.model_copy(update={"circuit_state": CircuitState.OPEN})
    )
    status = await tools.get_ingest_status(IngestStatusInput(source_id=SOURCE_ID))
    assert "unreachable since" in status.coverage_note


async def test_a_retired_source_reports_unreachable(service):
    tools, registry, _ = service
    await registry.set_status(SOURCE_ID, SourceStatus.UNREACHABLE)
    status = await tools.get_ingest_status(IngestStatusInput(source_id=SOURCE_ID))
    assert status.status == "unreachable"


async def test_failure_counts_come_from_the_dead_letter_store(service):
    tools, _, dead_letter = service
    await dead_letter.record(
        DeadLetterEntry(
            url="https://example.test/a",
            source_id=SOURCE_ID,
            reason=FailureReason.BLOCKED_PERSISTENT,
            stage="fetch",
            attempts=3,
            last_tier=FetchTier.STEALTH,
        )
    )
    status = await tools.get_ingest_status(IngestStatusInput(source_id=SOURCE_ID))
    assert status.docs_failed == 1


async def test_a_source_can_be_looked_up_by_domain(service):
    tools, _, _ = service
    status = await tools.get_ingest_status(IngestStatusInput(domain="example.test"))
    assert status.source_id == SOURCE_ID


async def test_the_server_exposes_exactly_two_tools(service):
    tools, _, _ = service
    server = build_server(tools)
    listed = await server.list_tools()
    assert len(listed) == 2


async def test_the_server_exposes_the_two_named_tools(service):
    tools, _, _ = service
    server = build_server(tools)
    listed = await server.list_tools()
    assert {tool.name for tool in listed} == {"search_corpus", "get_ingest_status"}


async def test_the_advertised_schema_carries_the_top_k_cap(service):
    """Field descriptions and bounds are the interface the model reads."""
    tools, _, _ = service
    server = build_server(tools)
    listed = await server.list_tools()
    search = next(tool for tool in listed if tool.name == "search_corpus")
    assert "20" in str(search.input_schema)


async def test_the_advertised_schema_has_no_tenant_field(service):
    tools, _, _ = service
    server = build_server(tools)
    listed = await server.list_tools()
    search = next(tool for tool in listed if tool.name == "search_corpus")
    assert "tenant" not in str(search.input_schema).lower()
