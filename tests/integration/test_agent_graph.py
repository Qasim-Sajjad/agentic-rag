"""Agent branching end to end, with a scripted LLM and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.agent.graph import AgentRunner
from rag.agent.llm import ScriptedClient
from rag.agent.nodes import NodeDependencies
from rag.clock import FakeClock
from rag.config.settings import (
    AgentSettings,
    LLMSettings,
    McpSettings,
    QdrantSettings,
    RetrieveSettings,
)
from rag.db.pool import Database
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import CircuitState, Source
from rag.index.embed import FakeEmbedder
from rag.index.store import QdrantStore
from rag.mcp.tools import ToolDependencies, ToolService
from rag.prompts.registry import PromptRegistry
from rag.retrieve.rerank import IdentityReranker
from rag.retrieve.service import RetrieveDependencies, SearchService

pytestmark = pytest.mark.integration

SOURCE_ID = "fixture"


def plan_json(tool: str, **fields) -> str:
    return json.dumps({"tool": tool, "reason": "test", **fields})


def answer_json(answer: str, citations: list[dict[str, str]] | None = None) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": citations or [],
            "confidence": "high",
            "unanswered_aspects": [],
        }
    )


@pytest.fixture
async def build(db: Database, tmp_path: Path):
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
            settings=McpSettings(),
            clock=FakeClock(),
        )
    )

    def make(responses: list[str], max_iterations: int = 1) -> AgentRunner:
        return AgentRunner(
            NodeDependencies(
                llm=ScriptedClient(responses=list(responses)),
                tools=tools,
                prompts=PromptRegistry(),
                llm_settings=LLMSettings(api_key="test"),
                agent_settings=AgentSettings(max_iterations=max_iterations),
            )
        )

    try:
        yield make, registry
    finally:
        await store.close()


async def test_a_question_needing_no_tool_skips_the_tool_executor(build):
    make, _ = build
    runner = make([plan_json("answer_directly"), answer_json("I search a corpus.")])
    result = await runner.run("what can you do")
    assert [step.node for step in result.trace] == ["router", "responder"]


async def test_a_search_question_reaches_the_tool_executor(build):
    make, _ = build
    runner = make(
        [plan_json("search_corpus", query="revenue"), answer_json("Nothing found.")]
    )
    result = await runner.run("what was revenue")
    assert "tool_executor" in [step.node for step in result.trace]


async def test_the_trace_records_the_tool_that_ran(build):
    make, _ = build
    runner = make(
        [
            plan_json("get_ingest_status", source_id=SOURCE_ID),
            answer_json("It is fine."),
        ]
    )
    result = await runner.run("is the source up to date")
    tools_used = [step.tool for step in result.trace if step.tool]
    assert tools_used == ["get_ingest_status"]


async def test_the_trace_records_the_prompt_version(build):
    """Checked against the active version rather than a literal, so activating
    a new router prompt is not a test failure. Which version is live is the
    registry's job to state, not this test's."""
    make, _ = build
    runner = make([plan_json("answer_directly"), answer_json("Hello.")])
    result = await runner.run("hello")
    active = PromptRegistry().active_version("router")
    assert result.trace[0].prompt_version == f"router/{active}"


async def test_low_confidence_triggers_exactly_one_retry(build):
    """Empty corpus means confidence none, so the retry path fires once."""
    make, _ = build
    runner = make(
        [
            plan_json("search_corpus", query="revenue", source_id=SOURCE_ID),
            plan_json("search_corpus", query="revenue broadened"),
            answer_json("Nothing relevant was found."),
        ]
    )
    result = await runner.run("what was revenue")
    assert [step.node for step in result.trace].count("router") == 2


async def test_the_retry_broadens_rather_than_repeating(build):
    make, _ = build
    runner = make(
        [
            plan_json("search_corpus", query="revenue", source_id=SOURCE_ID),
            plan_json("search_corpus", query="revenue broadened"),
            answer_json("Nothing relevant was found."),
        ]
    )
    result = await runner.run("what was revenue")
    assert [step.node for step in result.trace].count("tool_executor") == 2


async def test_an_identical_repeat_call_is_rejected(build):
    """The iteration cap alone does not stop this. Fingerprints do."""
    make, _ = build
    runner = make(
        [
            plan_json("search_corpus", query="revenue"),
            plan_json("search_corpus", query="revenue"),
            answer_json("Nothing found."),
        ]
    )
    result = await runner.run("what was revenue")
    notes = [step.note for step in result.trace if step.note]
    assert "duplicate call" in notes


async def test_a_tool_failure_produces_a_stated_answer_not_a_crash(build):
    make, registry = build
    state = await registry.state(SOURCE_ID)
    await registry.save_state(
        state.model_copy(update={"circuit_state": CircuitState.OPEN})
    )
    runner = make(
        [plan_json("get_ingest_status", source_id=SOURCE_ID), answer_json("Reported.")]
    )
    result = await runner.run("is the source healthy")
    assert result.answer


async def test_an_unavailable_model_still_returns_an_answer(build):
    """No queued responses stands in for no API key. Never a fabricated answer."""
    make, _ = build
    runner = make([])
    result = await runner.run("what was revenue")
    assert "could not be generated" in result.answer


async def test_an_unavailable_model_carries_no_citations(build):
    make, _ = build
    runner = make([])
    result = await runner.run("what was revenue")
    assert result.citations == []


async def test_a_fabricated_citation_never_reaches_the_caller(build):
    """Validation rejects it, the repair fails too, so the fallback answers."""
    make, _ = build
    poisoned = answer_json(
        "Revenue was 41.2 million [c_fake].",
        [{"chunk_id": "c_fake", "source_url": "https://evil.tld"}],
    )
    runner = make([plan_json("answer_directly"), poisoned, poisoned])
    result = await runner.run("what was revenue")
    assert result.citations == []


async def test_the_recursion_limit_is_not_reached_in_normal_operation(build):
    make, _ = build
    runner = make([plan_json("answer_directly"), answer_json("Fine.")])
    result = await runner.run("hello")
    assert len(result.trace) < 10
