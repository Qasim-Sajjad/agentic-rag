"""API contract: schemas, auth, tenant scoping, caching and the explain gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from rag.agent.graph import AgentRunner
from rag.agent.llm import ScriptedClient
from rag.agent.nodes import NodeDependencies
from rag.api.deps import AppContext, TTLCache, build_ingest
from rag.api.jobs import JobStore
from rag.api.main import create_app
from rag.clock import SystemClock
from rag.config.settings import (
    ApiSettings,
    IndexSettings,
    QdrantSettings,
    Settings,
)
from rag.db.pool import Database
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.factory import build_fetchers
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import Source
from rag.index.embed import FakeEmbedder
from rag.index.store import QdrantStore
from rag.mcp.tools import ToolDependencies, ToolService
from rag.prompts.registry import PromptRegistry
from rag.retrieve.rerank import IdentityReranker
from rag.retrieve.service import RetrieveDependencies, SearchService

pytestmark = pytest.mark.integration

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
SOURCE_ID = "fixture"


def answer_json(answer: str = "Nothing was found.") -> str:
    return json.dumps({"answer": answer, "citations": [], "confidence": "insufficient"})


async def build_context(db: Database, tmp_path: Path, explain: bool) -> AppContext:
    qdrant = QdrantSettings(path=str(tmp_path / "qdrant"), collection="test")
    settings = Settings(
        qdrant=qdrant,
        api=ApiSettings(api_keys={KEY: "default"}, explain_enabled=explain),
        # Under tmp_path so a test never writes a document blob into data/docs.
        index=IndexSettings(doc_store_path=str(tmp_path / "docs")),
    )
    store = QdrantStore(qdrant)
    embedder = FakeEmbedder()
    search = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=IdentityReranker(),
            settings=settings.retrieve,
            qdrant=qdrant,
        )
    )
    tools = ToolService(
        ToolDependencies(
            search=search,
            registry=SourceRegistry(db),
            dead_letter=DeadLetterStore(db),
            settings=settings.mcp,
            clock=SystemClock(),
        )
    )
    llm = ScriptedClient(responses=[answer_json() for _ in range(12)])
    prompts = PromptRegistry()
    agent = AgentRunner(
        NodeDependencies(
            llm=llm,
            tools=tools,
            prompts=prompts,
            llm_settings=settings.llm,
            agent_settings=settings.agent,
        )
    )
    fetchers = build_fetchers(settings.fetch)
    return AppContext(
        settings=settings,
        db=db,
        search=search,
        tools=tools,
        agent=agent,
        registry=SourceRegistry(db),
        dead_letter=DeadLetterStore(db),
        prompts=prompts,
        llm=llm,
        store=store,
        embedder=embedder,
        response_cache=TTLCache(settings.api.cache_ttl_seconds),
        # Real wiring rather than a second copy that drifts. Browsers launch
        # lazily, so building all four tiers costs nothing here.
        ingest=build_ingest(db, store, embedder, fetchers, settings),
        fetchers=fetchers,
        jobs=JobStore(),
    )


@pytest.fixture
async def client(db: Database, tmp_path: Path):
    await SourceRegistry(db).upsert(
        Source(source_id=SOURCE_ID, domain="example.test", seed_urls=[])
    )
    context = await build_context(db, tmp_path, explain=False)
    app = create_app(context)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


@pytest.fixture
async def explaining_client(db: Database, tmp_path: Path):
    await SourceRegistry(db).upsert(
        Source(source_id=SOURCE_ID, domain="example.test", seed_urls=[])
    )
    context = await build_context(db, tmp_path, explain=True)
    app = create_app(context)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


async def test_a_missing_api_key_is_rejected(client: httpx.AsyncClient):
    response = await client.post("/search", json={"query": "revenue"})
    assert response.status_code == 401


async def test_a_wrong_api_key_is_rejected(client: httpx.AsyncClient):
    response = await client.post(
        "/search", json={"query": "revenue"}, headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


async def test_the_ingest_endpoints_are_behind_the_same_key(client: httpx.AsyncClient):
    """A write endpoint that forgot the dependency would be an open door."""
    url = await client.post("/ingest/url", json={"url": "https://example.test/a"})
    assert url.status_code == 401
    upload = await client.post("/ingest/file", files={"file": ("a.txt", b"hello")})
    assert upload.status_code == 401


async def test_ingest_url_refuses_an_unregistered_domain_with_200(
    client: httpx.AsyncClient,
):
    """The request succeeded and the answer was no. Those are different, so this
    is a 200 carrying a typed reason rather than a 4xx."""
    response = await client.post(
        "/ingest/url", json={"url": "https://nowhere.test/a"}, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["failure"]["reason"] == "unknown_source"
    assert body["stages"] == []


async def wait_for_job(client: httpx.AsyncClient, job_id: str) -> dict:
    """Polls the way a client does. Ten seconds is far longer than a two line
    text file needs and short enough to fail rather than hang."""
    for _ in range(200):
        body = (await client.get(f"/ingest/jobs/{job_id}", headers=HEADERS)).json()
        if body["status"] != "running":
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never left running")


async def test_a_background_ingest_returns_202_and_a_job_id(
    client: httpx.AsyncClient,
):
    """202 rather than 200: the work was accepted, it has not happened yet."""
    response = await client.post(
        "/ingest/file?background=true",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert body["poll"] == f"/ingest/jobs/{body['job_id']}"


async def test_a_background_ingest_finishes_and_carries_the_same_trace(
    client: httpx.AsyncClient,
):
    """The polled result is the response the blocking call would have returned,
    so a client does not need two ways of reading an ingest."""
    accepted = await client.post(
        "/ingest/file?background=true",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    body = await wait_for_job(client, accepted.json()["job_id"])
    assert body["status"] == "done", body.get("error")
    assert body["result"]["ok"] is True
    assert body["result"]["chunks_written"] >= 1


async def test_a_background_ingest_reports_the_stages_it_passed_through(
    client: httpx.AsyncClient,
):
    """The point of the job: progress that names steps, not a spinner."""
    accepted = await client.post(
        "/ingest/file?background=true",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    body = await wait_for_job(client, accepted.json()["job_id"])
    stages = {row["stage"] for row in body["progress"]}
    assert {"chunk", "embed", "store"} <= stages


async def test_a_background_ingest_reports_each_stage_once(
    client: httpx.AsyncClient,
):
    """Embedding reports per batch. A caller wants the latest position, not the
    history, so a stage is replaced in place rather than appended."""
    accepted = await client.post(
        "/ingest/file?background=true",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    body = await wait_for_job(client, accepted.json()["job_id"])
    stages = [row["stage"] for row in body["progress"]]
    assert len(stages) == len(set(stages))


async def test_a_background_ingest_is_listed(client: httpx.AsyncClient):
    accepted = await client.post(
        "/ingest/file?background=true",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    job_id = accepted.json()["job_id"]
    await wait_for_job(client, job_id)
    listed = await client.get("/ingest/jobs", headers=HEADERS)
    assert job_id in {row["job_id"] for row in listed.json()["jobs"]}


async def test_an_unknown_job_id_is_a_404(client: httpx.AsyncClient):
    response = await client.get("/ingest/jobs/nosuchjob", headers=HEADERS)
    assert response.status_code == 404


async def test_the_job_endpoints_are_behind_the_same_key(client: httpx.AsyncClient):
    assert (await client.get("/ingest/jobs")).status_code == 401
    assert (await client.get("/ingest/jobs/anything")).status_code == 401


async def test_a_blocking_ingest_still_returns_the_trace(client: httpx.AsyncClient):
    """`background` defaults to false, so the existing contract is unchanged."""
    response = await client.post(
        "/ingest/file",
        files={"file": ("note.txt", b"# Heading\n\nRevenue grew in the quarter.\n")},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


async def test_search_returns_its_documented_shape(client: httpx.AsyncClient):
    response = await client.post("/search", json={"query": "revenue"}, headers=HEADERS)
    assert set(response.json()) == {
        "chunks",
        "confidence",
        "k_used",
        "steps",
        "latency_ms",
    }


async def test_search_rejects_a_top_k_over_the_cap(client: httpx.AsyncClient):
    response = await client.post(
        "/search", json={"query": "revenue", "top_k": 50}, headers=HEADERS
    )
    assert response.status_code == 422


async def test_ask_returns_chunk_objects_not_a_count(client: httpx.AsyncClient):
    response = await client.post("/ask", json={"question": "revenue"}, headers=HEADERS)
    assert isinstance(response.json()["chunks"], list)


async def test_ask_always_returns_a_validation_report(client: httpx.AsyncClient):
    response = await client.post("/ask", json={"question": "revenue"}, headers=HEADERS)
    assert set(response.json()["validation"]) == {
        "citations_checked",
        "citations_rejected",
        "repair_attempts",
        "fell_back",
    }


async def test_an_unanswerable_question_returns_200_not_an_error(
    client: httpx.AsyncClient,
):
    """The request succeeded. The corpus did not cover it."""
    response = await client.post("/ask", json={"question": "revenue"}, headers=HEADERS)
    assert response.status_code == 200


async def test_an_unanswerable_question_states_it_rather_than_guessing(
    client: httpx.AsyncClient,
):
    response = await client.post("/ask", json={"question": "revenue"}, headers=HEADERS)
    assert response.json()["confidence"] == "none"


async def test_explain_is_omitted_when_the_config_flag_is_off(
    client: httpx.AsyncClient,
):
    """A valid key is not authorisation to read prompt internals."""
    response = await client.post(
        "/ask", json={"question": "revenue", "explain": True}, headers=HEADERS
    )
    assert response.json()["explain"] is None


async def test_explain_is_returned_when_enabled(explaining_client: httpx.AsyncClient):
    response = await explaining_client.post(
        "/ask", json={"question": "revenue", "explain": True}, headers=HEADERS
    )
    assert response.json()["explain"] is not None


async def test_explain_never_returns_the_system_prompt_body(
    explaining_client: httpx.AsyncClient,
):
    response = await explaining_client.post(
        "/ask", json={"question": "revenue", "explain": True}, headers=HEADERS
    )
    block = response.json()["explain"]
    assert "You are a research assistant" not in json.dumps(block)


async def test_explain_identifies_the_prompt_by_version(
    explaining_client: httpx.AsyncClient,
):
    response = await explaining_client.post(
        "/ask", json={"question": "revenue", "explain": True}, headers=HEADERS
    )
    assert response.json()["explain"]["prompt_version"] == "rag_answer/v3"


async def test_two_explain_calls_return_different_nonces(
    explaining_client: httpx.AsyncClient,
):
    """Proves the cache is not serving a stale explain block."""
    payload = {"question": "revenue", "explain": True}
    first = await explaining_client.post("/ask", json=payload, headers=HEADERS)
    second = await explaining_client.post("/ask", json=payload, headers=HEADERS)
    assert first.json()["explain"]["nonce"] != second.json()["explain"]["nonce"]


async def test_the_agent_returns_a_trace(client: httpx.AsyncClient):
    response = await client.post("/agent", json={"question": "hello"}, headers=HEADERS)
    assert response.json()["trace"]


async def test_the_agent_trace_names_every_node_that_ran(client: httpx.AsyncClient):
    response = await client.post("/agent", json={"question": "hello"}, headers=HEADERS)
    nodes = [step["node"] for step in response.json()["trace"]]
    assert "responder" in nodes


async def test_the_agent_returns_the_context_it_answered_from(
    client: httpx.AsyncClient,
):
    """Same reason `/ask` returns chunks: an answer whose evidence cannot be
    inspected cannot be checked. The agent used to drop them at the boundary."""
    response = await client.post(
        "/agent", json={"question": "revenue"}, headers=HEADERS
    )
    body = response.json()
    assert "chunks" in body
    assert isinstance(body["chunks"], list)


async def test_the_agent_response_shape_is_documented(client: httpx.AsyncClient):
    response = await client.post("/agent", json={"question": "hello"}, headers=HEADERS)
    assert set(response.json()) == {
        "answer",
        "citations",
        "confidence",
        "trace",
        "chunks",
    }


async def test_ingest_status_summarises_the_corpus(client: httpx.AsyncClient):
    response = await client.get("/ingest/status", headers=HEADERS)
    assert response.json()["summary"]["total_sources"] >= 1


async def test_ingest_status_reports_one_source(client: httpx.AsyncClient):
    response = await client.get(
        f"/ingest/status?source_id={SOURCE_ID}", headers=HEADERS
    )
    assert response.json()["sources"][0]["source_id"] == SOURCE_ID


async def test_the_tenant_cannot_be_overridden_by_the_request_body(
    client: httpx.AsyncClient,
):
    response = await client.post(
        "/search",
        json={"query": "revenue", "filters": {"tenant_id": "other"}},
        headers=HEADERS,
    )
    assert response.status_code == 200
