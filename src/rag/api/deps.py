"""Auth, caching and the wiring the endpoints depend on."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, Request

from rag.agent.graph import AgentRunner
from rag.agent.llm import LLMClient, build_client
from rag.agent.nodes import NodeDependencies
from rag.api.ingest import IngestDependencies
from rag.api.jobs import JobStore
from rag.clock import SystemClock
from rag.config.settings import Settings, get_settings
from rag.db.pool import Database
from rag.extract.service import ExtractService
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.factory import build_fetchers, build_service
from rag.fetch.protocols import Fetcher
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import FetchTier
from rag.index.embed import Embedder, build_embedder
from rag.index.pipeline import IndexDependencies, IngestPipeline
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.store import QdrantStore
from rag.log import get_logger
from rag.mcp.tools import ToolDependencies, ToolService
from rag.prompts.registry import PromptRegistry
from rag.retrieve.rerank import MiniLMReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import Reranker

log = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


@dataclass
class TTLCache:
    """In memory, behind a small surface so Redis is a swap rather than a
    rewrite. `prompt_version` and `embed_model_version` are in every key,
    because a prompt or model change must not serve stale answers."""

    ttl_seconds: int
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            del self.entries[key]
            return None
        return entry.value

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = CacheEntry(value, time.monotonic() + self.ttl_seconds)


def cache_key(**parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class AppContext:
    settings: Settings
    db: Database
    search: SearchService
    tools: ToolService
    agent: AgentRunner
    registry: SourceRegistry
    dead_letter: DeadLetterStore
    prompts: PromptRegistry
    llm: LLMClient
    store: QdrantStore
    embedder: Embedder
    response_cache: TTLCache
    ingest: IngestDependencies
    # Held so the lifespan can close them. Browsers launch lazily, so an API
    # process that only answers questions never starts one.
    fetchers: dict[FetchTier, Fetcher]
    # In flight and recently finished background ingests. Per process, which is
    # correct while Qdrant in process makes this the only writer anyway.
    jobs: JobStore


def build_context(settings: Settings | None = None) -> AppContext:
    resolved = settings if settings is not None else get_settings()
    db = Database(resolved.postgres)
    embedder: Embedder = build_embedder(resolved.index)
    reranker: Reranker = MiniLMReranker(resolved.retrieve)
    store = QdrantStore(resolved.qdrant)
    search = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=resolved.retrieve,
            qdrant=resolved.qdrant,
        )
    )
    tools = ToolService(
        ToolDependencies(
            search=search,
            registry=SourceRegistry(db),
            dead_letter=DeadLetterStore(db),
            settings=resolved.mcp,
            clock=SystemClock(),
        )
    )
    prompts = PromptRegistry()
    llm = build_client(resolved.llm)
    agent = AgentRunner(
        NodeDependencies(
            llm=llm,
            tools=tools,
            prompts=prompts,
            llm_settings=resolved.llm,
            agent_settings=resolved.agent,
        )
    )
    fetchers = build_fetchers(resolved.fetch)
    return AppContext(
        settings=resolved,
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
        response_cache=TTLCache(resolved.api.cache_ttl_seconds),
        ingest=build_ingest(db, store, embedder, fetchers, resolved),
        fetchers=fetchers,
        jobs=JobStore(),
    )


def build_ingest(
    db: Database,
    store: QdrantStore,
    embedder: Embedder,
    fetchers: dict[FetchTier, Fetcher],
    resolved: Settings,
) -> IngestDependencies:
    """The write path. Shares the embedder and the store with retrieval on
    purpose: Qdrant in process is single writer, so a second client here would
    be a second process trying to hold the same collection.

    Public because the API tests build their own context and need the same
    wiring rather than a second, drifting copy of it.
    """
    pipeline = IngestPipeline(
        IndexDependencies(
            documents=DocumentRepository(db, Path(resolved.index.doc_store_path)),
            chunks=ChunkRepository(db),
            store=store,
            embedder=embedder,
            settings=resolved.index,
        )
    )
    return IngestDependencies(
        fetch=build_service(db, SystemClock(), resolved.fetch, fetchers),
        extract=ExtractService(resolved.extract),
        pipeline=pipeline,
        registry=SourceRegistry(db),
        settings=resolved,
    )


def context_of(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.context
    return ctx


async def tenant_from_key(request: Request, x_api_key: str = Header(default="")) -> str:
    """Compared with `compare_digest`, and the key maps to a tenant that the
    request body cannot override."""
    keys = context_of(request).settings.api.api_keys
    for candidate, tenant in keys.items():
        if secrets.compare_digest(x_api_key, candidate):
            return tenant
    raise HTTPException(status_code=401, detail="missing or invalid API key")
