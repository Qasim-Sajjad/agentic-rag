"""FastMCP server. Streamable HTTP, own process.

    python -m rag.mcp.server

Not stdio: the FastAPI `/agent` endpoint would otherwise spawn a subprocess per
request.
"""

from __future__ import annotations

from rag.clock import SystemClock
from rag.config.settings import Settings, get_settings
from rag.db.pool import Database
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.registry import SourceRegistry
from rag.index.embed import Embedder, build_embedder
from rag.index.store import QdrantStore
from rag.log import configure_logging, get_logger
from rag.mcp.schemas import (
    IngestStatusInput,
    IngestStatusOutput,
    SearchCorpusInput,
    SearchCorpusOutput,
)
from rag.mcp.tools import ToolDependencies, ToolService
from rag.retrieve.rerank import MiniLMReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import Reranker

log = get_logger(__name__)

SERVER_NAME = "agentic-rag"


def build_tool_service(db: Database, settings: Settings | None = None) -> ToolService:
    resolved = settings if settings is not None else get_settings()
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
    return ToolService(
        ToolDependencies(
            search=search,
            registry=SourceRegistry(db),
            dead_letter=DeadLetterStore(db),
            settings=resolved.mcp,
            clock=SystemClock(),
        )
    )


def build_server(service: ToolService, tenant_id: str = "default"):  # type: ignore[no-untyped-def]
    """Registers both tools. Tenant comes from here, never from the caller.

    `MCPServer` is what the SPEC calls `FastMCP`. The official SDK renamed the
    class in 2.0 and this repo pins 2.x.
    """
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(SERVER_NAME)

    @server.tool()
    async def search_corpus(request: SearchCorpusInput) -> SearchCorpusOutput:
        """Search the indexed corpus and return matching chunks.

        Returns source text, not an answer. Use it whenever a question needs
        facts from the document corpus.
        """
        return await service.search_corpus(request, tenant_id)

    @server.tool()
    async def get_ingest_status(request: IngestStatusInput) -> IngestStatusOutput:
        """Report whether a source is reachable and how much of it is indexed.

        Use it when a search returns nothing, to tell "the corpus does not
        cover this" apart from "that source has been blocked since a date".
        """
        return await service.get_ingest_status(request)

    return server


async def _serve() -> None:
    settings = get_settings()
    db = Database(settings.postgres)
    await db.connect()
    service = build_tool_service(db, settings)
    server = build_server(service)
    log.info("mcp server starting", port=settings.mcp.port)
    await server.run_streamable_http_async()


def main() -> None:
    import asyncio

    configure_logging()
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
