"""FastAPI surface. Separated by responsibility, not merged for convenience.

Four read endpoints: `/search` is retrieval with no LLM, `/ask` is single shot
RAG, `/agent` routes through LangGraph and MCP, `/ingest/status` reports scrape
state. Collapsing them would hide which layer failed and make it impossible to
benchmark retrieval independently of generation.

Two write endpoints, `/ingest/url` and `/ingest/file`, exist because Qdrant runs
in process and is single writer: the collection belongs to whichever process
opened it, so an ingest has to run here while the API is up.

    uvicorn rag.api.main:app --port 8000
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse

from rag.api.ask import AskDependencies, ask
from rag.api.deps import (
    AppContext,
    build_context,
    cache_key,
    context_of,
    tenant_from_key,
)
from rag.api.ingest import UploadedFile, ingest_upload, ingest_url
from rag.api.jobs import IngestJob, launch, status_of
from rag.api.models import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    ErrorBody,
    ErrorResponse,
    IngestJobAccepted,
    IngestJobList,
    IngestJobStatus,
    IngestStatusResponse,
    IngestSummary,
    IngestTraceResponse,
    IngestUrlRequest,
    SearchRequest,
    SearchResponse,
    SourceStatusRow,
)
from rag.db.pool import Database
from rag.fetch.factory import close_fetchers
from rag.log import configure_logging, get_logger
from rag.mcp.schemas import IngestStatusInput
from rag.retrieve.types import SearchFilters

log = get_logger(__name__)

Tenant = Annotated[str, Depends(tenant_from_key)]
# Both ingest endpoints answer either way: the whole trace when the caller
# waited, an id to poll when it asked not to.
IngestUrlResult = IngestTraceResponse | IngestJobAccepted
# Annotated rather than a `File(...)` default, which would be a function call in
# a default argument.
Upload = Annotated[UploadFile, File()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    context: AppContext = getattr(app.state, "context", None) or build_context()
    app.state.context = context
    await context.db.connect()
    await context.store.bootstrap(context.embedder.dims)
    yield
    await close_fetchers(context.fetchers)
    await context.store.close()
    await context.db.close()


def create_app(context: AppContext | None = None) -> FastAPI:
    app = FastAPI(title="agentic-rag", lifespan=lifespan)
    if context is not None:
        app.state.context = context
    _register_query(app)
    _register_ingest(app)
    _register_jobs(app)
    _register_errors(app)
    return app


def _register_query(app: FastAPI) -> None:
    """The read path. No endpoint here writes to the corpus."""

    @app.post("/search", response_model=SearchResponse)
    async def search(
        request: SearchRequest, http: Request, tenant: Tenant
    ) -> SearchResponse:
        """Pure retrieval. No LLM in the path, which is what lets retrieval be
        benchmarked on its own."""
        ctx = context_of(http)
        filters = (request.filters or SearchFilters()).model_copy(
            update={"tenant_id": tenant}
        )
        result = await ctx.search.search(request.query, filters, request.top_k)
        return SearchResponse(
            chunks=result.chunks,
            confidence=result.confidence,
            k_used=result.k_used,
            steps=result.steps,
            latency_ms=result.latency_ms,
        )

    @app.post("/ask", response_model=AskResponse)
    async def ask_endpoint(
        request: AskRequest, http: Request, tenant: Tenant
    ) -> AskResponse:
        ctx = context_of(http)
        key = _ask_cache_key(ctx, request, tenant)
        cached = None if request.explain else ctx.response_cache.get(key)
        if cached is not None:
            return AskResponse.model_validate(cached)
        response = await ask(
            request,
            AskDependencies(ctx.search, ctx.llm, ctx.prompts, ctx.settings),
            tenant,
        )
        if not request.explain:
            ctx.response_cache.put(key, response.model_dump())
        return response

    @app.post("/agent", response_model=AgentResponse)
    async def agent(
        request: AgentRequest, http: Request, tenant: Tenant
    ) -> AgentResponse:
        """Not cached. Tool state changes between calls, so a cached agent
        response can be wrong in a way a cached search result cannot."""
        result = await context_of(http).agent.run(request.question, tenant)
        return AgentResponse(
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            trace=result.trace,
            chunks=result.chunks,
        )


def _register_ingest(app: FastAPI) -> None:
    """The write path plus the status read, which reports on it."""

    @app.post("/ingest/url", response_model=IngestUrlResult)
    async def ingest_from_url(
        request: IngestUrlRequest,
        http: Request,
        tenant: Tenant,
        response: Response,
        background: bool = False,
    ) -> IngestUrlResult:
        """Fetch, extract, chunk, embed and store one URL, reporting each stage.

        Lives here rather than in a separate process because Qdrant in process
        is single writer: the collection is held by whoever opened it.

        `background=true` returns 202 and a job id instead of holding the
        connection open. A large document is minutes of work, which is a client
        timeout rather than a response.
        """
        ctx = context_of(http)
        if not background:
            return await ingest_url(request, ctx.ingest)
        job = launch(
            ctx.jobs,
            "url",
            request.url,
            lambda report: ingest_url(request, ctx.ingest, report),
        )
        return _accepted(job, response)

    @app.post("/ingest/file", response_model=IngestUrlResult)
    async def ingest_from_file(
        http: Request,
        tenant: Tenant,
        file: Upload,
        response: Response,
        background: bool = False,
    ) -> IngestUrlResult:
        """The same path minus the fetch. A PDF, DOCX, XLSX, CSV or text file."""
        ctx = context_of(http)
        upload = UploadedFile(
            filename=file.filename or "upload",
            content=await file.read(),
            content_type=file.content_type or "",
        )
        if not background:
            return await ingest_upload(upload, ctx.ingest)
        job = launch(
            ctx.jobs,
            "file",
            upload.filename,
            lambda report: ingest_upload(upload, ctx.ingest, report),
        )
        return _accepted(job, response)

    @app.get("/ingest/status", response_model=IngestStatusResponse)
    async def ingest_status(
        http: Request,
        tenant: Tenant,
        source_id: str | None = None,
        domain: str | None = None,
    ) -> IngestStatusResponse:
        ctx = context_of(http)
        everything = [
            await _one_status(ctx, source.source_id, None)
            for source in await ctx.registry.list_all()
        ]
        rows = await _filtered(ctx, everything, source_id, domain)
        # The summary always describes the corpus, never the filtered rows.
        # Reporting total_sources as the number of rows returned makes a
        # filtered request look like a one source corpus.
        return IngestStatusResponse(sources=rows, summary=_summarise(everything))


def _register_jobs(app: FastAPI) -> None:
    """Polling for a background ingest. Read only: these never start work."""

    @app.get("/ingest/jobs", response_model=IngestJobList)
    async def list_jobs(
        http: Request, tenant: Tenant, limit: int = 20
    ) -> IngestJobList:
        jobs = context_of(http).jobs.recent(limit)
        return IngestJobList(jobs=[status_of(job) for job in jobs])

    @app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatus)
    async def job_status(job_id: str, http: Request, tenant: Tenant) -> IngestJobStatus:
        """404 covers both never existed and evicted. The store is bounded, so a
        client that stops polling and comes back much later gets the same answer
        either way."""
        job = context_of(http).jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return status_of(job)


def _accepted(job: IngestJob, response: Response) -> IngestJobAccepted:
    """202, not 200. The work was accepted, it has not happened yet."""
    response.status_code = 202
    return IngestJobAccepted(
        job_id=job.job_id,
        status=job.status,
        poll=f"/ingest/jobs/{job.job_id}",
    )


def _ask_cache_key(ctx: AppContext, request: AskRequest, tenant: str) -> str:
    """`explain` is part of the key and bypasses the cache on read, because the
    nonce and the strip log describe one request and not the next one."""
    return cache_key(
        question=request.question,
        filters=request.filters.model_dump() if request.filters else None,
        tenant=tenant,
        prompt_version=ctx.prompts.get("rag_answer").content_hash,
        embed_model_version=ctx.embedder.model_name,
        explain=request.explain,
    )


async def _one_status(
    ctx: AppContext, source_id: str | None, domain: str | None
) -> SourceStatusRow:
    status = await ctx.tools.get_ingest_status(
        IngestStatusInput(source_id=source_id, domain=domain),
        session_id=f"api-{uuid.uuid4()}",
    )
    return SourceStatusRow(
        source_id=status.source_id,
        status=status.status,
        circuit_state=status.circuit_state,
        last_success_at=status.last_success_at,
        last_failure_reason=(
            str(status.last_failure_reason) if status.last_failure_reason else None
        ),
        docs_indexed=status.docs_indexed,
        docs_failed=status.docs_failed,
        pending=status.pending,
        in_flight=status.in_flight,
        requeued=status.requeued,
        coverage_note=status.coverage_note,
    )


async def _filtered(
    ctx: AppContext,
    rows: list[SourceStatusRow],
    source_id: str | None,
    domain: str | None,
) -> list[SourceStatusRow]:
    """A filter narrows which sources are listed, never what the summary counts."""
    wanted = source_id
    if wanted is None and domain is not None:
        source = await ctx.registry.by_domain(domain)
        wanted = source.source_id if source is not None else domain
    if wanted is None:
        return rows
    return [row for row in rows if row.source_id == wanted]


def _summarise(rows: list[SourceStatusRow]) -> IngestSummary:
    return IngestSummary(
        total_sources=len(rows),
        healthy=sum(1 for row in rows if row.status == "healthy"),
        degraded=sum(1 for row in rows if row.status == "degraded"),
        unreachable=sum(1 for row in rows if row.status == "unreachable"),
    )


def _register_errors(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Typed reason code, never a stack trace."""
        request_id = str(uuid.uuid4())
        log.error("unhandled error", request_id=request_id, error=str(exc))
        body = ErrorResponse(
            error=ErrorBody(
                code="internal_error",
                message="the request could not be completed",
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump())


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("rag.api.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()


__all__ = ["Database", "app", "create_app"]
