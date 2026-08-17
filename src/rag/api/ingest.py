"""URL and file ingestion with a stage trace.

This lives in the API process for one concrete reason: Qdrant runs in process
and is single writer, so the collection is held by whichever process opened it.
While the API is up, an ingest has to happen here or not at all.

The trace is the point. Every stage reports what it decided and how long it
took, so the pipeline is something a reviewer watches rather than something the
docs assert.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.api.models import (
    ChunkPreview,
    IngestFailure,
    IngestTraceResponse,
    IngestUrlRequest,
    PipelineStage,
)
from rag.config.settings import Settings
from rag.extract.protocols import (
    EmptyExtractionError,
    ParserUnavailableError,
    UnsupportedTypeError,
)
from rag.extract.router import resolve_mime
from rag.extract.service import ExtractService
from rag.extract.types import CanonicalDoc
from rag.fetch.registry import SourceRegistry
from rag.fetch.service import FetchService, UnknownSourceError, domain_of
from rag.fetch.types import FetchResult, FetchTier, Source, SourceStatus
from rag.index.pipeline import IngestPipeline, IngestResult, StageTiming
from rag.index.types import Chunk
from rag.log import get_logger

log = get_logger(__name__)

# A preview, not the corpus. Returning every chunk of a 900 page PDF over JSON
# would be the wrong default and the UI cannot render it anyway.
PREVIEW_LIMIT = 12
PREVIEW_CHARS = 1100

UPLOAD_SOURCE = "upload"
UPLOAD_DOMAIN = "upload.local"
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

TIER_NOTES: dict[FetchTier, str] = {
    FetchTier.STATIC: "curl_cffi with TLS impersonation, no browser started",
    FetchTier.BROWSER: "escalated to Playwright and Chromium, the page needed JS",
    FetchTier.STEALTH: "escalated to Camoufox, plain Chromium was fingerprinted",
    FetchTier.UNLOCKER: "managed unlocker, off unless a source opts in",
}

STAGE_NOTES: dict[str, str] = {
    "dedup": "checked before embedding, so a duplicate costs nothing to reject",
    "chunk": "structure aware, section paths kept, tables never split",
    "embed": "dense and sparse vectors in one pass",
    "store": "one point per chunk, both vectors on the same point",
}


@dataclass
class IngestDependencies:
    fetch: FetchService
    extract: ExtractService
    pipeline: IngestPipeline
    registry: SourceRegistry
    settings: Settings


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes
    content_type: str


@dataclass
class _Trace:
    """Carries the stage list and the wall clock together, so the helpers below
    stay inside the five argument limit."""

    began: float
    source_id: str = ""
    source_url: str = ""
    stages: list[PipelineStage] = field(default_factory=list)

    def add(self, stage: PipelineStage) -> None:
        self.stages.append(stage)

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self.began) * 1000)


class _StageStoppedError(Exception):
    """Raised by a stage that cannot hand anything to the next one.

    Caught once, at the entry point, and turned into a 200 with the failure
    named in the trace. A blocked fetch is a result, not a server error.
    """

    def __init__(self, stage: str, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.failure = IngestFailure(stage=stage, reason=reason, detail=detail)


async def ingest_url(
    request: IngestUrlRequest, deps: IngestDependencies
) -> IngestTraceResponse:
    trace = _Trace(time.perf_counter(), source_url=request.url)
    try:
        source = await _source_for_url(request, deps)
        trace.source_id = source.source_id
        result = await _fetch(request.url, deps, trace)
        trace.source_url = result.final_url
        doc = await _extract(
            result.content, result.final_url, result.content_type, deps, trace
        )
        return await _index(doc, source.source_id, int(result.tier_used), deps, trace)
    except _StageStoppedError as stop:
        return _stopped(stop, trace)


async def ingest_upload(
    upload: UploadedFile, deps: IngestDependencies
) -> IngestTraceResponse:
    """The same path as a crawled page, minus the fetch. Extraction takes bytes
    and a URL and cannot tell which one produced them."""
    trace = _Trace(time.perf_counter(), source_id=UPLOAD_SOURCE)
    try:
        await _ensure_upload_source(deps)
        trace.source_url = upload_url(upload.filename, upload.content)
        trace.add(_upload_stage(upload, trace.source_url))
        doc = await _extract(
            upload.content, trace.source_url, upload.content_type, deps, trace
        )
        return await _index(doc, UPLOAD_SOURCE, 0, deps, trace)
    except _StageStoppedError as stop:
        return _stopped(stop, trace)


def upload_url(filename: str, content: bytes) -> str:
    """Synthetic provenance. An uploaded file has no URL and a citation needs one.

    The content digest is in the path, so the same file uploaded twice resolves
    to the same `doc_id` and meets dedup instead of duplicating the corpus.
    """
    digest = hashlib.sha256(content).hexdigest()[:8]
    name = _UNSAFE_NAME.sub("-", Path(filename).name) or "file"
    return f"upload://{UPLOAD_DOMAIN}/{name}/{digest}"


def _upload_stage(upload: UploadedFile, url: str) -> PipelineStage:
    """No `latency_ms`: nothing was measured here. Bytes arrived."""
    return PipelineStage(
        name="upload",
        status="ok",
        detail={
            "filename": upload.filename,
            "bytes": len(upload.content),
            "declared_content_type": upload.content_type,
            "synthetic_url": url,
        },
        note="no fetch ran. Downstream, a file and a scraped page are the same bytes",
    )


async def _ensure_upload_source(deps: IngestDependencies) -> None:
    """`document.source_id` is a foreign key, so an upload needs a source row.

    Paused, so the scheduler never tries to crawl a synthetic domain.
    """
    if await deps.registry.get(UPLOAD_SOURCE) is not None:
        return
    await deps.registry.upsert(
        Source(
            source_id=UPLOAD_SOURCE,
            domain=UPLOAD_DOMAIN,
            seed_urls=[],
            status=SourceStatus.PAUSED,
            tos_note="synthetic source for uploaded files, never crawled",
        )
    )


async def _source_for_url(
    request: IngestUrlRequest, deps: IngestDependencies
) -> Source:
    if request.source_id is not None:
        named = await deps.registry.get(request.source_id)
        if named is None:
            raise _StageStoppedError(
                "fetch",
                "unknown_source",
                f"no source {request.source_id} is registered",
            )
        return named
    domain = domain_of(request.url)
    existing = await deps.registry.by_domain(domain)
    if existing is not None:
        return await _reconcile_unlocker(existing, request.allow_unlocker, deps)
    if not request.register_domain:
        raise _StageStoppedError(
            "fetch",
            "unknown_source",
            f"{domain} is not in the source registry. Seeding a domain is a "
            "deliberate decision, see config/sources.yaml",
        )
    return await _register(domain, request.allow_unlocker, deps)


async def _register(
    domain: str, allow_unlocker: bool, deps: IngestDependencies
) -> Source:
    """Registering is the deliberate act the caller opted into.

    It does not relax anything else on its own: robots.txt and the rate limiter
    are enforced by the fetch service and are not reachable here. The unlocker
    ban is the one thing this endpoint can lift, and only when `allow_unlocker`
    was set on this same request, never as a side effect of registration alone.
    """
    source = Source(
        source_id=f"adhoc-{domain}",
        domain=domain,
        seed_urls=[],
        status=SourceStatus.ACTIVE,
        max_tier=FetchTier.UNLOCKER if allow_unlocker else FetchTier.STEALTH,
        allow_unlocker=allow_unlocker,
        tos_note="registered at request time via POST /ingest/url",
    )
    await deps.registry.upsert(source)
    log.info(
        "source registered on request",
        source_id=source.source_id,
        domain=domain,
        allow_unlocker=allow_unlocker,
    )
    return source


async def _reconcile_unlocker(
    source: Source, allow_unlocker: bool, deps: IngestDependencies
) -> Source:
    """A caller can ask for the unlocker on a domain that is already registered
    without it. Upgrading is one directional and per request: asking for it
    turns it on, leaving the flag unset never turns it off, since a source that
    already earned tier 4 on a previous request should not silently lose it
    because a later request forgot to ask again."""
    if not allow_unlocker or source.allow_unlocker:
        return source
    upgraded = source.model_copy(
        update={"allow_unlocker": True, "max_tier": FetchTier.UNLOCKER}
    )
    await deps.registry.upsert(upgraded)
    log.info("unlocker enabled on request", source_id=source.source_id)
    return upgraded


async def _fetch(url: str, deps: IngestDependencies, trace: _Trace) -> FetchResult:
    began = time.perf_counter()
    try:
        outcome = await deps.fetch.fetch(url)
    except UnknownSourceError as exc:
        raise _StageStoppedError("fetch", "unknown_source", str(exc)) from exc
    latency = round((time.perf_counter() - began) * 1000)
    if not isinstance(outcome, FetchResult):
        trace.add(
            PipelineStage(
                name="fetch",
                status="failed",
                latency_ms=latency,
                detail={
                    "tier": int(outcome.last_tier),
                    "attempts": outcome.attempts,
                    "reason": str(outcome.reason),
                },
                note=outcome.detail,
            )
        )
        raise _StageStoppedError("fetch", str(outcome.reason), outcome.detail)
    trace.add(_fetch_stage(outcome, latency))
    return outcome


def _fetch_stage(result: FetchResult, latency: int) -> PipelineStage:
    return PipelineStage(
        name="fetch",
        status="ok",
        latency_ms=latency,
        detail={
            "tier": int(result.tier_used),
            "tier_name": result.tier_used.name.lower(),
            "http_status": result.status,
            "attempts": result.attempts,
            "bytes": len(result.content),
            "content_type": result.content_type,
            "final_url": result.final_url,
        },
        note=TIER_NOTES[result.tier_used],
    )


async def _extract(
    content: bytes,
    url: str,
    content_type: str,
    deps: IngestDependencies,
    trace: _Trace,
) -> CanonicalDoc:
    began = time.perf_counter()
    mime = resolve_mime(content_type, content)
    try:
        doc = await deps.extract.extract(content, url, content_type)
    except (UnsupportedTypeError, EmptyExtractionError, ParserUnavailableError) as exc:
        latency = round((time.perf_counter() - began) * 1000)
        trace.add(
            PipelineStage(
                name="extract",
                status="failed",
                latency_ms=latency,
                detail={"mime": mime, "declared_content_type": content_type},
                note=str(exc),
            )
        )
        raise _StageStoppedError("extract", type(exc).__name__, str(exc)) from exc
    trace.add(_extract_stage(doc, mime, content_type, began))
    return doc


def _extract_stage(
    doc: CanonicalDoc, mime: str, declared: str, began: float
) -> PipelineStage:
    return PipelineStage(
        name="extract",
        status="ok",
        latency_ms=round((time.perf_counter() - began) * 1000),
        detail={
            "mime": mime,
            "declared_content_type": declared,
            "parser": doc.extractor_name,
            "parser_version": doc.extractor_version,
            "doc_type": str(doc.doc_type),
            "blocks": len(doc.blocks),
            "block_types": _block_counts(doc),
            "chars": len(doc.text),
            "title": doc.title,
        },
        note=_mime_note(mime, declared),
    )


def _mime_note(mime: str, declared: str) -> str:
    """A corrected type is worth saying out loud. Routing on what the caller
    claimed is how a renamed PDF reaches the HTML parser."""
    stated = declared.split(";", maxsplit=1)[0].strip().lower()
    if not stated:
        return f"nothing was declared, routed as {mime} on the magic bytes"
    if stated != mime:
        return f"declared {stated}, routed as {mime} on the magic bytes"
    return "declared type and magic bytes agree"


def _block_counts(doc: CanonicalDoc) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in doc.blocks:
        counts[str(block.type)] = counts.get(str(block.type), 0) + 1
    return counts


async def _index(
    doc: CanonicalDoc,
    source_id: str,
    tier: int,
    deps: IngestDependencies,
    trace: _Trace,
) -> IngestTraceResponse:
    result = await deps.pipeline.ingest(doc, source_id, tier)
    for timing in result.stages:
        trace.add(_pipeline_stage(timing, result, deps.settings))
    log.info(
        "ingested",
        doc_id=result.doc_id,
        chunks=result.chunks_written,
        vectors=result.vectors_written,
        skipped=result.skipped_reason,
    )
    return IngestTraceResponse(
        ok=True,
        source_id=source_id,
        source_url=doc.source_url,
        doc_id=result.doc_id,
        doc_type=str(doc.doc_type),
        title=doc.title,
        stages=trace.stages,
        chunks_written=result.chunks_written,
        vectors_written=result.vectors_written,
        skipped_reason=result.skipped_reason,
        chunk_preview=_preview(result.chunks),
        latency_ms=trace.elapsed_ms(),
    )


def _pipeline_stage(
    timing: StageTiming, result: IngestResult, settings: Settings
) -> PipelineStage:
    if timing.stage == "dedup" and result.skipped:
        return PipelineStage(
            name="dedup",
            status="skipped",
            latency_ms=timing.ms,
            note=result.skipped_reason or "",
        )
    return PipelineStage(
        name=timing.stage,
        status="ok",
        latency_ms=timing.ms,
        detail=_stage_detail(timing.stage, result, settings),
        note=STAGE_NOTES[timing.stage],
    )


def _stage_detail(
    stage: str, result: IngestResult, settings: Settings
) -> dict[str, Any]:
    if stage == "chunk":
        return _chunk_detail(result, settings)
    if stage == "embed":
        return {
            "model": settings.index.embed_model,
            "dims": settings.index.embed_dims,
            "batch_size": settings.index.embed_batch_size,
            "chunks": result.chunks_written,
        }
    if stage == "store":
        return {
            "vectors": result.vectors_written,
            "collection": settings.qdrant.collection,
            "mode": "in process" if settings.qdrant.path else "server",
        }
    return {"duplicate": False}


def _chunk_detail(result: IngestResult, settings: Settings) -> dict[str, Any]:
    tokens = [chunk.token_count for chunk in result.chunks]
    return {
        "chunks": len(result.chunks),
        "target_tokens": settings.index.target_tokens,
        "tables": sum(1 for chunk in result.chunks if chunk.metadata.is_table),
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "mean_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
    }


def _preview(chunks: tuple[Chunk, ...]) -> list[ChunkPreview]:
    return [
        ChunkPreview(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            section_path=chunk.metadata.section_path,
            token_count=chunk.token_count,
            is_table=chunk.metadata.is_table,
            page_no=chunk.metadata.page_no,
            text=chunk.text[:PREVIEW_CHARS],
            truncated=len(chunk.text) > PREVIEW_CHARS,
        )
        for chunk in chunks[:PREVIEW_LIMIT]
    ]


def _stopped(stop: _StageStoppedError, trace: _Trace) -> IngestTraceResponse:
    log.warning(
        "ingest stopped",
        stage=stop.failure.stage,
        reason=stop.failure.reason,
        url=trace.source_url,
    )
    return IngestTraceResponse(
        ok=False,
        source_id=trace.source_id,
        source_url=trace.source_url,
        stages=trace.stages,
        failure=stop.failure,
        latency_ms=trace.elapsed_ms(),
    )
