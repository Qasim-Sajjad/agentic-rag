"""Demo entry point. Runs one URL or one local snippet through the pipeline.

    python -m rag.demo ingest <url> [--source-id ID]
    python -m rag.demo ingest-snippet <file> [--url URL] [--content-type TYPE]
    python -m rag.demo status

Exists for live demonstration and manual verification. It prints the decision
made at each stage, which is the part a reviewer wants to see.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag.clock import SystemClock
from rag.config.settings import Settings, get_settings
from rag.crawl import CrawlDependencies, Crawler
from rag.db.migrate import apply_migrations
from rag.db.pool import Database
from rag.extract.router import resolve_mime
from rag.extract.service import ExtractService
from rag.extract.types import CanonicalDoc
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.factory import build_fetchers, build_service, close_fetchers
from rag.fetch.frontier import Frontier
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import FetchResult, Source, SourceStatus
from rag.fetch.worker import FetchWorker, WorkerDeps
from rag.index.chunker import StructureAwareChunker
from rag.index.embed import build_embedder
from rag.index.pipeline import (
    IndexDependencies,
    IngestPipeline,
    IngestResult,
    chunk_metadata,
)
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.store import QdrantStore
from rag.index.types import ChunkMetadata
from rag.log import configure_logging, get_logger

log = get_logger("demo")

DEMO_SOURCE = "demo"
SNIPPET_SCHEME = "snippet://demo"


@dataclass
class Harness:
    db: Database
    settings: Settings
    pipeline: IngestPipeline
    store: QdrantStore


def snippet_url(content: bytes) -> str:
    """Synthetic provenance. A pasted snippet has none, and citations need one."""
    digest = hashlib.sha256(content).hexdigest()[:8]
    return f"{SNIPPET_SCHEME}/{digest}"


async def build_harness() -> Harness:
    settings = get_settings()
    db = Database(settings.postgres)
    await db.connect()
    await apply_migrations(db)
    embedder = build_embedder(settings.index)
    store = QdrantStore(settings.qdrant)
    await store.bootstrap(embedder.dims)
    deps = IndexDependencies(
        documents=DocumentRepository(db, Path("data/docs")),
        chunks=ChunkRepository(db),
        store=store,
        embedder=embedder,
        settings=settings.index,
    )
    return Harness(db, settings, IngestPipeline(deps), store)


async def ensure_demo_source(db: Database) -> None:
    """`document.source_id` is a foreign key, so a snippet needs a source row.

    Paused, so the scheduler never tries to crawl a synthetic domain.
    """
    await SourceRegistry(db).upsert(
        Source(
            source_id=DEMO_SOURCE,
            domain="demo.local",
            seed_urls=[],
            status=SourceStatus.PAUSED,
            tos_note="synthetic source for pasted snippets, never crawled",
        )
    )


def report(doc: CanonicalDoc, result: IngestResult) -> None:
    kinds: dict[str, int] = {}
    for block in doc.blocks:
        kinds[str(block.type)] = kinds.get(str(block.type), 0) + 1
    log.info(
        "extracted",
        parser=doc.extractor_name,
        blocks=len(doc.blocks),
        block_types=kinds,
        title=doc.title,
    )
    log.info(
        "indexed",
        doc_id=result.doc_id,
        chunks=result.chunks_written,
        vectors=result.vectors_written,
        skipped=result.skipped_reason,
    )


async def ingest_snippet(args: argparse.Namespace) -> None:
    content = await asyncio.to_thread(Path(args.file).read_bytes)
    url = args.url or snippet_url(content)
    harness = await build_harness()
    try:
        await ensure_demo_source(harness.db)
        mime = resolve_mime(args.content_type or "", content)
        log.info("routing", url=url, mime=mime, bytes=len(content))
        doc = await ExtractService().extract(content, url, args.content_type or mime)
        result = await harness.pipeline.ingest(doc, DEMO_SOURCE, fetch_tier=0)
        report(doc, result)
        _print_sample(doc, harness)
    finally:
        await harness.store.close()
        await harness.db.close()


def _print_sample(doc: CanonicalDoc, harness: Harness) -> None:
    chunks = StructureAwareChunker(harness.settings.index).chunk(
        doc, _sample_metadata(doc)
    )
    if chunks:
        log.info(
            "sample chunk",
            section_path=" > ".join(chunks[0].metadata.section_path),
            tokens=chunks[0].token_count,
            preview=chunks[0].text[:160],
        )


def _sample_metadata(doc: CanonicalDoc) -> ChunkMetadata:
    return chunk_metadata(doc, DEMO_SOURCE, 0, get_settings().index)


async def ingest_url(args: argparse.Namespace) -> None:
    harness = await build_harness()
    fetchers = build_fetchers(harness.settings.fetch)
    service = build_service(harness.db, SystemClock(), harness.settings.fetch, fetchers)
    try:
        outcome = await service.fetch(args.url)
        if not isinstance(outcome, FetchResult):
            log.warning(
                "fetch failed", reason=str(outcome.reason), detail=outcome.detail
            )
            return
        log.info(
            "fetched",
            tier=int(outcome.tier_used),
            status=outcome.status,
            attempts=outcome.attempts,
            bytes=len(outcome.content),
        )
        doc = await ExtractService().extract(
            outcome.content, outcome.final_url, outcome.content_type
        )
        result = await harness.pipeline.ingest(
            doc, args.source_id, int(outcome.tier_used)
        )
        report(doc, result)
    finally:
        await close_fetchers(fetchers)
        await harness.store.close()
        await harness.db.close()


async def status(args: argparse.Namespace) -> None:
    harness = await build_harness()
    try:
        chunks = await ChunkRepository(harness.db).count()
        vectors = await harness.store.count()
        sources = await SourceRegistry(harness.db).list_all()
        log.info("corpus", chunks=chunks, vectors=vectors, sources=len(sources))
    finally:
        await harness.store.close()
        await harness.db.close()


async def crawl(args: argparse.Namespace) -> None:
    """Fetch, extract, index and discover, until the page budget runs out."""
    harness = await build_harness()
    fetchers = build_fetchers(harness.settings.fetch)
    service = build_service(harness.db, SystemClock(), harness.settings.fetch, fetchers)
    source = await SourceRegistry(harness.db).get(args.source_id)
    if source is None:
        log.error("unknown source", source_id=args.source_id)
        return
    worker = FetchWorker(
        WorkerDeps(
            service=service,
            frontier=Frontier(harness.db, SystemClock()),
            dead_letter=DeadLetterStore(harness.db),
            clock=SystemClock(),
            settings=harness.settings.fetch,
        ),
        name=f"crawl-{args.source_id}",
        source_id=args.source_id,
    )
    crawler = Crawler(
        CrawlDependencies(
            worker=worker,
            extract=ExtractService(),
            ingest=harness.pipeline,
            frontier=Frontier(harness.db, SystemClock()),
            dead_letter=DeadLetterStore(harness.db),
        ),
        domain=source.domain,
        max_pages=args.max_pages,
    )
    try:
        await crawler.run()
    finally:
        await close_fetchers(fetchers)
        await harness.store.close()
        await harness.db.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rag.demo")
    subs = root.add_subparsers(dest="command", required=True)

    url_cmd = subs.add_parser("ingest", help="fetch, extract, chunk, embed, upsert")
    url_cmd.add_argument("url")
    url_cmd.add_argument("--source-id", default="books-toscrape")
    url_cmd.set_defaults(run=ingest_url)

    snippet = subs.add_parser("ingest-snippet", help="same path, no fetch")
    snippet.add_argument("file")
    snippet.add_argument("--url", default=None)
    snippet.add_argument("--content-type", default=None)
    snippet.set_defaults(run=ingest_snippet)

    status_cmd = subs.add_parser("status", help="corpus counts")
    status_cmd.set_defaults(run=status)

    crawl_cmd = subs.add_parser("crawl", help="crawl a registered source")
    crawl_cmd.add_argument("source_id")
    crawl_cmd.add_argument("--max-pages", type=int, default=200)
    crawl_cmd.set_defaults(run=crawl)

    return root


def main() -> None:
    configure_logging()
    args = parser().parse_args()
    asyncio.run(args.run(args))


if __name__ == "__main__":
    main()
