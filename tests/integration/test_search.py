"""Search against a real ingested index, ingest through retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.settings import IndexSettings, QdrantSettings, RetrieveSettings
from rag.db.pool import Database
from rag.extract.types import Block, BlockType, CanonicalDoc, content_hash
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import Source
from rag.index.embed import FakeEmbedder
from rag.index.pipeline import IndexDependencies, IngestPipeline
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.store import QdrantStore
from rag.retrieve.rerank import IdentityReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import SearchFilters

pytestmark = pytest.mark.integration

SOURCE_ID = "fixture"

DOCS = {
    "cyber": (
        "Cybersecurity incidents",
        "In March 2024 we disclosed an unauthorised access incident affecting "
        "an internal reporting system. No customer data was exfiltrated and the "
        "incident was contained within 48 hours by the security operations team.",
    ),
    "buyback": (
        "Capital returns",
        "The board approved a share buyback of up to 200 million dollars over "
        "eighteen months, funded from operating cash flow rather than new debt.",
    ),
    "segments": (
        "Segment performance",
        "Subscription revenue was 26.0 million dollars and services revenue was "
        "15.2 million dollars, so subscription is now 63 percent of the total.",
    ),
}


def make_doc(key: str) -> CanonicalDoc:
    heading, body = DOCS[key]
    blocks = [
        Block(type=BlockType.HEADING, text=heading, level=1),
        Block(type=BlockType.PARAGRAPH, text=body),
    ]
    return CanonicalDoc(
        doc_id=key,
        source_url=f"https://example.test/{key}",
        title="Annual report",
        blocks=blocks,
        content_hash=content_hash(blocks),
        extractor_name="test",
        extractor_version="1",
    )


@pytest.fixture
async def search(db: Database, tmp_path: Path):
    await SourceRegistry(db).upsert(
        Source(source_id=SOURCE_ID, domain="example.test", seed_urls=[])
    )
    qdrant = QdrantSettings(path=str(tmp_path / "qdrant"), collection="test")
    store = QdrantStore(qdrant)
    embedder = FakeEmbedder()
    await store.bootstrap(embedder.dims)
    pipeline = IngestPipeline(
        IndexDependencies(
            documents=DocumentRepository(db, tmp_path / "docs"),
            chunks=ChunkRepository(db),
            store=store,
            embedder=embedder,
            settings=IndexSettings(),
        )
    )
    for key in DOCS:
        await pipeline.ingest(make_doc(key), SOURCE_ID)
    service = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=IdentityReranker(),
            settings=RetrieveSettings(score_floor=0.0, low_floor=0.0, k_min=1),
            qdrant=qdrant,
        )
    )
    try:
        yield service
    finally:
        await store.close()


async def test_search_returns_chunks(search: SearchService):
    result = await search.search("cybersecurity incident disclosed")
    assert result.chunks


async def test_search_reports_how_many_it_used(search: SearchService):
    result = await search.search("share buyback")
    assert result.k_used == len(result.chunks)


async def test_search_records_latency(search: SearchService):
    result = await search.search("segment revenue")
    assert result.latency_ms >= 0


async def test_results_carry_their_source_url(search: SearchService):
    result = await search.search("share buyback")
    prefix = "https://example.test/"
    assert all(chunk.source_url.startswith(prefix) for chunk in result.chunks)


async def test_results_carry_their_section_path(search: SearchService):
    result = await search.search("share buyback")
    assert all(chunk.section_path for chunk in result.chunks)


async def test_a_source_filter_excludes_everything_else(search: SearchService):
    result = await search.search("revenue", SearchFilters(source_id="nonexistent"))
    assert result.chunks == []


async def test_a_matching_source_filter_still_returns_results(search: SearchService):
    result = await search.search("revenue", SearchFilters(source_id=SOURCE_ID))
    assert result.chunks
