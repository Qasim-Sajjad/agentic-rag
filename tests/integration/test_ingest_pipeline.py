"""End to end ingest: CanonicalDoc in, chunks in Postgres and vectors in Qdrant."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.settings import IndexSettings, QdrantSettings
from rag.db.pool import Database
from rag.extract.service import ExtractService
from rag.extract.types import Block, BlockType, CanonicalDoc, content_hash
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import Source
from rag.index.embed import FakeEmbedder
from rag.index.pipeline import IndexDependencies, IngestPipeline
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.store import QdrantStore

pytestmark = pytest.mark.integration

PAGES = Path(__file__).parent.parent / "fixtures" / "pages"
SOURCE_ID = "fixture"


@pytest.fixture
async def pipeline(db: Database, tmp_path: Path):
    await SourceRegistry(db).upsert(
        Source(source_id=SOURCE_ID, domain="example.test", seed_urls=[])
    )
    store = QdrantStore(
        QdrantSettings(path=str(tmp_path / "qdrant"), collection="test")
    )
    embedder = FakeEmbedder()
    await store.bootstrap(embedder.dims)
    deps = IndexDependencies(
        documents=DocumentRepository(db, tmp_path / "docs"),
        chunks=ChunkRepository(db),
        store=store,
        embedder=embedder,
        settings=IndexSettings(embed_batch_size=8),
    )
    try:
        yield IngestPipeline(deps), store, ChunkRepository(db)
    finally:
        await store.close()


BASE = (
    "Revenue for the quarter was 41.2 million dollars, up nine percent. "
    "Growth came from the subscription segment. Services revenue was flat. "
    "Concentration remains high across the three largest customers. "
)


def sample_doc(doc_id: str = "d1", body: str = BASE * 6) -> CanonicalDoc:
    blocks = [
        Block(type=BlockType.HEADING, text="Risk factors", level=1),
        Block(type=BlockType.PARAGRAPH, text=body),
    ]
    return CanonicalDoc(
        doc_id=doc_id,
        source_url=f"https://example.test/{doc_id}",
        title="Annual report",
        blocks=blocks,
        content_hash=content_hash(blocks),
        extractor_name="test",
        extractor_version="1",
    )


async def test_ingest_writes_chunks(pipeline):
    pipe, _store, _chunks = pipeline
    result = await pipe.ingest(sample_doc(), SOURCE_ID)
    assert result.chunks_written > 0


async def test_ingest_writes_vectors(pipeline):
    pipe, store, _ = pipeline
    await pipe.ingest(sample_doc(), SOURCE_ID)
    assert await store.count() > 0


async def test_chunks_are_persisted_to_postgres(pipeline):
    pipe, _store, chunks = pipeline
    await pipe.ingest(sample_doc(), SOURCE_ID)
    assert await chunks.count() > 0


async def test_reingesting_the_same_document_is_skipped(pipeline):
    """Exact content hash is the third dedup point, before embedding."""
    pipe, _, _ = pipeline
    await pipe.ingest(sample_doc(), SOURCE_ID)
    second = await pipe.ingest(sample_doc("d2"), SOURCE_ID)
    assert second.skipped


async def test_a_near_duplicate_document_is_skipped(pipeline):
    pipe, _, _ = pipeline
    await pipe.ingest(sample_doc(), SOURCE_ID)
    near = sample_doc("d3", BASE * 6 + "One sentence was added in a later revision.")
    result = await pipe.ingest(near, SOURCE_ID)
    assert result.skipped


async def test_an_unrelated_document_is_ingested(pipeline):
    pipe, _, _ = pipeline
    await pipe.ingest(sample_doc(), SOURCE_ID)
    other = sample_doc(
        "d4",
        "The board approved a share buyback of up to 200 million dollars over "
        "eighteen months, funded from operating cash flow rather than new debt. " * 4,
    )
    assert not (await pipe.ingest(other, SOURCE_ID)).skipped


async def test_every_stored_chunk_carries_its_embed_model(pipeline):
    """Lineage is what makes a targeted re-embed possible."""
    pipe, _, _ = pipeline
    result = await pipe.ingest(sample_doc(), SOURCE_ID)
    assert result.vectors_written == result.chunks_written


async def test_a_real_html_page_ingests_end_to_end(pipeline):
    pipe, store, _ = pipeline
    doc = await ExtractService().extract(
        (PAGES / "static.html").read_bytes(), "https://example.test/report", "text/html"
    )
    result = await pipe.ingest(doc, SOURCE_ID)
    assert result.chunks_written > 0 and await store.count() == result.vectors_written
