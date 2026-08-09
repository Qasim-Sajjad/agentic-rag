"""Ingest pipeline. CanonicalDoc in, chunks in Postgres and vectors in Qdrant.

Dedup runs at three points, each one saving the cost of the stage after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from rag.config.settings import IndexSettings
from rag.extract.types import CanonicalDoc, normalize_text
from rag.index.chunker import StructureAwareChunker
from rag.index.embed import Embedder
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.simhash import SimHashIndex, simhash
from rag.index.store import VectorStore
from rag.index.types import Chunk, ChunkMetadata
from rag.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    chunks_written: int
    vectors_written: int
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


@dataclass
class IndexDependencies:
    documents: DocumentRepository
    chunks: ChunkRepository
    store: VectorStore
    embedder: Embedder
    settings: IndexSettings


class IngestPipeline:
    def __init__(self, deps: IndexDependencies) -> None:
        self._deps = deps
        self._chunker = StructureAwareChunker(deps.settings)
        self._near = SimHashIndex(deps.settings.simhash_hamming_threshold)

    async def ingest(
        self, doc: CanonicalDoc, source_id: str, fetch_tier: int = 0
    ) -> IngestResult:
        skip = await self._duplicate_reason(doc)
        if skip is not None:
            return IngestResult(doc.doc_id, 0, 0, skip)
        await self._deps.documents.save(doc, source_id, fetch_tier)
        chunks = await self._fresh_chunks(doc, source_id, fetch_tier)
        if not chunks:
            return IngestResult(doc.doc_id, 0, 0, "all chunks were duplicates")
        written = await self._embed_and_write(chunks)
        return IngestResult(doc.doc_id, len(chunks), written)

    async def _duplicate_reason(self, doc: CanonicalDoc) -> str | None:
        if await self._deps.documents.exists_by_content_hash(doc.content_hash):
            return "exact duplicate content hash"
        fingerprint = simhash(normalize_text(doc.text))
        near = self._near.find_duplicate(fingerprint)
        if near is not None:
            return f"near duplicate of {near}"
        self._near.add(doc.doc_id, fingerprint)
        return None

    async def _fresh_chunks(
        self, doc: CanonicalDoc, source_id: str, fetch_tier: int
    ) -> list[Chunk]:
        chunks = self._chunker.chunk(
            doc, chunk_metadata(doc, source_id, fetch_tier, self._deps.settings)
        )
        known = await self._deps.chunks.known_hashes(
            [chunk.metadata.chunk_hash for chunk in chunks]
        )
        return [chunk for chunk in chunks if chunk.metadata.chunk_hash not in known]

    async def _embed_and_write(self, chunks: list[Chunk]) -> int:
        batch_size = self._deps.settings.embed_batch_size
        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await self._deps.embedder.embed([c.embed_text for c in batch])
            stamped = [
                chunk.model_copy(
                    update={
                        "metadata": chunk.metadata.model_copy(
                            update={
                                "embed_model_version": self._deps.embedder.model_name
                            }
                        )
                    }
                )
                for chunk in batch
            ]
            await self._deps.chunks.save_many(stamped)
            written += await self._deps.store.upsert(stamped, vectors)
        return written


def chunk_metadata(
    doc: CanonicalDoc, source_id: str, fetch_tier: int, settings: IndexSettings
) -> ChunkMetadata:
    return ChunkMetadata(
        doc_type=doc.doc_type,
        domain=urlsplit(doc.source_url).netloc.lower(),
        source_id=source_id,
        published_at=doc.published_at,
        language=doc.language,
        fetch_tier=fetch_tier,
        tenant_id=settings.tenant_id,
        source_url=doc.source_url,
        title=doc.title,
        content_hash=doc.content_hash,
        extractor_name=doc.extractor_name,
        extractor_version=doc.extractor_version,
    )
