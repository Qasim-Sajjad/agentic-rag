"""Document and chunk persistence. Qdrant is a derived index, this is the truth.

Keeping `CanonicalDoc` and chunks means a model swap is a backfill rather than
a re-scrape, which is the difference between hours and weeks.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.db.pool import Database
from rag.extract.types import CanonicalDoc
from rag.index.types import Chunk


class DocumentRepository:
    def __init__(self, db: Database, doc_store: Path | None = None) -> None:
        self._db = db
        self._doc_store = doc_store

    async def exists_by_content_hash(self, content_hash: str) -> bool:
        """Third exact dedup point: identical text never gets embedded twice."""
        found = await self._db.fetchval(
            "SELECT 1 FROM document WHERE content_hash = $1", content_hash
        )
        return found is not None

    async def save(self, doc: CanonicalDoc, source_id: str, fetch_tier: int) -> str:
        key = self._write_blob(doc)
        await self._db.execute(
            """
            INSERT INTO document (doc_id, source_id, source_url, title, published_at,
                language, doc_type, content_hash, canonical_doc_key, fetch_tier,
                extractor_name, extractor_version)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (doc_id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                canonical_doc_key = EXCLUDED.canonical_doc_key,
                ingested_at = now()
            """,
            doc.doc_id,
            source_id,
            doc.source_url,
            doc.title,
            doc.published_at,
            doc.language,
            str(doc.doc_type),
            doc.content_hash,
            key,
            fetch_tier,
            doc.extractor_name,
            doc.extractor_version,
        )
        return doc.doc_id

    def _write_blob(self, doc: CanonicalDoc) -> str:
        """Object storage stand in. One JSON blob per document, immutable."""
        key = f"docs/{doc.doc_id}.json"
        if self._doc_store is None:
            return key
        path = self._doc_store / f"{doc.doc_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.model_dump_json(), encoding="utf-8")
        return key


class ChunkRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def known_hashes(self, hashes: list[str]) -> set[str]:
        """Cross document chunk dedup. Keeps five copies of one disclaimer out
        of the top 10."""
        if not hashes:
            return set()
        rows = await self._db.fetch(
            "SELECT DISTINCT chunk_hash FROM chunk WHERE chunk_hash = ANY($1::text[])",
            hashes,
        )
        return {row["chunk_hash"] for row in rows}

    async def save_many(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            await self._save_one(chunk)
        return len(chunks)

    async def _save_one(self, chunk: Chunk) -> None:
        meta = chunk.metadata
        await self._db.execute(
            """
            INSERT INTO chunk (chunk_id, doc_id, chunk_index, text, embed_text,
                section_path, page_no, is_table, token_count, chunk_hash,
                chunker_version, embed_model_version, embedded_at, tenant_id)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
                    CASE WHEN $12 IS NULL THEN NULL ELSE now() END, $13)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embed_text = EXCLUDED.embed_text,
                embed_model_version = EXCLUDED.embed_model_version,
                embedded_at = EXCLUDED.embedded_at
            """,
            chunk.chunk_id,
            chunk.doc_id,
            chunk.chunk_index,
            chunk.text,
            chunk.embed_text,
            json.dumps(meta.section_path),
            meta.page_no,
            meta.is_table,
            chunk.token_count,
            meta.chunk_hash,
            meta.chunker_version,
            meta.embed_model_version,
            meta.tenant_id,
        )

    async def count(self) -> int:
        value = await self._db.fetchval("SELECT count(*) FROM chunk")
        return int(value or 0)
