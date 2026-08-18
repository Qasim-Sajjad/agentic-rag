"""Re-embed the stored corpus into the configured collection.

Changing `index.embed_model` changes the vector width, and a collection holds
one width, so the new model needs a new collection. Everything already indexed
would otherwise be invisible: Postgres still holds the documents and their
chunks, so dedup rejects a re-ingest as a duplicate, correctly, and the vectors
never arrive.

This is the backfill `src/rag/index/SPEC.md` describes. It streams
`chunk.embed_text` out of Postgres and writes vectors to the collection named in
config. No re-scrape, no re-extract, no LLM: for 500K chunks that is the
difference between hours and weeks, and the weeks would return different content
because the web moved.

    uv run python -m rag.index.reembed

Stop the API server first. Qdrant in process is single writer.
"""

from __future__ import annotations

import asyncio
import time

from rag.config.settings import Settings, get_settings
from rag.db.pool import Database
from rag.index.embed import Embedder, build_embedder
from rag.index.repository import ChunkRepository
from rag.index.store import QdrantStore
from rag.index.types import Chunk
from rag.log import configure_logging, get_logger

log = get_logger(__name__)


async def backfill(settings: Settings | None = None) -> int:
    """Returns the number of vectors written."""
    resolved = settings if settings is not None else get_settings()
    db = Database(resolved.postgres)
    await db.connect()
    embedder = build_embedder(resolved.index)
    store = QdrantStore(resolved.qdrant)
    try:
        await store.bootstrap(embedder.dims)
        chunks = await ChunkRepository(db).all_chunks()
        written = await _write(chunks, embedder, store, resolved)
    finally:
        await store.close()
        await db.close()
    return written


async def _write(
    chunks: list[Chunk], embedder: Embedder, store: QdrantStore, settings: Settings
) -> int:
    """Batched, and it logs per batch. A backfill over a real corpus is long
    enough that a silent run cannot be told apart from a hung one."""
    began = time.perf_counter()
    batch_size = settings.index.embed_batch_size
    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = _stamped(chunks[start : start + batch_size], embedder.model_name)
        vectors = await embedder.embed([chunk.embed_text for chunk in batch])
        written += await store.upsert(batch, vectors)
        log.info(
            "re-embed progress",
            done=written,
            total=len(chunks),
            elapsed_s=round(time.perf_counter() - began),
        )
    return written


def _stamped(chunks: list[Chunk], model: str) -> list[Chunk]:
    """The new model goes on the chunk, so the next backfill can tell which
    chunks it already covered."""
    return [
        chunk.model_copy(
            update={
                "metadata": chunk.metadata.model_copy(
                    update={"embed_model_version": model}
                )
            }
        )
        for chunk in chunks
    ]


def main() -> None:
    configure_logging()
    settings = get_settings()
    written = asyncio.run(backfill(settings))
    log.info(
        "re-embed complete",
        vectors=written,
        collection=settings.qdrant.collection,
        model=settings.index.embed_model,
    )


if __name__ == "__main__":
    main()
