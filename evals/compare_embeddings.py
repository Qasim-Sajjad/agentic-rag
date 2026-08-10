"""Embedding model comparison on the frozen gold set.

    python -m evals.compare_embeddings

Each model gets its own Qdrant collection, backfilled from the `chunk` table
rather than re-scraped. That is the re-embed path from the index SPEC, and it
is what makes this comparison cost hours instead of weeks.

Compared at a fixed context token budget rather than a fixed k would be the
next refinement. This run holds k policy constant and varies only the model,
which is enough to rank them but not enough to tune k.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.run_eval import append_row, config_hash, evaluate, load_goldset
from rag.config.settings import Settings, get_settings
from rag.db.pool import Database
from rag.index.embed import BGEM3Embedder, Embedder, SentenceTransformerEmbedder
from rag.index.repository import ChunkRepository
from rag.index.store import QdrantStore
from rag.index.types import Chunk
from rag.log import configure_logging, get_logger
from rag.retrieve.rerank import IdentityReranker, MiniLMReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import Reranker

log = get_logger("compare")

BATCH = 16


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    dims: int
    kind: str  # bge-m3 | sentence-transformers
    query_prefix: str = ""
    note: str = ""


MODELS = (
    ModelSpec(
        key="bge-m3",
        name="BAAI/bge-m3",
        dims=1024,
        kind="bge-m3",
        note="dense plus sparse from one model, 8192 token context",
    ),
    ModelSpec(
        key="bge-small",
        name="BAAI/bge-small-en-v1.5",
        dims=384,
        kind="sentence-transformers",
        note="dense only, 512 token context, roughly 130 MB",
    ),
)


def build_embedder(spec: ModelSpec, settings: Settings) -> Embedder:
    if spec.kind == "bge-m3":
        return BGEM3Embedder(settings.index, spec.name)
    return SentenceTransformerEmbedder(spec.name, spec.dims, spec.query_prefix)


async def backfill(
    store: QdrantStore, embedder: Embedder, chunks: list[Chunk]
) -> tuple[int, float]:
    """Streams `embed_text` from Postgres. No chunker logic in the path."""
    started = time.monotonic()
    written = 0
    for index in range(0, len(chunks), BATCH):
        batch = chunks[index : index + BATCH]
        vectors = await embedder.embed([chunk.embed_text for chunk in batch])
        written += await store.upsert(batch, vectors)
    return written, time.monotonic() - started


async def run_one(
    spec: ModelSpec, chunks: list[Chunk], settings: Settings, goldset: Path
) -> dict[str, Any]:
    qdrant = settings.qdrant.model_copy(
        update={"collection": f"{settings.qdrant.collection}_{spec.key}"}
    )
    store = QdrantStore(qdrant)
    embedder = build_embedder(spec, settings)
    reranker: Reranker = (
        MiniLMReranker(settings.retrieve)
        if settings.retrieve.rerank_pool
        else IdentityReranker()
    )
    await store.bootstrap(spec.dims)
    written, seconds = await backfill(store, embedder, chunks)
    log.info(
        "backfilled",
        model=spec.name,
        vectors=written,
        seconds=round(seconds, 1),
        chunks_per_second=round(written / seconds, 1) if seconds else 0,
    )
    service = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=settings.retrieve,
            qdrant=qdrant,
        )
    )
    metrics = await evaluate(service, load_goldset(goldset))
    await store.close()
    return {
        "run_id": f"embed-compare-{spec.key}",
        "config_hash": config_hash(settings, spec.name, reranker.name),
        "goldset_version": goldset.stem,
        "chunker": settings.index.target_tokens,
        "overlap": settings.index.overlap_ratio,
        "embed_model": spec.name,
        "dims": spec.dims,
        "reranker": reranker.name,
        "embed_seconds": round(seconds, 1),
        "note": spec.note,
        **metrics,
    }


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = get_settings()
    db = Database(settings.postgres)
    await db.connect()
    try:
        chunks = await ChunkRepository(db).all_chunks(args.max_chunks)
    finally:
        await db.close()
    log.info("comparing", models=[spec.name for spec in MODELS], chunks=len(chunks))
    rows: list[dict[str, Any]] = []
    for spec in MODELS:
        row = await run_one(spec, chunks, settings, Path(args.goldset))
        append_row(row, Path(args.results))
        rows.append(row)
        log.info("model done", **{k: v for k, v in row.items() if k != "note"})
    return rows


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evals.compare_embeddings")
    root.add_argument("--goldset", default="evals/goldset/v1.jsonl")
    root.add_argument("--results", default="evals/results.jsonl")
    root.add_argument("--max-chunks", type=int, default=None)
    return root


def main() -> None:
    configure_logging()
    asyncio.run(run(parser().parse_args()))


if __name__ == "__main__":
    main()
