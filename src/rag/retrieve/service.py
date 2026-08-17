"""Query in, ranked chunks out. Six stages, each testable on its own."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag.config.settings import QdrantSettings, RetrieveSettings
from rag.index.embed import Embedder
from rag.index.store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore
from rag.log import get_logger
from rag.retrieve.adaptive import adaptive_cut
from rag.retrieve.fusion import reciprocal_rank_fusion
from rag.retrieve.types import (
    Reranker,
    RetrievalStep,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

log = get_logger(__name__)


@dataclass
class RetrieveDependencies:
    store: QdrantStore
    embedder: Embedder
    reranker: Reranker
    settings: RetrieveSettings
    qdrant: QdrantSettings


@dataclass
class _Steps:
    """Stage timings collected as the search runs.

    Each `mark` measures the gap since the previous one rather than since the
    start, so the numbers add up to the total instead of being cumulative. Held
    in one object so `_finish` stays inside the argument limit.
    """

    began: float = field(default_factory=time.monotonic)
    rows: list[RetrievalStep] = field(default_factory=list)
    _at: float = field(default_factory=time.monotonic)

    def mark(self, stage: str, candidates: int, note: str = "") -> None:
        now = time.monotonic()
        self.rows.append(
            RetrievalStep(
                stage=stage,
                candidates=candidates,
                latency_ms=round((now - self._at) * 1000),
                note=note,
            )
        )
        self._at = now

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.began) * 1000)


class SearchService:
    def __init__(self, deps: RetrieveDependencies) -> None:
        self._deps = deps
        self._settings = deps.settings

    # get the ranked chunks of the query, with optional filters and top_k limit.
    # The returned chunks are already reranked.
    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> SearchResult:
        steps = _Steps()
        embedding = (await self._embed_query(query))[0]
        steps.mark("embed query", 1, self._deps.embedder.model_name)
        dense, sparse = await self._both_sides(embedding, filters)
        steps.mark(
            "vector search",
            len(dense) + len(sparse),
            f"dense {len(dense)}, sparse {len(sparse)}",
        )
        fused = reciprocal_rank_fusion([dense, sparse], self._settings.rrf_k)
        steps.mark("fuse", len(fused), f"reciprocal rank, k={self._settings.rrf_k}")
        pool = fused[: self._settings.rerank_pool]
        reranked = await self._deps.reranker.rerank(query, pool)
        steps.mark("rerank", len(reranked), self._deps.reranker.name)
        return self._finish(query, reranked, top_k, steps)

    async def _embed_query(self, query: str) -> list[Any]:
        """Some models want an instruction prefix on queries and not on
        documents. An embedder that needs the asymmetry exposes
        `embed_queries`, and getting it wrong silently destroys recall."""
        asymmetric = getattr(self._deps.embedder, "embed_queries", None)
        if asymmetric is not None:
            return list(await asymmetric([query]))
        return list(await self._deps.embedder.embed([query]))

    async def _both_sides(
        self, embedding: Any, filters: SearchFilters | None
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Two queries fused in code rather than server side fusion, so the
        fusion step is deterministic and unit testable."""
        condition = _filter(filters)
        pool = self._settings.candidate_pool
        dense = await self._query(DENSE_VECTOR, embedding.dense, pool, condition)
        sparse = await self._query(SPARSE_VECTOR, embedding.sparse, pool, condition)
        return dense, sparse

    async def _query(
        self, using: str, vector: Any, limit: int, condition: Any
    ) -> list[RetrievedChunk]:
        from qdrant_client import models

        query = (
            models.SparseVector(
                indices=list(vector.keys()), values=list(vector.values())
            )
            if using == SPARSE_VECTOR
            else vector
        )
        response = await self._deps.store.client().query_points(
            collection_name=self._deps.qdrant.collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=condition,
            with_payload=True,
        )
        return [_to_chunk(point) for point in response.points]

    def _finish(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None,
        steps: _Steps,
    ) -> SearchResult:
        cut = adaptive_cut(chunks, self._settings)
        selected = cut.chunks[:top_k] if top_k else cut.chunks
        note = cut.reason or f"confidence {cut.confidence}"
        steps.mark("adaptive cut", len(selected), note)
        latency = steps.elapsed_ms()
        # The query text, not just the outcome. Without it a log line answers
        # "did search work" but not "on what", which is what makes a live
        # instance's activity legible after the fact rather than only in the
        # moment a request is being watched.
        log.info(
            "search",
            query=query,
            k_used=len(selected),
            confidence=cut.confidence,
            latency_ms=latency,
        )
        return SearchResult(
            chunks=selected,
            confidence=cut.confidence,
            k_used=len(selected),
            reason=cut.reason,
            steps=steps.rows,
            latency_ms=latency,
        )


def _filter(filters: SearchFilters | None) -> Any:
    from qdrant_client import models

    if filters is None:
        return None
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in (
            ("doc_type", filters.doc_type),
            ("source_id", filters.source_id),
            ("tenant_id", filters.tenant_id),
        )
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


def _to_chunk(point: Any) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", point.id)),
        text=str(payload.get("text", "")),
        score=float(point.score),
        source_url=str(payload.get("source_url", "")),
        section_path=list(payload.get("section_path") or []),
        published_at=payload.get("published_at"),
        page_no=payload.get("page_no"),
        doc_id=str(payload.get("doc_id", "")),
    )
