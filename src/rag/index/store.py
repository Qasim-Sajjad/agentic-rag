"""Qdrant. One collection, dense plus sparse on the same point.

Not sharded by source: a general question searches across sources anyway, so
per source collections mean fan out plus a merge, and scores stop being
comparable. Filter on `source_id` instead.
"""

from __future__ import annotations

from typing import Any, Protocol

from rag.config.settings import QdrantSettings
from rag.index.embed import Embedding
from rag.index.types import Chunk
from rag.log import get_logger

log = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Filterable metadata needs a payload index. Display and lineage do not.
INDEXED_FIELDS = (
    "doc_type",
    "domain",
    "source_id",
    "published_at",
    "language",
    "is_table",
    "fetch_tier",
    "tenant_id",
)


class VectorStore(Protocol):
    async def bootstrap(self, dims: int) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: list[Embedding]) -> int: ...
    async def count(self) -> int: ...


class QdrantStore:
    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = (
                AsyncQdrantClient(path=self._settings.path)
                if self._settings.path
                else AsyncQdrantClient(
                    url=self._settings.url, timeout=int(self._settings.timeout_seconds)
                )
            )
        return self._client

    async def bootstrap(self, dims: int) -> None:
        from qdrant_client import models

        client = self.client()
        if await client.collection_exists(self._settings.collection):
            return
        await client.create_collection(
            collection_name=self._settings.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dims, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        await self._create_payload_indexes()

    async def _create_payload_indexes(self) -> None:
        from qdrant_client import models

        client = self.client()
        for field in INDEXED_FIELDS:
            schema = (
                models.PayloadSchemaType.KEYWORD
                if field != "is_table"
                else models.PayloadSchemaType.BOOL
            )
            await client.create_payload_index(
                collection_name=self._settings.collection,
                field_name=field,
                field_schema=schema,
            )

    async def upsert(self, chunks: list[Chunk], vectors: list[Embedding]) -> int:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR: vector.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(vector.sparse.keys()),
                        values=list(vector.sparse.values()),
                    ),
                },
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self.client().upsert(
            collection_name=self._settings.collection, points=points
        )
        return len(points)

    async def count(self) -> int:
        result = await self.client().count(
            collection_name=self._settings.collection, exact=True
        )
        return int(result.count)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _point_id(chunk_id: str) -> str:
    """Qdrant wants a UUID or an unsigned int. The chunk id is a 32 char hash."""
    parts = (
        chunk_id[:8],
        chunk_id[8:12],
        chunk_id[12:16],
        chunk_id[16:20],
        chunk_id[20:32],
    )
    return "-".join(parts)


def _payload(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata.model_dump(mode="json")
    meta["text"] = chunk.text
    meta["embed_text"] = chunk.embed_text
    meta["chunk_id"] = chunk.chunk_id
    meta["doc_id"] = chunk.doc_id
    return meta
