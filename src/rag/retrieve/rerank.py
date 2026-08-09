"""Cross encoder reranking behind a protocol.

MiniLM on CPU is the default because it is fast enough to run in the request
path. bge-reranker-v2-m3 is the better model on a GPU and the swap is a config
change, which is the reason this is a protocol and not a function.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag.config.settings import RetrieveSettings
from rag.log import get_logger
from rag.retrieve.types import RetrievedChunk

log = get_logger(__name__)

SHORT_CHUNK_TOKENS = 50


class IdentityReranker:
    """Passthrough. Lets retrieval be tested without loading a model."""

    name = "identity"

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return chunks


class MiniLMReranker:
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self, settings: RetrieveSettings, model_name: str | None = None
    ) -> None:
        self._settings = settings
        self.name = model_name or self.name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker", model=self.name)
            self._model = CrossEncoder(self.name, max_length=512)
        return self._model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pool = chunks[: self._settings.rerank_pool]
        pairs = [(query, _passage(chunk)) for chunk in pool]
        scores = await asyncio.to_thread(self._predict, pairs)
        scored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(pool, scores, strict=True)
        ]
        return sorted(scored, key=lambda chunk: -chunk.score)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """All candidates in one forward pass, roughly 150ms at pool 25."""
        return [float(score) for score in self._load().predict(pairs)]


def _passage(chunk: RetrievedChunk) -> str:
    """Pad a short chunk with its section path.

    Very short passages produce unstable cross encoder scores, and adaptive k
    keys off exactly those scores.
    """
    if len(chunk.text.split()) >= SHORT_CHUNK_TOKENS:
        return chunk.text
    return f"{' > '.join(chunk.section_path)}\n{chunk.text}"
