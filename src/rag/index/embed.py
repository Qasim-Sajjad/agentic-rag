"""Embedding behind a protocol. BGE-M3 for real, a hashing fake for tests.

BGE-M3 emits dense and sparse from one model, so hybrid retrieval is one
inference pass and one index rather than two systems that drift apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from rag.config.settings import IndexSettings
from rag.log import get_logger

log = get_logger(__name__)

BGE_M3 = "BAAI/bge-m3"
DENSE_DIMS = 1024


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]  # token id to weight


class Embedder(Protocol):
    model_name: str
    dims: int

    async def embed(self, texts: list[str]) -> list[Embedding]: ...


class FakeEmbedder:
    """Deterministic hashing embedder. Unit tests never download 2 GB.

    Same text gives the same vector, different text gives a different one,
    which is all the pipeline tests need to be meaningful.
    """

    model_name = "fake-hash-v1"
    dims = 64

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> Embedding:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=self.dims).digest()
        dense = [(byte / 255.0) - 0.5 for byte in digest]
        norm = sum(value * value for value in dense) ** 0.5 or 1.0
        tokens = {abs(hash(word)) % 30000: 1.0 for word in set(text.lower().split())}
        return Embedding([value / norm for value in dense], tokens)


class BGEM3Embedder:
    """Loads lazily, so importing this module costs nothing until first use."""

    model_name = BGE_M3
    dims = DENSE_DIMS

    def __init__(self, settings: IndexSettings, model_name: str = BGE_M3) -> None:
        self._settings = settings
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            log.info("loading embedding model", model=self.model_name)
            self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        return self._model

    async def embed(self, texts: list[str]) -> list[Embedding]:
        model = self._load()
        output = model.encode(
            texts,
            batch_size=self._settings.embed_batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            Embedding(
                dense=[float(value) for value in output["dense_vecs"][i]],
                sparse={
                    int(key): float(weight)
                    for key, weight in output["lexical_weights"][i].items()
                },
            )
            for i in range(len(texts))
        ]


async def embed_in_batches(
    embedder: Embedder, texts: list[str], batch_size: int
) -> list[Embedding]:
    """Batching plus a checkpoint boundary, so a crash at 400K chunks does not
    restart from zero. The caller persists after each batch."""
    results: list[Embedding] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        results.extend(await embedder.embed(batch))
    return results
