"""Embedding behind a protocol. BGE-M3 for real, a hashing fake for tests.

BGE-M3 emits dense and sparse from one model, so hybrid retrieval is one
inference pass and one index rather than two systems that drift apart.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from rag.config.settings import IndexSettings
from rag.log import get_logger

log = get_logger(__name__)

BGE_M3 = "BAAI/bge-m3"
DENSE_DIMS = 1024

# Exactly what BGE-M3 loads. The repo also ships onnx/model.onnx_data, a 2.2 GB
# copy of the same weights in a format we never touch, and FlagEmbedding's
# loader fetches the whole repo: its ignore list covers flax and TensorFlow
# weights but not ONNX. Resolving the snapshot ourselves with this filter is
# the difference between a 2.2 GB download and a 4.4 GB one.
BGE_M3_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "1_Pooling/config.json",
)


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

            path = self._resolve()
            log.info("loading embedding model", model=self.model_name, path=path)
            self._model = BGEM3FlagModel(path, use_fp16=False)
        return self._model

    def _resolve(self) -> str:
        """Local snapshot path for the model, downloading only what it loads.

        Handing FlagEmbedding a path that exists makes it skip its own
        whole-repo download, which is what otherwise drags in the ONNX copy.
        """
        from pathlib import Path

        if Path(self.model_name).exists():
            return self.model_name
        from huggingface_hub import snapshot_download

        return str(
            snapshot_download(self.model_name, allow_patterns=list(BGE_M3_FILES))
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        """Threaded, because this is synchronous CPU work of seconds per batch.
        On the event loop it blocks the whole API process, so a job reporting its
        progress cannot answer the request that polls for it: the work happens,
        and the caller watching it sees a frozen server.

        `_load` is inside the thread, not before it. Loading BGE-M3 is a two
        gigabyte read and the slowest single call in an ingest: awaiting it on
        the loop froze every poll for the length of the load, which is the exact
        symptom the job endpoint exists to remove.
        """
        output = await asyncio.to_thread(self._encode, texts)
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

    def _encode(self, texts: list[str]) -> Any:
        return self._load().encode(
            texts,
            batch_size=self._settings.embed_batch_size,
            # BGE-M3 defaults to its full 8192 token window. Chunks target
            # `target_tokens`, so encoding at 8192 pays for padding that is
            # never used and costs roughly an order of magnitude on CPU.
            max_length=self._settings.embed_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, dense only.

    Exists only for the embedding sweep in `evals/SPEC.md`, which names
    Qwen3-Embedding-0.6B as the model to compare against. It is never the
    production embedder: a dense only model leaves the sparse side of hybrid
    retrieval empty, which is exactly the cost DESIGN section 3 refuses to pay.
    """

    def __init__(
        self,
        model_name: str,
        dims: int,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.dims = dims
        # Qwen3 wants an instruction prefix on queries and not on documents.
        # Getting that asymmetry wrong silently destroys recall and looks like
        # a bad model, so it is a constructor argument rather than a guess.
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return await self._encode([f"{self._doc_prefix}{text}" for text in texts])

    async def embed_queries(self, texts: list[str]) -> list[Embedding]:
        return await self._encode([f"{self._query_prefix}{text}" for text in texts])

    async def _encode(self, prefixed: list[str]) -> list[Embedding]:
        """Threaded including the load, for the reason on `BGEM3Embedder.embed`."""
        vectors = await asyncio.to_thread(self._run, prefixed)
        return [Embedding([float(value) for value in vector], {}) for vector in vectors]

    def _run(self, prefixed: list[str]) -> Any:
        return self._load().encode(prefixed, normalize_embeddings=True)


def build_embedder(settings: IndexSettings) -> Embedder:
    """One place that turns config into an embedder. BGE-M3 is the only one
    that serves production, for the reason in docs/DESIGN.md section 3."""
    return BGEM3Embedder(settings, settings.embed_model)


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
