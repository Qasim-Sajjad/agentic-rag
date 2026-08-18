"""Embedding behind a protocol. Two real embedders and a hashing fake.

BGE-M3 emits dense and sparse from one model, so hybrid retrieval is one
inference pass and one index rather than two systems that drift apart. It is
also an XLM-R large, and on a CPU it measures roughly 8.6 seconds per 512 token
chunk: a 250 page document is 600 chunks and an hour and a half, which is not a
system anyone can demonstrate.

`SmallHybridEmbedder` is the same shape at a fraction of the cost: a 33M
parameter dense model, measured at 184 ms per chunk on the same machine, with
the sparse side computed lexically in `rag.index.lexical`. Which one runs is
`index.embed_model`, and the tradeoff is written down in docs/DESIGN.md.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from rag.config.settings import IndexSettings
from rag.index.lexical import lexical_sparse
from rag.log import get_logger

log = get_logger(__name__)

BGE_M3 = "BAAI/bge-m3"
DENSE_DIMS = 1024

BGE_SMALL = "BAAI/bge-small-en-v1.5"
SMALL_DIMS = 384

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


class SmallHybridEmbedder:
    """A small dense model plus a lexical sparse vector. Hybrid, without the
    large model.

    BGE-M3's sparse head is better than a bag of words: it weights a term by
    context rather than by frequency. It also costs 47 times as much per chunk
    on a CPU, and the difference between an hour and two minutes is the
    difference between a system that indexes a document and one that does not.
    See `rag.index.lexical` for what the sparse side gives up.

    `bge-small-en-v1.5` asks for an instruction prefix on queries and none on
    passages. Getting that asymmetry backwards silently costs recall, which is
    why it is stated here rather than assumed.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, settings: IndexSettings, model_name: str = BGE_SMALL) -> None:
        self._settings = settings
        self.model_name = model_name
        self.dims = settings.embed_dims
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model", model=self.model_name)
            model = SentenceTransformer(self.model_name)
            model.max_seq_length = self._settings.embed_max_length
            self._model = model
        return self._model

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return await self._encode(texts, texts)

    async def embed_queries(self, texts: list[str]) -> list[Embedding]:
        """The sparse side takes the bare query, not the prefixed one. The
        prefix is an instruction to the dense model and its words would
        otherwise become lexical terms that match every document."""
        prefixed = [f"{self.QUERY_PREFIX}{text}" for text in texts]
        return await self._encode(prefixed, texts)

    async def _encode(
        self, dense_in: list[str], sparse_in: list[str]
    ) -> list[Embedding]:
        """Threaded including the load, for the reason on `BGEM3Embedder.embed`."""
        vectors = await asyncio.to_thread(self._run, dense_in)
        return [
            Embedding(
                dense=[float(value) for value in vector],
                sparse=lexical_sparse(text),
            )
            for vector, text in zip(vectors, sparse_in, strict=True)
        ]

    def _run(self, texts: list[str]) -> Any:
        return self._load().encode(
            texts,
            batch_size=self._settings.embed_batch_size,
            normalize_embeddings=True,
        )


def build_embedder(settings: IndexSettings) -> Embedder:
    """One place that turns config into an embedder.

    Both are hybrid and both are swappable, and the choice is a latency for
    quality trade recorded in docs/DESIGN.md. Changing it changes `embed_dims`,
    which means a different Qdrant collection: vectors of two widths cannot
    share one, and mixing two models' vectors in one space would be worse than
    either model alone.
    """
    if settings.embed_model == BGE_M3:
        return BGEM3Embedder(settings, settings.embed_model)
    return SmallHybridEmbedder(settings, settings.embed_model)


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
