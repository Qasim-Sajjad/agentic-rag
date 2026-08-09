"""Dedup, chunking, embedding, upsert."""

from rag.index.chunker import StructureAwareChunker, split_table
from rag.index.embed import BGEM3Embedder, Embedder, Embedding, FakeEmbedder
from rag.index.pipeline import IndexDependencies, IngestPipeline, IngestResult
from rag.index.repository import ChunkRepository, DocumentRepository
from rag.index.simhash import SimHashIndex, hamming, simhash
from rag.index.store import QdrantStore, VectorStore
from rag.index.types import Chunk, ChunkMetadata, chunk_id_for
from rag.index.urls import canonicalize, url_hash

__all__ = [
    "BGEM3Embedder",
    "Chunk",
    "ChunkMetadata",
    "ChunkRepository",
    "DocumentRepository",
    "Embedder",
    "Embedding",
    "FakeEmbedder",
    "IndexDependencies",
    "IngestPipeline",
    "IngestResult",
    "QdrantStore",
    "SimHashIndex",
    "StructureAwareChunker",
    "VectorStore",
    "canonicalize",
    "chunk_id_for",
    "hamming",
    "simhash",
    "split_table",
    "url_hash",
]
