"""Hybrid search, rerank, adaptive k."""

from rag.retrieve.adaptive import adaptive_cut, confidence_for, elbow_index
from rag.retrieve.fusion import reciprocal_rank_fusion, rrf_score
from rag.retrieve.rerank import IdentityReranker, MiniLMReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import (
    Reranker,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)

__all__ = [
    "IdentityReranker",
    "MiniLMReranker",
    "Reranker",
    "RetrieveDependencies",
    "RetrievedChunk",
    "SearchFilters",
    "SearchResult",
    "SearchService",
    "adaptive_cut",
    "confidence_for",
    "elbow_index",
    "reciprocal_rank_fusion",
    "rrf_score",
]
