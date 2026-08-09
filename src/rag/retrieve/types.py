"""Retrieval contracts. This module never calls an LLM."""

from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["high", "low", "none"]


class SearchFilters(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_type: Literal["html", "pdf", "office", "text"] | None = None
    source_id: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    tenant_id: str | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    score: float
    source_url: str
    section_path: list[str] = Field(default_factory=list)
    published_at: date | None = None
    page_no: int | None = None
    doc_id: str = ""


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[RetrievedChunk]
    confidence: Confidence
    k_used: int
    reason: str | None = None  # set when confidence is not high
    latency_ms: int = 0


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...
