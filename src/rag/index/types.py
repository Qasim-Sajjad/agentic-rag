"""Chunk contracts and the three metadata classes."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.extract.types import DocType

CHUNKER_VERSION = "structure_aware_v1"


class TokenCounter(Protocol):
    """Two implementations: a heuristic and the real BGE-M3 tokenizer."""

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """About four characters per token. Good enough to plan chunk boundaries.

    Used everywhere the real tokenizer is not loaded, which keeps chunker tests
    free of a 2 GB model download.
    """

    chars_per_token = 4

    def count(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)


class ChunkMetadata(BaseModel):
    """Three classes with different operational roles, in one payload.

    Filterable needs a Qdrant payload index. Display is returned for
    attribution. Lineage is never queried and is what makes a targeted
    re-extract possible instead of reprocessing the corpus.
    """

    model_config = ConfigDict(frozen=True)

    # filterable
    doc_type: DocType
    domain: str
    source_id: str
    published_at: date | None = None
    language: str = "en"
    is_table: bool = False
    fetch_tier: int = 0
    tenant_id: str = "default"
    # display
    source_url: str = ""
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_no: int | None = None
    # lineage
    content_hash: str = ""
    chunk_hash: str = ""
    extractor_name: str = ""
    extractor_version: str = ""
    chunker_version: str = CHUNKER_VERSION
    embed_model_version: str | None = None


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str  # as extracted
    embed_text: str  # section path prepended, what actually gets embedded
    token_count: int
    metadata: ChunkMetadata


def chunk_id_for(
    doc_id: str, index: int, chunker_version: str = CHUNKER_VERSION
) -> str:
    """Deterministic, so re-running the chunker upserts instead of duplicating."""
    raw = f"{doc_id}:{index}:{chunker_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def chunk_hash_for(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
