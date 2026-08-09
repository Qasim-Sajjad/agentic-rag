"""Extraction contracts. The narrow waist of the whole pipeline.

Every parser emits `CanonicalDoc`. Chunking never knows which one ran, so
swapping an extractor changes nothing downstream.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    FIGURE_CAPTION = "figure_caption"


class DocType(StrEnum):
    HTML = "html"
    PDF = "pdf"
    OFFICE = "office"
    TEXT = "text"


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    css_path: str | None = None


class Block(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: BlockType
    text: str  # tables are markdown
    level: int | None = None  # heading depth
    provenance: Provenance = Provenance()
    confidence: float = 1.0  # below 1.0 only from OCR


class CanonicalDoc(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_url: str
    title: str | None = None
    published_at: date | None = None
    language: str = "en"
    blocks: list[Block] = Field(default_factory=list)
    content_hash: str = ""
    extractor_name: str = ""
    extractor_version: str = ""
    doc_type: DocType = DocType.HTML

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def content_hash(blocks: list[Block]) -> str:
    """Hash of normalized text, which is the third dedup point in `index`."""
    joined = "\n".join(normalize_text(block.text) for block in blocks)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def doc_id_for(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:32]
