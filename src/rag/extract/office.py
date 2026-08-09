"""Docling for office formats and table heavy PDFs, plus plain text and CSV.

Docling's table and OCR models are disabled unless a gate asked for them.
Running TableFormer on prose is the largest avoidable cost in this stage.
"""

from __future__ import annotations

import csv
import io

from rag.extract.html import blocks_from_markdown
from rag.extract.protocols import EmptyExtractionError, ParserUnavailableError
from rag.extract.types import (
    Block,
    BlockType,
    CanonicalDoc,
    DocType,
    content_hash,
    doc_id_for,
)


class DoclingParser:
    name = "docling"
    version = "2.0"

    def __init__(self, with_tables: bool = True) -> None:
        self._with_tables = with_tables

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        markdown = self._convert(content, source_url)
        blocks = blocks_from_markdown(markdown)
        if not blocks:
            raise EmptyExtractionError(f"docling found no content in {source_url}")
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.OFFICE,
        )

    def _convert(self, content: bytes, source_url: str) -> str:
        try:
            from docling.datamodel.base_models import DocumentStream
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ParserUnavailableError("docling is not installed") from exc
        stream = DocumentStream(name=source_url, stream=io.BytesIO(content))
        result = DocumentConverter().convert(stream)
        return str(result.document.export_to_markdown())


class TabularParser:
    """CSV to a markdown table, so it lands in the chunker as one table block."""

    name = "tabular"
    version = "1.0"

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        text = content.decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            raise EmptyExtractionError(f"empty csv at {source_url}")
        blocks = [Block(type=BlockType.TABLE, text=_to_markdown_table(rows))]
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.TEXT,
        )


class PlainTextParser:
    name = "plaintext"
    version = "1.0"

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            raise EmptyExtractionError(f"empty text at {source_url}")
        blocks = [
            Block(type=BlockType.PARAGRAPH, text=part.strip())
            for part in text.split("\n\n")
            if part.strip()
        ]
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.TEXT,
        )


def _to_markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:]
    lines = [_row(header), _row(separator), *(_row(row) for row in body)]
    return "\n".join(lines)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.strip() for cell in cells) + " |"
