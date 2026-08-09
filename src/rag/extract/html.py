"""HTML extraction with trafilatura, plus tables converted to markdown.

Boilerplate removal happens here, before near-dedup, because nav and footer
markup dominates the shingle space of raw HTML and would mark every page on a
domain as a duplicate of every other.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from rag.extract.protocols import EmptyExtractionError
from rag.extract.types import (
    Block,
    BlockType,
    CanonicalDoc,
    DocType,
    Provenance,
    content_hash,
    doc_id_for,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class TrafilaturaParser:
    name = "trafilatura"
    version = "1.12"

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        import trafilatura

        html = content.decode("utf-8", errors="replace")
        markdown = trafilatura.extract(
            html,
            output_format="markdown",
            include_tables=True,
            include_links=False,
            with_metadata=False,
            favor_recall=True,
        )
        if not markdown:
            raise EmptyExtractionError(f"trafilatura found no content in {source_url}")
        blocks = blocks_from_markdown(markdown)
        return self._document(blocks, html, source_url)

    def _document(self, blocks: list[Block], html: str, url: str) -> CanonicalDoc:
        meta = _metadata(html)
        return CanonicalDoc(
            doc_id=doc_id_for(url),
            source_url=url,
            title=meta.get("title"),
            published_at=_as_date(meta.get("date")),
            language=meta.get("language") or "en",
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.HTML,
        )


def _metadata(html: str) -> dict[str, Any]:
    import trafilatura

    extracted = trafilatura.extract_metadata(html)
    return extracted.as_dict() if extracted is not None else {}


def _as_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def blocks_from_markdown(markdown: str) -> list[Block]:
    """Markdown to typed blocks. Tables stay whole, one block each.

    Shared with the PDF path, which also produces markdown, so block typing is
    written once rather than per parser.
    """
    blocks: list[Block] = []
    for raw in _split_paragraphs(markdown):
        chunk = raw.strip()
        if chunk:
            blocks.append(_classify(chunk))
    return blocks


def _split_paragraphs(markdown: str) -> list[str]:
    """Blank lines separate blocks. Table rows are contiguous by definition, so
    they stay together without a special case."""
    parts: list[str] = []
    buffer: list[str] = []
    for line in markdown.splitlines():
        if not line.strip():
            parts.append("\n".join(buffer))
            buffer = []
            continue
        buffer.append(line)
    parts.append("\n".join(buffer))
    return [part for part in parts if part.strip()]


def _classify(text: str) -> Block:
    heading = _HEADING.match(text)
    if heading is not None:
        return Block(
            type=BlockType.HEADING,
            text=heading.group(2).strip(),
            level=len(heading.group(1)),
            provenance=Provenance(),
        )
    return Block(type=_body_type(text), text=text)


def _body_type(text: str) -> BlockType:
    first = text.splitlines()[0]
    if _TABLE_ROW.match(first):
        return BlockType.TABLE
    if _LIST_ITEM.match(first):
        return BlockType.LIST
    if text.startswith("```"):
        return BlockType.CODE
    return BlockType.PARAGRAPH
