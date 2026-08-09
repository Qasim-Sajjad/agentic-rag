"""Extraction against the real fixture files, with the real parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.extract.service import ExtractService
from rag.extract.types import BlockType, DocType

pytestmark = pytest.mark.integration

PAGES = Path(__file__).parent.parent / "fixtures" / "pages"
URL = "https://example.test/report"


@pytest.fixture(scope="module")
def service() -> ExtractService:
    return ExtractService()


async def test_html_extraction_produces_blocks(service: ExtractService):
    doc = await service.extract(
        (PAGES / "static.html").read_bytes(), URL, "text/html; charset=utf-8"
    )
    assert doc.blocks


async def test_html_extraction_keeps_the_heading(service: ExtractService):
    doc = await service.extract((PAGES / "static.html").read_bytes(), URL, "text/html")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
    assert any("Risk factors" in h for h in headings)


async def test_html_extraction_keeps_the_table_whole(service: ExtractService):
    doc = await service.extract((PAGES / "static.html").read_bytes(), URL, "text/html")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert tables and "Subscription" in tables[0].text and "Services" in tables[0].text


async def test_pdf_is_routed_by_magic_bytes_despite_a_wrong_header(
    service: ExtractService,
):
    """A URL like /download?id=8821 claims html and returns a PDF."""
    doc = await service.extract((PAGES / "doc.pdf").read_bytes(), URL, "text/html")
    assert doc.doc_type is DocType.PDF


async def test_pdf_extraction_reads_the_text_layer(service: ExtractService):
    doc = await service.extract(
        (PAGES / "doc.pdf").read_bytes(), URL, "application/pdf"
    )
    assert "Quarterly filing summary" in doc.text


async def test_extraction_sets_a_content_hash(service: ExtractService):
    doc = await service.extract((PAGES / "static.html").read_bytes(), URL, "text/html")
    assert len(doc.content_hash) == 64


async def test_the_same_bytes_produce_the_same_content_hash(service: ExtractService):
    content = (PAGES / "static.html").read_bytes()
    first = await service.extract(content, URL, "text/html")
    second = await service.extract(content, URL, "text/html")
    assert first.content_hash == second.content_hash


async def test_csv_becomes_one_markdown_table(service: ExtractService):
    csv = b"Segment,Revenue\nSubscription,26.0\nServices,15.2\n"
    doc = await service.extract(csv, "https://example.test/data.csv", "text/csv")
    assert doc.blocks[0].type is BlockType.TABLE
