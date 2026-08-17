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


def a_three_page_pdf() -> bytes:
    """Built here rather than committed, because the only thing this fixture
    needs to be is more than one page long."""
    import pymupdf

    doc = pymupdf.open()
    for number in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {number + 1} of the report. " * 20)
    content: bytes = doc.tobytes()
    return content


async def collect(service: ExtractService, content: bytes) -> list[tuple]:
    seen: list[tuple] = []

    def report(stage, done, total, detail=""):
        seen.append((stage, done, total))

    await service.extract(content, URL, "application/pdf", report)
    return seen


async def test_the_probe_reports_each_page_as_it_reads_it(service: ExtractService):
    """Table detection costs roughly 240 ms a page, so on a long document this
    stage is minutes. Reported per page or it is minutes of silence."""
    seen = await collect(service, a_three_page_pdf())
    probes = [row for row in seen if row[0] == "probe"]
    assert probes[:3] == [("probe", 1, 3), ("probe", 2, 3), ("probe", 3, 3)]


async def test_extraction_announces_itself_before_the_first_range_finishes(
    service: ExtractService,
):
    """A stage that is absent from the progress list reads as a stage that has
    not started, which on a long parse is the wrong thing to tell a caller."""
    seen = await collect(service, a_three_page_pdf())
    extracts = [row for row in seen if row[0] == "extract"]
    assert extracts[0][1] == 0
    assert extracts[-1][1] == extracts[-1][2]


async def test_extraction_without_a_progress_sink_still_parses(
    service: ExtractService,
):
    doc = await service.extract(a_three_page_pdf(), URL, "application/pdf")
    assert "Page 3 of the report" in doc.text


async def test_csv_becomes_one_markdown_table(service: ExtractService):
    csv = b"Segment,Revenue\nSubscription,26.0\nServices,15.2\n"
    doc = await service.extract(csv, "https://example.test/data.csv", "text/csv")
    assert doc.blocks[0].type is BlockType.TABLE
