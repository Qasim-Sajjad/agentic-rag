"""PDF ladder. Gates decide the parser per page range, not per document.

Real documents mix digital born pages with scanned appendices, so a single
per document decision sends either prose through OCR or scans through a text
extractor. Both are expensive mistakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from rag.config.settings import ExtractSettings
from rag.extract.html import blocks_from_markdown
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
from rag.log import get_logger

log = get_logger(__name__)

PRINTABLE_FLOOR = 32
REPLACEMENT = "�"
COLUMN_GAP_RATIO = 0.25


class PageClass(StrEnum):
    SIMPLE_TEXT = "simple_text"  # text layer, single column, no tables
    COMPLEX_TEXT = "complex_text"  # text layer with tables or columns
    SCANNED = "scanned"  # no usable text layer


@dataclass(frozen=True)
class PageProbe:
    page_no: int
    chars: int
    garbage_ratio: float
    has_tables: bool
    columns: int


@dataclass(frozen=True)
class PageRange:
    start: int  # inclusive, zero based
    end: int  # exclusive
    page_class: PageClass

    @property
    def pages(self) -> list[int]:
        return list(range(self.start, self.end))


def garbage_ratio(text: str) -> float:
    """Share of replacement or non printable characters.

    A PDF with broken font encoding extracts fluent looking nonsense, which is
    worse than no text at all because it passes every emptiness check.
    """
    if not text:
        return 0.0
    bad = sum(
        1
        for ch in text
        if ch == REPLACEMENT or (ord(ch) < PRINTABLE_FLOOR and ch not in "\n\r\t")
    )
    return bad / len(text)


def classify_page(probe: PageProbe, settings: ExtractSettings) -> PageClass:
    """Gate 1 decides whether text exists, gate 2 decides how hard it is."""
    if probe.chars < settings.min_chars_per_page:
        return PageClass.SCANNED
    if probe.garbage_ratio > settings.max_garbage_ratio:
        return PageClass.SCANNED
    if probe.has_tables or probe.columns > 1:
        return PageClass.COMPLEX_TEXT
    return PageClass.SIMPLE_TEXT


def probe_pages(
    content: bytes, on_page: Callable[[int, int], None] | None = None
) -> list[PageProbe]:
    """`on_page(done, total)` is called per page, because table detection costs
    roughly 240 ms a page and a 500 page document otherwise spends two silent
    minutes here before anything downstream has a step to report."""
    import pymupdf

    probes: list[PageProbe] = []
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        total = doc.page_count
        for number, page in enumerate(doc):
            probes.append(_probe_one(page, number))
            if on_page is not None:
                on_page(number + 1, total)
    return probes


def _probe_one(page: Any, number: int) -> PageProbe:
    text = str(page.get_text())
    return PageProbe(
        page_no=number,
        chars=len(text.strip()),
        garbage_ratio=garbage_ratio(text),
        has_tables=_has_tables(page),
        columns=_column_count(page),
    )


def _has_tables(page: Any) -> bool:
    try:
        return len(page.find_tables().tables) > 0
    except (ValueError, RuntimeError):
        return False


def _column_count(page: Any) -> int:
    """Two clusters of block left edges separated by a real gap means columns."""
    blocks = [block for block in page.get_text("blocks") if block[4].strip()]
    if len(blocks) < 4:
        return 1
    width = float(page.rect.width) or 1.0
    lefts = sorted(float(block[0]) / width for block in blocks)
    gaps = [(b - a, a) for a, b in pairwise(lefts)]
    widest, _ = max(gaps, default=(0.0, 0.0))
    return 2 if widest > COLUMN_GAP_RATIO else 1


def plan_ranges(probes: list[PageProbe], settings: ExtractSettings) -> list[PageRange]:
    """Contiguous runs of the same class, split at `pages_per_task`.

    Page range parallelism is what turns a 1000 page document into 20 tasks
    instead of one worker lock.
    """
    ranges: list[PageRange] = []
    for probe in probes:
        page_class = classify_page(probe, settings)
        if _extends(ranges, page_class, settings):
            last = ranges.pop()
            ranges.append(PageRange(last.start, probe.page_no + 1, page_class))
        else:
            ranges.append(PageRange(probe.page_no, probe.page_no + 1, page_class))
    return ranges


def _extends(
    ranges: list[PageRange], page_class: PageClass, settings: ExtractSettings
) -> bool:
    if not ranges:
        return False
    last = ranges[-1]
    if last.page_class is not page_class:
        return False
    return (last.end - last.start) < settings.pages_per_task


#: Memo for `configure_layout`. A dict rather than a module scalar so the
#: function needs no `global` statement to record what it already applied.
_layout_state: dict[str, bool] = {}


def configure_layout(use_layout: bool) -> None:
    """Select pymupdf4llm's extraction path. Called once, before first parse.

    `use_layout(False)` is public API on pymupdf4llm and drops it from the GNN
    layout plus OCR path to the cheaper one. This is the single biggest cost in
    the whole ingest for a text layer PDF, so the choice is config rather than a
    library default: see `ExtractSettings.pymupdf_use_layout` for the numbers.

    Idempotent by module flag. The switch mutates pymupdf4llm globals, so
    calling it per page range would be both wasteful and a race.
    """
    if _layout_state.get("use_layout") == use_layout:
        return
    import pymupdf4llm

    pymupdf4llm.use_layout(use_layout)
    _layout_state["use_layout"] = use_layout
    log.info("pymupdf4llm layout path", use_layout=use_layout)


class PyMuPDF4LLMParser:
    """Fast path. Text layer, single column, no tables."""

    name = "pymupdf4llm"
    version = "0.0.17"

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        blocks = self.parse_pages(content, None)
        if not blocks:
            raise EmptyExtractionError(f"no text layer in {source_url}")
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.PDF,
        )

    def parse_pages(self, content: bytes, pages: list[int] | None) -> list[Block]:
        import pymupdf
        import pymupdf4llm

        with pymupdf.open(stream=content, filetype="pdf") as doc:
            markdown = pymupdf4llm.to_markdown(doc, pages=pages, show_progress=False)
        first_page = pages[0] if pages else 0
        return [
            block.model_copy(update={"provenance": Provenance(page=first_page)})
            for block in blocks_from_markdown(str(markdown))
        ]


def merge_split_tables(blocks: list[Block]) -> list[Block]:
    """Reassembly fixup for a table cut by a page range boundary.

    A table at the top of a range with the same column count as the table that
    ended the previous range, and no header row of its own, is a continuation.
    """
    merged: list[Block] = []
    for block in blocks:
        if _is_continuation(merged, block):
            previous = merged.pop()
            merged.append(
                previous.model_copy(update={"text": f"{previous.text}\n{block.text}"})
            )
            continue
        merged.append(block)
    return merged


def _is_continuation(merged: list[Block], block: Block) -> bool:
    if not merged or block.type is not BlockType.TABLE:
        return False
    previous = merged[-1]
    if previous.type is not BlockType.TABLE:
        return False
    if _has_header(block.text):
        return False
    return _columns(previous.text) == _columns(block.text)


def _columns(table_markdown: str) -> int:
    first = table_markdown.splitlines()[0]
    return first.count("|") - 1


def _has_header(table_markdown: str) -> bool:
    """A markdown table header is the `|---|---|` separator on the second line."""
    lines = table_markdown.splitlines()
    return len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ")
