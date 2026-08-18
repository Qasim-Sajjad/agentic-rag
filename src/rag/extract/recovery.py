"""Deciding which pages the fast Markdown path read badly.

pymupdf4llm's cheap path silently drops text on some layouts. Measured on a 62
page handbook: 42 of its 48 text pages came back holding 40 to 60 percent of
what the page says, and one sentence stopped mid word with nothing anywhere to
say it had been cut. That is the worst failure mode a retrieval system has,
because the answer is simply absent and every layer downstream reports success.

The check is free. The probe already counted each page's text layer, so a page
whose extracted blocks hold far less than that was read badly, whatever the
parser thinks. What to do about it is in `rag.extract.service`: a table page is
worth re-reading on the expensive layout path, and everything else is cheaper to
take from the text layer.
"""

from __future__ import annotations

from rag.config.settings import ExtractSettings
from rag.extract.pdf import PageProbe
from rag.extract.types import Block


def captured_chars(blocks: list[Block]) -> dict[int, int]:
    """Characters extracted per page. Blocks with no page are not counted,
    because there is no page to compare them against."""
    counts: dict[int, int] = {}
    for block in blocks:
        page = block.provenance.page
        if page is not None:
            counts[page] = counts.get(page, 0) + len(block.text)
    return counts


def is_thin(
    probe: PageProbe, captured: dict[int, int], settings: ExtractSettings
) -> bool:
    """Text layer pages only.

    A scanned page has almost no text to compare against, and it has already
    been sent to OCR, which is the right answer there rather than a second parse.
    """
    if probe.chars < settings.min_chars_per_page:
        return False
    got = captured.get(probe.page_no + 1, 0)
    return got < probe.chars * settings.min_capture_ratio


def thin_pages(
    blocks: list[Block], probes: list[PageProbe], settings: ExtractSettings
) -> list[PageProbe]:
    captured = captured_chars(blocks)
    return [probe for probe in probes if is_thin(probe, captured, settings)]


def replace_pages(
    blocks: list[Block], rescued: list[Block], pages: set[int]
) -> list[Block]:
    """Swap the named pages' blocks for the recovered ones, keeping page order.

    Rebuilt by page rather than concatenated, so a recovered page stays where it
    belongs in the document and the table fixup still sees its neighbours.
    """
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        if block.provenance.page not in pages:
            by_page.setdefault(block.provenance.page or 0, []).append(block)
    for block in rescued:
        by_page.setdefault(block.provenance.page or 0, []).append(block)
    return [block for page in sorted(by_page) for block in by_page[page]]
