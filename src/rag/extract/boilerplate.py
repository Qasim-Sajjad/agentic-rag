"""Repeated page furniture, removed before chunking.

A faxed clinical note carries the patient banner on every page, the sending
machine's line at the foot of every page, and a form artifact repeated wherever
a field was blank. A prospectus carries a running header. None of it is content,
and all of it survives extraction, because it is printed on the page and both
the text layer and OCR read the page as it is.

Left alone it does three things, all bad: it fills chunks that say nothing, it
dilutes the chunks it shares space with, and identical banner only chunks
collide on `chunk_hash`, so dedup drops them and the count of what was indexed
stops describing the document.

The rule is deliberately narrow, and narrow in two directions:

- Only short paragraphs. A repeated heading is document structure and the
  sections under it would lose their section path; a repeated table is data.
- Digits are masked when matching, because a page number, a timestamp and a fax
  counter change per page while the line does not. But a line whose digits vary
  in more than `repeat_max_varying` places is data, not furniture: three
  measurements that differ only in their numbers are three measurements, and
  masking alone would delete all of them.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rag.config.settings import ExtractSettings
from rag.extract.types import Block, BlockType
from rag.log import get_logger

log = get_logger(__name__)

_DIGITS = re.compile(r"\d+")


def mask_digits(text: str) -> str:
    """`P.003/018` and `P.004/018` are the same line on two pages."""
    return _DIGITS.sub(lambda match: "#" * len(match.group()), text.strip())


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(_DIGITS.findall(text))


def strip_repeated(blocks: list[Block], settings: ExtractSettings) -> list[Block]:
    """Drop repeated page furniture. Returns the blocks unchanged when off."""
    if not settings.strip_repeated_blocks:
        return blocks
    furniture = _furniture(blocks, settings)
    if not furniture:
        return blocks
    kept = [
        block
        for block in blocks
        if not (
            _short_paragraph(block, settings) and mask_digits(block.text) in furniture
        )
    ]
    log.info(
        "page furniture removed",
        distinct_lines=len(furniture),
        blocks_dropped=len(blocks) - len(kept),
        blocks_kept=len(kept),
    )
    return kept


def _furniture(blocks: list[Block], settings: ExtractSettings) -> set[str]:
    """The masked lines that repeat often enough and vary little enough."""
    seen: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for block in blocks:
        if _short_paragraph(block, settings):
            seen[mask_digits(block.text)].append(_numbers(block.text))
    return {
        text
        for text, numbers in seen.items()
        if text
        and len(numbers) >= settings.repeat_min_count
        and _varying(numbers) <= settings.repeat_max_varying
    }


def _varying(numbers: list[tuple[str, ...]]) -> int:
    """How many number positions actually differ between the occurrences.

    A footer's clock and page counter move. A row of readings moves in every
    number it has, which is what separates the two.
    """
    return sum(1 for column in zip(*numbers, strict=True) if len(set(column)) > 1)


def _short_paragraph(block: Block, settings: ExtractSettings) -> bool:
    return (
        block.type is BlockType.PARAGRAPH
        and len(block.text.strip()) <= settings.repeat_max_chars
    )
