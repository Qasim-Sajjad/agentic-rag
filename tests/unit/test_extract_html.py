"""Markdown to typed blocks. Tables stay whole, headings keep their depth."""

from __future__ import annotations

from rag.extract.html import blocks_from_markdown
from rag.extract.types import BlockType

MARKDOWN = """# Annual report

Revenue rose nine percent.

## Risk factors

We consider this risk material.

| Segment | Revenue |
| --- | --- |
| Subscription | 26.0 |
| Services | 15.2 |

- first item
- second item
"""


def blocks():
    return blocks_from_markdown(MARKDOWN)


def test_headings_are_typed():
    assert blocks()[0].type is BlockType.HEADING


def test_heading_depth_is_kept():
    depths = [b.level for b in blocks() if b.type is BlockType.HEADING]
    assert depths == [1, 2]


def test_a_table_is_one_block():
    tables = [b for b in blocks() if b.type is BlockType.TABLE]
    assert len(tables) == 1


def test_a_table_block_keeps_every_row():
    table = next(b for b in blocks() if b.type is BlockType.TABLE)
    assert "Subscription" in table.text and "Services" in table.text


def test_lists_are_typed():
    assert any(b.type is BlockType.LIST for b in blocks())


def test_paragraphs_are_typed():
    assert any(b.type is BlockType.PARAGRAPH for b in blocks())


def test_blank_lines_do_not_produce_empty_blocks():
    assert all(b.text.strip() for b in blocks())
