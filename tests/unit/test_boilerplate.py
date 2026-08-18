"""Page furniture removal: what counts as furniture, and what must never."""

from __future__ import annotations

from rag.config.settings import ExtractSettings
from rag.extract.boilerplate import mask_digits, strip_repeated
from rag.extract.types import Block, BlockType

SETTINGS = ExtractSettings()

BANNER = "Name: Ortega, Alba DOB: 11/28/1939 Date:"
FAX = "From:IM Data Centers LLC 9545332152 02/22/2023 12:19 #097 P.003/018"


def para(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text)


def heading(text: str) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=2)


def table(text: str) -> Block:
    return Block(type=BlockType.TABLE, text=text)


def texts(blocks: list[Block]) -> list[str]:
    return [block.text for block in blocks]


def test_a_banner_on_every_page_is_dropped():
    blocks = [para(BANNER), para("Real content."), para(BANNER), para(BANNER)]
    assert texts(strip_repeated(blocks, SETTINGS)) == ["Real content."]


def test_digits_are_masked_so_a_per_page_footer_still_matches():
    """The fax counter and the timestamp change on every page. The line does
    not, and matching on the literal text would keep all of them."""
    pages = [FAX.replace("P.003/018", f"P.00{n}/018") for n in range(1, 4)]
    blocks = [para(text) for text in pages] + [para("Real content.")]
    assert texts(strip_repeated(blocks, SETTINGS)) == ["Real content."]


def test_a_line_below_the_count_is_kept():
    """Twice is a coincidence, and the threshold is what separates furniture
    from a sentence a document happens to repeat."""
    blocks = [para("Signed."), para("Body."), para("Signed.")]
    assert len(strip_repeated(blocks, SETTINGS)) == 3


def test_a_repeated_heading_is_kept():
    """A repeated heading is document structure. Dropping it would leave the
    sections under it with no section path."""
    blocks = [heading("Assessment"), heading("Assessment"), heading("Assessment")]
    assert len(strip_repeated(blocks, SETTINGS)) == 3


def test_a_repeated_table_is_kept():
    """Two pages of a form can carry the same empty table. That is data the
    document contains, not furniture printed around it."""
    rows = "| A | B |\n|---|---|\n| 1 | 2 |"
    blocks = [table(rows), table(rows), table(rows)]
    assert len(strip_repeated(blocks, SETTINGS)) == 3


def test_a_long_repeated_paragraph_is_kept():
    """A legal boilerplate paragraph repeated in a contract is content. The
    length bound is what keeps this rule to page furniture."""
    long_text = "This agreement is governed by the laws of the state. " * 6
    blocks = [para(long_text) for _ in range(3)]
    assert len(strip_repeated(blocks, SETTINGS)) == 3


def test_the_surviving_order_is_unchanged():
    blocks = [para(BANNER), para("First."), para(BANNER), para("Second."), para(BANNER)]
    assert texts(strip_repeated(blocks, SETTINGS)) == ["First.", "Second."]


def test_config_can_turn_it_off():
    blocks = [para(BANNER) for _ in range(4)]
    settings = ExtractSettings(strip_repeated_blocks=False)
    assert len(strip_repeated(blocks, settings)) == 4


def test_masking_leaves_the_words_alone():
    assert mask_digits(" Page 12 of 18 ") == "Page ## of ##"


def test_lines_whose_numbers_all_move_are_data_not_furniture():
    """Three readings that differ only in their numbers are three readings.
    Masking digits alone would delete every one of them."""
    blocks = [
        para(f"Reading {n}: unit {n * 3} at {40 + n} degrees, {0.1 + n:.2f} mm")
        for n in range(1, 5)
    ]
    assert len(strip_repeated(blocks, SETTINGS)) == 4


def test_a_footer_whose_clock_and_counter_move_is_still_furniture():
    """Two numbers move, the line does not. That is the case masking exists
    for, and the varying bound has to leave room for it."""
    blocks = [
        para(f"From:IM Data Centers LLC 9545332152 02/22/2023 12:{19 + n} P.00{n}/018")
        for n in range(1, 5)
    ]
    blocks.append(para("Real content."))
    assert texts(strip_repeated(blocks, SETTINGS)) == ["Real content."]
