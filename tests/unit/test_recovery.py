"""Which pages count as badly read, and how a recovered page is put back."""

from __future__ import annotations

from rag.config.settings import ExtractSettings
from rag.extract.pdf import PageProbe
from rag.extract.recovery import captured_chars, is_thin, replace_pages, thin_pages
from rag.extract.types import Block, BlockType, Provenance

SETTINGS = ExtractSettings()


def block(text: str, page: int | None) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text, provenance=Provenance(page=page))


def probe(page_no: int, chars: int, tables: bool = False) -> PageProbe:
    return PageProbe(
        page_no=page_no, chars=chars, garbage_ratio=0.0, has_tables=tables, columns=1
    )


def test_a_page_that_lost_half_its_text_is_thin():
    """The probe counted the text layer, so a page holding far less than that
    was read badly whatever the parser reports."""
    captured = {1: 500}
    assert is_thin(probe(0, 1200), captured, SETTINGS) is True


def test_a_page_that_kept_its_text_is_not_thin():
    assert is_thin(probe(0, 1000), {1: 950}, SETTINGS) is False


def test_a_page_missing_entirely_is_thin():
    """Nothing extracted at all is the strongest possible case, and a missing
    key must not read as satisfied."""
    assert is_thin(probe(0, 1000), {}, SETTINGS) is True


def test_a_scanned_page_is_never_thin():
    """It has no text layer to compare against and has already gone to OCR.
    Re-parsing it with a text extractor would produce nothing twice."""
    assert is_thin(probe(0, 12), {}, SETTINGS) is False


def test_captured_chars_counts_per_page():
    blocks = [block("aaa", 1), block("bb", 1), block("cccc", 2)]
    assert captured_chars(blocks) == {1: 5, 2: 4}


def test_blocks_with_no_page_are_not_counted():
    """There is no page to compare them against, and counting them against
    page zero would make a real page look satisfied."""
    assert captured_chars([block("aaa", None)]) == {}


def test_thin_pages_selects_only_the_pages_that_failed():
    blocks = [block("x" * 900, 1), block("y" * 100, 2)]
    probes = [probe(0, 1000), probe(1, 1000)]
    assert [p.page_no for p in thin_pages(blocks, probes, SETTINGS)] == [1]


def test_a_recovered_page_goes_back_where_it_belongs():
    """Page order, not appended at the end, so the table fixup still sees a
    split table's two halves next to each other."""
    original = [block("page one", 1), block("bad page two", 2), block("page three", 3)]
    rescued = [block("good page two", 2)]
    result = replace_pages(original, rescued, {2})
    assert [b.text for b in result] == ["page one", "good page two", "page three"]


def test_only_the_named_pages_are_replaced():
    original = [block("keep", 1), block("drop", 2)]
    result = replace_pages(original, [block("new", 2)], {2})
    assert [b.text for b in result] == ["keep", "new"]


def test_replacing_nothing_leaves_the_document_alone():
    original = [block("one", 1), block("two", 2)]
    assert replace_pages(original, [], set()) == original
