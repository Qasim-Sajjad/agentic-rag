"""PDF gates and the page range reassembly fixup."""

from __future__ import annotations

from rag.config.settings import ExtractSettings
from rag.extract.pdf import (
    PageClass,
    PageProbe,
    classify_page,
    garbage_ratio,
    merge_split_tables,
    plan_ranges,
)
from rag.extract.types import Block, BlockType

SETTINGS = ExtractSettings()


def probe(
    chars: int = 500, garbage: float = 0.0, tables: bool = False, columns: int = 1
):
    return PageProbe(0, chars, garbage, tables, columns)


def test_a_digital_born_page_is_simple_text():
    assert classify_page(probe(), SETTINGS) is PageClass.SIMPLE_TEXT


def test_a_page_with_no_text_layer_is_scanned():
    assert classify_page(probe(chars=10), SETTINGS) is PageClass.SCANNED


def test_broken_font_encoding_is_treated_as_scanned():
    """Fluent nonsense passes every emptiness check, so the gate must catch it."""
    assert classify_page(probe(garbage=0.5), SETTINGS) is PageClass.SCANNED


def test_a_page_with_tables_is_complex():
    assert classify_page(probe(tables=True), SETTINGS) is PageClass.COMPLEX_TEXT


def test_a_two_column_page_is_complex():
    assert classify_page(probe(columns=2), SETTINGS) is PageClass.COMPLEX_TEXT


def test_garbage_ratio_is_zero_for_clean_text():
    assert garbage_ratio("Revenue rose nine percent.") == 0.0


def test_garbage_ratio_counts_replacement_characters():
    assert garbage_ratio("ab��") == 0.5


def test_ranges_split_at_the_task_size():
    settings = ExtractSettings(pages_per_task=2)
    probes = [PageProbe(n, 500, 0.0, False, 1) for n in range(5)]
    assert len(plan_ranges(probes, settings)) == 3


def test_ranges_break_when_the_page_class_changes():
    probes = [
        PageProbe(0, 500, 0.0, False, 1),
        PageProbe(1, 5, 0.0, False, 1),
        PageProbe(2, 500, 0.0, False, 1),
    ]
    classes = [r.page_class for r in plan_ranges(probes, SETTINGS)]
    assert classes == [
        PageClass.SIMPLE_TEXT,
        PageClass.SCANNED,
        PageClass.SIMPLE_TEXT,
    ]


def table(text: str) -> Block:
    return Block(type=BlockType.TABLE, text=text)


HEADED = "| Segment | Revenue |\n| --- | --- |\n| Subscription | 26.0 |"
CONTINUATION = "| Services | 15.2 |\n| Other | 1.1 |"


def test_a_headerless_table_continuation_is_merged():
    merged = merge_split_tables([table(HEADED), table(CONTINUATION)])
    assert len(merged) == 1


def test_the_merged_table_keeps_both_halves():
    merged = merge_split_tables([table(HEADED), table(CONTINUATION)])
    assert "Services" in merged[0].text and "Subscription" in merged[0].text


def test_a_table_with_its_own_header_is_not_merged():
    assert len(merge_split_tables([table(HEADED), table(HEADED)])) == 2


def test_a_table_with_a_different_column_count_is_not_merged():
    other = "| A | B | C |\n| 1 | 2 | 3 |"
    assert len(merge_split_tables([table(HEADED), table(other)])) == 2


def test_a_paragraph_between_tables_prevents_merging():
    para = Block(type=BlockType.PARAGRAPH, text="Some prose.")
    assert len(merge_split_tables([table(HEADED), para, table(CONTINUATION)])) == 3
