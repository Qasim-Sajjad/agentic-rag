"""Chunker invariants. These are the three that must never break."""

from __future__ import annotations

from rag.config.settings import IndexSettings
from rag.extract.types import Block, BlockType, CanonicalDoc, DocType
from rag.index.chunker import StructureAwareChunker, split_table
from rag.index.types import ChunkMetadata, HeuristicTokenCounter

SETTINGS = IndexSettings()
COUNTER = HeuristicTokenCounter()

META = ChunkMetadata(doc_type=DocType.HTML, domain="example.test", source_id="s")

TABLE = (
    "| Segment | Revenue |\n| --- | --- |\n| Subscription | 26.0 |\n| Services | 15.2 |"
)
PROSE = "We consider this risk material because concentration remains high. " * 4


def doc(blocks: list[Block], title: str | None = "Annual report") -> CanonicalDoc:
    return CanonicalDoc(
        doc_id="d1", source_url="https://example.test/a", title=title, blocks=blocks
    )


def chunk(
    blocks: list[Block], settings: IndexSettings = SETTINGS, title="Annual report"
):
    return StructureAwareChunker(settings, COUNTER).chunk(doc(blocks, title), META)


def heading(text: str, level: int = 1) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=level)


def para(text: str = PROSE) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text)


def test_every_chunk_has_a_non_empty_section_path():
    chunks = chunk(
        [heading("Risk factors"), para(), Block(type=BlockType.TABLE, text=TABLE)]
    )
    assert all(c.metadata.section_path for c in chunks)


def test_a_document_with_no_headings_falls_back_to_its_title():
    chunks = chunk([para()])
    assert chunks[0].metadata.section_path == ["Annual report"]


def test_a_document_with_no_headings_and_no_title_still_has_a_path():
    chunks = chunk([para()], title=None)
    assert chunks[0].metadata.section_path == ["untitled"]


def test_the_heading_path_is_prepended_to_the_embedded_text():
    """A chunk saying "we consider this risk material" is useless alone."""
    chunks = chunk([heading("Annual report"), heading("Risk factors", 2), para()])
    assert chunks[0].embed_text.startswith("Annual report > Risk factors")


def test_a_deeper_heading_extends_the_path():
    chunks = chunk([heading("A"), heading("B", 2), para()])
    assert chunks[0].metadata.section_path == ["A", "B"]


def test_a_sibling_heading_replaces_the_deeper_one():
    chunks = chunk([heading("A"), heading("B", 2), para(), heading("C", 2), para()])
    assert chunks[-1].metadata.section_path == ["A", "C"]


def test_a_table_is_never_split_when_it_fits():
    chunks = chunk([heading("Data"), Block(type=BlockType.TABLE, text=TABLE)])
    tables = [c for c in chunks if c.metadata.is_table]
    assert len(tables) == 1


def test_a_table_chunk_is_marked_as_a_table():
    chunks = chunk([Block(type=BlockType.TABLE, text=TABLE)])
    assert chunks[0].metadata.is_table


def test_a_heading_forces_a_chunk_boundary():
    chunks = chunk([heading("A"), para("short one"), heading("B"), para("short two")])
    assert len(chunks) == 2


def test_no_chunk_exceeds_the_target_by_more_than_one_block():
    settings = IndexSettings(target_tokens=64)
    chunks = chunk([para() for _ in range(10)], settings)
    assert all(c.token_count <= settings.target_tokens * 2 for c in chunks)


def test_chunk_ids_are_deterministic():
    first = chunk([heading("A"), para()])
    second = chunk([heading("A"), para()])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_an_oversized_table_repeats_its_header_in_every_part():
    rows = "\n".join(f"| Row {n} | {n}.0 |" for n in range(200))
    big = f"| Segment | Revenue |\n| --- | --- |\n{rows}"
    parts = split_table(big, max_tokens=200, counter=COUNTER)
    assert len(parts) > 1
    assert all(part.startswith("| Segment | Revenue |") for part in parts)


def test_an_oversized_table_keeps_the_separator_row_in_every_part():
    rows = "\n".join(f"| Row {n} | {n}.0 |" for n in range(200))
    big = f"| Segment | Revenue |\n| --- | --- |\n{rows}"
    parts = split_table(big, max_tokens=200, counter=COUNTER)
    assert all("| --- | --- |" in part for part in parts)


def test_a_table_with_no_header_is_left_whole_rather_than_split():
    """Splitting a headerless table produces unreadable fragments."""
    rows = "\n".join(f"| Row {n} | {n}.0 |" for n in range(200))
    assert len(split_table(rows, max_tokens=50, counter=COUNTER)) == 1
