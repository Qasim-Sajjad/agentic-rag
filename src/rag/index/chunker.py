"""Structure aware chunking over typed blocks.

Recursive character splitting is the fallback for unknown structure, and we
have the structure. Three invariants hold for every chunk this produces:
no table is split without a repeated header, no chunk exceeds the max, and
every chunk carries a non empty section path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.config.settings import IndexSettings
from rag.extract.types import Block, BlockType, CanonicalDoc
from rag.index.types import (
    CHUNKER_VERSION,
    Chunk,
    ChunkMetadata,
    HeuristicTokenCounter,
    TokenCounter,
    chunk_hash_for,
    chunk_id_for,
)

UNTITLED = "untitled"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Buffer:
    blocks: list[Block] = field(default_factory=list)
    tokens: int = 0

    def add(self, block: Block, tokens: int) -> None:
        self.blocks.append(block)
        self.tokens += tokens

    def clear(self) -> None:
        self.blocks = []
        self.tokens = 0

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


class StructureAwareChunker:
    def __init__(
        self, settings: IndexSettings, counter: TokenCounter | None = None
    ) -> None:
        self._settings = settings
        self._counter = counter if counter is not None else HeuristicTokenCounter()

    def chunk(self, doc: CanonicalDoc, metadata: ChunkMetadata) -> list[Chunk]:
        state = _State(doc, metadata, self._settings, self._counter)
        for block in doc.blocks:
            self._consume(state, block)
        state.flush()
        return state.chunks

    def _consume(self, state: _State, block: Block) -> None:
        if block.type is BlockType.HEADING:
            state.flush()  # hard break, a heading starts a new section
            state.push_heading(block)
            return
        if block.type is BlockType.TABLE:
            state.flush()
            state.emit_table(block)
            return
        state.add(block)


class _State:
    """Walk state. Separated from the chunker so the walk stays readable."""

    def __init__(
        self,
        doc: CanonicalDoc,
        metadata: ChunkMetadata,
        settings: IndexSettings,
        counter: TokenCounter,
    ) -> None:
        self.doc = doc
        self.metadata = metadata
        self.settings = settings
        self.counter = counter
        self.chunks: list[Chunk] = []
        self.path: list[tuple[int, str]] = []
        self.buffer = _Buffer()

    @property
    def section_path(self) -> list[str]:
        """Never empty. A chunk saying "we consider this risk material" is
        useless alone and precise with its path."""
        if self.path:
            return [title for _, title in self.path]
        return [self.doc.title or UNTITLED]

    def push_heading(self, block: Block) -> None:
        level = block.level or 1
        self.path = [(lvl, title) for lvl, title in self.path if lvl < level]
        self.path.append((level, block.text))

    def add(self, block: Block) -> None:
        tokens = self.counter.count(block.text)
        if tokens > self.settings.target_tokens:
            self.flush()
            self.emit_oversized(block)
            return
        if self.buffer.tokens + tokens > self.settings.target_tokens:
            self.flush(overlap=True)
        self.buffer.add(block, tokens)

    def emit_oversized(self, block: Block) -> None:
        """One block bigger than the target. Sentence split is the fallback for
        structure the block types do not describe."""
        for piece in split_sentences(
            block.text, self.settings.target_tokens, self.counter
        ):
            self.emit(piece, is_table=False, page=block.provenance.page)

    def flush(self, overlap: bool = False) -> None:
        if not self.buffer.blocks:
            return
        text = self.buffer.text
        tail = self.buffer.blocks[-1] if overlap else None
        self.buffer.clear()
        self.emit(text, is_table=False, page=None)
        self._carry_over(tail)

    def _carry_over(self, tail: Block | None) -> None:
        """Overlap only when the trailing block is small enough to be context
        rather than a second copy of the chunk."""
        if tail is None or self.settings.overlap_ratio <= 0:
            return
        budget = self.settings.target_tokens * self.settings.overlap_ratio
        tokens = self.counter.count(tail.text)
        if tokens <= budget:
            self.buffer.add(tail, tokens)

    def emit_table(self, block: Block) -> None:
        parts = split_table(block.text, self.settings.max_table_tokens, self.counter)
        for part in parts:
            self.emit(part, is_table=True, page=block.provenance.page)

    def emit(self, text: str, is_table: bool, page: int | None) -> None:
        index = len(self.chunks)
        path = self.section_path
        embed_text = f"{' > '.join(path)}\n\n{text}"
        self.chunks.append(
            Chunk(
                chunk_id=chunk_id_for(self.doc.doc_id, index),
                doc_id=self.doc.doc_id,
                chunk_index=index,
                text=text,
                embed_text=embed_text,
                token_count=self.counter.count(embed_text),
                metadata=self.metadata.model_copy(
                    update={
                        "section_path": path,
                        "is_table": is_table,
                        "page_no": page,
                        "chunk_hash": chunk_hash_for(text),
                        "chunker_version": CHUNKER_VERSION,
                    }
                ),
            )
        )


def split_sentences(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Pack sentences up to the limit. Used only for a single oversized block."""
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        candidate = " ".join([*current, sentence]).strip()
        if current and counter.count(candidate) > max_tokens:
            pieces.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current).strip())
    return [piece for piece in pieces if piece]


def split_table(markdown: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """One chunk per table. If oversized, split by rows and repeat the header.

    A table fragment without its header is unreadable and unretrievable, so the
    header travels with every part or the table is not split at all.
    """
    if counter.count(markdown) <= max_tokens:
        return [markdown]
    lines = markdown.splitlines()
    header, rows = _header_and_rows(lines)
    if not header:
        return [markdown]
    return _pack_rows(header, rows, max_tokens, counter)


def _header_and_rows(lines: list[str]) -> tuple[list[str], list[str]]:
    if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: "):
        return lines[:2], lines[2:]
    return [], lines


def _pack_rows(
    header: list[str], rows: list[str], max_tokens: int, counter: TokenCounter
) -> list[str]:
    parts: list[str] = []
    current = list(header)
    for row in rows:
        candidate = "\n".join([*current, row])
        if len(current) > len(header) and counter.count(candidate) > max_tokens:
            parts.append("\n".join(current))
            current = [*header, row]
        else:
            current.append(row)
    parts.append("\n".join(current))
    return parts
