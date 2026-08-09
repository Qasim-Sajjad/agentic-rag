"""Parser protocol and the typed failure extraction can produce."""

from __future__ import annotations

from typing import Protocol

from rag.extract.types import CanonicalDoc


class DocumentParser(Protocol):
    name: str
    version: str

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc: ...


class UnsupportedTypeError(Exception):
    """No parser is registered for this MIME type. Extending support is one entry."""

    def __init__(self, mime: str) -> None:
        super().__init__(f"no parser registered for {mime}")
        self.mime = mime


class ParserUnavailableError(Exception):
    """The parser exists but its dependency or backend is not installed."""


class EmptyExtractionError(Exception):
    """The parser produced no usable content. Dead letter rather than index it."""
