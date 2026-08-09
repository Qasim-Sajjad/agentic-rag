"""Content routing, parsers, CanonicalDoc."""

from rag.extract.protocols import (
    DocumentParser,
    EmptyExtractionError,
    ParserUnavailableError,
    UnsupportedTypeError,
)
from rag.extract.service import ExtractService, PdfRouter, parser_registry
from rag.extract.types import (
    Block,
    BlockType,
    CanonicalDoc,
    DocType,
    Provenance,
    content_hash,
    doc_id_for,
)

__all__ = [
    "Block",
    "BlockType",
    "CanonicalDoc",
    "DocType",
    "DocumentParser",
    "EmptyExtractionError",
    "ExtractService",
    "ParserUnavailableError",
    "PdfRouter",
    "Provenance",
    "UnsupportedTypeError",
    "content_hash",
    "doc_id_for",
    "parser_registry",
]
