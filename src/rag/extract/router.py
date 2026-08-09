"""Content type routing on headers plus magic bytes. Never on the URL extension.

URLs like `/download?id=8821` return PDFs, and plenty of servers label a PDF
`application/octet-stream`. Magic bytes win when the two disagree.
"""

from __future__ import annotations

from rag.extract.types import DocType

HTML = "text/html"
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV = "text/csv"
PLAIN = "text/plain"

# Magic bytes are authoritative. A Content-Type header is a claim, not a fact.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", PDF),
    (b"PK\x03\x04", DOCX),  # any OOXML container, refined below
    (b"<!DOCTYPE html", HTML),
    (b"<!doctype html", HTML),
    (b"<html", HTML),
    (b"<HTML", HTML),
)

DOC_TYPES: dict[str, DocType] = {
    HTML: DocType.HTML,
    PDF: DocType.PDF,
    DOCX: DocType.OFFICE,
    XLSX: DocType.OFFICE,
    CSV: DocType.TEXT,
    PLAIN: DocType.TEXT,
}


def normalize_mime(content_type: str) -> str:
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def sniff(content: bytes) -> str | None:
    """MIME from the first bytes, or None when nothing matches."""
    head = content[:512]
    for signature, mime in _MAGIC:
        if head.startswith(signature):
            return mime
    stripped = head.lstrip()[:64].lower()
    if stripped.startswith((b"<!doctype html", b"<html")):
        return HTML
    return None


def resolve_mime(content_type: str, content: bytes) -> str:
    """Header first, magic bytes overrule it when they disagree.

    An `octet-stream` header on a real PDF is common enough that trusting the
    header alone sends documents to the dead letter store for no reason.
    """
    declared = normalize_mime(content_type)
    sniffed = sniff(content)
    if sniffed is None:
        return declared
    return sniffed if _magic_wins(declared, sniffed) else declared


def _magic_wins(declared: str, sniffed: str) -> bool:
    if declared in ("", "application/octet-stream", "binary/octet-stream"):
        return True
    if sniffed == PDF and declared != PDF:
        return True
    return declared not in DOC_TYPES


def doc_type_for(mime: str) -> DocType:
    return DOC_TYPES.get(mime, DocType.TEXT)
