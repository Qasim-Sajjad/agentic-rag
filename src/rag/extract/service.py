"""Routing and the extract entry point.

`extract()` takes bytes and a URL. It cannot tell whether a fetcher or a local
file produced them, which is what makes a pasted snippet indistinguishable from
a crawled page downstream.
"""

from __future__ import annotations

from rag.config.settings import ExtractSettings, Settings, get_settings
from rag.extract import router
from rag.extract.ocr import VLMOCRParser
from rag.extract.office import DoclingParser, PlainTextParser, TabularParser
from rag.extract.pdf import (
    PageClass,
    PageRange,
    PyMuPDF4LLMParser,
    merge_split_tables,
    plan_ranges,
    probe_pages,
)
from rag.extract.protocols import (
    DocumentParser,
    EmptyExtractionError,
    ParserUnavailableError,
    UnsupportedTypeError,
)
from rag.extract.types import Block, CanonicalDoc, DocType, content_hash, doc_id_for
from rag.log import get_logger

log = get_logger(__name__)


class PdfRouter:
    """Picks a parser per page range, per the gates in the SPEC."""

    name = "pdf_router"
    version = "1.1"

    def __init__(
        self,
        settings: ExtractSettings | None = None,
        ocr: VLMOCRParser | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings().extract
        self._simple = PyMuPDF4LLMParser()
        self._complex = DoclingParser()
        self._ocr = ocr

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        ranges = plan_ranges(probe_pages(content), self._settings)
        blocks: list[Block] = []
        for page_range in ranges:
            blocks.extend(await self._parse_range(content, page_range, source_url))
        if not blocks:
            raise EmptyExtractionError(f"no usable pages in {source_url}")
        merged = merge_split_tables(blocks)
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=merged,
            content_hash=content_hash(merged),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.PDF,
        )

    async def _parse_range(
        self, content: bytes, page_range: PageRange, source_url: str
    ) -> list[Block]:
        """A range we cannot parse is skipped, not fatal. A 900 page report
        with a scanned appendix is still worth the 880 pages we can read."""
        if page_range.page_class is PageClass.SCANNED:
            return await self._ocr_range(content, page_range, source_url)
        return self._simple.parse_pages(content, page_range.pages)

    async def _ocr_range(
        self, content: bytes, page_range: PageRange, source_url: str
    ) -> list[Block]:
        pages = f"{page_range.start}-{page_range.end}"
        if self._ocr is None:
            log.warning(
                "scanned range skipped, no OCR wired", url=source_url, pages=pages
            )
            return []
        blocks = await self._ocr.parse_pages(content, page_range.pages)
        if not blocks:
            log.warning("scanned range skipped, OCR returned nothing", pages=pages)
        else:
            log.info("scanned range read by OCR", pages=pages, blocks=len(blocks))
        return blocks


def build_ocr(settings: Settings | None = None) -> VLMOCRParser | None:
    """None when OCR is off, which the router turns into a skipped range.

    The Anthropic key is shared with the agent rather than configured twice: it
    is the same account, and a second key would be a second thing to rotate.
    """
    resolved = settings if settings is not None else get_settings()
    if not resolved.extract.ocr.enabled:
        return None
    from rag.extract.vision import AnthropicDocumentReader

    reader = AnthropicDocumentReader(resolved.extract.ocr, resolved.llm.api_key)
    return VLMOCRParser(reader, resolved.extract.ocr)


def parser_registry(
    settings: ExtractSettings | None = None,
    ocr: VLMOCRParser | None = None,
) -> dict[str, DocumentParser]:
    """Extending support is one entry. That is the whole point of the router."""
    from rag.extract.html import TrafilaturaParser

    return {
        router.HTML: TrafilaturaParser(),
        router.PDF: PdfRouter(settings, ocr),
        router.DOCX: DoclingParser(),
        router.XLSX: DoclingParser(),
        router.CSV: TabularParser(),
        router.PLAIN: PlainTextParser(),
    }


class ExtractService:
    def __init__(
        self,
        settings: ExtractSettings | None = None,
        ocr: VLMOCRParser | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings().extract
        resolved_ocr = ocr if ocr is not None else build_ocr()
        self._parsers = parser_registry(self._settings, resolved_ocr)

    def parser_for(
        self, content_type: str, content: bytes
    ) -> tuple[str, DocumentParser]:
        mime = router.resolve_mime(content_type, content)
        parser = self._parsers.get(mime)
        if parser is None:
            raise UnsupportedTypeError(mime)
        return mime, parser

    async def extract(
        self, content: bytes, source_url: str, content_type: str
    ) -> CanonicalDoc:
        """Raises `UnsupportedTypeError` or `EmptyExtractionError`, both of
        which the caller writes to the dead letter store with `stage=extract`."""
        mime, parser = self.parser_for(content_type, content)
        doc = await parser.parse(content, source_url)
        log.info(
            "extracted",
            url=source_url,
            mime=mime,
            parser=parser.name,
            blocks=len(doc.blocks),
        )
        return doc


__all__ = [
    "EmptyExtractionError",
    "ExtractService",
    "ParserUnavailableError",
    "PdfRouter",
    "UnsupportedTypeError",
    "build_ocr",
    "parser_registry",
]
