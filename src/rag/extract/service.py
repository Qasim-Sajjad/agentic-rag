"""Routing and the extract entry point.

`extract()` takes bytes and a URL. It cannot tell whether a fetcher or a local
file produced them, which is what makes a pasted snippet indistinguishable from
a crawled page downstream.
"""

from __future__ import annotations

import asyncio

from rag.config.settings import ExtractSettings, Settings, get_settings
from rag.extract import router
from rag.extract.boilerplate import strip_repeated
from rag.extract.ocr import VLMOCRParser
from rag.extract.office import DoclingParser, PlainTextParser, TabularParser
from rag.extract.pdf import (
    PageClass,
    PageRange,
    PyMuPDF4LLMParser,
    configure_layout,
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
from rag.progress import Progress
from rag.progress import silent as _silent

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
        """The `DocumentParser` protocol shape. Reports nothing."""
        return await self.parse_progress(content, source_url, _silent)

    async def parse_progress(
        self, content: bytes, source_url: str, progress: Progress
    ) -> CanonicalDoc:
        """Same work, reporting each stage as it completes.

        A separate method rather than an optional argument on `parse`, so the
        protocol every other parser implements stays two arguments. `ExtractService`
        looks for this method the way retrieval looks for `embed_queries`.
        """
        configure_layout(self._settings.pymupdf_use_layout)
        report = progress

        def probed(done: int, total: int) -> None:
            """Called from the worker thread. Reporting is a dict assignment on
            the other side, and the reader is a separate request, so there is
            nothing here for a lock to protect."""
            report("probe", done, total, "reading the text layer, detecting tables")

        # Probing opens every page and runs table detection, which is seconds of
        # CPU on a long document. Off the event loop or the whole API stalls.
        probes = await asyncio.to_thread(probe_pages, content, probed)
        ranges = plan_ranges(probes, self._settings)
        report("probe", len(probes), len(probes), f"{len(ranges)} ranges planned")
        blocks = await self._parse_ranges(content, ranges, source_url, report)
        if not blocks:
            raise EmptyExtractionError(f"no usable pages in {source_url}")
        # After reassembly, because furniture is counted across the whole
        # document and a range on its own cannot tell a banner from a sentence.
        merged = strip_repeated(merge_split_tables(blocks), self._settings)
        if not merged:
            raise EmptyExtractionError(f"only page furniture in {source_url}")
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=merged,
            content_hash=content_hash(merged),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.PDF,
        )

    async def _parse_ranges(
        self,
        content: bytes,
        ranges: list[PageRange],
        source_url: str,
        report: Progress,
    ) -> list[Block]:
        """Ranges are independent by construction, so they run concurrently.

        `plan_ranges` has always split a long document into tasks; until now the
        caller awaited them one at a time, which made the split bookkeeping
        rather than parallelism. `gather` preserves input order, so the blocks
        still come back in page order and `merge_split_tables` still sees a
        table's two halves adjacent.
        """
        limit = asyncio.Semaphore(max(1, self._settings.max_parallel_ranges))
        done = 0
        parallel = max(1, self._settings.max_parallel_ranges)
        # Announced before the first range finishes, so the stage exists in the
        # progress list while it is still working rather than appearing at the
        # end. A stage that is absent reads as a stage that has not started.
        report("extract", 0, len(ranges), f"{parallel} ranges at a time")

        async def one(page_range: PageRange) -> list[Block]:
            nonlocal done
            async with limit:
                blocks = await self._parse_range(content, page_range, source_url)
            done += 1
            report(
                "extract",
                done,
                len(ranges),
                f"pages {page_range.start}-{page_range.end}, "
                f"{page_range.page_class}, {len(blocks)} blocks",
            )
            return blocks

        results = await asyncio.gather(*(one(entry) for entry in ranges))
        return [block for group in results for block in group]

    async def _parse_range(
        self, content: bytes, page_range: PageRange, source_url: str
    ) -> list[Block]:
        """A range we cannot parse is skipped, not fatal. A 900 page report
        with a scanned appendix is still worth the 880 pages we can read."""
        if page_range.page_class is PageClass.SCANNED:
            return await self._ocr_range(content, page_range, source_url)
        # Synchronous PyMuPDF work. Threaded so a 500 page extract does not hold
        # the event loop and time out every other request on the server.
        return await asyncio.to_thread(
            self._simple.parse_pages, content, page_range.pages
        )

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
        self,
        content: bytes,
        source_url: str,
        content_type: str,
        progress: Progress | None = None,
    ) -> CanonicalDoc:
        """Raises `UnsupportedTypeError` or `EmptyExtractionError`, both of
        which the caller writes to the dead letter store with `stage=extract`."""
        mime, parser = self.parser_for(content_type, content)
        doc = await self._run(parser, content, source_url, progress)
        log.info(
            "extracted",
            url=source_url,
            mime=mime,
            parser=parser.name,
            blocks=len(doc.blocks),
        )
        return doc

    async def _run(
        self,
        parser: DocumentParser,
        content: bytes,
        source_url: str,
        progress: Progress | None,
    ) -> CanonicalDoc:
        """Only the PDF router reports stages, because only it has stages worth
        reporting. Everything else is one pass over one document."""
        detailed = getattr(parser, "parse_progress", None)
        if progress is not None and detailed is not None:
            result: CanonicalDoc = await detailed(content, source_url, progress)
            return result
        return await parser.parse(content, source_url)


__all__ = [
    "EmptyExtractionError",
    "ExtractService",
    "ParserUnavailableError",
    "PdfRouter",
    "UnsupportedTypeError",
    "build_ocr",
    "parser_registry",
]
