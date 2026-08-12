"""VLM OCR for page ranges with no usable text layer.

Two rules are written down here because they are the design, and neither is
obvious from the code alone:

1. Pass any extracted text layer alongside the image, even a poor one. The model
   aligns to it instead of free generating, which is the difference between a
   wrong digit and a plausible invented one.
2. Set confidence below 1.0 on every OCR block. VLM errors are fluent and pass
   spell checks, so they reach the index looking correct. Confidence propagation
   is the mitigation, not accuracy.

A vision model rather than a classic OCR engine because the scanned pages that
survive the text layer gate are the ones with tables and multi column layout,
which is exactly where character level OCR loses the structure the chunker needs.
"""

from __future__ import annotations

from rag.config.settings import OcrSettings
from rag.extract.html import blocks_from_markdown
from rag.extract.protocols import EmptyExtractionError, ParserUnavailableError
from rag.extract.types import (
    Block,
    CanonicalDoc,
    DocType,
    Provenance,
    content_hash,
    doc_id_for,
)
from rag.extract.vision import DocumentReader, DocumentReading, VisionUnavailableError
from rag.log import get_logger
from rag.prompts.registry import PromptRegistry

log = get_logger(__name__)

#: Never 1.0. See rule 2 above: this number is what tells everything downstream
#: that the text was transcribed rather than read.
OCR_CONFIDENCE = 0.7

OCR_ROLE = "ocr"


class VLMOCRParser:
    name = "vlm_ocr"
    version = "1.0"

    def __init__(
        self,
        reader: DocumentReader,
        settings: OcrSettings,
        prompts: PromptRegistry | None = None,
    ) -> None:
        self._reader = reader
        self._settings = settings
        self._prompts = prompts if prompts is not None else PromptRegistry()

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        blocks = await self.parse_pages(content, None)
        if not blocks:
            raise EmptyExtractionError(f"OCR returned nothing for {source_url}")
        return CanonicalDoc(
            doc_id=doc_id_for(source_url),
            source_url=source_url,
            blocks=blocks,
            content_hash=content_hash(blocks),
            extractor_name=self.name,
            extractor_version=self.version,
            doc_type=DocType.PDF,
        )

    async def parse_pages(self, content: bytes, pages: list[int] | None) -> list[Block]:
        """Returns an empty list rather than raising when OCR is off or refused.

        The caller treats a scanned range it cannot read as skipped, which is the
        established behaviour: a 900 page report with an unreadable appendix is
        still worth the 880 pages that read cleanly.
        """
        if not self._settings.enabled:
            return []
        selected = pages if pages is not None else _all_pages(content)
        blocks: list[Block] = []
        for batch in _batched(selected, self._settings.max_pages_per_call):
            blocks.extend(await self._read_batch(content, batch))
        return blocks

    async def _read_batch(self, content: bytes, pages: list[int]) -> list[Block]:
        try:
            reading = await self._reader.read(
                extract_pages(content, pages),
                self._prompts.get(OCR_ROLE).text,
                _user_message(_text_layer(content, pages, self._settings)),
            )
        except VisionUnavailableError as exc:
            log.warning("ocr unavailable", pages=_label(pages), error=str(exc))
            return []
        return _to_blocks(reading, pages)


def extract_pages(content: bytes, pages: list[int]) -> bytes:
    """A new single range PDF, so the request carries only the pages that failed
    the gate rather than the whole document."""
    import pymupdf

    with pymupdf.open(stream=content, filetype="pdf") as source:
        extracted = pymupdf.open()
        try:
            extracted.insert_pdf(source, from_page=min(pages), to_page=max(pages))
            return bytes(extracted.tobytes())
        finally:
            extracted.close()


def _all_pages(content: bytes) -> list[int]:
    import pymupdf

    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return list(range(doc.page_count))


def _text_layer(content: bytes, pages: list[int], settings: OcrSettings) -> str:
    """Rule 1. Whatever the failed extraction did manage to read, truncated,
    because past a few thousand characters it stops disambiguating and starts
    competing with the image."""
    import pymupdf

    with pymupdf.open(stream=content, filetype="pdf") as doc:
        parts = [str(doc[number].get_text()) for number in pages if number < len(doc)]
    return "\n".join(parts)[: settings.text_layer_hint_chars].strip()


def _user_message(hint: str) -> str:
    """The hint is delimited and named, so the model treats it as reference
    material rather than as part of its instructions."""
    if not hint:
        return "Transcribe these pages. No text layer could be extracted from them."
    return f"Transcribe these pages.\n\n<text_layer_hint>\n{hint}\n</text_layer_hint>"


def _batched(pages: list[int], size: int) -> list[list[int]]:
    step = max(1, size)
    return [pages[start : start + step] for start in range(0, len(pages), step)]


def _to_blocks(reading: DocumentReading, pages: list[int]) -> list[Block]:
    """Reuses the Markdown block parser, so an OCR'd table is the same `Block`
    shape as a table from the text layer and nothing downstream can tell which
    parser produced it. Confidence is the one field that can."""
    fallback = min(pages) if pages else 0
    blocks: list[Block] = []
    for page in reading.pages:
        number = page.page_no if page.page_no is not None else fallback
        blocks.extend(
            block.model_copy(
                update={
                    "provenance": Provenance(page=number),
                    "confidence": OCR_CONFIDENCE,
                }
            )
            for block in blocks_from_markdown(page.text)
        )
    return blocks


def _label(pages: list[int]) -> str:
    return f"{min(pages)}-{max(pages)}" if pages else "none"


__all__ = [
    "OCR_CONFIDENCE",
    "EmptyExtractionError",
    "ParserUnavailableError",
    "VLMOCRParser",
    "extract_pages",
]
