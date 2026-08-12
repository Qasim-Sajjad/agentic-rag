"""Document vision access behind a protocol.

Separate from `rag.agent.llm` on purpose. That module serves the agent, and
`extract` must not import from `agent`: extraction runs during a crawl, long
before anything answers a question, and a dependency in that direction would
drag the graph and the MCP layer into the ingest path.

Two implementations, the same split as the agent's client. `AnthropicDocumentReader`
is the real one. `ScriptedDocumentReader` returns queued text, so the OCR parser
and its failure paths are testable without a key, a network call or a bill.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.config.settings import OcrSettings
from rag.log import get_logger

log = get_logger(__name__)


class VisionUnavailableError(RuntimeError):
    """No API key, or the provider rejected the request."""


@dataclass(frozen=True)
class PageText:
    """Transcribed text and the page it was cited from.

    `page_no` is zero based to match `Provenance.page` and the rest of the PDF
    module. The API reports 1 indexed page numbers, so the reader converts.
    """

    text: str
    page_no: int | None = None


@dataclass(frozen=True)
class DocumentReading:
    pages: tuple[PageText, ...]
    model: str

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


class DocumentReader(Protocol):
    async def read(self, pdf: bytes, system: str, user: str) -> DocumentReading: ...


@dataclass
class ScriptedDocumentReader:
    """Deterministic stand in. Pops one queued response per call."""

    responses: list[str] = field(default_factory=list)
    calls: list[tuple[int, str]] = field(default_factory=list)
    model: str = "scripted"

    async def read(self, pdf: bytes, system: str, user: str) -> DocumentReading:
        self.calls.append((len(pdf), user))
        if not self.responses:
            raise VisionUnavailableError(
                "ScriptedDocumentReader ran out of queued responses"
            )
        return DocumentReading((PageText(self.responses.pop(0)),), self.model)


class AnthropicDocumentReader:
    """Sends the page range as a PDF rather than rasterizing it here.

    The API accepts a document block and renders the pages itself, so there is
    no image pipeline to maintain and no resolution constant to get wrong.
    Citations are enabled because the per page numbers they return are what
    fills `Provenance.page`, which is otherwise unknowable from one blob of
    transcribed Markdown.
    """

    def __init__(self, settings: OcrSettings, api_key: str) -> None:
        self._settings = settings
        self._api_key = api_key
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            import anthropic

            if not self._api_key:
                raise VisionUnavailableError(
                    "no API key. Set ANTHROPIC_API_KEY in .env to enable OCR, or "
                    "set extract.ocr.enabled false to skip scanned pages"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def read(self, pdf: bytes, system: str, user: str) -> DocumentReading:
        import anthropic

        client = self._load()
        try:
            response = await client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": _content(pdf, user)}],
                timeout=self._settings.timeout_seconds,
            )
        except anthropic.APIError as exc:
            raise VisionUnavailableError(f"OCR request failed: {exc}") from exc
        _log_refusal(response, self._settings.model)
        return DocumentReading(_pages_of(response), self._settings.model)


def _content(pdf: bytes, user: str) -> list[dict[str, Any]]:
    """Document first, instruction second. The API is clear that a document
    placed after the text block is attended to less reliably."""
    return [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf).decode("ascii"),
            },
            "citations": {"enabled": True},
        },
        {"type": "text", "text": user},
    ]


def _log_refusal(response: Any, model: str) -> None:
    """A refusal is a 200 with an empty body, so it has to be checked rather
    than caught. Reported here and returned as no pages, which the caller turns
    into a skipped range rather than a silently empty document."""
    if getattr(response, "stop_reason", None) == "refusal":
        log.warning("ocr refused", model=model, detail=_refusal_detail(response))


def _refusal_detail(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    return str(getattr(details, "category", "unknown"))


def _pages_of(response: Any) -> tuple[PageText, ...]:
    """One `PageText` per text block, carrying the cited page where the model
    gave one. Blocks without a citation keep `page_no` unset rather than
    inheriting a neighbour's, because a wrong page number is worse than none."""
    pages: list[PageText] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if not text:
            continue
        pages.append(PageText(text=str(text), page_no=_cited_page(block)))
    return tuple(pages)


def _cited_page(block: Any) -> int | None:
    for citation in getattr(block, "citations", None) or []:
        start = getattr(citation, "start_page_number", None)
        if isinstance(start, int):
            # The API is 1 indexed here, `Provenance.page` is 0 indexed.
            return max(0, start - 1)
    return None
