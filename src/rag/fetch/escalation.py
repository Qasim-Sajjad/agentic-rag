"""Decides what a response means. The only place escalation policy lives.

Fetchers return whatever the server said. This module turns that into one of a
few verdicts, so adding a rung to the ladder never means editing detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from rag.config.settings import FetchSettings
from rag.fetch.types import FetchResult

BLOCK_STATUSES = frozenset({403, 503})
RATE_LIMIT_STATUS = 429
BLOCK_HEADERS = ("cf-mitigated", "x-datadome")

# An empty single page app shell. Rendering is the only way to read these.
SPA_MARKERS = ('<div id="root"></div>', '<div id="app"></div>', "__NEXT_DATA__")

_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.DOTALL)
_NOSCRIPT = re.compile(r"<noscript.*?</noscript>", re.DOTALL | re.IGNORECASE)


class Verdict(StrEnum):
    OK = "ok"
    ESCALATE = "escalate"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"


@dataclass(frozen=True)
class Assessment:
    verdict: Verdict
    detail: str


def visible_text(html: str) -> str:
    """Rough text extraction, enough to decide whether a page rendered.

    Deliberately not trafilatura. This runs before the extract module and must
    stay cheap, since it runs on every response including the ones we discard.
    """
    without_noscript = _NOSCRIPT.sub(" ", html)
    return " ".join(_TAGS.sub(" ", without_noscript).split())


def is_html(content_type: str) -> bool:
    return "html" in content_type.lower() or "xml" in content_type.lower()


def assess(result: FetchResult, settings: FetchSettings) -> Assessment:
    """Verdict for one response. Never raises.

    Order matters. Rate limiting and 404 are decided before block signatures,
    because a 429 requeues and a 404 stops, and neither is worth a browser.
    """
    checks = (_terminal_status, _block_verdict, _error_status, _emptiness_verdict)
    for check in checks:
        found = check(result, settings)
        if found is not None:
            return found
    return Assessment(Verdict.OK, "")


def _terminal_status(result: FetchResult, settings: FetchSettings) -> Assessment | None:
    if result.status == RATE_LIMIT_STATUS:
        return Assessment(Verdict.RATE_LIMITED, "status 429")
    if result.status == 404:
        return Assessment(Verdict.NOT_FOUND, "status 404")
    return None


def _block_verdict(result: FetchResult, settings: FetchSettings) -> Assessment | None:
    blocked = _block_signature(result, settings)
    return Assessment(Verdict.ESCALATE, blocked) if blocked is not None else None


def _error_status(result: FetchResult, settings: FetchSettings) -> Assessment | None:
    """A 5xx is retried in place. Escalating a broken server buys nothing."""
    if result.status >= 500:
        return Assessment(Verdict.SERVER_ERROR, f"status {result.status}")
    if result.status >= 400:
        return Assessment(Verdict.ESCALATE, f"status {result.status}")
    return None


def _emptiness_verdict(
    result: FetchResult, settings: FetchSettings
) -> Assessment | None:
    empty = _emptiness(result, settings)
    return Assessment(Verdict.ESCALATE, empty) if empty is not None else None


def _block_signature(result: FetchResult, settings: FetchSettings) -> str | None:
    if result.status in BLOCK_STATUSES:
        return f"block status {result.status}"
    header = next((h for h in BLOCK_HEADERS if h in result.headers), None)
    if header is not None:
        return f"block header {header!r}"
    marker = _challenge_marker(result, settings.challenge_markers)
    if marker is not None:
        return f"challenge marker {marker!r}"
    return None


def _challenge_marker(result: FetchResult, markers: tuple[str, ...]) -> str | None:
    if not is_html(result.content_type):
        return None
    body = result.content.decode("utf-8", errors="replace").lower()
    return next((marker for marker in markers if marker in body), None)


def _emptiness(result: FetchResult, settings: FetchSettings) -> str | None:
    """Emptiness only means something for HTML. A PDF has no rendered text."""
    if not is_html(result.content_type):
        return None
    body = result.content.decode("utf-8", errors="replace")
    text = visible_text(body)
    if len(text) < settings.min_text_chars:
        spa = next((m for m in SPA_MARKERS if m in body), None)
        if spa is not None:
            return f"empty spa root {spa!r}"
        return f"only {len(text)} chars of text, under {settings.min_text_chars}"
    return None
