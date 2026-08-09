"""Escalation decisions. Block signatures, emptiness, and what must not escalate."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.config.settings import FetchSettings
from rag.fetch.escalation import Verdict, assess, visible_text
from rag.fetch.types import FetchResult, FetchTier

SETTINGS = FetchSettings()

REAL_PAGE = (
    "<html><body><article><h1>Quarterly filing</h1><p>"
    + "Revenue rose nine percent against the same quarter last year. " * 6
    + "</p></article></body></html>"
)


def result(
    body: str = REAL_PAGE,
    status: int = 200,
    content_type: str = "text/html",
    headers: dict[str, str] | None = None,
) -> FetchResult:
    return FetchResult(
        url="https://example.test/a",
        final_url="https://example.test/a",
        status=status,
        content=body.encode("utf-8"),
        content_type=content_type,
        tier_used=FetchTier.STATIC,
        attempts=1,
        fetched_at=datetime.now(UTC),
        headers=headers or {},
    )


def test_clean_page_does_not_escalate():
    assert assess(result(), SETTINGS).verdict is Verdict.OK


def test_403_is_a_block_signature():
    assert assess(result(status=403), SETTINGS).verdict is Verdict.ESCALATE


def test_503_is_a_block_signature():
    assert assess(result(status=503), SETTINGS).verdict is Verdict.ESCALATE


def test_429_is_rate_limited_not_escalation():
    """Rate limiting is temporary. Escalating to a browser would not help."""
    assert assess(result(status=429), SETTINGS).verdict is Verdict.RATE_LIMITED


def test_404_never_escalates():
    assert assess(result(status=404), SETTINGS).verdict is Verdict.NOT_FOUND


def test_500_retries_in_place_rather_than_escalating():
    assert assess(result(status=500), SETTINGS).verdict is Verdict.SERVER_ERROR


def test_cf_mitigated_header_escalates():
    blocked = result(headers={"cf-mitigated": "challenge"})
    assert assess(blocked, SETTINGS).verdict is Verdict.ESCALATE


def test_challenge_marker_in_body_escalates():
    page = "<html><body><h1>Just a moment...</h1></body></html>"
    assert assess(result(body=page), SETTINGS).verdict is Verdict.ESCALATE


def test_challenge_marker_detection_is_case_insensitive():
    page = "<html><body>JUST A MOMENT while we check</body></html>"
    assert assess(result(body=page), SETTINGS).verdict is Verdict.ESCALATE


def test_empty_spa_root_escalates():
    page = '<html><body><div id="root"></div></body></html>'
    assert "empty spa root" in assess(result(body=page), SETTINGS).detail


def test_short_page_escalates_on_the_configured_floor():
    assert assess(result(body="<html><body>hi</body></html>"), SETTINGS).verdict is (
        Verdict.ESCALATE
    )


def test_a_pdf_is_never_judged_empty():
    """Emptiness is an HTML heuristic. A PDF has no rendered text to count."""
    pdf = result(body="%PDF-1.4 tiny", content_type="application/pdf")
    assert assess(pdf, SETTINGS).verdict is Verdict.OK


def test_noscript_only_page_escalates():
    page = (
        "<html><body><noscript>Enable JavaScript to continue</noscript></body></html>"
    )
    assert assess(result(body=page), SETTINGS).verdict is Verdict.ESCALATE


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>one two</p>", "one two"),
        ("<script>var x = 'hidden'</script><p>shown</p>", "shown"),
        ("<style>p{color:red}</style><p>shown</p>", "shown"),
    ],
)
def test_visible_text_drops_markup_and_scripts(html, expected):
    assert visible_text(html) == expected
