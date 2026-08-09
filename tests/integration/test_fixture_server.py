"""Contract tests for the fixture server. Every later fetch test rests on these.

Thresholds come from config, not from literals, so a settings change that would
invalidate a fixture page fails here rather than three phases later.
"""

from __future__ import annotations

import re

import httpx
import pytest

from rag.config import get_settings

pytestmark = pytest.mark.integration

_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.DOTALL)


def visible_text(html: str) -> str:
    return " ".join(_TAGS.sub(" ", html).split())


def test_static_returns_ok(client: httpx.Client):
    assert client.get("/static").status_code == 200


def test_static_is_html(client: httpx.Client):
    assert client.get("/static").headers["content-type"].startswith("text/html")


def test_static_has_enough_text_to_skip_escalation(client: httpx.Client):
    text = visible_text(client.get("/static").text)
    assert len(text) > get_settings().fetch.min_text_chars


def test_js_only_shell_has_no_rendered_text(client: httpx.Client):
    text = visible_text(client.get("/js-only").text)
    assert len(text) < get_settings().fetch.min_text_chars


def test_js_only_serves_an_empty_spa_root(client: httpx.Client):
    assert '<div id="root"></div>' in client.get("/js-only").text


def test_rate_limited_returns_429(client: httpx.Client):
    assert client.get("/rate-limited").status_code == 429


def test_rate_limited_defaults_over_the_sleep_cap(client: httpx.Client):
    retry_after = int(client.get("/rate-limited").headers["retry-after"])
    assert retry_after > get_settings().fetch.max_retry_after_seconds


def test_rate_limited_honours_a_requested_delay(client: httpx.Client):
    response = client.get("/rate-limited", params={"retry_after": 2})
    assert response.headers["retry-after"] == "2"


def test_challenge_blocks_the_static_tier(client: httpx.Client):
    assert client.get("/challenge").status_code == 503


def test_challenge_body_carries_a_challenge_marker(client: httpx.Client):
    assert "cf_chl_opt" in client.get("/challenge").text


def test_challenge_still_blocks_the_browser_tier(client: httpx.Client):
    response = client.get("/challenge", headers={"x-fixture-tier": "2"})
    assert response.status_code == 503


def test_challenge_passes_at_the_stealth_tier(client: httpx.Client):
    response = client.get("/challenge", headers={"x-fixture-tier": "3"})
    assert response.status_code == 200


def test_challenge_reads_the_user_agent_when_no_tier_header(client: httpx.Client):
    response = client.get("/challenge", headers={"user-agent": "Camoufox/135.0"})
    assert response.status_code == 200


def test_flaky_fails_the_first_two_attempts(client: httpx.Client):
    statuses = [client.get("/flaky").status_code for _ in range(2)]
    assert statuses == [500, 500]


def test_flaky_succeeds_on_the_third_attempt(client: httpx.Client):
    for _ in range(2):
        client.get("/flaky")
    assert client.get("/flaky").status_code == 200


def test_flaky_counter_is_reset_between_tests(client: httpx.Client):
    assert client.get("/__stats").json()["flaky_attempts"] == 0


def test_always_500_never_succeeds(client: httpx.Client):
    statuses = [client.get("/always-500").status_code for _ in range(3)]
    assert statuses == [500, 500, 500]


def test_robots_txt_disallows_the_blocked_path(client: httpx.Client):
    assert "Disallow: /robots-blocked" in client.get("/robots.txt").text


def test_robots_blocked_serves_content_when_asked(client: httpx.Client):
    assert client.get("/robots-blocked").status_code == 200


def test_stats_report_no_hit_for_an_untouched_path(client: httpx.Client):
    hits = client.get("/__stats").json()["hits"]
    assert "/robots-blocked" not in hits


def test_stats_count_every_request(client: httpx.Client):
    for _ in range(2):
        client.get("/static")
    assert client.get("/__stats").json()["hits"]["/static"] == 2


def test_reset_clears_the_hit_counters(client: httpx.Client):
    client.get("/static")
    client.post("/__reset")
    assert client.get("/__stats").json()["hits"] == {}


def test_doc_pdf_is_served_with_the_pdf_content_type(client: httpx.Client):
    assert client.get("/doc.pdf").headers["content-type"] == "application/pdf"


def test_doc_pdf_starts_with_the_pdf_magic_bytes(client: httpx.Client):
    assert client.get("/doc.pdf").content.startswith(b"%PDF")


def test_doc_pdf_carries_a_text_layer(client: httpx.Client):
    assert b"Quarterly filing summary" in client.get("/doc.pdf").content
