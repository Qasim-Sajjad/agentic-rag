"""Tier 4. Provider side failures are ours, target site failures are the site's.

Nothing here touches the network: `httpx.MockTransport` answers every request,
so the account error mapping and the origin status handling are testable without
spending a credit.
"""

from __future__ import annotations

import httpx
import pytest

from rag.config.settings import FetchSettings, UnlockerSettings
from rag.fetch.protocols import (
    FetchTimeoutError,
    FetchTransportError,
    UnlockerNotConfiguredError,
)
from rag.fetch.types import FetchTier
from rag.fetch.unlocker import UnlockerFetcher

KEY = "test-key"


def settings(api_key: str = KEY, **overrides) -> FetchSettings:
    unlocker = UnlockerSettings(api_key=api_key, **overrides)
    return FetchSettings(unlocker=unlocker)


def fetcher_returning(response: httpx.Response, **overrides) -> UnlockerFetcher:
    """Wires a mock transport into the fetcher's own client."""
    fetcher = UnlockerFetcher(settings(**overrides))
    fetcher._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response)
    )
    return fetcher


def capturing_fetcher(response: httpx.Response) -> tuple[UnlockerFetcher, list]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    fetcher = UnlockerFetcher(settings())
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return fetcher, seen


def raising_fetcher(error: Exception) -> UnlockerFetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    fetcher = UnlockerFetcher(settings())
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return fetcher


def test_no_key_means_the_tier_is_not_configured():
    assert UnlockerFetcher(settings(api_key="")).configured is False


def test_a_key_configures_the_tier():
    assert UnlockerFetcher(settings()).configured is True


async def test_a_missing_key_refuses_before_any_request():
    """Cheaper and clearer than letting the provider reject it."""
    fetcher = UnlockerFetcher(settings(api_key=""))
    with pytest.raises(UnlockerNotConfiguredError, match="no API key"):
        await fetcher.fetch("https://blocked.test/a", 10.0)


async def test_the_page_comes_back_as_a_tier_four_result():
    body = b"<html><body>the real page</body></html>"
    fetcher = fetcher_returning(
        httpx.Response(200, content=body, headers={"content-type": "text/html"})
    )
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.tier_used is FetchTier.UNLOCKER
    assert result.content == body
    assert result.status == 200
    await fetcher.close()


async def test_the_api_key_and_url_are_sent_as_parameters():
    fetcher, seen = capturing_fetcher(httpx.Response(200, content=b"ok"))
    await fetcher.fetch("https://blocked.test/a", 10.0)
    params = seen[0].url.params
    assert params["api_key"] == KEY
    assert params["url"] == "https://blocked.test/a"
    await fetcher.close()


async def test_booleans_are_sent_lowercased():
    """`True` would reach the provider as the string 'True', which it rejects."""
    fetcher, seen = capturing_fetcher(httpx.Response(200, content=b"ok"))
    await fetcher.fetch("https://blocked.test/a", 10.0)
    params = seen[0].url.params
    assert params["render_js"] == "true"
    assert params["premium_proxy"] == "true"
    await fetcher.close()


async def test_an_empty_country_code_is_omitted_rather_than_sent_blank():
    fetcher, seen = capturing_fetcher(httpx.Response(200, content=b"ok"))
    await fetcher.fetch("https://blocked.test/a", 10.0)
    assert "country_code" not in seen[0].url.params
    await fetcher.close()


async def test_the_origin_status_is_reported_over_the_proxy_status():
    """The ladder's block signatures key off what the site said. A 200 from the
    proxy wrapping a 403 from the origin must read as 403."""
    fetcher = fetcher_returning(
        httpx.Response(200, content=b"denied", headers={"spb-original-status": "403"})
    )
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.status == 403
    await fetcher.close()


async def test_a_missing_origin_status_falls_back_to_the_proxy_status():
    fetcher = fetcher_returning(httpx.Response(200, content=b"ok"))
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.status == 200
    await fetcher.close()


async def test_an_unparseable_origin_status_falls_back_rather_than_raising():
    fetcher = fetcher_returning(
        httpx.Response(200, content=b"ok", headers={"spb-original-status": "unknown"})
    )
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.status == 200
    await fetcher.close()


@pytest.mark.parametrize("status", [401, 402, 403])
async def test_an_account_failure_stops_the_ladder_rather_than_retrying(status):
    """A bad key or an empty balance is permanent. Retrying spends another
    request to learn the same thing."""
    fetcher = fetcher_returning(httpx.Response(status, content=b"{}"))
    with pytest.raises(UnlockerNotConfiguredError):
        await fetcher.fetch("https://blocked.test/a", 10.0)
    await fetcher.close()


async def test_provider_rate_limiting_is_transient_and_retryable():
    fetcher = fetcher_returning(httpx.Response(429, content=b"{}"))
    with pytest.raises(FetchTransportError, match="rate limiting"):
        await fetcher.fetch("https://blocked.test/a", 10.0)
    await fetcher.close()


async def test_an_empty_provider_error_is_transient():
    fetcher = fetcher_returning(httpx.Response(503, content=b""))
    with pytest.raises(FetchTransportError):
        await fetcher.fetch("https://blocked.test/a", 10.0)
    await fetcher.close()


async def test_a_five_hundred_carrying_a_body_is_the_origin_speaking():
    """The provider reached the site and the site returned a 5xx. That is a
    result to classify, not a provider outage to retry here."""
    fetcher = fetcher_returning(httpx.Response(500, content=b"origin error page"))
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.status == 500
    await fetcher.close()


async def test_a_timeout_is_reported_as_a_timeout():
    fetcher = raising_fetcher(httpx.ReadTimeout("too slow"))
    with pytest.raises(FetchTimeoutError, match="tier 4 timeout"):
        await fetcher.fetch("https://blocked.test/a", 10.0)
    await fetcher.close()


async def test_a_transport_failure_is_reported_as_a_transport_error():
    fetcher = raising_fetcher(httpx.ConnectError("no route"))
    with pytest.raises(FetchTransportError, match="tier 4 transport error"):
        await fetcher.fetch("https://blocked.test/a", 10.0)
    await fetcher.close()


async def test_the_result_never_carries_the_provider_url_or_the_key():
    """Regression. `final_url` was `response.url`, which is the provider endpoint
    with `api_key` in the query string. It became `CanonicalDoc.source_url` and
    wrote the credential into Postgres, Qdrant, every citation, and the model's
    context."""
    fetcher = fetcher_returning(httpx.Response(200, content=b"the page"))
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.final_url == "https://blocked.test/a"
    for field in (result.url, result.final_url):
        assert KEY not in field
        assert "scrapingbee" not in field
    await fetcher.close()


async def test_the_resolved_origin_url_is_used_when_the_provider_reports_it():
    """Redirects still resolve, they just resolve to the origin rather than to
    the proxy."""
    fetcher = fetcher_returning(
        httpx.Response(
            200,
            content=b"the page",
            headers={"spb-resolved-url": "https://blocked.test/final"},
        )
    )
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.final_url == "https://blocked.test/final"
    await fetcher.close()


async def test_a_blank_resolved_url_falls_back_to_the_requested_url():
    fetcher = fetcher_returning(
        httpx.Response(200, content=b"ok", headers={"spb-resolved-url": "   "})
    )
    result = await fetcher.fetch("https://blocked.test/a", 10.0)
    assert result.final_url == "https://blocked.test/a"
    await fetcher.close()
