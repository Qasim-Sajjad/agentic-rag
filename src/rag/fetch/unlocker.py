"""Tier 4. A managed unlocker, currently ScrapingBee.

This tier is different in kind from the three below it. Tiers 1 to 3 change how
we present ourselves and let the site decide. Tier 4 pays a third party to get
through a challenge on our behalf, which is why it is gated twice: a key must be
configured, and the source must set `allow_unlocker`. Neither default is on. See
the legal boundaries section of the SPEC: point this at a domain whose terms
permit automated access, or do not point it anywhere.

Swapping providers is this class plus the `unlocker` config block. Nothing above
it knows which service answered.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from rag.config.settings import FetchSettings, UnlockerSettings
from rag.fetch.protocols import (
    FetchTimeoutError,
    FetchTransportError,
    UnlockerNotConfiguredError,
)
from rag.fetch.types import FetchResult, FetchTier
from rag.log import get_logger

log = get_logger(__name__)

DEFAULT_CONTENT_TYPE = "application/octet-stream"

# The provider's own status codes, which describe our account rather than the
# target site. Retrying an exhausted quota just spends another request.
_ACCOUNT_ERRORS = {
    401: "the unlocker rejected the API key",
    402: "the unlocker account is out of credits",
    403: "the unlocker key is not permitted for this request",
}

#: ScrapingBee reports the origin's status here. Absent on some responses, so
#: it is read defensively and the proxy's own status is the fallback.
_ORIGIN_STATUS_HEADER = "spb-original-status"

#: Where the provider says it actually landed, after redirects.
_RESOLVED_URL_HEADER = "spb-resolved-url"


class UnlockerFetcher:
    tier = FetchTier.UNLOCKER

    def __init__(self, settings: FetchSettings) -> None:
        self._settings = settings
        self._unlocker = settings.unlocker
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self._unlocker.api_key)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client

    async def fetch(self, url: str, timeout: float) -> FetchResult:
        if not self.configured:
            raise UnlockerNotConfiguredError(
                "tier 4 has no API key. Set SCRAPINGBEE_API_KEY in .env, or leave "
                "allow_unlocker false on every source to keep the tier unreachable"
            )
        # Logged on every call because every call is billable. A crawl that
        # quietly escalated to tier 4 should be visible in the log, not on an
        # invoice at the end of the month.
        log.info("unlocker request", url=url, provider=self._unlocker.provider)
        response = await self._request(url, timeout)
        _raise_for_account_error(response)
        return _to_result(url, response, self.tier)

    async def _request(self, url: str, timeout: float) -> httpx.Response:
        try:
            return await self._ensure_client().get(
                self._unlocker.endpoint,
                params=_params(self._unlocker, url),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise FetchTimeoutError(f"tier 4 timeout after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise FetchTransportError(f"tier 4 transport error: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _params(unlocker: UnlockerSettings, url: str) -> dict[str, Any]:
    """Booleans go over the wire lowercased, which is what the provider expects."""
    params: dict[str, Any] = {
        "api_key": unlocker.api_key,
        "url": url,
        "render_js": str(unlocker.render_js).lower(),
        "premium_proxy": str(unlocker.premium_proxy).lower(),
    }
    if unlocker.country_code:
        params["country_code"] = unlocker.country_code
    return params


def _raise_for_account_error(response: httpx.Response) -> None:
    """A provider side failure is ours, not the target site's.

    Splitting these out matters for the ladder: a bad key or an empty balance is
    permanent and must stop, while concurrency limits and provider outages are
    transient and worth a retry.
    """
    detail = _ACCOUNT_ERRORS.get(response.status_code)
    if detail is not None:
        raise UnlockerNotConfiguredError(f"{detail} (HTTP {response.status_code})")
    if response.status_code == 429:
        raise FetchTransportError("the unlocker is rate limiting our account")
    if response.status_code >= 500 and not response.content:
        raise FetchTransportError(f"the unlocker returned HTTP {response.status_code}")


def _to_result(url: str, response: httpx.Response, tier: FetchTier) -> FetchResult:
    """Reports the origin's status when the provider tells us, so escalation and
    the block signature checks see what the site said rather than what the proxy
    said about having spoken to it."""
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return FetchResult(
        url=url,
        final_url=_origin_url(headers, url),
        status=_origin_status(headers, response.status_code),
        content=bytes(response.content),
        content_type=headers.get("content-type", DEFAULT_CONTENT_TYPE),
        tier_used=tier,
        attempts=1,
        fetched_at=datetime.now(UTC),
        headers=headers,
    )


def _origin_url(headers: dict[str, str], requested: str) -> str:
    """Never `response.url`.

    On every other tier the response url is the origin after redirects, which is
    exactly what `final_url` means. Here it is the provider's endpoint, and that
    endpoint carries `api_key` as a query parameter. `final_url` becomes
    `CanonicalDoc.source_url`, so returning it would write the credential into
    Postgres, into the Qdrant payload, into every citation, and into the context
    sent to the model. It did, once. This function is why it cannot again.
    """
    resolved = headers.get(_RESOLVED_URL_HEADER, "").strip()
    return resolved or requested


def _origin_status(headers: dict[str, str], fallback: int) -> int:
    raw = headers.get(_ORIGIN_STATUS_HEADER, "")
    try:
        return int(raw)
    except ValueError:
        return fallback
