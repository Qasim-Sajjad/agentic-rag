"""Tier 1. curl_cffi with TLS impersonation, 50 to 150ms per page.

TLS impersonation without an honest user agent would be evasion. The
fingerprint matches a real browser because most bot walls reject the Python
default outright, while the `User-Agent` header says exactly who we are and
where to complain. See the legal boundaries section of the SPEC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from curl_cffi.requests import AsyncSession
from curl_cffi.requests import exceptions as curl_exceptions

from rag.config.settings import FetchSettings
from rag.fetch.protocols import FetchTimeoutError, FetchTransportError
from rag.fetch.types import FetchResult, FetchTier

DEFAULT_CONTENT_TYPE = "application/octet-stream"


class StaticFetcher:
    tier = FetchTier.STATIC

    def __init__(self, settings: FetchSettings) -> None:
        self._settings = settings
        self._session: AsyncSession[Any] | None = None

    def _ensure_session(self) -> AsyncSession[Any]:
        if self._session is None:
            self._session = AsyncSession(
                impersonate=self._settings.impersonate_profile,
                headers={"User-Agent": self._settings.user_agent},
            )
        return self._session

    async def fetch(self, url: str, timeout: float) -> FetchResult:
        session = self._ensure_session()
        try:
            response: Any = await session.get(
                url, timeout=timeout, allow_redirects=True
            )
        except curl_exceptions.Timeout as exc:
            raise FetchTimeoutError(f"tier 1 timeout after {timeout}s") from exc
        except curl_exceptions.RequestException as exc:
            raise FetchTransportError(f"tier 1 transport error: {exc}") from exc
        return _to_result(url, response, self.tier)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def _to_result(url: str, response: Any, tier: FetchTier) -> FetchResult:
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return FetchResult(
        url=url,
        final_url=str(response.url),
        status=int(response.status_code),
        content=bytes(response.content),
        content_type=headers.get("content-type", DEFAULT_CONTENT_TYPE),
        tier_used=tier,
        attempts=1,
        fetched_at=datetime.now(UTC),
        headers=headers,
    )
