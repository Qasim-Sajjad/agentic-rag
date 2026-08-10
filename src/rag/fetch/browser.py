"""Tiers 2 and 3. Chromium for rendering, Camoufox for interstitials.

One browser per process with a bounded pool of contexts, acquired for the
duration of one page load. A browser per URL is the fastest way to turn a
crawl into an out of memory kill.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from rag.config.settings import FetchSettings
from rag.fetch.protocols import FetchTimeoutError, FetchTransportError
from rag.fetch.types import FetchResult, FetchTier

HTML_CONTENT_TYPE = "text/html; charset=utf-8"


class BrowserPool:
    """Bounded context pool over one launched browser.

    The browser is launched on first use, not at construction, so importing
    this module never costs 300 MB of Chromium.
    """

    def __init__(self, settings: FetchSettings, stealth: bool) -> None:
        self._settings = settings
        self._stealth = stealth
        self._semaphore = asyncio.Semaphore(settings.browser_pool_size)
        self._browser: Any = None
        self._owner: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_browser(self) -> Any:
        async with self._lock:
            if self._browser is None:
                self._browser = await self._launch()
            return self._browser

    async def _launch(self) -> Any:
        if self._stealth:
            from camoufox.async_api import AsyncCamoufox

            self._owner = AsyncCamoufox(  # type: ignore[no-untyped-call]
                headless=self._settings.browser_headless
            )
            return await self._owner.__aenter__()
        from playwright.async_api import async_playwright

        self._owner = async_playwright()
        playwright = await self._owner.__aenter__()
        return await playwright.chromium.launch(
            headless=self._settings.browser_headless
        )

    async def page(self) -> _PageLease:
        browser = await self._ensure_browser()
        return _PageLease(browser, self._semaphore, self._settings)

    async def close(self) -> None:
        if self._browser is not None and not self._stealth:
            await self._browser.close()
        if self._owner is not None:
            await self._owner.__aexit__(None, None, None)
        self._browser = None
        self._owner = None


class _PageLease:
    """Holds one pool slot and one browser context for one page load."""

    def __init__(
        self, browser: Any, semaphore: asyncio.Semaphore, settings: FetchSettings
    ) -> None:
        self._browser = browser
        self._semaphore = semaphore
        self._settings = settings
        self._context: Any = None

    async def __aenter__(self) -> Any:
        await self._semaphore.acquire()
        self._context = await self._browser.new_context(
            extra_http_headers={"X-Crawler-Contact": self._settings.user_agent}
        )
        return await self._context.new_page()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._context is not None:
            await self._context.close()
        self._semaphore.release()


class BrowserFetcher:
    """Tier 2. Renders with Chromium, which is what `/js-only` needs."""

    tier = FetchTier.BROWSER

    def __init__(
        self, settings: FetchSettings, pool: BrowserPool | None = None
    ) -> None:
        self._settings = settings
        self._pool = pool if pool is not None else BrowserPool(settings, stealth=False)

    async def fetch(self, url: str, timeout: float) -> FetchResult:
        return await _render(self._pool, url, timeout, self.tier)

    async def close(self) -> None:
        await self._pool.close()


class StealthFetcher:
    """Tier 3. Camoufox presents as a real Firefox build, fingerprint included.

    This renders a page the way a normal browser would. It does not solve
    CAPTCHAs and does not target any specific vendor's protection.
    """

    tier = FetchTier.STEALTH

    def __init__(
        self, settings: FetchSettings, pool: BrowserPool | None = None
    ) -> None:
        self._settings = settings
        self._pool = pool if pool is not None else BrowserPool(settings, stealth=True)

    async def fetch(self, url: str, timeout: float) -> FetchResult:
        return await _render(self._pool, url, timeout, self.tier)

    async def close(self) -> None:
        await self._pool.close()


async def _render(
    pool: BrowserPool, url: str, timeout: float, tier: FetchTier
) -> FetchResult:
    """Launching the browser is inside the try on purpose.

    A missing or stale browser binary is an environment problem, but it must
    still arrive as a typed transport failure. Letting a Playwright error
    escape kills the whole crawl instead of dead lettering one URL.
    """
    try:
        lease = await pool.page()
        async with lease as page:
            response = await _goto(page, url, timeout)
            body = await page.content()
            return _to_result(url, page, response, body, tier)
    except FetchTransportError:
        raise
    except Exception as exc:
        raise FetchTransportError(f"tier {int(tier)} render failed: {exc}") from exc


async def _goto(page: Any, url: str, timeout: float) -> Any:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeout

    try:
        return await page.goto(
            url, timeout=timeout * 1000, wait_until="domcontentloaded"
        )
    except PlaywrightTimeout as exc:
        raise FetchTimeoutError(f"render timeout after {timeout}s") from exc
    except PlaywrightError as exc:
        raise FetchTransportError(f"navigation failed: {exc}") from exc


def _to_result(
    url: str, page: Any, response: Any, body: str, tier: FetchTier
) -> FetchResult:
    headers = {str(k).lower(): str(v) for k, v in (response.headers or {}).items()}
    return FetchResult(
        url=url,
        final_url=str(page.url),
        status=int(response.status),
        content=body.encode("utf-8"),
        content_type=headers.get("content-type", HTML_CONTENT_TYPE),
        tier_used=tier,
        attempts=1,
        fetched_at=datetime.now(UTC),
        headers=headers,
    )
