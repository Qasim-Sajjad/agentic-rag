"""Fixture server. Makes the fetch ladder testable offline and deterministically.

Serves the eight endpoints named in `tests/SPEC.md`, plus `/robots.txt` (which
`/robots-blocked` is defined in terms of) and two control endpoints, `/__stats`
and `/__reset`. `/__stats` is how a test asserts that a robots disallowed URL
produced zero HTTP requests, and `/__reset` keeps the stateful endpoints from
leaking attempt counts between tests.

Run standalone with: python -m tests.fixtures.server
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.base import RequestResponseEndpoint

PAGES = Path(__file__).parent / "pages"

FLAKY_FAILURES = 2
DEFAULT_RETRY_AFTER = 600
CHALLENGE_TIER = 3
TIER_HEADER = "x-fixture-tier"

ROBOTS_TXT = "User-agent: *\nDisallow: /robots-blocked\n"

# The control plane is not part of what a fetch test measures.
CONTROL_PATHS = frozenset({"/__stats", "/__reset"})

# A stealth tier client presents as a real Firefox build, a browser tier as
# Chromium. That is the only signal the server needs to decide who gets past
# the interstitial, and it keeps the fetchers free of fixture specific code.
STEALTH_AGENTS = ("camoufox", "firefox")
BROWSER_AGENTS = ("chrome", "chromium")


@dataclass
class FixtureState:
    """Per app mutable state. One instance per `create_app()` call."""

    hits: Counter[str] = field(default_factory=Counter)
    flaky_attempts: int = 0

    def reset(self) -> None:
        self.hits.clear()
        self.flaky_attempts = 0


# read a file from tests/fixtures/pages
def _page(name: str) -> str:
    return (PAGES / name).read_text(encoding="utf-8")


# Convert to typed Accessor.
def _state(request: Request) -> FixtureState:
    state: FixtureState = request.app.state.fixture
    return state


# Header Instruction for the fixture server to decide which fetcher tier the
# caller looks like.
def _client_tier(request: Request) -> int:
    """Which ladder tier the caller looks like, from an explicit header or UA."""
    declared = request.headers.get(TIER_HEADER)
    if declared is not None and declared.isdigit():
        return int(declared)
    agent = request.headers.get("user-agent", "").lower()
    if any(marker in agent for marker in STEALTH_AGENTS):
        return 3
    if any(marker in agent for marker in BROWSER_AGENTS):
        return 2
    return 1


def create_app() -> FastAPI:
    app = FastAPI(title="fetch fixture server", docs_url=None, redoc_url=None)
    app.state.fixture = FixtureState()
    _register_middleware(app)
    _register_pages(app)
    _register_failures(app)
    _register_control(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def count_hits(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path not in CONTROL_PATHS:
            _state(request).hits[request.url.path] += 1
        return await call_next(request)


def _register_pages(app: FastAPI) -> None:
    @app.get("/static", response_class=HTMLResponse)
    async def static_page() -> HTMLResponse:
        return HTMLResponse(_page("static.html"))

    @app.get("/js-only", response_class=HTMLResponse)
    async def js_only() -> HTMLResponse:
        return HTMLResponse(_page("js_only.html"))

    @app.get("/robots-blocked", response_class=HTMLResponse)
    async def robots_blocked() -> HTMLResponse:
        """Serves real content. Only robots.txt is meant to stop a fetch here."""
        return HTMLResponse(_page("static.html"))

    @app.get("/robots.txt", response_class=PlainTextResponse)
    async def robots_txt() -> PlainTextResponse:
        return PlainTextResponse(ROBOTS_TXT)

    @app.get("/doc.pdf")
    async def doc_pdf() -> Response:
        return Response(
            content=(PAGES / "doc.pdf").read_bytes(),
            media_type="application/pdf",
        )


def _register_failures(app: FastAPI) -> None:
    @app.get("/rate-limited")
    async def rate_limited(retry_after: int = DEFAULT_RETRY_AFTER) -> Response:
        """Default is over `fetch.max_retry_after_seconds`, so the bare URL
        exercises the requeue path. Pass `?retry_after=` for the honour path."""
        return PlainTextResponse(
            "rate limited",
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    @app.get("/challenge")
    async def challenge(request: Request) -> Response:
        if _client_tier(request) >= CHALLENGE_TIER:
            return HTMLResponse(_page("challenge_passed.html"))
        return HTMLResponse(_page("challenge.html"), status_code=503)

    @app.get("/flaky")
    async def flaky(request: Request) -> Response:
        state = _state(request)
        state.flaky_attempts += 1
        if state.flaky_attempts <= FLAKY_FAILURES:
            return PlainTextResponse(
                f"attempt {state.flaky_attempts} failed", status_code=500
            )
        return HTMLResponse(_page("static.html"))

    @app.get("/always-500")
    async def always_500() -> Response:
        return PlainTextResponse("server error", status_code=500)


def _register_control(app: FastAPI) -> None:
    @app.get("/__stats")
    async def stats(request: Request) -> dict[str, object]:
        state = _state(request)
        return {"hits": dict(state.hits), "flaky_attempts": state.flaky_attempts}

    @app.post("/__reset")
    async def reset(request: Request) -> dict[str, str]:
        _state(request).reset()
        return {"status": "reset"}


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8099)
