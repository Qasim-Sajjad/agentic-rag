"""The nine ladder cases named in src/rag/fetch/SPEC.md.

Against the fixture server and a real Postgres. No test touches the network.
Time is a fake clock, so backoff and rate limiting cost nothing in wall time.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from rag.clock import FakeClock
from rag.config.settings import FetchSettings
from rag.db.pool import Database
from rag.fetch.deadletter import DeadLetterStore
from rag.fetch.factory import build_fetchers, build_service, close_fetchers
from rag.fetch.registry import SourceRegistry
from rag.fetch.service import FetchService, UnknownSourceError
from rag.fetch.types import (
    FailureReason,
    FetchFailure,
    FetchResult,
    FetchTier,
    Source,
    SourceStatus,
)

pytestmark = pytest.mark.integration

SOURCE_ID = "fixture"


@dataclass
class Env:
    service: FetchService
    registry: SourceRegistry
    dead_letter: DeadLetterStore
    clock: FakeClock
    base_url: str
    http: httpx.Client

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def hits(self, path: str) -> int:
        stats = self.http.get("/__stats").json()
        return int(stats["hits"].get(path, 0))


@pytest.fixture
async def env(db: Database, fixture_server: str, client: httpx.Client):
    settings = FetchSettings(default_requests_per_second=1000.0)
    clock = FakeClock()
    fetchers = build_fetchers(settings)
    registry = SourceRegistry(db)
    await registry.upsert(
        Source(
            source_id=SOURCE_ID,
            domain="127.0.0.1",
            seed_urls=[],
            max_tier=FetchTier.STEALTH,
            requests_per_second=1000.0,
            tos_note="local fixture server",
        )
    )
    service = build_service(db, clock=clock, settings=settings, fetchers=fetchers)
    try:
        yield Env(service, registry, DeadLetterStore(db), clock, fixture_server, client)
    finally:
        await close_fetchers(fetchers)


async def test_static_resolves_at_tier_one(env: Env):
    outcome = await env.service.fetch(env.url("/static"))
    assert isinstance(outcome, FetchResult) and outcome.tier_used is FetchTier.STATIC


async def test_static_does_not_retry(env: Env):
    outcome = await env.service.fetch(env.url("/static"))
    assert isinstance(outcome, FetchResult) and outcome.attempts == 1


@pytest.mark.slow
async def test_js_only_escalates_to_the_browser_tier(env: Env):
    outcome = await env.service.fetch(env.url("/js-only"))
    assert isinstance(outcome, FetchResult) and outcome.tier_used is FetchTier.BROWSER


@pytest.mark.slow
async def test_js_only_returns_rendered_text(env: Env):
    outcome = await env.service.fetch(env.url("/js-only"))
    assert isinstance(outcome, FetchResult)
    assert b"Operating margin" in outcome.content


@pytest.mark.slow
async def test_challenge_escalates_to_the_stealth_tier(env: Env):
    outcome = await env.service.fetch(env.url("/challenge"))
    assert isinstance(outcome, FetchResult) and outcome.tier_used is FetchTier.STEALTH


async def test_rate_limited_returns_rate_limited(env: Env):
    outcome = await env.service.fetch(env.url("/rate-limited"))
    assert isinstance(outcome, FetchFailure)
    assert outcome.reason is FailureReason.RATE_LIMITED


async def test_rate_limited_requeues_rather_than_sleeping_past_the_cap(env: Env):
    await env.service.fetch(env.url("/rate-limited"))
    assert env.clock.total_slept == 0.0


async def test_rate_limited_carries_the_requeue_delay(env: Env):
    outcome = await env.service.fetch(env.url("/rate-limited"))
    assert isinstance(outcome, FetchFailure) and outcome.retry_after_seconds == 600.0


async def test_a_short_retry_after_is_slept_not_requeued(env: Env):
    await env.service.fetch(env.url("/rate-limited?retry_after=5"))
    assert env.clock.total_slept == pytest.approx(15.0)


async def test_flaky_succeeds_on_the_third_attempt(env: Env):
    outcome = await env.service.fetch(env.url("/flaky"))
    assert isinstance(outcome, FetchResult) and outcome.attempts == 3


async def test_always_500_gives_up_with_server_error(env: Env):
    outcome = await env.service.fetch(env.url("/always-500"))
    assert isinstance(outcome, FetchFailure)
    assert outcome.reason is FailureReason.SERVER_ERROR


async def test_always_500_lands_in_the_dead_letter_store(env: Env):
    url = env.url("/always-500")
    await env.service.fetch(url)
    entry = await env.dead_letter.get(url)
    assert entry is not None and entry["reason"] == "server_error"


async def test_robots_blocked_returns_robots_disallowed(env: Env):
    outcome = await env.service.fetch(env.url("/robots-blocked"))
    assert isinstance(outcome, FetchFailure)
    assert outcome.reason is FailureReason.ROBOTS_DISALLOWED


async def test_robots_blocked_makes_zero_requests_to_the_url(env: Env):
    await env.service.fetch(env.url("/robots-blocked"))
    assert env.hits("/robots-blocked") == 0


async def test_five_consecutive_failures_open_the_circuit(env: Env):
    for _ in range(5):
        await env.service.fetch(env.url("/always-500"))
    state = await env.registry.state(SOURCE_ID)
    assert state.circuit_state == "open"


async def test_an_open_circuit_makes_no_request(env: Env):
    url = env.url("/always-500")
    for _ in range(5):
        await env.service.fetch(url)
    before = env.hits("/always-500")
    await env.service.fetch(url)
    assert env.hits("/always-500") == before


async def test_an_open_circuit_still_returns_a_typed_failure(env: Env):
    url = env.url("/always-500")
    for _ in range(5):
        await env.service.fetch(url)
    outcome = await env.service.fetch(url)
    assert isinstance(outcome, FetchFailure) and "circuit open" in outcome.detail


@pytest.mark.slow
async def test_policy_cache_starts_the_next_url_at_the_learned_tier(env: Env):
    """The second URL on a tier 2 domain must not pay for tier 1 first."""
    await env.service.fetch(env.url("/js-only"))
    outcome = await env.service.fetch(env.url("/static"))
    assert isinstance(outcome, FetchResult) and outcome.tier_used is FetchTier.BROWSER


async def test_an_unregistered_domain_is_never_fetched(env: Env):
    """Seeding a domain is a legal decision, so fetch refuses to invent one."""
    with pytest.raises(UnknownSourceError):
        await env.service.fetch("https://not-registered.test/page")


async def test_a_successful_fetch_records_the_source_as_healthy(env: Env):
    await env.service.fetch(env.url("/static"))
    source = await env.registry.get(SOURCE_ID)
    assert source is not None and source.status is SourceStatus.ACTIVE
