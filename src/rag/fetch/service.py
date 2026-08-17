"""The fetch orchestrator. Ties the ladder, robots, limits and the breaker together.

Written last on purpose. Every decision it makes is delegated to a module that
can be tested without it: `escalation` decides what a response means, `circuit`
decides whether to allow a request, `backoff` decides how long to wait.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from urllib.parse import urlsplit

from rag.clock import Clock
from rag.config.settings import FetchSettings
from rag.fetch import circuit
from rag.fetch.backoff import backoff_seconds, decide_retry_after, parse_retry_after
from rag.fetch.deadletter import DeadLetterEntry, DeadLetterStore
from rag.fetch.escalation import Verdict, assess
from rag.fetch.protocols import (
    Fetcher,
    FetchTimeoutError,
    FetchTransportError,
    UnlockerNotConfiguredError,
)
from rag.fetch.ratelimit import DomainLimiter
from rag.fetch.registry import SourceRegistry
from rag.fetch.robots import RobotsChecker
from rag.fetch.types import (
    FailureReason,
    FetchFailure,
    FetchResult,
    FetchTier,
    Source,
    SourceState,
    SourceStatus,
)
from rag.log import get_logger

log = get_logger(__name__)


class UnknownSourceError(RuntimeError):
    """The domain is not in the registry. Seeding a domain is a legal decision."""


class Action(StrEnum):
    OK = "ok"
    RETRY = "retry"
    ESCALATE = "escalate"
    STOP = "stop"


# What each verdict does to the ladder. OK and rate limiting are handled
# separately, because both need the response itself and not just its class.
_VERDICT_ACTIONS: dict[Verdict, tuple[Action, FailureReason]] = {
    Verdict.NOT_FOUND: (Action.STOP, FailureReason.NOT_FOUND),
    Verdict.SERVER_ERROR: (Action.RETRY, FailureReason.SERVER_ERROR),
    Verdict.ESCALATE: (Action.ESCALATE, FailureReason.BLOCKED_PERSISTENT),
}


@dataclass(frozen=True)
class Attempt:
    action: Action
    result: FetchResult | None = None
    failure: FetchFailure | None = None
    sleep_seconds: float | None = None


@dataclass(frozen=True)
class TierOutcome:
    result: FetchResult | None
    failure: FetchFailure | None
    escalate: bool
    attempts: int


@dataclass
class FetchDependencies:
    """Collected into one object because a constructor takes at most 5 arguments."""

    registry: SourceRegistry
    dead_letter: DeadLetterStore
    robots: RobotsChecker
    limiter: DomainLimiter
    fetchers: dict[FetchTier, Fetcher]
    clock: Clock
    settings: FetchSettings
    rng: random.Random


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.split("@")[-1].split(":")[0]


class FetchService:
    def __init__(self, deps: FetchDependencies) -> None:
        self._deps = deps
        self._settings = deps.settings

    async def fetch(self, url: str) -> FetchResult | FetchFailure:
        """Never raises for an expected failure. Returns `FetchFailure` instead."""
        source = await self._source_for(url)
        state = await self._deps.registry.state(source.source_id)
        # If the circuit breaker is open, that means its last failure was persistent and the site is still blocking us.
        denied = self._circuit_denial(url, state)
        if denied is not None:
            return denied
        # Robot Disallowed is a legal decision.
        disallowed = await self._robots_denial(source, url)
        if disallowed is not None:
            return disallowed
        #
        await self._deps.limiter.acquire(source.domain, source.effective_rate)
        return await self._climb(url, source, state)

    async def _source_for(self, url: str) -> Source:
        source = await self._deps.registry.by_domain(domain_of(url))
        if source is None:
            raise UnknownSourceError(f"{domain_of(url)} is not in the source registry")
        return source

    def _circuit_denial(self, url: str, state: SourceState) -> FetchFailure | None:
        """An open circuit answers without making a request. That is the point."""
        decision = circuit.gate(state, self._deps.clock.now(), self._settings)
        if decision is not circuit.Gate.DENY:
            return None
        return FetchFailure(
            url=url,
            reason=state.last_failure_reason or FailureReason.BLOCKED_PERSISTENT,
            last_tier=state.preferred_tier,
            attempts=0,
            detail=f"circuit open for {state.source_id}, no request made",
        )

    async def _robots_denial(self, source: Source, url: str) -> FetchFailure | None:
        decision = await self._deps.robots.check(source, url)
        if decision.fetched_text is not None:
            await self._deps.registry.save_robots(
                source.source_id,
                decision.fetched_text,
                self._deps.clock.now(),
                decision.crawl_delay,
            )
        if decision.allowed:
            return None
        return FetchFailure(
            url=url,
            reason=FailureReason.ROBOTS_DISALLOWED,
            last_tier=FetchTier.STATIC,
            attempts=0,
            detail=decision.detail,
        )

    def _tiers(self, source: Source, state: SourceState) -> list[FetchTier]:
        """Start at the cached tier, stop at the source's ceiling."""
        ceiling = int(source.max_tier)
        if not source.allow_unlocker:
            ceiling = min(ceiling, int(FetchTier.STEALTH))
        start = int(self._start_tier(state))
        return [FetchTier(t) for t in range(min(start, ceiling), ceiling + 1)]

    def _start_tier(self, state: SourceState) -> FetchTier:
        """Expired policy resets to tier 1, so a site that dropped its
        protection stops paying for a browser forever."""
        if state.tier_learned_at is None:
            return FetchTier.STATIC
        ttl = timedelta(hours=self._settings.policy_cache_ttl_hours)
        if self._deps.clock.now() - state.tier_learned_at > ttl:
            return FetchTier.STATIC
        return state.preferred_tier

    # Climb all the tiers of fetch ladder from start to ceiling.
    async def _climb(
        self, url: str, source: Source, state: SourceState
    ) -> FetchResult | FetchFailure:
        attempts = 0
        failure: FetchFailure | None = None
        for tier in self._tiers(source, state):
            outcome = await self._run_tier(url, tier, source)
            attempts += outcome.attempts
            if outcome.result is not None:
                return await self._on_success(source, state, outcome.result, attempts)
            failure = outcome.failure
            if not outcome.escalate:
                break
        return await self._on_failure(source, state, url, failure, attempts)

    async def _run_tier(self, url: str, tier: FetchTier, source: Source) -> TierOutcome:
        fetcher = self._deps.fetchers[tier]
        timeout = self._timeout(tier)
        last: FetchFailure | None = None
        for attempt in range(self._settings.max_attempts_per_tier):
            outcome = await self._attempt(fetcher, url, timeout, tier)
            last = outcome.failure or last
            if outcome.action is Action.OK:
                return TierOutcome(outcome.result, None, False, attempt + 1)
            if outcome.action is not Action.RETRY:
                escalate = outcome.action is Action.ESCALATE
                return TierOutcome(None, last, escalate, attempt + 1)
            await self._wait(attempt, outcome)
        return TierOutcome(
            None,
            last,
            _escalate_after_exhaustion(last),
            self._settings.max_attempts_per_tier,
        )

    async def _wait(self, attempt: int, outcome: Attempt) -> None:
        seconds = outcome.sleep_seconds
        if seconds is None:
            seconds = backoff_seconds(attempt, self._settings, self._deps.rng)
        await self._deps.clock.sleep(seconds)

    def _timeout(self, tier: FetchTier) -> float:
        timeouts = self._settings.timeouts
        return {
            FetchTier.STATIC: timeouts.static,
            FetchTier.BROWSER: timeouts.browser,
            FetchTier.STEALTH: timeouts.stealth,
            FetchTier.UNLOCKER: timeouts.unlocker,
        }[tier]

    async def _attempt(
        self, fetcher: Fetcher, url: str, timeout: float, tier: FetchTier
    ) -> Attempt:
        try:
            result = await fetcher.fetch(url, timeout)
        except FetchTimeoutError as exc:
            return _transport_attempt(url, tier, FailureReason.TIMEOUT, str(exc))
        except UnlockerNotConfiguredError as exc:
            return Attempt(
                Action.STOP,
                failure=_failure(url, tier, FailureReason.BLOCKED_PERSISTENT, str(exc)),
            )
        except FetchTransportError as exc:
            return _transport_attempt(url, tier, FailureReason.SERVER_ERROR, str(exc))
        return self._classify(result, url, tier)

    def _classify(self, result: FetchResult, url: str, tier: FetchTier) -> Attempt:
        verdict = assess(result, self._settings)
        if verdict.verdict is Verdict.OK:
            return Attempt(Action.OK, result=result)
        if verdict.verdict is Verdict.RATE_LIMITED:
            return self._rate_limited(result, url, tier)
        action, reason = _VERDICT_ACTIONS[verdict.verdict]
        return Attempt(action, failure=_failure(url, tier, reason, verdict.detail))

    def _rate_limited(self, result: FetchResult, url: str, tier: FetchTier) -> Attempt:
        """Honour a short Retry-After. Requeue a long one, never sleep on it."""
        retry_after = parse_retry_after(result.headers.get("retry-after"))
        decision = decide_retry_after(retry_after, self._settings)
        failure = _failure(url, tier, FailureReason.RATE_LIMITED, decision.detail)
        if decision.sleep_seconds is not None:
            return Attempt(
                Action.RETRY, failure=failure, sleep_seconds=decision.sleep_seconds
            )
        requeue = decision.requeue_after_seconds
        return Attempt(
            Action.STOP,
            failure=failure.model_copy(update={"retry_after_seconds": requeue}),
        )

    async def _on_success(
        self, source: Source, state: SourceState, result: FetchResult, attempts: int
    ) -> FetchResult:
        now = self._deps.clock.now()
        closed = circuit.on_success(state, now, self._settings)
        learned = closed.model_copy(
            update={"preferred_tier": result.tier_used, "tier_learned_at": now}
        )
        await self._deps.registry.save_state(learned)
        log.info(
            "fetch ok",
            url=result.url,
            tier=int(result.tier_used),
            attempts=attempts,
            status=result.status,
        )
        return result.model_copy(update={"attempts": attempts})

    async def _on_failure(
        self,
        source: Source,
        state: SourceState,
        url: str,
        failure: FetchFailure | None,
        attempts: int,
    ) -> FetchFailure:
        resolved = failure or _failure(
            url, FetchTier.STATIC, FailureReason.SERVER_ERROR, "no tier attempted"
        )
        resolved = resolved.model_copy(update={"attempts": attempts})
        await self._record_failure(source, state, resolved)
        return resolved

    async def _record_failure(
        self, source: Source, state: SourceState, failure: FetchFailure
    ) -> None:
        now = self._deps.clock.now()
        updated = circuit.on_failure(state, now, failure.reason, self._settings)
        await self._deps.registry.save_state(updated)
        if circuit.should_mark_unreachable(updated, now, self._settings):
            await self._deps.registry.set_status(
                source.source_id, SourceStatus.UNREACHABLE
            )
        if failure.reason is not FailureReason.RATE_LIMITED:
            await self._write_dead_letter(source, failure)
        log.warning(
            "fetch failed",
            url=failure.url,
            reason=str(failure.reason),
            tier=int(failure.last_tier),
            attempts=failure.attempts,
        )

    async def _write_dead_letter(self, source: Source, failure: FetchFailure) -> None:
        await self._deps.dead_letter.record(
            DeadLetterEntry(
                url=failure.url,
                source_id=source.source_id,
                reason=failure.reason,
                stage="fetch",
                attempts=failure.attempts,
                last_tier=failure.last_tier,
                detail=failure.detail,
            )
        )


def _failure(
    url: str, tier: FetchTier, reason: FailureReason, detail: str
) -> FetchFailure:
    return FetchFailure(
        url=url, reason=reason, last_tier=tier, attempts=1, detail=detail
    )


def _transport_attempt(
    url: str, tier: FetchTier, reason: FailureReason, detail: str
) -> Attempt:
    return Attempt(Action.RETRY, failure=_failure(url, tier, reason, detail))


#: Reasons that a more expensive tier cannot fix, so exhausting a tier on one
#: of them ends the climb instead of continuing it.
_NO_ESCALATION = frozenset(
    {
        # The server said "slower", not "prove you are a browser". Paying 2 to
        # 10 seconds for a rendered tier just collects the same 429.
        FailureReason.RATE_LIMITED,
        # A broken server is broken at every tier.
        FailureReason.SERVER_ERROR,
    }
)


def _escalate_after_exhaustion(last: FetchFailure | None) -> bool:
    """A tier that ran out of retries escalates, unless nothing up the ladder
    would help. A timeout still escalates: a slow page may well render."""
    return last is None or last.reason not in _NO_ESCALATION
