"""Token bucket pacing, measured on the fake clock rather than the wall."""

from __future__ import annotations

from rag.clock import FakeClock
from rag.fetch.ratelimit import DomainLimiter, TokenBucket


async def test_the_first_request_does_not_wait():
    bucket = TokenBucket(rate=1.0, clock=FakeClock())
    assert await bucket.acquire() == 0.0


async def test_the_second_request_waits_for_the_refill():
    bucket = TokenBucket(rate=1.0, clock=FakeClock())
    await bucket.acquire()
    assert await bucket.acquire() == 1.0


async def test_a_faster_rate_waits_proportionally_less():
    bucket = TokenBucket(rate=10.0, clock=FakeClock())
    await bucket.acquire()
    assert await bucket.acquire() == 0.1


async def test_elapsed_time_refills_the_bucket():
    clock = FakeClock()
    bucket = TokenBucket(rate=1.0, clock=clock)
    await bucket.acquire()
    clock.advance(5)
    assert await bucket.acquire() == 0.0


async def test_domains_are_limited_independently():
    """A slow domain must not throttle a fast one."""
    limiter = DomainLimiter(FakeClock(), default_rate=1.0)
    await limiter.acquire("a.test")
    assert await limiter.acquire("b.test") == 0.0


async def test_the_same_domain_shares_one_bucket():
    limiter = DomainLimiter(FakeClock(), default_rate=1.0)
    await limiter.acquire("a.test")
    assert await limiter.acquire("a.test") > 0.0
