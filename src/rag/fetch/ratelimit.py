"""Per domain token bucket, plus the global concurrency semaphore.

Politeness is per domain, throughput is global. One semaphore over everything
would either starve fast domains or hammer slow ones.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from rag.clock import Clock


@dataclass
class TokenBucket:
    """Refills continuously at `rate` per second, capacity `burst`."""

    rate: float
    clock: Clock
    burst: float = 1.0
    _tokens: float = field(default=1.0, init=False)
    _last: float | None = field(default=None, init=False)

    async def acquire(self) -> float:
        """Waits until a token is available. Returns the seconds waited."""
        waited = 0.0
        self._refill()
        if self._tokens < 1.0:
            waited = (1.0 - self._tokens) / self.rate
            await self.clock.sleep(waited)
            self._refill()
        self._tokens -= 1.0
        return waited

    def _refill(self) -> None:
        stamp = self.clock.now().timestamp()
        if self._last is None:
            self._last = stamp
            return
        elapsed = max(0.0, stamp - self._last)
        self._last = stamp
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)


class DomainLimiter:
    """One bucket per domain, created on first use."""

    def __init__(self, clock: Clock, default_rate: float) -> None:
        self._clock = clock
        self._default_rate = default_rate
        self._buckets: dict[str, TokenBucket] = {}

    def bucket(self, domain: str, rate: float | None = None) -> TokenBucket:
        existing = self._buckets.get(domain)
        if existing is None:
            existing = TokenBucket(rate or self._default_rate, self._clock)
            self._buckets[domain] = existing
        return existing

    async def acquire(self, domain: str, rate: float | None = None) -> float:
        return await self.bucket(domain, rate).acquire()


def global_semaphore(limit: int) -> asyncio.Semaphore:
    return asyncio.Semaphore(limit)
