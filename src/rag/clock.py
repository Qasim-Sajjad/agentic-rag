"""Time behind a protocol, so backoff and circuit breakers are testable.

`tests/SPEC.md` says no sleeps, fake the clock. Every module that waits or
expires something takes a `Clock` rather than calling `asyncio.sleep` or
`datetime.now` directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, always timezone aware and in UTC."""
        ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Real time. The only implementation that reaches production."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    """Advances only when told to. Records what it was asked to sleep for."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start if start is not None else datetime(2026, 1, 1, tzinfo=UTC)
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    @property
    def total_slept(self) -> float:
        return sum(self.slept)
