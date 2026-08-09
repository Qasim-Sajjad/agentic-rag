"""Exponential backoff with full jitter, and the Retry-After decision.

Full jitter rather than plain exponential: without it, N workers that failed
together retry together, and the retry storm looks exactly like the outage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rag.config.settings import FetchSettings


def backoff_seconds(attempt: int, settings: FetchSettings, rng: random.Random) -> float:
    """Full jitter: uniform in [0, min(cap, base * 2^attempt)].

    `attempt` is zero based, so the first retry samples from [0, base].
    """
    ceiling = min(
        settings.backoff_cap_seconds, settings.backoff_base_seconds * (2**attempt)
    )
    return rng.uniform(0.0, ceiling)


@dataclass(frozen=True)
class RetryDecision:
    """What to do about a 429."""

    sleep_seconds: float | None  # None means do not sleep
    requeue_after_seconds: float | None  # None means do not requeue
    detail: str


def decide_retry_after(
    retry_after: float | None, settings: FetchSettings
) -> RetryDecision:
    """Honour a short Retry-After, requeue a long one. Never block a worker.

    A 30 second wait is cheaper than releasing and reclaiming the URL. A 10
    minute wait held in memory is a worker doing nothing, so that one goes back
    to the queue with a delayed visibility timestamp instead.
    """
    if retry_after is None:
        return RetryDecision(None, None, "429 without Retry-After")
    if retry_after <= settings.max_retry_after_seconds:
        return RetryDecision(retry_after, None, f"honouring Retry-After {retry_after}s")
    return RetryDecision(
        None,
        retry_after,
        f"Retry-After {retry_after}s over cap {settings.max_retry_after_seconds}s",
    )


def parse_retry_after(value: str | None) -> float | None:
    """Seconds form only. HTTP-date form is rare and is treated as absent."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
