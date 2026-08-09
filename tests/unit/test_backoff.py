"""Backoff bounds and the Retry-After cap decision."""

from __future__ import annotations

import random

from rag.config.settings import FetchSettings
from rag.fetch.backoff import backoff_seconds, decide_retry_after, parse_retry_after

SETTINGS = FetchSettings()


def test_first_retry_never_exceeds_the_base():
    rng = random.Random(1)
    samples = [backoff_seconds(0, SETTINGS, rng) for _ in range(50)]
    assert max(samples) <= SETTINGS.backoff_base_seconds


def test_backoff_is_capped():
    rng = random.Random(1)
    assert backoff_seconds(20, SETTINGS, rng) <= SETTINGS.backoff_cap_seconds


def test_full_jitter_produces_different_waits():
    """Without jitter, workers that failed together retry together."""
    rng = random.Random(7)
    samples = {backoff_seconds(3, SETTINGS, rng) for _ in range(20)}
    assert len(samples) > 1


def test_a_short_retry_after_is_honoured():
    decision = decide_retry_after(30.0, SETTINGS)
    assert decision.sleep_seconds == 30.0


def test_a_short_retry_after_does_not_requeue():
    assert decide_retry_after(30.0, SETTINGS).requeue_after_seconds is None


def test_a_long_retry_after_requeues_instead_of_sleeping():
    decision = decide_retry_after(600.0, SETTINGS)
    assert decision.sleep_seconds is None


def test_a_long_retry_after_carries_the_requeue_delay():
    assert decide_retry_after(600.0, SETTINGS).requeue_after_seconds == 600.0


def test_a_missing_retry_after_neither_sleeps_nor_requeues():
    decision = decide_retry_after(None, SETTINGS)
    assert (decision.sleep_seconds, decision.requeue_after_seconds) == (None, None)


def test_retry_after_seconds_form_parses():
    assert parse_retry_after("120") == 120.0


def test_retry_after_http_date_form_is_treated_as_absent():
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
