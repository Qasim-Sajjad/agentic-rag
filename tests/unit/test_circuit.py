"""Circuit breaker transitions, driven by a fake clock."""

from __future__ import annotations

from datetime import timedelta

from rag.clock import FakeClock
from rag.config.settings import FetchSettings
from rag.fetch.circuit import (
    Gate,
    gate,
    on_failure,
    on_success,
    should_mark_unreachable,
)
from rag.fetch.types import CircuitState, FailureReason, SourceState

SETTINGS = FetchSettings()
REASON = FailureReason.SERVER_ERROR


def fail_times(state: SourceState, count: int, clock: FakeClock) -> SourceState:
    for _ in range(count):
        state = on_failure(state, clock.now(), REASON, SETTINGS)
    return state


def test_closed_circuit_allows_requests():
    clock = FakeClock()
    assert gate(SourceState(source_id="s"), clock.now(), SETTINGS) is Gate.ALLOW


def test_four_failures_leave_the_circuit_closed():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 4, clock)
    assert state.circuit_state is CircuitState.CLOSED


def test_five_consecutive_failures_open_the_circuit():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    assert state.circuit_state is CircuitState.OPEN


def test_an_open_circuit_denies_without_a_request():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    assert gate(state, clock.now(), SETTINGS) is Gate.DENY


def test_the_circuit_half_opens_after_the_wait():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    clock.advance(SETTINGS.circuit_open_minutes * 60)
    assert gate(state, clock.now(), SETTINGS) is Gate.PROBE


def test_success_closes_the_circuit():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    assert on_success(state, clock.now(), SETTINGS).circuit_state is CircuitState.CLOSED


def test_success_resets_the_failure_count():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    assert on_success(state, clock.now(), SETTINGS).consecutive_failures == 0


def test_a_failed_probe_doubles_the_wait():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    opened = state.circuit_open_seconds
    reopened = on_failure(state, clock.now(), REASON, SETTINGS)
    assert reopened.circuit_open_seconds == opened * 2


def test_the_doubled_wait_is_capped():
    clock = FakeClock()
    state = SourceState(
        source_id="s",
        circuit_state=CircuitState.OPEN,
        circuit_open_seconds=SETTINGS.circuit_open_cap_hours * 3600,
        circuit_opened_at=clock.now(),
    )
    reopened = on_failure(state, clock.now(), REASON, SETTINGS)
    assert reopened.circuit_open_seconds == SETTINGS.circuit_open_cap_hours * 3600


def test_three_reopens_inside_a_day_mark_the_source_unreachable():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    for _ in range(3):
        state = on_failure(state, clock.now(), REASON, SETTINGS)
    assert should_mark_unreachable(state, clock.now(), SETTINGS)


def test_reopens_spread_over_more_than_a_day_do_not():
    clock = FakeClock()
    state = fail_times(SourceState(source_id="s"), 5, clock)
    for _ in range(3):
        state = on_failure(state, clock.now(), REASON, SETTINGS)
    later = clock.now() + timedelta(hours=25)
    assert not should_mark_unreachable(state, later, SETTINGS)
