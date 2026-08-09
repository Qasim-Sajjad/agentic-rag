"""Circuit breaker, as pure functions over `SourceState`.

No IO here on purpose. The repository writes what these return, which makes
every transition testable with a fake clock and no database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from rag.config.settings import FetchSettings
from rag.fetch.types import CircuitState, FailureReason, SourceState

UNREACHABLE_WINDOW_HOURS = 24


class Gate(StrEnum):
    ALLOW = "allow"
    PROBE = "probe"  # half open, this is the single trial request
    DENY = "deny"  # open, return a failure without making a request


def gate(state: SourceState, now: datetime, settings: FetchSettings) -> Gate:
    """Whether a request may be made, without changing anything."""
    if state.circuit_state is CircuitState.CLOSED:
        return Gate.ALLOW
    if state.circuit_state is CircuitState.HALF_OPEN:
        return Gate.PROBE
    if state.circuit_opened_at is None:
        return Gate.DENY
    elapsed = (now - state.circuit_opened_at).total_seconds()
    return Gate.PROBE if elapsed >= state.circuit_open_seconds else Gate.DENY


def on_success(
    state: SourceState, now: datetime, settings: FetchSettings
) -> SourceState:
    """Success closes the circuit and resets the open duration to base."""
    return state.model_copy(
        update={
            "circuit_state": CircuitState.CLOSED,
            "consecutive_failures": 0,
            "circuit_opened_at": None,
            "circuit_open_seconds": settings.circuit_open_minutes * 60,
            "last_success_at": now,
        }
    )


def on_failure(
    state: SourceState, now: datetime, reason: FailureReason, settings: FetchSettings
) -> SourceState:
    """Counts the failure and opens or reopens the circuit if it is time."""
    failures = state.consecutive_failures + 1
    base = state.model_copy(
        update={
            "consecutive_failures": failures,
            "last_failure_at": now,
            "last_failure_reason": reason,
        }
    )
    if _is_probe_failure(state):
        return _reopen(base, now, settings)
    if failures >= settings.circuit_failure_threshold:
        return _open(base, now, settings)
    return base


def _is_probe_failure(state: SourceState) -> bool:
    return state.circuit_state in (CircuitState.HALF_OPEN, CircuitState.OPEN)


def _open(state: SourceState, now: datetime, settings: FetchSettings) -> SourceState:
    return state.model_copy(
        update={
            "circuit_state": CircuitState.OPEN,
            "circuit_opened_at": now,
            "circuit_open_seconds": settings.circuit_open_minutes * 60,
            "circuit_first_open_at": state.circuit_first_open_at or now,
        }
    )


def _reopen(state: SourceState, now: datetime, settings: FetchSettings) -> SourceState:
    """A failed probe doubles the wait, capped, so a dead site backs off fast."""
    cap = settings.circuit_open_cap_hours * 3600
    return state.model_copy(
        update={
            "circuit_state": CircuitState.OPEN,
            "circuit_opened_at": now,
            "circuit_open_seconds": min(cap, state.circuit_open_seconds * 2),
            "circuit_reopen_count": state.circuit_reopen_count + 1,
            "circuit_first_open_at": state.circuit_first_open_at or now,
        }
    )


def should_mark_unreachable(
    state: SourceState, now: datetime, settings: FetchSettings
) -> bool:
    """Three reopens inside 24 hours means stop scheduling this source."""
    if state.circuit_reopen_count < settings.circuit_reopen_limit:
        return False
    if state.circuit_first_open_at is None:
        return False
    window = timedelta(hours=UNREACHABLE_WINDOW_HOURS)
    return now - state.circuit_first_open_at <= window
