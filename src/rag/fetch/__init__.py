"""Tiered fetching, retry, circuit breaker."""

from rag.fetch.factory import build_fetchers, build_service, close_fetchers
from rag.fetch.service import FetchService, UnknownSourceError
from rag.fetch.types import (
    CircuitState,
    FailureReason,
    FetchFailure,
    FetchResult,
    FetchTier,
    Source,
    SourceState,
    SourceStatus,
)

__all__ = [
    "CircuitState",
    "FailureReason",
    "FetchFailure",
    "FetchResult",
    "FetchService",
    "FetchTier",
    "Source",
    "SourceState",
    "SourceStatus",
    "UnknownSourceError",
    "build_fetchers",
    "build_service",
    "close_fetchers",
]
