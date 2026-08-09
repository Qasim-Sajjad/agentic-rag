"""Fetch contracts. Defined in src/rag/fetch/SPEC.md.

Nothing here does work. Types and enums only, so every other module in the
package can import them without an import cycle.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FetchTier(IntEnum):
    STATIC = 1  # curl_cffi with TLS impersonation
    BROWSER = 2  # Playwright plus Chromium
    STEALTH = 3  # Playwright plus Camoufox
    UNLOCKER = 4  # managed third party API


class FailureReason(StrEnum):
    BLOCKED_PERSISTENT = "blocked_persistent"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ROBOTS_DISALLOWED = "robots_disallowed"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    UNSUPPORTED_TYPE = "unsupported_type"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class SourceStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    UNREACHABLE = "unreachable"
    RETIRED = "retired"


class FetchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    final_url: str
    status: int
    content: bytes
    content_type: str
    tier_used: FetchTier
    attempts: int
    fetched_at: datetime
    # Block signatures include `cf-mitigated`, and 429 handling needs
    # Retry-After, so headers are part of the result rather than dropped.
    headers: dict[str, str] = Field(default_factory=dict)


class FetchFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    reason: FailureReason
    last_tier: FetchTier
    attempts: int
    detail: str
    retry_after_seconds: float | None = None


class Source(BaseModel):
    """One row per source, not per URL. Crawl policy, human edited."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    domain: str
    seed_urls: list[str] = Field(default_factory=list)
    status: SourceStatus = SourceStatus.ACTIVE
    max_tier: FetchTier = FetchTier.STEALTH
    allow_unlocker: bool = False
    requests_per_second: float = 1.0
    crawl_delay_seconds: float | None = None
    robots_txt: str | None = None
    robots_fetched_at: datetime | None = None
    tos_note: str | None = None
    priority: int = 5

    @property
    def effective_rate(self) -> float:
        """Crawl-delay from robots.txt overrides the configured rate."""
        if self.crawl_delay_seconds is not None and self.crawl_delay_seconds > 0:
            return 1.0 / self.crawl_delay_seconds
        return self.requests_per_second


class SourceState(BaseModel):
    """Machine written runtime state. Circuit breaker plus policy cache."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    circuit_opened_at: datetime | None = None
    circuit_open_seconds: int = 1800
    circuit_reopen_count: int = 0
    circuit_first_open_at: datetime | None = None
    preferred_tier: FetchTier = FetchTier.STATIC
    tier_learned_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_failure_reason: FailureReason | None = None
    docs_indexed: int = 0
    docs_failed: int = 0


class FrontierEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    url_hash: str
    url: str
    source_id: str
    attempts: int = 0
    passes: int = 0
    priority: int = 5
