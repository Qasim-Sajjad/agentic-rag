"""Source registry and per source runtime state.

Two tables, split because `source` is human edited policy and `source_state` is
machine written runtime state. Wiping state must never lose crawl policy.
"""

from __future__ import annotations

import json
from datetime import datetime

import asyncpg

from rag.db.pool import Database
from rag.fetch.types import (
    CircuitState,
    FailureReason,
    FetchTier,
    Source,
    SourceState,
    SourceStatus,
)

_SOURCE_COLUMNS = """
    source_id, domain, seed_urls, status, max_tier, allow_unlocker,
    requests_per_second, crawl_delay_seconds, robots_txt, robots_fetched_at,
    tos_note, priority
"""

_STATE_COLUMNS = """
    source_id, circuit_state, consecutive_failures, circuit_opened_at,
    circuit_open_seconds, circuit_reopen_count, circuit_first_open_at,
    preferred_tier, tier_learned_at, last_success_at, last_failure_at,
    last_failure_reason, docs_indexed, docs_failed
"""


def _to_source(row: asyncpg.Record) -> Source:
    return Source(
        source_id=row["source_id"],
        domain=row["domain"],
        seed_urls=json.loads(row["seed_urls"]),
        status=SourceStatus(row["status"]),
        max_tier=FetchTier(row["max_tier"]),
        allow_unlocker=row["allow_unlocker"],
        requests_per_second=row["requests_per_second"],
        crawl_delay_seconds=row["crawl_delay_seconds"],
        robots_txt=row["robots_txt"],
        robots_fetched_at=row["robots_fetched_at"],
        tos_note=row["tos_note"],
        priority=row["priority"],
    )


def _to_state(row: asyncpg.Record) -> SourceState:
    reason = row["last_failure_reason"]
    return SourceState(
        source_id=row["source_id"],
        circuit_state=CircuitState(row["circuit_state"]),
        consecutive_failures=row["consecutive_failures"],
        circuit_opened_at=row["circuit_opened_at"],
        circuit_open_seconds=row["circuit_open_seconds"],
        circuit_reopen_count=row["circuit_reopen_count"],
        circuit_first_open_at=row["circuit_first_open_at"],
        preferred_tier=FetchTier(row["preferred_tier"]),
        tier_learned_at=row["tier_learned_at"],
        last_success_at=row["last_success_at"],
        last_failure_at=row["last_failure_at"],
        last_failure_reason=FailureReason(reason) if reason else None,
        docs_indexed=row["docs_indexed"],
        docs_failed=row["docs_failed"],
    )


class SourceRegistry:
    """Owned by `fetch`. `mcp` and `api` read from it, neither writes."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, source_id: str) -> Source | None:
        row = await self._db.fetchrow(
            f"SELECT {_SOURCE_COLUMNS} FROM source WHERE source_id = $1", source_id
        )
        return _to_source(row) if row is not None else None

    async def by_domain(self, domain: str) -> Source | None:
        row = await self._db.fetchrow(
            f"SELECT {_SOURCE_COLUMNS} FROM source WHERE domain = $1", domain.lower()
        )
        return _to_source(row) if row is not None else None

    async def list_all(self) -> list[Source]:
        rows = await self._db.fetch(
            f"SELECT {_SOURCE_COLUMNS} FROM source ORDER BY priority, source_id"
        )
        return [_to_source(row) for row in rows]

    async def upsert(self, source: Source) -> None:
        await self._db.execute(
            """
            INSERT INTO source (source_id, domain, seed_urls, status, max_tier,
                allow_unlocker, requests_per_second, tos_note, priority)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (source_id) DO UPDATE SET
                domain = EXCLUDED.domain,
                seed_urls = EXCLUDED.seed_urls,
                status = EXCLUDED.status,
                max_tier = EXCLUDED.max_tier,
                allow_unlocker = EXCLUDED.allow_unlocker,
                requests_per_second = EXCLUDED.requests_per_second,
                tos_note = EXCLUDED.tos_note,
                priority = EXCLUDED.priority,
                updated_at = now()
            """,
            source.source_id,
            source.domain.lower(),
            json.dumps(source.seed_urls),
            str(source.status),
            int(source.max_tier),
            source.allow_unlocker,
            source.requests_per_second,
            source.tos_note,
            source.priority,
        )
        await self._db.execute(
            "INSERT INTO source_state (source_id) VALUES ($1) ON CONFLICT DO NOTHING",
            source.source_id,
        )

    async def save_robots(
        self, source_id: str, text: str, fetched_at: datetime, crawl_delay: float | None
    ) -> None:
        await self._db.execute(
            """
            UPDATE source SET robots_txt = $2, robots_fetched_at = $3,
                   crawl_delay_seconds = $4, updated_at = now()
            WHERE source_id = $1
            """,
            source_id,
            text,
            fetched_at,
            crawl_delay,
        )

    async def set_status(self, source_id: str, status: SourceStatus) -> None:
        await self._db.execute(
            "UPDATE source SET status = $2, updated_at = now() WHERE source_id = $1",
            source_id,
            str(status),
        )

    async def state(self, source_id: str) -> SourceState:
        row = await self._db.fetchrow(
            f"SELECT {_STATE_COLUMNS} FROM source_state WHERE source_id = $1", source_id
        )
        if row is None:
            await self._db.execute(
                "INSERT INTO source_state (source_id) VALUES ($1) "
                "ON CONFLICT DO NOTHING",
                source_id,
            )
            return SourceState(source_id=source_id)
        return _to_state(row)

    async def save_state(self, state: SourceState) -> None:
        await self._db.execute(
            """
            UPDATE source_state SET
                circuit_state = $2, consecutive_failures = $3,
                circuit_opened_at = $4, circuit_open_seconds = $5,
                circuit_reopen_count = $6, circuit_first_open_at = $7,
                preferred_tier = $8, tier_learned_at = $9,
                last_success_at = $10, last_failure_at = $11,
                last_failure_reason = $12
            WHERE source_id = $1
            """,
            state.source_id,
            str(state.circuit_state),
            state.consecutive_failures,
            state.circuit_opened_at,
            state.circuit_open_seconds,
            state.circuit_reopen_count,
            state.circuit_first_open_at,
            int(state.preferred_tier),
            state.tier_learned_at,
            state.last_success_at,
            state.last_failure_at,
            str(state.last_failure_reason) if state.last_failure_reason else None,
        )
