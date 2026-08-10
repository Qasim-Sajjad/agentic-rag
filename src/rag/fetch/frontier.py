"""Frontier queue in Postgres, claimed with SELECT ... FOR UPDATE SKIP LOCKED.

No broker. The claim, the failure write and the circuit update are one
transaction, which is the property a separate queue would give up.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import asyncpg

from rag.clock import Clock
from rag.db.pool import Database
from rag.fetch.types import FrontierEntry


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _to_entry(row: asyncpg.Record) -> FrontierEntry:
    return FrontierEntry(
        url_hash=row["url_hash"],
        url=row["url"],
        source_id=row["source_id"],
        attempts=row["attempts"],
        passes=row["passes"],
        priority=row["priority"],
    )


class Frontier:
    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    async def add(self, url: str, source_id: str, priority: int = 5) -> str:
        """Idempotent on canonical URL, which is the first dedup point."""
        digest = url_hash(url)
        await self._db.execute(
            """
            INSERT INTO frontier (url_hash, url, source_id, priority)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (url_hash) DO NOTHING
            """,
            digest,
            url,
            source_id,
            priority,
        )
        return digest

    async def claim(
        self,
        worker: str,
        limit: int,
        lease_minutes: int,
        source_id: str | None = None,
    ) -> list[FrontierEntry]:
        """SKIP LOCKED is what makes this safe across workers without a broker.

        `source_id` scopes a worker to one source, which is what keeps a crawl
        of one domain from picking up another domain's queue and applying the
        wrong rate limit to it.
        """
        rows = await self._db.fetch(
            """
            UPDATE frontier SET status = 'leased', leased_by = $1,
                   lease_expires_at = now() + ($3 || ' minutes')::interval,
                   attempts = attempts + 1, updated_at = now()
            WHERE url_hash IN (
                SELECT url_hash FROM frontier
                WHERE status = 'pending' AND visible_at <= now()
                  AND ($4::text IS NULL OR source_id = $4::text)
                ORDER BY priority, visible_at
                LIMIT $2 FOR UPDATE SKIP LOCKED)
            RETURNING url_hash, url, source_id, attempts, passes, priority
            """,
            worker,
            limit,
            str(lease_minutes),
            source_id,
        )
        return [_to_entry(row) for row in rows]

    async def complete(self, digest: str) -> None:
        await self._db.execute(
            "UPDATE frontier SET status = 'done', updated_at = now() "
            "WHERE url_hash = $1",
            digest,
        )

    async def requeue(self, digest: str, delay_seconds: float) -> datetime:
        """Delayed visibility. The worker moves on instead of sleeping."""
        visible_at = self._clock.now() + timedelta(seconds=delay_seconds)
        await self._db.execute(
            """
            UPDATE frontier SET status = 'pending', visible_at = $2,
                   leased_by = NULL, lease_expires_at = NULL, updated_at = now()
            WHERE url_hash = $1
            """,
            digest,
            visible_at,
        )
        return visible_at

    async def record_pass(self, digest: str) -> int:
        """A scheduling pass that exhausted every tier. Two of these is give up."""
        value = await self._db.fetchval(
            """
            UPDATE frontier SET passes = passes + 1, last_pass_at = now(),
                   updated_at = now()
            WHERE url_hash = $1 RETURNING passes
            """,
            digest,
        )
        return int(value) if value is not None else 0

    async def kill(self, digest: str) -> None:
        await self._db.execute(
            "UPDATE frontier SET status = 'dead', updated_at = now() "
            "WHERE url_hash = $1",
            digest,
        )

    async def sweep_expired_leases(self) -> int:
        """Recovers work from a crashed worker."""
        result = await self._db.execute(
            """
            UPDATE frontier SET status = 'pending', leased_by = NULL,
                   lease_expires_at = NULL, updated_at = now()
            WHERE status = 'leased' AND lease_expires_at < now()
            """
        )
        return int(result.split()[-1]) if result else 0

    async def pending_count(self) -> int:
        value = await self._db.fetchval(
            "SELECT count(*) FROM frontier WHERE status = 'pending'"
        )
        return int(value or 0)
