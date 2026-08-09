"""Dead letter store. A row here is a decision, not a crash.

Every row carries the reason code that produced it, so ingest status can report
"412 blocked, 89 unsupported type, 23 robots disallowed" instead of one number.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.db.pool import Database
from rag.fetch.frontier import url_hash
from rag.fetch.types import FailureReason, FetchTier


@dataclass(frozen=True)
class DeadLetterEntry:
    url: str
    source_id: str
    reason: FailureReason
    stage: str  # fetch | extract
    attempts: int
    last_tier: FetchTier | None = None
    http_status: int | None = None
    detail: str | None = None


class DeadLetterStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(self, entry: DeadLetterEntry) -> None:
        """Repeat failures update the row rather than raising on the primary key."""
        await self._db.execute(
            """
            INSERT INTO dead_letter (url_hash, url, source_id, reason, stage,
                last_tier, http_status, attempts, detail)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (url_hash) DO UPDATE SET
                reason = EXCLUDED.reason,
                stage = EXCLUDED.stage,
                last_tier = EXCLUDED.last_tier,
                http_status = EXCLUDED.http_status,
                attempts = EXCLUDED.attempts,
                detail = EXCLUDED.detail,
                last_failed_at = now()
            """,
            url_hash(entry.url),
            entry.url,
            entry.source_id,
            str(entry.reason),
            entry.stage,
            int(entry.last_tier) if entry.last_tier is not None else None,
            entry.http_status,
            entry.attempts,
            entry.detail,
        )

    async def counts_by_reason(self, source_id: str | None = None) -> dict[str, int]:
        query = "SELECT reason, count(*) AS n FROM dead_letter"
        rows = (
            await self._db.fetch(
                query + " WHERE source_id = $1 GROUP BY reason", source_id
            )
            if source_id is not None
            else await self._db.fetch(query + " GROUP BY reason")
        )
        return {row["reason"]: int(row["n"]) for row in rows}

    async def get(self, url: str) -> dict[str, object] | None:
        row = await self._db.fetchrow(
            "SELECT url, reason, stage, last_tier, http_status, attempts, detail "
            "FROM dead_letter WHERE url_hash = $1",
            url_hash(url),
        )
        return dict(row) if row is not None else None
