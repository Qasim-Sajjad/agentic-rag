"""Migration runner. Applies every .sql file in `migrations/` once, in order.

Run with: python -m rag.db.migrate
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rag.db.pool import Database
from rag.log import configure_logging, get_logger

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migration (
    name        TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

log = get_logger(__name__)


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """Sorted by filename, which is why they are numbered."""
    return sorted(directory.glob("*.sql"))


async def apply_migrations(db: Database, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Applies pending migrations and returns the names applied this run."""
    await db.execute(_TRACKING_TABLE)
    applied: list[str] = []
    for path in migration_files(directory):
        if await _already_applied(db, path.name):
            continue
        async with db.transaction() as conn:
            await conn.execute(path.read_text(encoding="utf-8"))
            await conn.execute(
                "INSERT INTO schema_migration (name) VALUES ($1)", path.name
            )
        applied.append(path.name)
        log.info("migration applied", migration=path.name)
    return applied


async def _already_applied(db: Database, name: str) -> bool:
    found = await db.fetchval("SELECT 1 FROM schema_migration WHERE name = $1", name)
    return found is not None


async def _main() -> None:
    configure_logging()
    async with Database() as db:
        applied = await apply_migrations(db)
    log.info("migrations complete", applied=len(applied))


if __name__ == "__main__":
    asyncio.run(_main())
