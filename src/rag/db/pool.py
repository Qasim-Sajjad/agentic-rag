"""Connection pool. One per process, created at startup and closed at shutdown."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from types import TracebackType
from typing import Any

import asyncpg

from rag.config.settings import PostgresSettings, get_settings


class DatabaseNotConnectedError(RuntimeError):
    """Query attempted before `connect()`. A wiring bug, not a runtime failure."""


class Database:
    """Thin wrapper over an asyncpg pool.

    Exists so callers depend on one small surface rather than on asyncpg, and
    so tests can point every repository at a throwaway database by passing a
    different DSN.
    """

    def __init__(self, settings: PostgresSettings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings().postgres
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise DatabaseNotConnectedError("call connect() before querying")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.dsn,
            min_size=self._settings.pool_min_size,
            max_size=self._settings.pool_max_size,
            command_timeout=self._settings.command_timeout_seconds,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[asyncpg.pool.PoolConnectionProxy[asyncpg.Record]]:
        """A claim, a failure write and a circuit update are one transaction."""
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    async def execute(self, query: str, *args: object) -> str:
        return await self.pool.execute(query, *args)

    async def fetch(self, query: str, *args: object) -> list[asyncpg.Record]:
        return await self.pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args: object) -> asyncpg.Record | None:
        return await self.pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: object) -> Any:
        """Returns `Any` because a scalar's type depends on the query."""
        return await self.pool.fetchval(query, *args)


@lru_cache(maxsize=1)
def get_database() -> Database:
    """Process wide database handle. Connect it once during startup."""
    return Database()
