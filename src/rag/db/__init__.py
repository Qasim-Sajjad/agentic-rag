"""Postgres access shared by the modules that own tables in the same database.

`fetch` owns the source registry, frontier and dead letter. `index` owns
documents and chunks. The connection pool and the migration runner belong to
neither, so they live here.
"""

from rag.db.migrate import apply_migrations
from rag.db.pool import Database, get_database

__all__ = ["Database", "apply_migrations", "get_database"]
