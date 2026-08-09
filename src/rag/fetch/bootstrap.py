"""Loads `config/sources.yaml` into the registry and seeds the frontier.

Run with: python -m rag.fetch.bootstrap

Adding a domain here is a legal decision, not a crawler decision. A link on
page 40,000 must not be able to enrol a site whose terms forbid access, which
is why discovery inside a source is automatic and this step is not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rag.clock import SystemClock
from rag.config.settings import REPO_ROOT
from rag.db.pool import Database
from rag.fetch.frontier import Frontier
from rag.fetch.registry import SourceRegistry
from rag.fetch.types import Source
from rag.log import configure_logging, get_logger

SOURCES_FILE = REPO_ROOT / "config" / "sources.yaml"
EXAMPLE_SOURCES_FILE = REPO_ROOT / "config" / "sources.example.yaml"

log = get_logger(__name__)


class SourcesFileError(RuntimeError):
    """The sources file is missing, unreadable, or not a list of sources."""


def sources_path() -> Path:
    for candidate in (SOURCES_FILE, EXAMPLE_SOURCES_FILE):
        if candidate.is_file():
            return candidate
    raise SourcesFileError(f"no sources file at {SOURCES_FILE}")


def load_sources(path: Path | None = None) -> list[Source]:
    resolved = path if path is not None else sources_path()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourcesFileError(f"could not read {resolved}") from exc
    if not isinstance(raw, list):
        raise SourcesFileError(f"{resolved} must contain a list of sources")
    try:
        return [Source.model_validate(entry) for entry in raw]
    except ValidationError as exc:
        raise SourcesFileError(f"invalid source entry in {resolved}") from exc


async def bootstrap(db: Database, path: Path | None = None) -> int:
    """Upserts every source and seeds its URLs. Returns the seed URL count."""
    registry = SourceRegistry(db)
    frontier = Frontier(db, SystemClock())
    seeded = 0
    for source in load_sources(path):
        await registry.upsert(source)
        for url in source.seed_urls:
            await frontier.add(url, source.source_id, source.priority)
            seeded += 1
        log.info(
            "source registered", source_id=source.source_id, seeds=len(source.seed_urls)
        )
    return seeded


async def _main() -> None:
    configure_logging()
    async with Database() as db:
        seeded = await bootstrap(db)
    log.info("bootstrap complete", seed_urls=seeded)


if __name__ == "__main__":
    asyncio.run(_main())
