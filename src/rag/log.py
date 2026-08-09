"""structlog setup. Call `configure_logging()` once per process, at startup.

Stdlib records go through the same renderer as structlog ones, so uvicorn and
library logs land in the same format as ours instead of two mixed shapes.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import Processor

from rag.config.settings import LoggingSettings, get_settings

_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _renderer(output_format: str) -> Processor:
    if output_format == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=False)


def _handler(output_format: str) -> logging.Handler:
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            _renderer(output_format),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Idempotent. Repeated calls replace the root handler rather than stack."""
    resolved = settings if settings is not None else get_settings().logging
    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    root = logging.getLogger()
    root.handlers = [_handler(resolved.format)]
    root.setLevel(_LEVELS[resolved.level])


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if name is None:
        return structlog.stdlib.get_logger()
    return structlog.stdlib.get_logger(name)
