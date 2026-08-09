"""Logging setup: one handler, both renderers, stdlib records included."""

from __future__ import annotations

import json
import logging

import pytest

from rag.config.settings import LoggingSettings
from rag.log import configure_logging, get_logger


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


def test_json_format_emits_parseable_lines(capsys: pytest.CaptureFixture[str]):
    configure_logging(LoggingSettings(level="info", format="json"))
    get_logger("test").info("fetch complete", url="/static", tier=1)
    assert json.loads(capsys.readouterr().out)["url"] == "/static"


def test_bound_context_reaches_the_output(capsys: pytest.CaptureFixture[str]):
    configure_logging(LoggingSettings(level="info", format="json"))
    get_logger("test").bind(source_id="sec-edgar").warning("circuit open")
    assert json.loads(capsys.readouterr().out)["source_id"] == "sec-edgar"


def test_console_format_is_plain_text(capsys: pytest.CaptureFixture[str]):
    configure_logging(LoggingSettings(level="info", format="console"))
    get_logger("test").info("fetch complete")
    assert "fetch complete" in capsys.readouterr().out


def test_level_filters_below_threshold(capsys: pytest.CaptureFixture[str]):
    configure_logging(LoggingSettings(level="warning", format="json"))
    get_logger("test").info("should not appear")
    assert capsys.readouterr().out == ""


def test_stdlib_records_use_the_same_renderer(capsys: pytest.CaptureFixture[str]):
    configure_logging(LoggingSettings(level="info", format="json"))
    logging.getLogger("third_party.client").info("startup complete")
    assert json.loads(capsys.readouterr().out)["event"] == "startup complete"


def test_repeated_configuration_does_not_stack_handlers():
    configure_logging(LoggingSettings(level="info", format="json"))
    configure_logging(LoggingSettings(level="info", format="json"))
    assert len(logging.getLogger().handlers) == 1
