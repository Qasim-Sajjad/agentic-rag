"""Settings loader: precedence, fallback, and the ways loading is meant to fail."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config.settings import (
    EXAMPLE_SETTINGS_FILE,
    SETTINGS_PATH_ENV,
    Settings,
    SettingsFileError,
    settings_path,
)


@pytest.fixture
def yaml_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def write(body: str) -> Path:
        path = tmp_path / "settings.yaml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv(SETTINGS_PATH_ENV, str(path))
        return path

    return write

# Tests for the Settings Loader

def test_yaml_values_override_defaults(yaml_file):
    yaml_file("fetch:\n  min_text_chars: 400\n")
    assert Settings().fetch.min_text_chars == 400


def test_untouched_sections_keep_their_defaults(yaml_file):
    yaml_file("fetch:\n  min_text_chars: 400\n")
    assert Settings().retrieve.k_max == 15


def test_env_overrides_yaml(yaml_file, monkeypatch: pytest.MonkeyPatch):
    yaml_file("fetch:\n  min_text_chars: 400\n")
    monkeypatch.setenv("RAG__FETCH__MIN_TEXT_CHARS", "999")
    assert Settings().fetch.min_text_chars == 999


def test_empty_file_is_not_an_error(yaml_file):
    yaml_file("")
    assert Settings().index.target_tokens == 512


def test_unknown_key_is_rejected(yaml_file):
    yaml_file("fetch:\n  min_text_chars: 400\n  typo_key: 1\n")
    with pytest.raises(ValidationError):
        Settings()


def test_wrong_type_is_rejected(yaml_file):
    yaml_file("fetch:\n  min_text_chars: not_a_number\n")
    with pytest.raises(ValidationError):
        Settings()


def test_non_mapping_file_is_rejected(yaml_file):
    yaml_file("- one\n- two\n")
    with pytest.raises(SettingsFileError):
        Settings()


def test_missing_explicit_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(tmp_path / "absent.yaml"))
    with pytest.raises(SettingsFileError):
        settings_path()


def test_falls_back_to_a_committed_file(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(SETTINGS_PATH_ENV, raising=False)
    path = settings_path()
    assert path is not None and path.is_file()


def test_settings_are_frozen():
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.fetch = settings.fetch


def test_example_file_matches_the_model(monkeypatch: pytest.MonkeyPatch):
    """The committed example must stay loadable, or a clean clone breaks."""
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(EXAMPLE_SETTINGS_FILE))
    assert Settings().mcp.port == 8765
