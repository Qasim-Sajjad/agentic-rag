"""Typed settings loaded from YAML, overridable by environment variables.

Precedence, highest first: constructor arguments, environment, .env file,
`config/settings.yaml`, field defaults. Sections forbid unknown keys so a typo
in the YAML fails at startup instead of silently keeping the default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SETTINGS_PATH_ENV = "RAG_SETTINGS_FILE"
REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_FILE = REPO_ROOT / "config" / "settings.yaml"
EXAMPLE_SETTINGS_FILE = REPO_ROOT / "config" / "settings.example.yaml"


class SettingsFileError(RuntimeError):
    """The settings file is missing, unreadable or not a mapping."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingSettings(_Section):
    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["json", "console"] = "json"


class PostgresSettings(_Section):
    dsn: str = "postgresql://rag@127.0.0.1:5432/agentic_rag"
    pool_min_size: int = 1
    pool_max_size: int = 10
    command_timeout_seconds: float = 30.0


class TierTimeouts(_Section):
    """Seconds allowed per tier. Every external call has one, no exceptions."""

    static: float = 15.0
    browser: float = 30.0
    stealth: float = 45.0
    unlocker: float = 60.0


class FetchSettings(_Section):
    min_text_chars: int = 200
    max_attempts_per_tier: int = 3
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 60.0
    max_retry_after_seconds: float = 300.0
    circuit_failure_threshold: int = 5
    circuit_open_minutes: int = 30
    circuit_open_cap_hours: int = 6
    circuit_reopen_limit: int = 3
    policy_cache_ttl_hours: int = 168
    browser_pool_size: int = 4
    default_requests_per_second: float = 1.0
    robots_cache_ttl_hours: int = 24
    lease_minutes: int = 10
    give_up_passes: int = 2
    give_up_pass_gap_hours: int = 1
    user_agent: str = "agentic-rag/0.1 (+https://example.invalid/crawler)"
    impersonate_profile: str = "chrome124"
    browser_headless: bool = True
    timeouts: TierTimeouts = TierTimeouts()
    challenge_markers: tuple[str, ...] = (
        "just a moment",
        "cf_chl_opt",
        "checking your browser",
        "datadome",
        "_abck",
    )


class ExtractSettings(_Section):
    min_chars_per_page: int = 100
    max_garbage_ratio: float = 0.2
    pages_per_task: int = 50


class QdrantSettings(_Section):
    """`path` runs Qdrant in process, `url` talks to a server. Path wins."""

    url: str = "http://127.0.0.1:6333"
    path: str | None = None
    collection: str = "corpus"
    timeout_seconds: float = 30.0


class IndexSettings(_Section):
    target_tokens: int = 512
    overlap_ratio: float = 0.1
    max_table_tokens: int = 2048
    simhash_hamming_threshold: int = 3
    embed_batch_size: int = 32
    embed_model: str = "BAAI/bge-m3"
    embed_dims: int = 1024
    tenant_id: str = "default"


class RetrieveSettings(_Section):
    candidate_pool: int = 50
    rerank_pool: int = 25
    rrf_k: int = 60
    score_floor: float = 0.3
    low_floor: float = 0.15
    elbow_delta: float = 0.15
    k_min: int = 3
    k_max: int = 15


class AgentSettings(_Section):
    max_iterations: int = 1
    recursion_limit: int = 10


class McpSettings(_Section):
    port: int = 8765
    session_call_budget: int = 20


def settings_path() -> Path | None:
    """Resolve which YAML file to load, or None when there is none.

    `config/settings.yaml` is gitignored, so a clean clone falls back to the
    committed example rather than starting with defaults nobody declared.
    """
    override = os.environ.get(SETTINGS_PATH_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise SettingsFileError(f"{SETTINGS_PATH_ENV} points at {path}, not a file")
        return path
    for candidate in (SETTINGS_FILE, EXAMPLE_SETTINGS_FILE):
        if candidate.is_file():
            return candidate
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SettingsFileError(f"could not read settings file {path}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SettingsFileError(f"settings file {path} must contain a mapping")
    return raw


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Feeds a settings file into the pydantic-settings source chain."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        path = settings_path()
        self.path = path
        self.data: dict[str, Any] = _read_yaml(path) if path is not None else {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return self.data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self.data


# ===================================================================================
# Override the BaseSettings of Pydantic to load from YAML, Environment and .env file
# ==================================================================================
class Settings(BaseSettings):
    """Root settings object. Read it through `get_settings()`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    logging: LoggingSettings = LoggingSettings()
    postgres: PostgresSettings = PostgresSettings()
    qdrant: QdrantSettings = QdrantSettings()
    fetch: FetchSettings = FetchSettings()
    extract: ExtractSettings = ExtractSettings()
    index: IndexSettings = IndexSettings()
    retrieve: RetrieveSettings = RetrieveSettings()
    agent: AgentSettings = AgentSettings()
    mcp: McpSettings = McpSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache and read the file again. For tests and config reloads."""
    get_settings.cache_clear()
    return get_settings()
