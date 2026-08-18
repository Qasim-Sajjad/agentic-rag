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
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
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


class UnlockerSettings(_Section):
    """Tier 4, a managed unlocker. Off unless a key is present AND the source
    sets `allow_unlocker`, so enabling it stays a per source decision.

    `api_key` comes from SCRAPINGBEE_API_KEY in .env, never from a committed
    file. Every call costs credits, which is why the fetcher logs each one.
    """

    provider: Literal["scrapingbee"] = "scrapingbee"
    endpoint: str = "https://app.scrapingbee.com/api/v1/"
    api_key: str = ""
    # The provider renders and solves on its side. That is the whole product,
    # and it is why this tier is opt in per source rather than a global default.
    render_js: bool = True
    premium_proxy: bool = True
    country_code: str = ""


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
    unlocker: UnlockerSettings = UnlockerSettings()


class OcrSettings(_Section):
    """VLM OCR for page ranges with no usable text layer.

    A vision model, not a classic OCR engine, because the scanned pages that
    reach this gate are the ones with tables and multi column layout, which is
    exactly where character level OCR loses the structure that chunking needs.
    """

    enabled: bool = True
    model: str = "claude-sonnet-5"
    max_tokens: int = 8192
    timeout_seconds: float = 180.0
    # One request per range, so a range wider than this is split. Guards both
    # the model's output budget and the blast radius of a failed call.
    max_pages_per_call: int = 20
    # Characters of the existing, poor text layer passed alongside the image.
    # See rule 1 in src/rag/extract/ocr.py: the model aligns to it.
    text_layer_hint_chars: int = 4000


class ExtractSettings(_Section):
    # Page furniture: a banner, a running header, a fax footer, a form artifact.
    # Printed on the page, so both the text layer and OCR read it, and it is not
    # content. See src/rag/extract/boilerplate.py.
    strip_repeated_blocks: bool = True
    repeat_min_count: int = 3
    repeat_max_chars: int = 200
    # How many number positions may differ between occurrences of the same
    # masked line. A footer's clock and page counter move, a row of readings
    # moves in every number it has.
    repeat_max_varying: int = 3
    min_chars_per_page: int = 100
    max_garbage_ratio: float = 0.2
    pages_per_task: int = 50
    # pymupdf4llm takes a GNN layout plus OCR path on every page when
    # `pymupdf-layout` is installed. Measured on a 9 page PDF: 2583 ms/page with
    # it against 227 ms/page without, for 0.7% less extracted text. Scanned
    # ranges go to the VLM, never to pymupdf4llm, so the expensive path buys
    # nothing here. On means 21 minutes for a 500 page PDF, off means 2.
    pymupdf_use_layout: bool = False
    # Page ranges are independent, so they extract concurrently. Capped because
    # each one is a CPU bound thread and oversubscribing thrashes.
    max_parallel_ranges: int = 4
    ocr: OcrSettings = OcrSettings()


class QdrantSettings(_Section):
    """`path` runs Qdrant in process, `url` talks to a server. Path wins."""

    url: str = "http://127.0.0.1:6333"
    path: str | None = None
    # Named for the vectors in it. A collection holds one width and one model:
    # 384 dimension points and 1024 dimension points cannot share it, and two
    # models' vectors in one space would retrieve worse than either alone.
    collection: str = "corpus-small"
    timeout_seconds: float = 30.0


class IndexSettings(_Section):
    target_tokens: int = 512
    overlap_ratio: float = 0.1
    max_table_tokens: int = 2048
    simhash_hamming_threshold: int = 3
    embed_batch_size: int = 32
    # Measured on this machine, 512 token chunks: bge-m3 8.6 s per chunk,
    # bge-small-en-v1.5 0.18 s. A 250 page document is roughly 600 chunks, which
    # is 90 minutes against 2. The sparse half of hybrid retrieval moves to
    # `rag.index.lexical` when the small model is selected. Set this back to
    # "BAAI/bge-m3" with `embed_dims: 1024` and a different `qdrant.collection`
    # to take the better model instead. See docs/DESIGN.md section 3.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dims: int = 384
    # Chunks target `target_tokens`, so a longer window pays for padding that is
    # never used and costs roughly an order of magnitude on CPU.
    embed_max_length: int = 512
    tenant_id: str = "default"
    # Object storage stand in. `CanonicalDoc` is written here as one immutable
    # JSON blob per document, which is what makes a re-embed a backfill rather
    # than a re-scrape.
    doc_store_path: str = "data/docs"


class RetrieveSettings(_Section):
    candidate_pool: int = 50
    rerank_pool: int = 25
    rrf_k: int = 60
    # Compared against a reranker score in 0 to 1, see src/rag/retrieve/rerank.py.
    # Low because ms-marco-MiniLM was trained on short web queries and scores a
    # conversational question against table heavy text far below what its
    # ranking quality suggests. Raising these silently returns nothing.
    score_floor: float = 0.10
    low_floor: float = 0.02
    elbow_delta: float = 0.15
    k_min: int = 3
    k_max: int = 15


class LLMSettings(_Section):
    """Model per role, see src/rag/agent/SPEC.md. The key comes from the
    environment, never from a committed file."""

    api_key: str = ""
    router_model: str = "claude-haiku-4-5-20251001"
    responder_model: str = "claude-sonnet-5"
    judge_model: str = "claude-opus-5"
    # The responder must echo a chunk id per claim inside JSON, so its output
    # scales with the number of retrieved chunks, not with answer length. At
    # 2048 a 15 chunk context truncated mid JSON: schema validation failed, the
    # repair failed identically, and every answer fell back. Measured: 4798
    # output tokens on that context.
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout_seconds: float = 60.0


class AgentSettings(_Section):
    max_iterations: int = 1
    recursion_limit: int = 10


class McpSettings(_Section):
    port: int = 8765
    session_call_budget: int = 20


class ApiSettings(_Section):
    """`api_keys` maps a key to a tenant. Keys belong in .env, not in YAML."""

    api_keys: dict[str, str] = {"dev-key": "default"}
    explain_enabled: bool = False
    cache_ttl_seconds: int = 300
    embed_cache_ttl_seconds: int = 3600


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
    llm: LLMSettings = LLMSettings()
    agent: AgentSettings = AgentSettings()
    mcp: McpSettings = McpSettings()
    api: ApiSettings = ApiSettings()

    # Declared rather than tolerated: `extra="forbid"` is what catches a typo
    # in the YAML, so the conventional unprefixed name has to be a real field.
    # The alias bypasses the RAG__ prefix, which is the point of using it.
    anthropic_api_key: str = Field(
        "", validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    scrapingbee_api_key: str = Field(
        "", validation_alias=AliasChoices("SCRAPINGBEE_API_KEY", "scrapingbee_api_key")
    )

    @model_validator(mode="after")
    def _api_key_from_environment(self) -> Settings:
        """Provider names are the conventional ones, so honour them directly
        rather than forcing `RAG__LLM__API_KEY` on top of them. A key never
        appears in a committed file, only in .env or the environment."""
        self._inject_llm_key()
        self._inject_unlocker_key()
        return self

    def _inject_llm_key(self) -> None:
        if self.llm.api_key:
            return
        key = self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            object.__setattr__(
                self, "llm", self.llm.model_copy(update={"api_key": key})
            )

    def _inject_unlocker_key(self) -> None:
        if self.fetch.unlocker.api_key:
            return
        key = self.scrapingbee_api_key or os.environ.get("SCRAPINGBEE_API_KEY", "")
        if not key:
            return
        unlocker = self.fetch.unlocker.model_copy(update={"api_key": key})
        object.__setattr__(
            self, "fetch", self.fetch.model_copy(update={"unlocker": unlocker})
        )

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
