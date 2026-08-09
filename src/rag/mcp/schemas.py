"""Tool input and output models.

Field descriptions are read by the calling model. They are part of the
interface, not documentation, so they are written for the agent.

What is absent matters as much as what is present. `tenant_id` appears in no
input schema: it is injected server side, so an agent cannot request another
tenant's corpus. `top_k` is capped in the schema rather than in code, so the
cap is discoverable and the violation is a validation error.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.fetch.types import FailureReason
from rag.retrieve.types import RetrievedChunk

MAX_TOP_K = 20

SourceHealth = Literal["healthy", "degraded", "unreachable", "never_ingested"]


class SearchCorpusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Natural language search query")
    top_k: int = Field(
        8, ge=1, le=MAX_TOP_K, description="Maximum number of chunks to return"
    )
    doc_type: Literal["html", "pdf", "office"] | None = Field(
        None, description="Restrict to one document type"
    )
    source_id: str | None = Field(None, description="Restrict to one registered source")
    date_from: date | None = Field(
        None, description="Only documents published on or after this date"
    )


class SearchCorpusOutput(BaseModel):
    chunks: list[RetrievedChunk]
    confidence: Literal["high", "low", "none"]
    k_used: int
    reason: str | None = None


class IngestStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str | None = Field(
        None, description="Registered source id. Omit for a corpus summary"
    )
    domain: str | None = Field(
        None, description="Look up by domain instead of source id"
    )


class IngestStatusOutput(BaseModel):
    source_id: str
    status: SourceHealth
    circuit_state: Literal["closed", "open", "half_open"]
    last_success_at: datetime | None = None
    last_failure_reason: FailureReason | None = None
    docs_indexed: int = 0
    docs_failed: int = 0
    coverage_note: str = ""


class CallBudgetExceededError(RuntimeError):
    """Per session cap, enforced server side and independent of the agent loop.

    A runaway agent cannot exhaust the retrieval backend even if its own
    iteration cap fails.
    """
