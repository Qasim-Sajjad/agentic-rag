"""Request and response shapes. Four endpoints, deliberately separated."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.state import TraceStep
from rag.mcp.schemas import MAX_TOP_K
from rag.prompts.render import StrippedMarker
from rag.prompts.validate import Citation, ValidationReport
from rag.retrieve.types import RetrievedChunk, SearchFilters


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(8, ge=1, le=MAX_TOP_K)
    filters: SearchFilters | None = None


class SearchResponse(BaseModel):
    chunks: list[RetrievedChunk]
    confidence: Literal["high", "low", "none"]
    k_used: int
    latency_ms: int


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    filters: SearchFilters | None = None
    explain: bool = False


class ExplainBlock(BaseModel):
    """Debug affordance, gated by config as well as by the API key.

    Carries the assembled context and the strip log, never the system prompt
    body. `prompt_version` identifies it. An endpoint that echoes the prompt is
    a prompt exfiltration endpoint.
    """

    nonce: str
    prompt_version: str
    task_position: str
    rendered_context: str
    stripped: list[StrippedMarker]


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "low", "none", "insufficient"]
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    validation: ValidationReport = ValidationReport()
    explain: ExplainBlock | None = None
    latency_ms: int = 0


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "low", "none"]
    trace: list[TraceStep] = Field(default_factory=list)


class SourceStatusRow(BaseModel):
    source_id: str
    status: str
    circuit_state: str
    last_success_at: Any = None
    last_failure_reason: str | None = None
    docs_indexed: int = 0
    docs_failed: int = 0


class IngestSummary(BaseModel):
    total_sources: int = 0
    healthy: int = 0
    degraded: int = 0
    unreachable: int = 0


class IngestStatusResponse(BaseModel):
    sources: list[SourceStatusRow] = Field(default_factory=list)
    summary: IngestSummary = IngestSummary()


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    """Consistent shape, typed reason codes, never a stack trace."""

    error: ErrorBody
