"""Request and response shapes. Four endpoints, deliberately separated."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.state import TraceStep
from rag.mcp.schemas import MAX_TOP_K
from rag.prompts.render import StrippedMarker
from rag.prompts.validate import Citation, ValidationReport
from rag.retrieve.types import RetrievalStep, RetrievedChunk, SearchFilters


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(8, ge=1, le=MAX_TOP_K)
    filters: SearchFilters | None = None


class SearchResponse(BaseModel):
    chunks: list[RetrievedChunk]
    confidence: Literal["high", "low", "none"]
    k_used: int
    # The funnel, stage by stage. Retrieval is five steps and a total latency
    # cannot say which of them cost the time or dropped the chunk.
    steps: list[RetrievalStep] = Field(default_factory=list)
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
    # The context the responder saw. Same shape and same reason as on `/ask`.
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class IngestUrlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    # Omitted means "resolve the domain in the registry". Passing it explicitly
    # attributes the document to a source that already exists.
    source_id: str | None = None
    # Seeding a domain is a legal decision, see src/rag/fetch/SPEC.md, so an
    # unregistered domain fails by default and this flag is the deliberate act.
    # It registers a source. It does not relax robots, rate limits or the
    # unlocker ban, none of which this flag can reach.
    register_domain: bool = False
    # A second, separate decision from `register_domain`. Tier 4 is a paid
    # service that solves a challenge on the caller's behalf, so it costs money
    # and is only appropriate for a domain whose terms permit automated access.
    # Bare registration must never imply it: this flag exists so a reviewer of
    # the request can see the unlocker was chosen deliberately, not inherited
    # from clicking "register" on an unfamiliar URL.
    allow_unlocker: bool = False


class PipelineStage(BaseModel):
    """One observed step. `latency_ms` is present only where it was measured."""

    name: str
    status: Literal["ok", "skipped", "failed"]
    latency_ms: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class ChunkPreview(BaseModel):
    """What the chunker produced, as returned by the pipeline rather than
    recomputed. Recomputing risks showing chunks that were never written."""

    chunk_id: str
    chunk_index: int
    section_path: list[str] = Field(default_factory=list)
    token_count: int
    is_table: bool = False
    page_no: int | None = None
    text: str
    truncated: bool = False


class IngestFailure(BaseModel):
    """Typed reason code and the stage that produced it, never a stack trace."""

    stage: str
    reason: str
    detail: str


class IngestTraceResponse(BaseModel):
    """The whole pipeline as a list of stages, which is the part a reviewer
    wants to see. Returns 200 on a fetch that was blocked: the request
    succeeded, the site refused, and the trace says so."""

    ok: bool
    source_id: str = ""
    source_url: str = ""
    doc_id: str | None = None
    doc_type: str | None = None
    title: str | None = None
    stages: list[PipelineStage] = Field(default_factory=list)
    chunks_written: int = 0
    vectors_written: int = 0
    skipped_reason: str | None = None
    chunk_preview: list[ChunkPreview] = Field(default_factory=list)
    failure: IngestFailure | None = None
    latency_ms: int = 0


class ProgressRow(BaseModel):
    """One stage's latest position while the job is still running.

    `total` is 0 when the stage cannot know it in advance, which a client should
    render as indeterminate rather than as 0%.
    """

    stage: str
    done: int
    total: int
    detail: str = ""


class IngestJobAccepted(BaseModel):
    """The 202 body. Deliberately minimal: an id and where to poll it."""

    job_id: str
    status: str
    poll: str


class IngestJobStatus(BaseModel):
    """`progress` is the live view, `result` only appears once status is done.

    Both are present rather than one replacing the other, so a client polling
    the same shape throughout never has to switch how it reads the response.
    """

    job_id: str
    kind: str
    label: str
    status: str
    elapsed_ms: int
    progress: list[ProgressRow] = Field(default_factory=list)
    result: IngestTraceResponse | None = None
    error: str | None = None


class IngestJobList(BaseModel):
    jobs: list[IngestJobStatus] = Field(default_factory=list)


class SourceStatusRow(BaseModel):
    source_id: str
    status: str
    circuit_state: str
    last_success_at: Any = None
    last_failure_reason: str | None = None
    docs_indexed: int = 0
    docs_failed: int = 0
    # Queue state, so an ingest in progress is visible while it runs.
    pending: int = 0
    in_flight: int = 0
    requeued: int = 0
    coverage_note: str = ""


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
