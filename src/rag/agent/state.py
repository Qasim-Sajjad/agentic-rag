"""Agent state and the plan the router produces.

What is kept: normalized chunks, the trace, call fingerprints, iteration count.
What is dropped: raw MCP envelopes after normalization, the full text of chunks
the reranker cut, and conversation history, because each call is stateless. On
retry the previous chunks are replaced rather than accumulated, so the
responder never sees two overlapping context sets.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from rag.prompts.validate import Citation
from rag.retrieve.types import RetrievedChunk

ToolName = Literal["search_corpus", "get_ingest_status", "answer_directly"]


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: ToolName
    query: str | None = None
    source_id: str | None = None
    doc_type: Literal["html", "pdf", "office"] | None = None
    reason: str = ""

    def fingerprint(self) -> str:
        """Hash of (tool, sorted args). A repeat is rejected before MCP sees it.

        The iteration cap alone does not prevent this: a router will happily
        reissue an identical query on the retry it was given.
        """
        args = {
            "query": self.query,
            "source_id": self.source_id,
            "doc_type": self.doc_type,
        }
        payload = json.dumps([self.tool, sorted(args.items())], default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TraceStep(BaseModel):
    node: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    latency_ms: int = 0
    model: str | None = None
    prompt_version: str | None = None
    note: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    tenant_id: str
    plan: Plan | None
    chunks: list[RetrievedChunk]
    trace: list[TraceStep]
    seen_call_hashes: set[str]
    iteration: int
    confidence: Literal["high", "low", "none"]
    error: str | None
    answer: str | None
    citations: list[Citation]
    coverage_note: str | None


def initial_state(question: str, tenant_id: str = "default") -> AgentState:
    return AgentState(
        question=question,
        tenant_id=tenant_id,
        plan=None,
        chunks=[],
        trace=[],
        seen_call_hashes=set(),
        iteration=0,
        confidence="none",
        error=None,
        answer=None,
        citations=[],
        coverage_note=None,
    )


class AgentAnswer(BaseModel):
    """What `/agent` returns. The trace is a first class part of it."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "low", "none"] = "none"
    trace: list[TraceStep] = Field(default_factory=list)
    # The chunks the responder actually saw, for the same reason `/ask` returns
    # them: an answer whose evidence cannot be inspected cannot be checked. On
    # retry these are the replacement set, not both sets, which is what makes
    # them the context the answer was written from.
    chunks: list[RetrievedChunk] = Field(default_factory=list)
