"""Structured output validation, the deterministic defence layer.

An injection that changes the model's wording is survivable. One that
fabricates a source is not. Every cited chunk id is checked in code against the
actual retrieved set, so a fabricated source cannot leave the system regardless
of how the model behaved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CITATION_MARKER = re.compile(r"\[([a-zA-Z0-9_-]{4,64})\]")


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_url: str


class AnswerPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "partial", "insufficient"] = "partial"
    unanswered_aspects: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    citations_checked: int = 0  # ids the model emitted, before repair
    citations_rejected: int = 0  # ids not in the retrieved set
    repair_attempts: int = 0  # 0 or 1, never more
    fell_back: bool = False


@dataclass(frozen=True)
class ValidationOutcome:
    payload: AnswerPayload | None
    error: str | None
    report: ValidationReport

    @property
    def ok(self) -> bool:
        return self.payload is not None and self.error is None


def strip_fence(text: str) -> str:
    """Drop a markdown code fence around model output.

    Models wrap JSON in ```json blocks whatever the prompt says. Parsing that
    as-is fails, burns the one repair turn, and doubles cost on every call, so
    this runs before validation on every path rather than on some of them.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    return cleaned.strip()


def parse_payload(raw: str) -> tuple[AnswerPayload | None, str | None]:
    try:
        return AnswerPayload.model_validate_json(raw), None
    except ValidationError as exc:
        return None, f"schema validation failed: {exc.error_count()} errors"


def unresolved_markers(answer: str, retrieved_ids: set[str]) -> list[str]:
    """Every `[id]` marker in the prose must resolve to a retrieved chunk."""
    found = {match.group(1) for match in CITATION_MARKER.finditer(answer)}
    return sorted(marker for marker in found if marker not in retrieved_ids)


def validate(raw: str, retrieved_ids: set[str], repairs: int = 0) -> ValidationOutcome:
    """Schema, then citations, then inline markers. Order is cheapest first."""
    payload, error = parse_payload(strip_fence(raw))
    if payload is None:
        return ValidationOutcome(None, error, ValidationReport(repair_attempts=repairs))
    rejected = [c for c in payload.citations if c.chunk_id not in retrieved_ids]
    dangling = unresolved_markers(payload.answer, retrieved_ids)
    report = ValidationReport(
        citations_checked=len(payload.citations),
        citations_rejected=len(rejected),
        repair_attempts=repairs,
    )
    problem = _describe(rejected, dangling)
    return ValidationOutcome(None if problem else payload, problem, report)


def _describe(rejected: list[Citation], dangling: list[str]) -> str | None:
    if rejected:
        ids = ", ".join(sorted(c.chunk_id for c in rejected))
        return f"cited chunk ids not in the retrieved set: {ids}"
    if dangling:
        return f"answer references unknown chunk ids: {', '.join(dangling)}"
    return None


def fallback_answer(chunks_summary: str, reason: str) -> AnswerPayload:
    """Deterministic template used when validation fails twice. Never a loop."""
    return AnswerPayload(
        answer=(
            "A grounded answer could not be generated for this question. "
            f"Reason: {reason}. The retrieved passages are listed below so the "
            f"question can be answered by hand.\n\n{chunks_summary}"
        ),
        citations=[],
        confidence="insufficient",
    )
