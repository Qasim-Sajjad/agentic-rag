"""The /ask path: retrieve, generate, validate, report what happened.

Separate from `main.py` because it is the only endpoint with real logic. It
returns the chunks it used, what the renderer stripped, and whether the
citation check fired, which is what makes the pipeline inspectable rather than
something you have to trust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rag.agent.llm import LLMClient, LLMUnavailableError
from rag.api.models import AskRequest, AskResponse, ExplainBlock
from rag.config.settings import Settings
from rag.log import get_logger
from rag.prompts.registry import PromptRegistry
from rag.prompts.render import RenderedContext, assemble, new_nonce, render_context
from rag.prompts.validate import (
    ValidationOutcome,
    ValidationReport,
    fallback_answer,
    validate,
)
from rag.retrieve.service import SearchService
from rag.retrieve.types import SearchFilters, SearchResult

log = get_logger(__name__)


@dataclass
class AskDependencies:
    search: SearchService
    llm: LLMClient
    prompts: PromptRegistry
    settings: Settings


@dataclass(frozen=True)
class _Attempt:
    """Everything the response needs, collected so `_respond` stays in bounds."""

    result: SearchResult
    rendered: RenderedContext
    outcome: ValidationOutcome
    prompt_version: str
    started: float


async def ask(
    request: AskRequest, deps: AskDependencies, tenant_id: str
) -> AskResponse:
    started = time.monotonic()
    filters = _with_tenant(request, tenant_id)
    result = await deps.search.search(request.question, filters)
    prompt = deps.prompts.get("rag_answer")
    rendered = render_context(result.chunks, new_nonce())
    outcome = await _generate(request, deps, prompt.text, rendered, result)
    attempt = _Attempt(result, rendered, outcome, prompt.identifier, started)
    return _respond(request, deps, attempt)


def _with_tenant(request: AskRequest, tenant_id: str) -> SearchFilters:
    """The tenant comes from the API key, so it overwrites whatever the body
    said rather than merging with it."""
    base = request.filters or SearchFilters()
    return base.model_copy(update={"tenant_id": tenant_id})


async def _generate(
    request: AskRequest,
    deps: AskDependencies,
    system: str,
    rendered: RenderedContext,
    result: SearchResult,
) -> ValidationOutcome:
    """One repair carrying the specific error, then a deterministic fallback."""
    if not result.chunks:
        return ValidationOutcome(None, "no relevant documents", ValidationReport())
    retrieved = {chunk.chunk_id for chunk in result.chunks}
    user = assemble(system, rendered, request.question)
    first = await _once(deps, system, user, retrieved, 0)
    if first.ok or first.error is None:
        return first
    repair = f"{user}\n\nYour previous answer was rejected: {first.error}. Fix it."
    return await _once(deps, system, repair, retrieved, 1)


async def _once(
    deps: AskDependencies, system: str, user: str, retrieved: set[str], repairs: int
) -> ValidationOutcome:
    try:
        completion = await deps.llm.complete(
            system, user, deps.settings.llm.responder_model
        )
    except LLMUnavailableError as exc:
        return ValidationOutcome(
            None, str(exc), ValidationReport(repair_attempts=repairs)
        )
    return validate(completion.text, retrieved, repairs)


def _respond(
    request: AskRequest, deps: AskDependencies, attempt: _Attempt
) -> AskResponse:
    outcome = attempt.outcome
    payload = outcome.payload or fallback_answer(
        _summary(attempt.result), outcome.error or "validation failed twice"
    )
    report = outcome.report.model_copy(update={"fell_back": outcome.payload is None})
    return AskResponse(
        answer=payload.answer,
        citations=list(payload.citations),
        confidence="none" if not attempt.result.chunks else attempt.result.confidence,
        chunks=attempt.result.chunks,
        validation=report,
        explain=_explain(request, deps, attempt.rendered, attempt.prompt_version),
        latency_ms=int((time.monotonic() - attempt.started) * 1000),
    )


def _explain(
    request: AskRequest,
    deps: AskDependencies,
    rendered: RenderedContext,
    prompt_version: str,
) -> ExplainBlock | None:
    """Two gates, not one. A valid key is not authorisation to read prompt
    internals, so the config flag has to be on as well."""
    if not (request.explain and deps.settings.api.explain_enabled):
        return None
    return ExplainBlock(
        nonce=rendered.nonce,
        prompt_version=prompt_version,
        task_position=rendered.task_position,
        rendered_context=rendered.text,
        stripped=rendered.stripped,
    )


def _summary(result: SearchResult) -> str:
    return "\n".join(
        f"- [{chunk.chunk_id}] {chunk.source_url}: {chunk.text[:200]}"
        for chunk in result.chunks
    )
