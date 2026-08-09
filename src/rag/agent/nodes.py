"""The four nodes. Only the responder ever sees retrieved content."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from rag.agent.assess import broaden_plan, is_retry
from rag.agent.llm import Completion, LLMClient, LLMUnavailableError
from rag.agent.state import AgentState, Plan, TraceStep
from rag.config.settings import AgentSettings, LLMSettings
from rag.log import get_logger
from rag.mcp.schemas import IngestStatusInput, SearchCorpusInput
from rag.mcp.tools import ToolService
from rag.prompts.registry import PromptRegistry
from rag.prompts.render import assemble, new_nonce, render_context
from rag.prompts.validate import (
    ValidationOutcome,
    ValidationReport,
    fallback_answer,
    validate,
)

log = get_logger(__name__)

MAX_REPAIRS = 1


@dataclass
class NodeDependencies:
    llm: LLMClient
    tools: ToolService
    prompts: PromptRegistry
    llm_settings: LLMSettings
    agent_settings: AgentSettings


def _step(node: str, started: float, **fields: Any) -> TraceStep:
    latency = int((time.monotonic() - started) * 1000)
    return TraceStep(node=node, latency_ms=latency, **fields)


async def router(state: AgentState, deps: NodeDependencies) -> AgentState:
    """Reads the question only. Never sees retrieved content.

    That is structural, not incidental: a poisoned chunk cannot reach this
    node, so the routing decision is immune to injection by construction
    rather than by instruction.
    """
    started = time.monotonic()
    prompt = deps.prompts.get("router")
    model = deps.llm_settings.router_model
    retry = is_retry(state)
    plan, note = await _plan_from_llm(deps, prompt.text, state["question"], model)
    if retry:
        plan = broaden_plan(plan)
        state["iteration"] = state.get("iteration", 0) + 1
    state["plan"] = plan
    state["trace"].append(
        _step(
            "router",
            started,
            model=model,
            prompt_version=prompt.identifier,
            args={"tool": plan.tool},
            note=note,
        )
    )
    return state


async def _plan_from_llm(
    deps: NodeDependencies, system: str, question: str, model: str
) -> tuple[Plan, str | None]:
    try:
        completion = await deps.llm.complete(system, question, model)
    except LLMUnavailableError as exc:
        return Plan(
            tool="search_corpus", query=question, reason="router unavailable"
        ), str(exc)
    return _parse_plan(completion, question)


def _parse_plan(completion: Completion, question: str) -> tuple[Plan, str | None]:
    """A malformed plan falls back to search rather than failing the request."""
    try:
        plan = Plan.model_validate_json(_strip_fence(completion.text))
    except (ValidationError, ValueError) as exc:
        fallback = Plan(tool="search_corpus", query=question, reason="unparsable plan")
        return fallback, f"router output invalid: {exc}"
    if plan.tool == "search_corpus" and not plan.query:
        plan = plan.model_copy(update={"query": question})
    return plan, None


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    return cleaned.strip()


async def tool_executor(state: AgentState, deps: NodeDependencies) -> AgentState:
    """Catches tool failures, writes `error`, never raises."""
    started = time.monotonic()
    plan = state["plan"]
    assert plan is not None
    fingerprint = plan.fingerprint()
    if fingerprint in state["seen_call_hashes"]:
        state["error"] = "repeat tool call rejected"
        state["trace"].append(
            _step("tool_executor", started, tool=plan.tool, note="duplicate call")
        )
        return state
    state["seen_call_hashes"].add(fingerprint)
    return await _run_tool(state, deps, plan, started)


async def _run_tool(
    state: AgentState, deps: NodeDependencies, plan: Plan, started: float
) -> AgentState:
    try:
        note = await _dispatch(state, deps, plan)
    except Exception as exc:  # noqa: BLE001 a tool failure is data, not a crash
        note = f"{plan.tool} failed: {exc}"
        state["error"] = note
    state["trace"].append(
        _step(
            "tool_executor",
            started,
            tool=plan.tool,
            args={"query": plan.query, "source_id": plan.source_id},
            note=note,
        )
    )
    return state


async def _dispatch(state: AgentState, deps: NodeDependencies, plan: Plan) -> str:
    if plan.tool == "get_ingest_status":
        status = await deps.tools.get_ingest_status(
            IngestStatusInput(source_id=plan.source_id)
        )
        state["coverage_note"] = status.coverage_note
        state["confidence"] = "high"
        return f"{status.source_id}: {status.status}"
    request = SearchCorpusInput(
        query=plan.query or state["question"],
        source_id=plan.source_id,
        doc_type=plan.doc_type,
    )
    result = await deps.tools.search_corpus(request, state["tenant_id"])
    state["chunks"] = result.chunks  # replaced, never accumulated
    state["confidence"] = result.confidence
    return f"{result.k_used} chunks, confidence {result.confidence}"


async def responder(state: AgentState, deps: NodeDependencies) -> AgentState:
    """The only node that sees untrusted content."""
    started = time.monotonic()
    prompt = deps.prompts.get("rag_answer")
    nonce = new_nonce()
    rendered = render_context(state["chunks"], nonce)
    user = assemble(prompt.text, rendered, state["question"])
    outcome = await _answer_with_one_repair(deps, prompt.text, user, state)
    _apply(state, outcome)
    state["trace"].append(
        _step(
            "responder",
            started,
            model=deps.llm_settings.responder_model,
            prompt_version=prompt.identifier,
            note=f"stripped {len(rendered.stripped)} markers",
        )
    )
    return state


async def _answer_with_one_repair(
    deps: NodeDependencies, system: str, user: str, state: AgentState
) -> ValidationOutcome:
    """Native output, one repair carrying the specific error, then fallback.
    Never a loop, because a second repair is more likely to produce a confident
    wrong answer than a correct one."""
    retrieved = {chunk.chunk_id for chunk in state["chunks"]}
    model = deps.llm_settings.responder_model
    outcome = await _try_once(deps, system, user, retrieved, 0)
    if outcome.ok or outcome.error is None:
        return outcome
    repair = f"{user}\n\nYour previous answer was rejected: {outcome.error}. Fix it."
    log.info("repairing answer", model=model, reason=outcome.error)
    return await _try_once(deps, system, repair, retrieved, MAX_REPAIRS)


async def _try_once(
    deps: NodeDependencies, system: str, user: str, retrieved: set[str], repairs: int
) -> ValidationOutcome:
    try:
        completion = await deps.llm.complete(
            system, user, deps.llm_settings.responder_model
        )
    except LLMUnavailableError as exc:
        return ValidationOutcome(None, str(exc), _report(repairs))
    return validate(_strip_fence(completion.text), retrieved, repairs)


def _report(repairs: int) -> ValidationReport:
    return ValidationReport(repair_attempts=repairs)


def _apply(state: AgentState, outcome: ValidationOutcome) -> None:
    if outcome.payload is not None:
        state["answer"] = outcome.payload.answer
        state["citations"] = list(outcome.payload.citations)
        return
    summary = _summarise(state)
    payload = fallback_answer(summary, outcome.error or "validation failed twice")
    state["answer"] = _with_context_note(payload.answer, state)
    state["citations"] = []


def _summarise(state: AgentState) -> str:
    return "\n".join(
        f"- [{chunk.chunk_id}] {chunk.source_url}: {chunk.text[:160]}"
        for chunk in state["chunks"]
    )


def _with_context_note(answer: str, state: AgentState) -> str:
    """A tool failure or a coverage gap is stated, never silently dropped."""
    parts = [answer]
    if state.get("error"):
        parts.append(f"The corpus could not be reached: {state['error']}.")
    if state.get("coverage_note"):
        parts.append(str(state["coverage_note"]))
    return "\n\n".join(parts)


def answer_json(text: str) -> str:
    """Helper for scripted tests and the fallback template."""
    return json.dumps({"answer": text, "citations": [], "confidence": "insufficient"})
