"""The branching decision, deterministic and with no LLM in it.

Keeping this out of the model means the branching logic is unit testable across
every path and cannot be talked out of a decision by anything it reads.
"""

from __future__ import annotations

from enum import StrEnum

from rag.agent.state import AgentState, Plan


class Edge(StrEnum):
    RESPOND = "responder"
    RETRY = "router"


def assess(state: AgentState, max_iterations: int) -> Edge:
    """Reads confidence, error and iteration. Returns the next edge name.

    An error goes to the responder rather than a retry: retrying a tool that
    just failed is how a loop starts, and the user is better served by a stated
    failure than by a second identical timeout.
    """
    if state.get("error"):
        return Edge.RESPOND
    if state.get("confidence") == "high":
        return Edge.RESPOND
    if state.get("iteration", 0) < max_iterations:
        return Edge.RETRY
    return Edge.RESPOND


def broaden_plan(plan: Plan) -> Plan:
    """The retry rule: drop filters rather than reason about why it failed.

    A richer planner would ask why retrieval came back empty. This is a fixed
    rule, recorded as a known gap rather than dressed up as a decision.
    """
    return plan.model_copy(update={"source_id": None, "doc_type": None})


def is_retry(state: AgentState) -> bool:
    """A second visit to the router. Counted there rather than in the edge,
    because LangGraph hands conditional edges a copy of the state and a
    mutation made in one would be silently discarded."""
    return state.get("plan") is not None
