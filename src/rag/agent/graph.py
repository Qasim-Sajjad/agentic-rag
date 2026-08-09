"""Graph assembly and loop prevention.

The only loop is assess -> router -> tool_executor -> assess, and three things
bound it: the iteration cap in the conditional edge, call fingerprints that
reject an identical repeat before it reaches MCP, and LangGraph's recursion
limit as the backstop for anything unanticipated.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from rag.agent.assess import assess
from rag.agent.nodes import NodeDependencies, responder, router, tool_executor
from rag.agent.state import AgentAnswer, AgentState, initial_state
from rag.log import get_logger

log = get_logger(__name__)

NO_TOOL = "answer_directly"


def build_graph(deps: NodeDependencies) -> Any:
    from langgraph.graph import END, StateGraph

    # `partial` rather than a lambda: LangGraph checks whether a node is a
    # coroutine function, and a lambda returning a coroutine fails that check.
    graph = StateGraph(AgentState)
    graph.add_node("router", partial(router, deps=deps))
    graph.add_node("tool_executor", partial(tool_executor, deps=deps))
    graph.add_node("responder", partial(responder, deps=deps))
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _after_router,
        {"tool_executor": "tool_executor", "responder": "responder"},
    )
    graph.add_conditional_edges(
        "tool_executor",
        lambda state: _after_tools(state, deps),
        {"router": "router", "responder": "responder"},
    )
    graph.add_edge("responder", END)
    return graph.compile()


def _after_router(state: AgentState) -> str:
    plan = state.get("plan")
    if plan is None or plan.tool == NO_TOOL:
        return "responder"
    return "tool_executor"


def _after_tools(state: AgentState, deps: NodeDependencies) -> str:
    """Pure. The router counts the retry and broadens the plan, because a
    mutation made here would be discarded with the state copy."""
    return str(assess(state, deps.agent_settings.max_iterations))


class AgentRunner:
    """Wraps the compiled graph so callers deal in questions and answers."""

    def __init__(self, deps: NodeDependencies) -> None:
        self._deps = deps
        self._graph = build_graph(deps)

    async def run(self, question: str, tenant_id: str = "default") -> AgentAnswer:
        state = initial_state(question, tenant_id)
        final: AgentState = await self._graph.ainvoke(
            state, {"recursion_limit": self._deps.agent_settings.recursion_limit}
        )
        return AgentAnswer(
            answer=final.get("answer") or "No answer was produced.",
            citations=final.get("citations", []),
            confidence=final.get("confidence", "none"),
            trace=final.get("trace", []),
        )
