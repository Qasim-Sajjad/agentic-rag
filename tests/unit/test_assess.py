"""`assess` as a pure function, across every branch. No graph, no LLM."""

from __future__ import annotations

from rag.agent.assess import Edge, assess, broaden_plan, is_retry
from rag.agent.state import Plan, initial_state

MAX_ITERATIONS = 1


def state(**overrides):
    base = initial_state("what was revenue")
    base.update(overrides)
    return base


def test_high_confidence_goes_to_the_responder():
    assert assess(state(confidence="high"), MAX_ITERATIONS) is Edge.RESPOND


def test_low_confidence_on_the_first_pass_retries():
    assert assess(state(confidence="low", iteration=0), MAX_ITERATIONS) is Edge.RETRY


def test_low_confidence_after_the_cap_stops_retrying():
    assert assess(state(confidence="low", iteration=1), MAX_ITERATIONS) is Edge.RESPOND


def test_no_confidence_on_the_first_pass_retries():
    assert assess(state(confidence="none", iteration=0), MAX_ITERATIONS) is Edge.RETRY


def test_an_error_goes_straight_to_the_responder():
    """Retrying a tool that just failed is how a loop starts."""
    branch = assess(state(error="tool timeout", confidence="none"), MAX_ITERATIONS)
    assert branch is Edge.RESPOND


def test_an_error_wins_over_a_retry_that_would_otherwise_fire():
    branch = assess(
        state(error="tool timeout", confidence="low", iteration=0), MAX_ITERATIONS
    )
    assert branch is Edge.RESPOND


def test_zero_iterations_allowed_never_retries():
    assert assess(state(confidence="low", iteration=0), 0) is Edge.RESPOND


def test_broadening_drops_the_source_filter():
    plan = Plan(tool="search_corpus", query="q", source_id="sec-edgar")
    assert broaden_plan(plan).source_id is None


def test_broadening_drops_the_doc_type_filter():
    plan = Plan(tool="search_corpus", query="q", doc_type="pdf")
    assert broaden_plan(plan).doc_type is None


def test_broadening_keeps_the_query():
    plan = Plan(tool="search_corpus", query="revenue")
    assert broaden_plan(plan).query == "revenue"


def test_a_first_visit_to_the_router_is_not_a_retry():
    assert not is_retry(state())


def test_a_second_visit_to_the_router_is_a_retry():
    assert is_retry(state(plan=Plan(tool="search_corpus", query="q")))


def test_identical_plans_share_a_fingerprint():
    left = Plan(tool="search_corpus", query="revenue", source_id="a")
    right = Plan(tool="search_corpus", query="revenue", source_id="a")
    assert left.fingerprint() == right.fingerprint()


def test_a_different_filter_changes_the_fingerprint():
    left = Plan(tool="search_corpus", query="revenue", source_id="a")
    right = Plan(tool="search_corpus", query="revenue", source_id="b")
    assert left.fingerprint() != right.fingerprint()


def test_a_different_tool_changes_the_fingerprint():
    left = Plan(tool="search_corpus", query="revenue")
    right = Plan(tool="get_ingest_status", query="revenue")
    assert left.fingerprint() != right.fingerprint()
