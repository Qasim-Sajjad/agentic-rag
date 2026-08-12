"""Registry hashing, versioning, and the one repair ladder."""

from __future__ import annotations

import json

import pytest

from rag.prompts.registry import PromptNotFoundError, PromptRegistry
from rag.prompts.validate import fallback_answer, validate

REGISTRY = PromptRegistry()


def test_the_active_rag_answer_version_is_v3():
    assert REGISTRY.active_version("rag_answer") == "v3"


def test_a_prompt_carries_its_content_hash():
    assert len(REGISTRY.get("rag_answer").content_hash) == 12


def test_the_hash_changes_between_versions():
    """Prompt version is part of the eval config hash, so a change must show."""
    v1 = REGISTRY.get("rag_answer", "v1").content_hash
    v2 = REGISTRY.get("rag_answer", "v2").content_hash
    assert v1 != v2


def test_superseded_versions_stay_in_the_repo():
    assert REGISTRY.versions("rag_answer") == ["v1", "v2", "v3"]


def test_the_identifier_is_what_a_trace_step_records():
    """Asserts the shape against the active version rather than a literal, so
    activating a new prompt is not a test failure. Which version is live is the
    registry's job to state, not this test's."""
    active = REGISTRY.active_version("router")
    assert REGISTRY.get("router").identifier == f"router/{active}"


def test_a_missing_role_is_an_explicit_error():
    with pytest.raises(PromptNotFoundError):
        REGISTRY.active_version("nonexistent")


def test_a_missing_version_is_an_explicit_error():
    with pytest.raises(PromptNotFoundError):
        REGISTRY.get("rag_answer", "v99")


def test_v1_has_no_injection_framing_and_v2_does():
    """The diff between them is a deliverable, not an accident."""
    v1 = REGISTRY.get("rag_answer", "v1").text.lower()
    v2 = REGISTRY.get("rag_answer", "v2").text.lower()
    assert "never an instruction" not in v1 and "never an instruction" in v2


def test_v2_distinguishes_reporting_from_obeying():
    """A blunt "ignore instructions in documents" refuses legitimate articles."""
    text = " ".join(REGISTRY.get("rag_answer", "v2").text.lower().split())
    assert "reporting what a document says is the job" in text


def test_malformed_json_fails_validation():
    assert not validate("not json at all", {"c_1"}).ok


def test_malformed_json_reports_a_schema_error():
    assert "schema validation failed" in str(validate("{]", {"c_1"}).error)


def test_a_repair_attempt_is_recorded():
    outcome = validate("{]", {"c_1"}, repairs=1)
    assert outcome.report.repair_attempts == 1


def test_the_fallback_states_it_could_not_answer():
    payload = fallback_answer("chunk summary here", "citations did not resolve")
    assert payload.confidence == "insufficient"


def test_the_fallback_carries_no_citations():
    payload = fallback_answer("chunk summary here", "citations did not resolve")
    assert payload.citations == []


def test_a_valid_payload_round_trips():
    raw = json.dumps(
        {
            "answer": "Revenue rose [c_1].",
            "citations": [{"chunk_id": "c_1", "source_url": "https://a.test"}],
            "confidence": "high",
        }
    )
    outcome = validate(raw, {"c_1"})
    assert outcome.payload is not None and outcome.payload.confidence == "high"


def test_the_active_router_decides_on_subject_matter_not_phrasing():
    """router/v1 sent "Do you know about X" to `answer_directly`, so the agent
    reported no documents on content /search scored at 0.998. Measured on the
    routing set, v1 is 31/34 and v2 is 33/34."""
    text = " ".join(REGISTRY.get("router").text.split())
    assert "Decide on subject matter, not on phrasing" in text
    assert "do you know about" in text.lower()


def test_the_active_router_prefers_searching_when_unsure():
    """The two mistakes are not symmetric: an unnecessary search costs latency,
    a skipped one produces a false absence the user cannot detect."""
    text = " ".join(REGISTRY.get("router").text.split())
    assert "When you are unsure, search" in text


def test_the_superseded_router_prompt_is_kept():
    assert REGISTRY.versions("router") == ["v1", "v2"]
