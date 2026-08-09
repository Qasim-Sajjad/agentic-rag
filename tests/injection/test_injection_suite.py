"""Injection defence suite. `pytest -k injection` must run in front of a reviewer.

What this measures, precisely: the two layers that do not depend on a model
behaving. The structural layer (nonce containers and delimiter stripping) and
the validation layer (citations resolved in code against the retrieved set).

What it does not measure: whether a given model follows an injected
instruction. That needs a live model, and the instructional layer is assumed
bypassable anyway, which is why the load bearing defences are the other two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.prompts.registry import get_registry
from rag.prompts.render import (
    FORGED_CLOSE_TAG,
    ROLE_MARKER,
    assemble,
    new_nonce,
    render_context,
    strip_delimiters,
)
from rag.prompts.validate import validate
from rag.retrieve.types import RetrievedChunk

pytestmark = pytest.mark.injection

CASES_FILE = Path(__file__).parent.parent.parent / "evals" / "injection" / "cases.jsonl"
CANARY = "CANARY_7F3A"


def load_cases() -> list[dict[str, object]]:
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CASES = load_cases()
CASE_IDS = [str(case["id"]) for case in CASES]


def chunk_for(case: dict[str, object], chunk_id: str = "c_8821") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=str(case["chunk_text"]),
        score=0.9,
        source_url="https://example.test/filing",
        section_path=["Annual report", "Risk factors"],
    )


def test_the_suite_has_fifteen_attacks_plus_a_benign_lookalike():
    attacks = [case for case in CASES if not case["benign"]]
    assert len(attacks) == 15 and len(CASES) == 16


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_a_chunk_cannot_close_its_own_container(case):
    """The nonce is the reason a forged `</doc>` cannot escape into
    instruction space, and stripping is the belt to that braces."""
    nonce = new_nonce()
    rendered = render_context([chunk_for(case)], nonce)
    body = rendered.text.split(f"</doc_{nonce}>")[0]
    assert body.count(f"<doc_{nonce}") == 1


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_context_closes_exactly_once_per_chunk(case):
    nonce = new_nonce()
    rendered = render_context([chunk_for(case)], nonce)
    assert rendered.text.count(f"</doc_{nonce}>") == 1


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["expect_stripped"]],
    ids=[str(c["id"]) for c in CASES if c["expect_stripped"]],
)
def test_forged_markers_are_stripped_and_reported(case):
    """A case where the model behaves but nothing was stripped is a silent
    failure of layer 2. The strip log is what makes it visible."""
    rendered = render_context([chunk_for(case)], new_nonce())
    patterns = {marker.pattern for marker in rendered.stripped}
    assert set(case["expect_stripped"]) <= patterns


def test_a_forged_close_tag_is_recorded_against_its_chunk():
    case = next(c for c in CASES if c["id"] == "forged_close_1")
    rendered = render_context([chunk_for(case, "c_1234")], new_nonce())
    markers = [m for m in rendered.stripped if m.pattern == FORGED_CLOSE_TAG]
    assert markers and markers[0].chunk_id == "c_1234"


def test_role_markers_are_removed_from_the_rendered_text():
    case = next(c for c in CASES if c["id"] == "forged_turn_1")
    rendered = render_context([chunk_for(case)], new_nonce())
    assert "Assistant:" not in rendered.text


def test_the_strip_log_never_echoes_attacker_text():
    """A field carrying attacker bytes into a response is a second surface."""
    rendered = render_context([chunk_for(CASES[2])], new_nonce())
    assert all(CANARY not in marker.pattern for marker in rendered.stripped)


def test_the_task_is_restated_after_the_context():
    """If the last thing in the window is attacker text, the attacker gets the
    strongest position in the prompt."""
    prompt = get_registry().get("rag_answer")
    nonce = new_nonce()
    rendered = render_context([chunk_for(CASES[0])], nonce)
    assembled = assemble(prompt.text, rendered, "What was revenue?")
    assert assembled.rindex("Question:") > assembled.rindex("</context>") - 1


def test_a_fabricated_citation_is_rejected_in_code():
    """The load bearing guarantee. It holds regardless of model behaviour."""
    raw = json.dumps(
        {
            "answer": "Revenue was 41.2 million [c_authoritative].",
            "citations": [
                {
                    "chunk_id": "c_authoritative",
                    "source_url": "https://evil.tld/official",
                }
            ],
            "confidence": "high",
        }
    )
    outcome = validate(raw, {"c_8821"})
    assert outcome.report.citations_rejected == 1


def test_a_rejected_citation_blocks_the_answer():
    raw = json.dumps(
        {
            "answer": "Revenue was 41.2 million.",
            "citations": [{"chunk_id": "c_fake", "source_url": "https://evil.tld"}],
            "confidence": "high",
        }
    )
    assert not validate(raw, {"c_8821"}).ok


def test_an_inline_marker_for_an_unretrieved_chunk_is_rejected():
    raw = json.dumps(
        {
            "answer": "Margin was 12 percent [c_fabricated_999].",
            "citations": [],
            "confidence": "high",
        }
    )
    assert not validate(raw, {"c_8821"}).ok


def test_a_real_citation_passes():
    raw = json.dumps(
        {
            "answer": "Revenue was 41.2 million [c_8821].",
            "citations": [
                {"chunk_id": "c_8821", "source_url": "https://example.test/filing"}
            ],
            "confidence": "high",
        }
    )
    assert validate(raw, {"c_8821"}).ok


def test_a_valid_answer_reports_zero_rejections():
    raw = json.dumps(
        {
            "answer": "Revenue was 41.2 million [c_8821].",
            "citations": [
                {"chunk_id": "c_8821", "source_url": "https://example.test/filing"}
            ],
            "confidence": "high",
        }
    )
    assert validate(raw, {"c_8821"}).report.citations_rejected == 0


def test_the_benign_lookalike_is_not_mangled():
    """The class most implementations fail. A legitimate article about prompt
    injection must survive rendering intact, or the defence has censored it."""
    case = next(c for c in CASES if c["benign"])
    rendered = render_context([chunk_for(case)], new_nonce())
    assert "ignore previous instructions" in rendered.text


def test_the_benign_lookalike_strips_nothing():
    case = next(c for c in CASES if c["benign"])
    assert render_context([chunk_for(case)], new_nonce()).stripped == []


def test_stripping_leaves_ordinary_prose_untouched():
    text = "Revenue rose nine percent. The board approved a buyback."
    cleaned, removed = strip_delimiters(text, new_nonce())
    assert cleaned == text and removed == []


def test_every_attack_case_carries_a_class():
    assert all(case["class"] for case in CASES)


def test_role_marker_pattern_is_the_class_name_not_the_match():
    case = next(c for c in CASES if c["id"] == "forged_inst_1")
    rendered = render_context([chunk_for(case)], new_nonce())
    assert ROLE_MARKER in {m.pattern for m in rendered.stripped}
