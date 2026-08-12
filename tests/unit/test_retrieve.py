"""RRF ordering, adaptive k and confidence, all on fixed fixtures."""

from __future__ import annotations

import pytest

from rag.config.settings import RetrieveSettings
from rag.retrieve.adaptive import adaptive_cut, elbow_index
from rag.retrieve.fusion import reciprocal_rank_fusion, rrf_score
from rag.retrieve.rerank import _sigmoid
from rag.retrieve.types import RetrievedChunk

SETTINGS = RetrieveSettings()


def chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text {chunk_id}", score=score, source_url="u"
    )


def test_rrf_score_matches_the_hand_computed_value():
    assert rrf_score(0, 60) == 1 / 61


def test_a_chunk_in_both_rankings_beats_one_that_tops_only_one():
    """The exact case hybrid exists for: agreement outranks a single strong hit."""
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("d"), chunk("b"), chunk("e")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0].chunk_id == "b"


def test_fusion_ordering_matches_the_hand_computed_order():
    dense = [chunk("a"), chunk("b")]
    sparse = [chunk("b"), chunk("a")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert [c.chunk_id for c in fused] == ["a", "b"]


def test_fusion_deduplicates():
    fused = reciprocal_rank_fusion([[chunk("a")], [chunk("a")]], k=60)
    assert len(fused) == 1


def test_fusion_of_nothing_is_nothing():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_the_elbow_is_the_first_large_gap():
    assert elbow_index([0.9, 0.85, 0.4, 0.35], delta=0.15) == 2


def test_no_elbow_when_scores_decay_smoothly():
    assert elbow_index([0.9, 0.85, 0.8, 0.75], delta=0.15) is None


def test_a_narrow_question_returns_fewer_chunks_than_a_broad_one():
    narrow = [
        chunk("a", 0.9),
        chunk("b", 0.88),
        *[chunk(f"n{i}", 0.2) for i in range(8)],
    ]
    broad = [chunk(f"b{i}", 0.9 - i * 0.01) for i in range(10)]
    assert len(adaptive_cut(narrow, SETTINGS).chunks) < len(
        adaptive_cut(broad, SETTINGS).chunks
    )


def test_the_cut_never_goes_below_k_min():
    chunks = [chunk("a", 0.9), chunk("b", 0.1), chunk("c", 0.05), chunk("d", 0.04)]
    assert len(adaptive_cut(chunks, SETTINGS).chunks) >= SETTINGS.k_min


def test_the_cut_never_exceeds_k_max():
    chunks = [chunk(f"c{i}", 0.9) for i in range(40)]
    assert len(adaptive_cut(chunks, SETTINGS).chunks) <= SETTINGS.k_max


# Explicit thresholds, so these test the branch logic rather than whatever the
# configured floors happen to be this week.
BANDS = RetrieveSettings(score_floor=0.30, low_floor=0.15)


def test_a_strong_top_score_is_high_confidence():
    assert adaptive_cut([chunk("a", 0.8)], BANDS).confidence == "high"


def test_a_weak_top_score_is_low_confidence():
    assert adaptive_cut([chunk("a", 0.2)], BANDS).confidence == "low"


def test_a_top_score_under_the_low_floor_is_no_confidence():
    assert adaptive_cut([chunk("a", 0.05)], BANDS).confidence == "none"


def test_an_unanswerable_query_still_returns_the_candidates_it_rejected():
    """Not a bad guess, a visible refusal. The retrieval layer hands back what
    it scored so a failed lookup can be told apart from a broken one. Refusing
    to generate from it is `src/rag/api/ask.py`, which gates on confidence."""
    cut = adaptive_cut([chunk("a", 0.05), chunk("b", 0.04), chunk("c", 0.01)], BANDS)
    assert cut.confidence == "none"
    assert [c.chunk_id for c in cut.chunks] == ["a", "b", "c"]


def test_an_unanswerable_query_is_bounded_to_k_min():
    """Every score is under the floor, so the cut is the smallest it can be
    rather than the whole rerank pool."""
    chunks = [chunk(f"c{i}", 0.05) for i in range(20)]
    assert len(adaptive_cut(chunks, BANDS).chunks) == BANDS.k_min


def test_zero_results_is_still_an_empty_list():
    assert adaptive_cut([], BANDS).chunks == []


def test_no_results_is_no_confidence():
    assert adaptive_cut([], SETTINGS).confidence == "none"


def test_no_confidence_carries_a_reason():
    assert adaptive_cut([], SETTINGS).reason == "no relevant documents"


def test_the_reranker_squashes_logits_into_the_range_the_floors_use():
    """The cross encoder emits an unbounded logit. `score_floor` is a 0 to 1
    threshold, so comparing the two directly made the cut depend on phrasing.
    """
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(2.094) == pytest.approx(0.890, abs=0.001)
    assert _sigmoid(0.675) == pytest.approx(0.663, abs=0.001)
    assert _sigmoid(-9.033) == pytest.approx(0.0001, abs=0.0001)


def test_the_squash_is_monotonic_so_the_reranked_order_never_changes():
    logits = [-11.0, -9.033, -0.5, 0.0, 0.675, 2.094, 11.0]
    squashed = [_sigmoid(value) for value in logits]
    assert squashed == sorted(squashed)


def test_an_extreme_logit_saturates_rather_than_raising():
    """`exp` of a large positive raises OverflowError, which is why the
    implementation branches on the sign."""
    assert _sigmoid(800.0) == 1.0
    assert _sigmoid(-800.0) == pytest.approx(0.0, abs=1e-12)
