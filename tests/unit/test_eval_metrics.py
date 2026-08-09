"""Metric correctness on hand computed cases."""

from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    unanswerable_accuracy,
)
from evals.run_eval import GoldItem, append_row, load_goldset

GOLD = {"c1"}
RANKED = ["c9", "c1", "c4"]


def test_recall_at_1_misses_a_second_place_hit():
    assert recall_at_k(RANKED, GOLD, 1) == 0.0


def test_recall_at_5_finds_it():
    assert recall_at_k(RANKED, GOLD, 5) == 1.0


def test_recall_counts_the_share_of_gold_found():
    assert recall_at_k(["a", "b"], {"a", "z"}, 5) == 0.5


def test_reciprocal_rank_is_one_over_the_position():
    assert reciprocal_rank(RANKED, GOLD) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_is_found():
    assert reciprocal_rank(["x", "y"], GOLD) == 0.0


def test_ndcg_is_one_when_the_gold_chunk_is_first():
    assert ndcg_at_k(["c1", "c2"], GOLD, 10) == 1.0


def test_ndcg_falls_when_the_gold_chunk_is_lower():
    assert ndcg_at_k(RANKED, GOLD, 10) < 1.0


def test_unanswerable_accuracy_counts_correct_refusals():
    assert unanswerable_accuracy([True, True, False, True]) == 0.75


def test_percentile_picks_the_middle():
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0


def test_goldset_round_trips(tmp_path: Path):
    path = tmp_path / "v1.jsonl"
    row = {
        "qid": "q1",
        "question": "what was revenue",
        "gold_chunk_ids": ["c1"],
        "gold_doc_id": "d1",
        "content_type": "prose",
        "source_quality": "clean",
        "answerable": True,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_goldset(path) == [GoldItem(**row)]


def test_results_file_is_append_only(tmp_path: Path):
    """That file is the regression suite, so a run must never overwrite it."""
    path = tmp_path / "results.jsonl"
    append_row({"run_id": "a"}, path)
    append_row({"run_id": "b"}, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
