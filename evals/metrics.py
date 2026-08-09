"""Retrieval metrics. Pure functions over ranked ids, so they are testable.

Kept out of `run_eval.py` on purpose: a metric that only runs inside a harness
is a metric nobody checks.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Share of gold chunks that appear in the top k."""
    if not gold:
        return 0.0
    hits = len(set(retrieved[:k]) & gold)
    return hits / len(gold)


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / index
    return 0.0


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    relevances = [1 if chunk_id in gold else 0 for chunk_id in retrieved[:k]]
    ideal = [1] * min(len(gold), k)
    best = dcg(ideal)
    return dcg(relevances) / best if best else 0.0


def unanswerable_accuracy(said_nothing: list[bool]) -> float:
    """Share of unanswerable questions that correctly returned no answer.

    The unanswerable slice is the only measure of the low confidence branch and
    the only false positive rate available for the score floor.
    """
    if not said_nothing:
        return 0.0
    return sum(1 for value in said_nothing if value) / len(said_nothing)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]
