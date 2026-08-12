"""Adaptive k and confidence assignment.

Static k returns ten chunks whether the question is "who is the CEO" or
"compare the 2023 and 2024 risk disclosures". The retrieval signal decides
instead: cut at the score floor or at the first real gap, whichever comes first.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config.settings import RetrieveSettings
from rag.retrieve.types import Confidence, RetrievedChunk


@dataclass(frozen=True)
class Cut:
    chunks: list[RetrievedChunk]
    confidence: Confidence
    reason: str | None


def elbow_index(scores: list[float], delta: float) -> int | None:
    """First position where the score drops by more than `delta`."""
    for index in range(1, len(scores)):
        if scores[index - 1] - scores[index] > delta:
            return index
    return None


def floor_index(scores: list[float], floor: float) -> int | None:
    for index, score in enumerate(scores):
        if score < floor:
            return index
    return None


def confidence_for(
    top_score: float, settings: RetrieveSettings
) -> tuple[Confidence, str | None]:
    if top_score >= settings.score_floor:
        return "high", None
    if top_score >= settings.low_floor:
        return "low", "weak match"
    return "none", "no relevant documents"


def adaptive_cut(chunks: list[RetrievedChunk], settings: RetrieveSettings) -> Cut:
    """Cut the reranked list, then assign confidence from the top score.

    `confidence == "none"` still returns the candidates it cut to, and the cut
    lands on `k_min` because every score is under the floor. Returning an empty
    list instead made a failed retrieval indistinguishable from a broken one:
    nothing to inspect in `/search`, nothing in the `/ask` response, nothing in
    the agent trace. Not generating an answer from weak chunks is the caller's
    job, and `src/rag/api/ask.py` does it by checking `confidence`, not by
    checking whether the list happens to be empty.
    """
    if not chunks:
        return Cut([], "none", "no relevant documents")
    scores = [chunk.score for chunk in chunks]
    confidence, reason = confidence_for(scores[0], settings)
    return Cut(chunks[: _k(scores, settings)], confidence, reason)


def _k(scores: list[float], settings: RetrieveSettings) -> int:
    candidates = [
        index
        for index in (
            floor_index(scores, settings.score_floor),
            elbow_index(scores, settings.elbow_delta),
        )
        if index is not None
    ]
    cut = min(candidates) if candidates else len(scores)
    return max(settings.k_min, min(cut or settings.k_min, settings.k_max))
