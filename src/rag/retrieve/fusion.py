"""Reciprocal Rank Fusion.

No score normalization and no per query tuning. Weighted score fusion needs a
weight tuned per query class, and that work has not been done, so the honest
default is the one with no free parameters beyond k.
"""

from __future__ import annotations

from rag.retrieve.types import RetrievedChunk


def rrf_score(rank: int, k: int) -> float:
    """Rank is zero based, so the top result contributes 1/(k+1)."""
    return 1.0 / (k + rank + 1)


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Fuse ranked lists. A chunk in both lists beats one that tops only one.

    Dense fails on exact identifiers and rare tokens, lexical fails on
    paraphrase. Fusion is what makes a corpus full of jargon searchable by
    either kind of question.
    """
    scores: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + rrf_score(
                rank, k
            )
            seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        seen[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in ordered
    ]
