"""Eval harness. Takes a config, returns metrics, appends one row to results.

    python -m evals.run_eval --goldset evals/goldset/v1.jsonl

Built before any tuning. Tuning without a harness is guessing, and a number
you cannot reproduce is not a result.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    unanswerable_accuracy,
)
from rag.config.settings import Settings, get_settings
from rag.index.embed import BGEM3Embedder, Embedder, FakeEmbedder
from rag.index.store import QdrantStore
from rag.log import configure_logging, get_logger
from rag.retrieve.rerank import IdentityReranker, MiniLMReranker
from rag.retrieve.service import RetrieveDependencies, SearchService
from rag.retrieve.types import Reranker

log = get_logger("eval")

RESULTS_FILE = Path(__file__).parent / "results.jsonl"
DEFAULT_GOLDSET = Path(__file__).parent / "goldset" / "v1.jsonl"
K_VALUES = (1, 5, 10)


@dataclass(frozen=True)
class GoldItem:
    qid: str
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_doc_id: str | None = None
    content_type: str = "prose"
    source_quality: str = "clean"
    answerable: bool = True


def load_goldset(path: Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldItem(**json.loads(line)))
    return items


def config_hash(settings: Settings, embed_model: str, reranker: str) -> str:
    """Identity of a run. Any change to chunking, model, retrieval params or
    prompt version must produce a new hash, or results become incomparable."""
    payload = {
        "chunker": settings.index.model_dump(),
        "retrieve": settings.retrieve.model_dump(),
        "embed_model": embed_model,
        "reranker": reranker,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


async def evaluate(service: SearchService, items: list[GoldItem]) -> dict[str, Any]:
    answerable = [item for item in items if item.answerable]
    unanswerable = [item for item in items if not item.answerable]
    scores: dict[str, list[float]] = {f"recall@{k}": [] for k in K_VALUES}
    scores["mrr"] = []
    scores["ndcg@10"] = []
    latencies: list[float] = []
    for item in answerable:
        result = await service.search(item.question)
        ranked = [chunk.chunk_id for chunk in result.chunks]
        gold = set(item.gold_chunk_ids)
        latencies.append(result.latency_ms)
        for k in K_VALUES:
            scores[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        scores["mrr"].append(reciprocal_rank(ranked, gold))
        scores["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
    refusals = [await _refused(service, item) for item in unanswerable]
    metrics = {name: _mean(values) for name, values in scores.items()}
    metrics["unanswerable_accuracy"] = unanswerable_accuracy(refusals)
    metrics["p50_latency_ms"] = percentile(latencies, 0.5)
    metrics["items"] = len(items)
    return metrics


async def _refused(service: SearchService, item: GoldItem) -> bool:
    result = await service.search(item.question)
    return result.confidence == "none"


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_service(
    settings: Settings, fake: bool
) -> tuple[SearchService, Embedder, Reranker]:
    embedder: Embedder = (
        FakeEmbedder()
        if fake
        else BGEM3Embedder(settings.index, settings.index.embed_model)
    )
    reranker: Reranker = (
        IdentityReranker() if fake else MiniLMReranker(settings.retrieve)
    )
    store = QdrantStore(settings.qdrant)
    service = SearchService(
        RetrieveDependencies(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=settings.retrieve,
            qdrant=settings.qdrant,
        )
    )
    return service, embedder, reranker


def append_row(row: dict[str, Any], path: Path = RESULTS_FILE) -> None:
    """Append only. This file is the regression suite, not a scratch pad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    items = load_goldset(Path(args.goldset))
    service, embedder, reranker = build_service(settings, args.fake_models)
    metrics = await evaluate(service, items)
    row = {
        "run_id": args.run_id,
        "config_hash": config_hash(settings, embedder.model_name, reranker.name),
        "goldset_version": Path(args.goldset).stem,
        "chunker": settings.index.model_dump()["target_tokens"],
        "overlap": settings.index.overlap_ratio,
        "embed_model": embedder.model_name,
        "reranker": reranker.name,
        **metrics,
    }
    append_row(row, Path(args.results))
    log.info("eval complete", **{k: v for k, v in row.items() if k != "run_id"})
    return row


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evals.run_eval")
    root.add_argument("--goldset", default=str(DEFAULT_GOLDSET))
    root.add_argument("--results", default=str(RESULTS_FILE))
    root.add_argument("--run-id", default="manual")
    root.add_argument(
        "--fake-models",
        action="store_true",
        help="hashing embedder and no reranker, for harness smoke tests",
    )
    return root


def main() -> None:
    configure_logging()
    asyncio.run(run(parser().parse_args()))


if __name__ == "__main__":
    main()


__all__ = ["GoldItem", "append_row", "config_hash", "evaluate", "load_goldset", "run"]
