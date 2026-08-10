"""Question generation and the auto filter, the two LLM stages of the gold set.

    python -m evals.generate --out evals/goldset/v1.jsonl --per-stratum 12

Stage 3 is the one that matters. Every generated question is asked again with
no context at all, and any question the model answers correctly is discarded:
it tested world knowledge rather than retrieval. That step removes 25 to 35
percent of a naive set and is the difference between a real eval and a vanity
metric.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from evals.build_goldset import Candidate, sample, write
from rag.agent.llm import AnthropicClient, LLMClient, LLMUnavailableError
from rag.config.settings import get_settings
from rag.db.pool import Database
from rag.log import configure_logging, get_logger

log = get_logger("goldset")

GENERATE_SYSTEM = (
    "You write evaluation questions for a retrieval system. Given one passage, "
    "write a single specific question that the passage answers and that cannot "
    "be answered without it. Use concrete details from the passage: names, "
    "figures, dates. Return only the question, no preamble."
)

CLOSED_BOOK_SYSTEM = (
    "Answer from your own knowledge only. If you do not know the answer with "
    "confidence, reply with exactly: UNKNOWN. Keep any answer under 30 words."
)

JUDGE_SYSTEM = (
    "You compare a candidate answer against a source passage. Reply with "
    "exactly YES if the candidate states the same specific fact the passage "
    "does, otherwise reply with exactly NO."
)

UNKNOWN = "UNKNOWN"
UNANSWERABLE = [
    "What was the chief executive's total compensation last year?",
    "Which manufacturing plants were closed during the restructuring?",
    "How many employees left after the merger was announced?",
    "What penalty did the regulator impose in the settlement?",
    "Which competitor filed the patent infringement claim?",
]


@dataclass
class GeneratedItem:
    candidate: Candidate
    question: str
    kept: bool
    reason: str


async def generate_question(llm: LLMClient, candidate: Candidate, model: str) -> str:
    completion = await llm.complete(GENERATE_SYSTEM, candidate.text[:4000], model)
    return completion.text.strip().strip('"')


async def answers_without_context(
    llm: LLMClient, item: GeneratedItem, model: str
) -> bool:
    """The auto filter. True means the question tested world knowledge."""
    closed = await llm.complete(CLOSED_BOOK_SYSTEM, item.question, model)
    guess = closed.text.strip()
    if guess.upper().startswith(UNKNOWN):
        return False
    verdict = await llm.complete(
        JUDGE_SYSTEM,
        f"Passage:\n{item.candidate.text[:3000]}\n\nCandidate answer:\n{guess}",
        model,
    )
    return verdict.text.strip().upper().startswith("YES")


async def build_item(llm: LLMClient, candidate: Candidate, model: str) -> GeneratedItem:
    question = await generate_question(llm, candidate, model)
    item = GeneratedItem(candidate, question, True, "generated")
    if await answers_without_context(llm, item, model):
        return GeneratedItem(candidate, question, False, "answerable without context")
    return item


def to_row(item: GeneratedItem) -> dict[str, object]:
    return {
        "qid": f"q{item.candidate.chunk_id[:8]}",
        "question": item.question,
        "gold_chunk_ids": [item.candidate.chunk_id],
        "gold_doc_id": item.candidate.doc_id,
        "content_type": item.candidate.content_type,
        "source_quality": item.candidate.source_quality,
        "answerable": True,
    }


def unanswerable_rows(count: int) -> list[dict[str, object]]:
    """Plausible, on topic, and nothing in the corpus answers them.

    The only way to measure the low confidence branch, and the only false
    positive rate available for the reranker score floor.
    """
    return [
        {
            "qid": f"q9{index:02d}",
            "question": question,
            "gold_chunk_ids": [],
            "gold_doc_id": None,
            "content_type": "prose",
            "source_quality": "clean",
            "answerable": False,
        }
        for index, question in enumerate(UNANSWERABLE[:count])
    ]


async def build(args: argparse.Namespace) -> int:
    settings = get_settings()
    llm = AnthropicClient(settings.llm)
    db = Database(settings.postgres)
    await db.connect()
    try:
        candidates = await sample(db, args.per_stratum)
    finally:
        await db.close()
    items = await _generate_all(llm, candidates, settings.llm.judge_model)
    kept = [item for item in items if item.kept]
    rows = [to_row(item) for item in kept] + unanswerable_rows(args.unanswerable)
    write(rows, Path(args.out))
    _report(items, kept, rows, args.out)
    return len(rows)


async def _generate_all(
    llm: LLMClient, candidates: list[Candidate], model: str
) -> list[GeneratedItem]:
    items: list[GeneratedItem] = []
    for candidate in candidates:
        try:
            items.append(await build_item(llm, candidate, model))
        except LLMUnavailableError as exc:
            log.warning(
                "generation failed", chunk_id=candidate.chunk_id, error=str(exc)
            )
    return items


def _report(
    items: list[GeneratedItem],
    kept: list[GeneratedItem],
    rows: list[dict[str, object]],
    out: str,
) -> None:
    discarded = len(items) - len(kept)
    share = discarded / len(items) if items else 0.0
    log.info(
        "goldset written",
        path=out,
        generated=len(items),
        discarded_by_filter=discarded,
        filter_share=round(share, 3),
        answerable=len(kept),
        total=len(rows),
        note="hand verify the survivors before freezing",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evals.generate")
    root.add_argument("--out", default="evals/goldset/v1.jsonl")
    root.add_argument("--per-stratum", type=int, default=12)
    root.add_argument("--unanswerable", type=int, default=5)
    return root


def main() -> None:
    configure_logging()
    asyncio.run(build(parser().parse_args()))


if __name__ == "__main__":
    main()
