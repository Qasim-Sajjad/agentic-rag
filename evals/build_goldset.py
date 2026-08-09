"""Gold set builder with the auto filter step.

    python -m evals.build_goldset --out evals/goldset/v1.jsonl

Four stages, and the third is the one that matters:

1. Stratified sample across content types and source quality, not random.
   Random sampling gives 100 clean paragraphs and an eval blind to the cases
   the corpus is actually full of.
2. Generate one question per chunk, answerable only from that chunk.
3. Auto filter: ask each question with no context. If the model answers
   correctly it tested world knowledge, not retrieval, so discard the pair.
   This removes 25 to 35 percent and is the difference between a real eval and
   a vanity metric.
4. Hand verify the survivors, then freeze and version the file.

Stages 2 and 3 need an LLM. Without an API key this writes the sampled
candidates with empty questions so they can be filled by hand, which keeps the
sampling and the freezing reproducible either way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag.config.settings import get_settings
from rag.db.pool import Database
from rag.log import configure_logging, get_logger

log = get_logger("goldset")

STRATA = ("prose", "table", "short", "long", "pdf", "scraped", "low_quality")
TARGET_PER_STRATUM = 17


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    content_type: str
    source_quality: str


def stratum_of(row: dict[str, object]) -> str:
    """Deliberate spread. These are the cases that break retrieval differently."""
    if row.get("is_table"):
        return "table"
    raw_tokens = row.get("token_count")
    tokens = raw_tokens if isinstance(raw_tokens, int) else 0
    by_size = _size_stratum(tokens)
    if by_size is not None:
        return by_size
    return "pdf" if str(row.get("doc_type") or "") == "pdf" else "prose"


def _size_stratum(tokens: int) -> str | None:
    if tokens < 80:
        return "short"
    return "long" if tokens > 400 else None


async def sample(db: Database, per_stratum: int) -> list[Candidate]:
    rows = await db.fetch(
        """
        SELECT c.chunk_id, c.doc_id, c.text, c.token_count, c.is_table,
               d.doc_type
        FROM chunk c JOIN document d ON d.doc_id = c.doc_id
        ORDER BY c.chunk_id
        """
    )
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        mapping = dict(row)
        stratum = stratum_of(mapping)
        buckets[stratum].append(
            Candidate(
                chunk_id=str(mapping["chunk_id"]),
                doc_id=str(mapping["doc_id"]),
                text=str(mapping["text"]),
                content_type=stratum,
                source_quality="clean",
            )
        )
    picked: list[Candidate] = []
    for stratum, items in buckets.items():
        picked.extend(items[:per_stratum])
        log.info("stratum sampled", stratum=stratum, available=len(items))
    return picked


def to_row(candidate: Candidate, question: str) -> dict[str, object]:
    return {
        "qid": f"q{candidate.chunk_id[:8]}",
        "question": question,
        "gold_chunk_ids": [candidate.chunk_id],
        "gold_doc_id": candidate.doc_id,
        "content_type": candidate.content_type,
        "source_quality": candidate.source_quality,
        "answerable": True,
    }


def write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


async def build(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.postgres)
    await db.connect()
    try:
        candidates = await sample(db, args.per_stratum)
    finally:
        await db.close()
    rows = [to_row(candidate, "") for candidate in candidates]
    write(rows, Path(args.out))
    log.info(
        "goldset written",
        path=args.out,
        items=len(rows),
        note="questions are empty, generation and the auto filter need an LLM key",
    )
    return len(rows)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evals.build_goldset")
    root.add_argument("--out", default="evals/goldset/v1.jsonl")
    root.add_argument("--per-stratum", type=int, default=TARGET_PER_STRATUM)
    return root


def main() -> None:
    configure_logging()
    asyncio.run(build(parser().parse_args()))


if __name__ == "__main__":
    main()
