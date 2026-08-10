"""Tool routing set and tool selection accuracy.

    python -m evals.routing build
    python -m evals.routing score

30 questions labelled with the expected first tool. Ambiguous cases carry a set
of acceptable answers rather than one, because forcing a single label on a
genuinely ambiguous question measures the labeller, not the router.

The router sees the question only, so this measures exactly what it decides on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from rag.agent.llm import AnthropicClient, LLMClient, LLMUnavailableError
from rag.agent.nodes import parse_plan_text
from rag.config.settings import get_settings
from rag.db.pool import Database
from rag.log import configure_logging, get_logger
from rag.prompts.registry import PromptRegistry

log = get_logger("routing")

ROUTING_FILE = Path(__file__).parent / "goldset" / "tool_routing.jsonl"

# Questions about the crawler and the corpus itself, not about content.
STATUS_QUESTIONS = (
    "Is the {source} source up to date?",
    "Has the {source} crawl been failing recently?",
    "How many documents have we indexed from {source}?",
    "Is {source} currently reachable?",
    "When did we last successfully fetch anything from {source}?",
    "Are any of our sources blocked right now?",
    "How many documents failed to ingest?",
    "Which sources are unreachable?",
)

# No document lookup needed at all.
DIRECT_QUESTIONS = (
    "Hello, what can you help me with?",
    "What kind of questions can you answer?",
    "Can you search documents for me?",
    "What is 17 plus 25?",
    "Thanks, that is all for now.",
    "Who are you?",
    "Explain what you do in one sentence.",
)

# Content questions, templated from real document titles so they are answerable
# from the corpus rather than invented.
SEARCH_TEMPLATES = (
    "What does the corpus say about {title}?",
    "Summarise what we have on {title}.",
    "Find passages about {title}.",
)

# Genuinely ambiguous: could be answered from content or from crawler state.
AMBIGUOUS = (
    (
        "Do we have anything on quantum computing?",
        ["search_corpus", "get_ingest_status"],
    ),
    ("Why did that search return nothing?", ["get_ingest_status", "search_corpus"]),
)


@dataclass
class RoutingItem:
    qid: str
    question: str
    expected_tools: list[str] = field(default_factory=list)


def _rows(items: list[RoutingItem]) -> list[dict[str, object]]:
    return [
        {
            "qid": item.qid,
            "question": item.question,
            "expected_tools": item.expected_tools,
        }
        for item in items
    ]


async def titles(db: Database, limit: int) -> list[str]:
    rows = await db.fetch(
        "SELECT DISTINCT title FROM document WHERE title IS NOT NULL "
        "AND length(title) > 8 ORDER BY title LIMIT $1",
        limit,
    )
    return [str(row["title"])[:70] for row in rows]


async def build(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.postgres)
    await db.connect()
    try:
        found = await titles(db, 13)
        sources = await db.fetch("SELECT source_id FROM source ORDER BY source_id")
    finally:
        await db.close()
    names = [str(row["source_id"]) for row in sources] or ["books-toscrape"]
    items = _search_items(found) + _status_items(names) + _direct_items() + _ambiguous()
    _write(items, Path(args.out))
    log.info("routing set written", path=args.out, questions=len(items))
    return len(items)


def _search_items(found: list[str]) -> list[RoutingItem]:
    items: list[RoutingItem] = []
    for index, title in enumerate(found):
        template = SEARCH_TEMPLATES[index % len(SEARCH_TEMPLATES)]
        items.append(
            RoutingItem(
                f"r{index:03d}", template.format(title=title), ["search_corpus"]
            )
        )
    return items


def _status_items(names: list[str]) -> list[RoutingItem]:
    return [
        RoutingItem(
            f"r1{index:02d}",
            question.format(source=names[index % len(names)]),
            ["get_ingest_status"],
        )
        for index, question in enumerate(STATUS_QUESTIONS)
    ]


def _direct_items() -> list[RoutingItem]:
    return [
        RoutingItem(f"r2{index:02d}", question, ["answer_directly"])
        for index, question in enumerate(DIRECT_QUESTIONS)
    ]


def _ambiguous() -> list[RoutingItem]:
    return [
        RoutingItem(f"r3{index:02d}", question, list(accepted))
        for index, (question, accepted) in enumerate(AMBIGUOUS)
    ]


def _write(items: list[RoutingItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in _rows(items):
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_routing(path: Path = ROUTING_FILE) -> list[RoutingItem]:
    if not path.is_file():
        return []
    items: list[RoutingItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(RoutingItem(**json.loads(line)))
    return items


async def score(
    llm: LLMClient, prompts: PromptRegistry, items: list[RoutingItem], model: str
) -> dict[str, float]:
    """Tool selection accuracy. Runs the router prompt, nothing else."""
    if not items:
        return {"tool_selection_accuracy": 0.0, "routing_items": 0}
    system = prompts.get("router").text
    correct = 0
    for item in items:
        chosen = await _route(llm, system, item.question, model)
        if chosen in item.expected_tools:
            correct += 1
        else:
            log.info(
                "routing miss",
                qid=item.qid,
                chose=chosen,
                expected=item.expected_tools,
            )
    return {
        "tool_selection_accuracy": round(correct / len(items), 4),
        "routing_items": len(items),
    }


async def _route(llm: LLMClient, system: str, question: str, model: str) -> str:
    try:
        completion = await llm.complete(system, question, model)
    except LLMUnavailableError:
        return "unavailable"
    plan, _ = parse_plan_text(completion.text, question)
    return str(plan.tool)


async def _score_cli(args: argparse.Namespace) -> None:
    settings = get_settings()
    metrics = await score(
        AnthropicClient(settings.llm),
        PromptRegistry(),
        load_routing(Path(args.out)),
        settings.llm.router_model,
    )
    log.info("routing scored", **metrics)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evals.routing")
    root.add_argument("--out", default=str(ROUTING_FILE))
    subs = root.add_subparsers(dest="command", required=True)
    subs.add_parser("build").set_defaults(run=build)
    subs.add_parser("score").set_defaults(run=_score_cli)
    return root


def main() -> None:
    configure_logging()
    args = parser().parse_args()
    asyncio.run(args.run(args))


if __name__ == "__main__":
    main()
