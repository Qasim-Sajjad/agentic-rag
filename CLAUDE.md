# Project conventions

Agentic RAG over a 100K document corpus scraped from bot-protected sources.
Read this before writing any code in this repo.

## Stack

- Python 3.11+, asyncio for all IO
- FastAPI, Pydantic v2, pydantic-settings for config
- Qdrant (vector store), BGE-M3 (embeddings), MiniLM cross-encoder (rerank)
- LangGraph (agents), official `mcp` SDK (tool server)
- pytest for correctness, Jupyter notebooks for benchmarks only

## Writing style in code and docs

- No em dashes anywhere. Use a comma, a colon, or a new sentence.
- No filler. No "leverage", "robust", "seamless", "comprehensive".
- Sentence case in headings and log messages.

## Code rules

- Type hints on every function signature and every dataclass field.
- Pydantic models at every module boundary. Plain dataclasses inside a module.
- Functions do one thing. If a function needs a section comment, split it.
- Target 40 lines per function. Over 60 means it should have been two.
- Comments explain why, not what. No comment blocks above obvious code.
- No inline prompt strings. Prompts live in `src/rag/prompts/` and load via the registry.
- No hardcoded model names, URLs, thresholds, or paths. Config only.
- No bare `except:`. Catch the specific exception or let it propagate.
- No `print`. Use `structlog` with bound context.
- Every module exposes a Protocol for anything with more than one implementation.

## Config

Single source: `config/settings.yaml` plus env overrides via pydantic-settings.
Anything a reviewer might want to change without editing code goes here:
model IDs, tier thresholds, backoff parameters, chunk sizes, k values.

## Errors

- Failures produce a typed reason code, never a bare string.
- Nothing fails silently. A swallowed exception is a bug.
- Every external call has a timeout. No exceptions.

## Tests

- Unit tests never touch the network. Use `tests/fixtures/server.py`.
- Integration tests run against the local fixture server only.
- Every failure path in a SPEC gets a test. Happy paths are the easy half.
- Notebooks import from `src/`. They never define pipeline logic.

## Layout

```
src/rag/
  fetch/       tiered fetching, retry, circuit breaker
  extract/     content routing, parsers, CanonicalDoc
  index/       dedup, chunking, embedding, upsert
  retrieve/    hybrid search, rerank, adaptive k
  mcp/         MCP server and tool schemas
  agent/       LangGraph graph, state, nodes
  api/         FastAPI endpoints
  prompts/     versioned prompt files plus registry
evals/         gold set, eval harness, results
tests/         unit, integration, fixtures
notebooks/     benchmark notebooks, no logic
docs/          design doc, architecture, AI usage log
```

Each `src/rag/*/SPEC.md` defines that module's contract. Read the SPEC before
implementing or changing a module. If the code needs to diverge from the SPEC,
update the SPEC in the same commit.

## Shortcuts

Stubs and shortcuts are fine. Record every one in `docs/DESIGN.md` under
"Known gaps" with a one line reason. An unrecorded shortcut is the problem,
not the shortcut.
