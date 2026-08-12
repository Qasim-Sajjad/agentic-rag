# Project conventions

Agentic RAG over a 100K document corpus scraped from bot-protected sources.
Read this before writing any code in this repo.

## Start here

If this is your first session on this repo, read in this order before writing
anything:

1. This file, all of it
2. `docs/BUILD_ORDER.md` for the phase sequence and checkpoints
3. `docs/ARCHITECTURE.md` for the system shape
4. The SPEC of the module you are about to build

Then state which phase you are on and what the checkpoint is, before writing
code. If the phase is unclear, ask rather than guessing.

## Session protocol

Every session:

1. Say which phase and which module, and read that module's SPEC
2. Build in the order the phase lists. Types and protocols first, orchestration
   last
3. Write the tests named in the SPEC. Failure paths, not just happy paths
4. Run `ruff check . && mypy src && pytest` before saying you are done
5. Append one entry to `docs/AI_USAGE.md`

Do not start the next phase until the current checkpoint passes.

## Non negotiable

These are enforced by ruff, mypy and pre-commit hooks. Failing them fails the
commit, so write to them the first time.

- Max 25 statements per function. Over that, split it
- Max complexity 8, max 5 arguments, max 8 branches
- Type annotations on every signature. mypy runs in strict mode
- No `print`, no bare `except`, no em dashes anywhere
- No inline prompt strings in `src/`. Prompts load from the registry

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
ui/            Streamlit demo front end, HTTP client only
```

`ui/` imports nothing from `rag` and never will. Qdrant runs in process and is
single writer, so the API server is the only process that may hold the
collection. A front end with pipeline code in it could not run alongside it.

Each `src/rag/*/SPEC.md` defines that module's contract. Read the SPEC before
implementing or changing a module. If the code needs to diverge from the SPEC,
update the SPEC in the same commit.

## Shortcuts

Stubs and shortcuts are fine. Record every one in `docs/DESIGN.md` under
"Known gaps" with a one line reason. An unrecorded shortcut is the problem,
not the shortcut.
