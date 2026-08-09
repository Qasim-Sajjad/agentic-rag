# Agentic RAG over a bot-protected corpus

Search and RAG over a 100K document corpus, with a resilient scraping pipeline,
an MCP tool server, and a LangGraph multi-agent layer.

## Status

Specs first. Each module under `src/rag/` has a `SPEC.md` defining its contract.
Read `CLAUDE.md` before writing code.

## Setup

TODO after implementation:

```bash
uv sync
docker compose up -d qdrant
cp config/settings.example.yaml config/settings.yaml
uv run python -m rag.index.bootstrap
```

## Run

```bash
uv run uvicorn rag.api.main:app --port 8000    # API
uv run python -m rag.mcp.server                # MCP server, port 8765
```

## Endpoints

Full request and response shapes with curl examples in `src/rag/api/SPEC.md`.

| Endpoint | Purpose |
|---|---|
| `POST /search` | Semantic search, no LLM |
| `POST /ask` | Full RAG, answer plus citations |
| `POST /agent` | LangGraph route, answer plus trace |
| `GET /ingest/status` | Scrape and ingestion state |

## Tests

```bash
pytest                       # correctness
pytest -k injection          # injection defence suite
pytest tests/integration     # fetch ladder against local fixture server
```

Benchmarks are notebooks under `notebooks/`, committed with outputs.

## Docs

- `docs/DESIGN.md` design decisions and known gaps
- `docs/ARCHITECTURE.md` diagram source, Mermaid
- `docs/AI_USAGE.md` AI tool usage log
