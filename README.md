# Agentic RAG over a bot-protected corpus

Search and RAG over a 100K document corpus, with a resilient scraping pipeline,
an MCP tool server, and a LangGraph multi-agent layer.

## Status

Specs first. Each module under `src/rag/` has a `SPEC.md` defining its contract.
Read `CLAUDE.md` before writing code.

## Setup

Needs Postgres. Qdrant runs in process by default, so no container is required.

```bash
uv venv --python 3.12
uv sync --extra fetch --extra extract --extra index --extra retrieve --group dev
uv run python -m playwright install chromium
uv run python -m camoufox fetch
```

Create the database and put the DSN in `.env`:

```bash
createdb agentic_rag
echo "RAG__POSTGRES__DSN=postgresql://USER:PASSWORD@127.0.0.1:5432/agentic_rag" > .env
uv run python -m rag.db.migrate
```

`config/settings.yaml` is optional. Without it the loader reads
`config/settings.example.yaml`, so a clean clone runs.

## Run the pipeline

```bash
uv run python -m rag.fetch.bootstrap
```

```bash
uv run python -m rag.demo ingest-snippet path/to/page.html --url https://demo.local/snippet-1 --content-type text/html
```

```bash
uv run python -m rag.demo ingest https://books.toscrape.com/ --source-id books-toscrape
```

```bash
uv run python -m rag.demo crawl books-toscrape --max-pages 300
```

```bash
uv run python -m rag.demo status
```

```bash
uv run python -m evals.run_eval --run-id local
```

`crawl` fetches, extracts, indexes and follows links within the source's own
domain until the page budget runs out. Adding a new domain stays a manual
decision in `config/sources.yaml`.

## Run the services

The MCP server and the API are separate processes on purpose. Start the MCP
server first if you want `/agent` to route through it.

```bash
uv run python -m rag.mcp.server
```

```bash
uv run python -m rag.mcp.client
```

```bash
uv run uvicorn rag.api.main:app --port 8000
```

`rag.mcp.client` is the discovery proof: it connects, lists both tools with
their JSON schemas, calls both and prints the results.

`/ask` and `/agent` need an Anthropic key. Put it in `.env`:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

Without a key both endpoints still answer, with the deterministic fallback
rather than a fabricated answer. That is the designed behaviour, not a crash.

## Endpoints

```bash
curl -X POST localhost:8000/search -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"query": "cybersecurity risk disclosure", "top_k": 5}'
```

```bash
curl -X POST localhost:8000/ask -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"question": "What cybersecurity incidents were disclosed in 2024?"}'
```

```bash
curl -X POST localhost:8000/ask -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"question": "What was revenue?", "explain": true}'
```

```bash
curl -X POST localhost:8000/agent -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"question": "Is the SEC filings source up to date?"}'
```

```bash
curl "localhost:8000/ingest/status?source_id=books-toscrape" -H "X-API-Key: dev-key"
```

The `explain` block needs `api.explain_enabled: true` in config as well as a
valid key. It returns the assembled context and the list of what the renderer
stripped, never the system prompt body.

## Tests

```bash
uv run pytest
```

```bash
uv run pytest -k injection
```

```bash
uv run pytest -m slow
```

`-k injection` is the defence suite: 15 attack cases plus a benign lookalike.
`-m slow` launches Chromium and Camoufox against the fixture server, which is
the fetch escalation demo.

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
