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

`/ask` and `/agent` need an Anthropic key. So does OCR, which uses the same key.
Put it in `.env`:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

Tier 4, the paid unlocker, needs a ScrapingBee key in the same file:

```bash
echo "SCRAPINGBEE_API_KEY=..." >> .env
```

Neither key belongs in `config/settings.yaml`, which is why neither is read from
there. Without the ScrapingBee key tier 4 refuses before making a request, and
the ladder stops at tier 3 exactly as it did before.

Without a key both endpoints still answer, with the deterministic fallback
rather than a fabricated answer. That is the designed behaviour, not a crash.

## Run the demo front end

A Streamlit page over the same endpoints. Start the API first, then:

```bash
uv run streamlit run ui/app.py
```

Four tabs: run a URL or a file through the pipeline and watch every stage, the
agent with its full path through the graph, pure retrieval, and the corpus state.

`/ask` is deliberately not a tab. Next to Search and Agent it reads as a third
answer source, when what it actually is is the benchmark baseline: retrieval
plus generation with no routing, so a bad answer is attributable to one of two
layers instead of four. It stays available over HTTP, and it is also the only
endpoint that returns the `explain` block with the strip log, so it is the one
to call when demonstrating injection defence.

`ui/app.py` imports nothing from `rag`. It is an HTTP client, and that is a
requirement rather than a style choice: `qdrant.path` runs Qdrant in process,
an in process collection is single writer, and the API server holds it. Two
processes touching the pipeline means a lock error instead of a document.

The same rule applies to everything else. While the API is up, do not run a
crawl, `python -m rag.demo`, or `pytest`. Stop the server first.

Point it somewhere else with `RAG_API_BASE` and `RAG_API_KEY`, or type both into
the sidebar.

**First ingest after a restart is slow.** BGE-M3 loads on the first embed, which
is roughly 70 seconds on CPU. Every ingest after that is fast. Warm it with one
throwaway `/search` call before a demo.

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

```bash
curl -X POST localhost:8000/ingest/url -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"url": "https://quotes.toscrape.com/page/9/"}'
```

```bash
curl -X POST localhost:8000/ingest/file -H "X-API-Key: dev-key" -F "file=@report.pdf"
```

The `explain` block needs `api.explain_enabled: true` in config as well as a
valid key. It returns the assembled context and the list of what the renderer
stripped, never the system prompt body.

Full request and response shapes in `src/rag/api/SPEC.md`.

| Endpoint | Purpose |
|---|---|
| `POST /search` | Semantic search, no LLM |
| `POST /ask` | Full RAG, answer plus citations |
| `POST /agent` | LangGraph route, answer plus trace |
| `GET /ingest/status` | Scrape and ingestion state |
| `POST /ingest/url` | One URL through the pipeline, stage by stage |
| `POST /ingest/file` | The same, from an uploaded file |

Both ingest endpoints return HTTP 200 when a fetch is blocked or a document is a
duplicate. The request succeeded, and what the site or the corpus decided is
named in the trace.

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
the fetch escalation demo. Stop the API server before running any of these: the
suite opens the same in process Qdrant collection.

## Docs

- `docs/DESIGN.md` design decisions and known gaps
- `docs/ARCHITECTURE.md` diagram source, Mermaid
- `docs/AI_USAGE.md` AI tool usage log
