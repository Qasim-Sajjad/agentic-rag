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
uv run python -m rag.demo ingest https://www.sec.gov/some/filing.htm --source-id sec-edgar
```

```bash
uv run python -m rag.demo status
```

```bash
uv run python -m evals.run_eval --run-id local
```

Add `--fake-embedder` to any demo command to skip the 2.2 GB BGE-M3 download.
The vectors are then meaningless, which is fine for watching the pipeline run
and useless for judging retrieval quality.

Phases 6 to 9 (MCP, agent, prompts, API) are not built yet.

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
