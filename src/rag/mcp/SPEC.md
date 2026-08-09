# mcp

MCP server exposing the corpus and the ingestion subsystem as two tools. Runs as
its own process. The LangGraph agent connects as a client and discovers tools at
runtime.

## Transport

Streamable HTTP on `mcp_port` (default 8765). Not stdio, because the FastAPI
`/agent` endpoint would otherwise spawn a subprocess per request.

Built on the official `mcp` Python SDK using `FastMCP`.

## Tool 1: search_corpus

Hybrid retrieval over the indexed corpus. Returns chunks, never a generated
answer.

```python
class SearchCorpusInput(BaseModel):
    query: str = Field(description="Natural language search query")
    top_k: int = Field(8, ge=1, le=20,
                       description="Maximum number of chunks to return")
    doc_type: Literal["html", "pdf", "office"] | None = Field(
        None, description="Restrict to one document type")
    source_id: str | None = Field(
        None, description="Restrict to one registered source")
    date_from: date | None = Field(
        None, description="Only documents published on or after this date")

class SearchCorpusOutput(BaseModel):
    chunks: list[RetrievedChunk]
    confidence: Literal["high", "low", "none"]
    k_used: int
    reason: str | None
```

Field descriptions are read by the calling model. They are part of the
interface, not documentation. Write them for the agent, not for a human.

## Tool 2: get_ingest_status

Scrape and ingestion state for a source.

```python
class IngestStatusInput(BaseModel):
    source_id: str | None = Field(
        None, description="Registered source id. Omit for a corpus summary")
    domain: str | None = Field(
        None, description="Look up by domain instead of source id")

class IngestStatusOutput(BaseModel):
    source_id: str
    status: Literal["healthy", "degraded", "unreachable", "never_ingested"]
    circuit_state: Literal["closed", "open", "half_open"]
    last_success_at: datetime | None
    last_failure_reason: FailureReason | None
    docs_indexed: int
    docs_failed: int
    coverage_note: str
```

**Why this tool exists.** When `search_corpus` returns `confidence: "none"`,
there are two very different causes: the corpus genuinely does not cover the
topic, or the source that would cover it has been blocked since a given date.
The status tool turns "I do not know" into "that source has been unreachable
for six days, so the corpus does not cover it". It is what makes the low
confidence branch informative rather than a dead end.

`get_ingest_status` reads `source`, `source_state` and `dead_letter`, all
defined in `src/rag/fetch/SPEC.md`. `docs_failed` is a count over `dead_letter`
grouped by reason. This module never writes to any of them.

## Boundary scoping

What is deliberately not exposed:

- **Tenant.** `tenant_id` is injected server side from session context. It is
  not in any input schema. An agent cannot request another tenant's corpus.
- **Unbounded k.** `top_k` is capped at 20 in the schema.
- **Writes.** The surface is read only. No ingest triggers, no deletes, no
  config changes.
- **Arbitrary filters.** Only the enumerated fields above. No filter expression
  passthrough, no raw query language.
- **Call budget.** Per session cap enforced server side, independent of the
  agent's own loop limit. A runaway agent cannot exhaust the retrieval backend.

## Client

`src/rag/mcp/client.py` provides a standalone demonstration script that
connects, lists tools with their schemas, calls both, and prints the results.
This is a deliverable, not a test helper. It proves discovery works.

The agent wires MCP tools into LangGraph through `langchain-mcp-adapters`, which
converts discovered tools into callable LangGraph tools. Discovery is live, not
hardcoded.

## Tests

- Server starts and lists exactly two tools
- Each tool's advertised input schema matches the Pydantic model
- `search_corpus` with `top_k=50` is rejected by schema validation
- `search_corpus` output validates against `SearchCorpusOutput`
- `get_ingest_status` on an unknown source returns `never_ingested`, not an error
- `get_ingest_status` on a source with an open circuit reports `unreachable`
- Tenant is not settable from the client
- Session call budget rejects the call after the cap

## Known gaps

Authentication at the MCP boundary is a shared secret header, not OAuth. Fine
for a single trusted client on localhost. Multi client deployment would need
per client credentials and per client scoping.
