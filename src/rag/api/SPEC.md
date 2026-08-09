# api

FastAPI surface. Four endpoints, deliberately separated by responsibility.

## Why four and not one

`/search` is pure retrieval with no LLM. `/ask` is single shot RAG. `/agent`
routes through LangGraph and MCP. They have different latency profiles,
different failure modes, and different costs. Collapsing them would hide which
layer failed, and would make it impossible to benchmark retrieval independently
of generation.

## Endpoints

### POST /search

Semantic search. No LLM in the path.

```
Request:  {query, top_k?, filters?}
Response: {chunks: [...], confidence, k_used, latency_ms}
```

```bash
curl -X POST localhost:8000/search \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "cybersecurity risk disclosure", "top_k": 5,
       "filters": {"doc_type": "pdf", "date_from": "2024-01-01"}}'
```

### POST /ask

Full RAG. Retrieval plus generation plus citations. No agent, no tool routing.

```
Request:  {question, filters?}
Response: {answer, citations: [...], confidence, chunks_used, latency_ms}
```

On `confidence == "none"`, returns a stated non answer with the retrieved
chunks, not a generated guess. HTTP 200 with `confidence: "insufficient"`, not
an error status. The request succeeded, the corpus did not cover it.

```bash
curl -X POST localhost:8000/ask \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What cybersecurity incidents were disclosed in 2024?"}'
```

### POST /agent

Routes through the LangGraph graph, which calls MCP tools.

```
Request:  {question}
Response: {answer, citations: [...], confidence,
           trace: [{node, tool, args, latency_ms, model, prompt_version}]}
```

The trace is a first class part of the response, not a debug flag. It is how a
reviewer sees which agent and tool steps were taken.

```bash
curl -X POST localhost:8000/agent \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Is the SEC filings source up to date?"}'
```

### GET /ingest/status

Scrape and ingestion state.

```
Query:    ?source_id=... or ?domain=... , omit both for a corpus summary
Response: {sources: [{source_id, status, circuit_state, last_success_at,
                      last_failure_reason, docs_indexed, docs_failed}],
           summary: {total_sources, healthy, degraded, unreachable}}
```

Includes sources that failed and sources currently being retried. Reads the same
store as the MCP `get_ingest_status` tool.

```bash
curl "localhost:8000/ingest/status?source_id=sec-edgar" -H "X-API-Key: $API_KEY"
```

## Auth

Static API key in an `X-API-Key` header, validated by a FastAPI dependency,
compared with `secrets.compare_digest`. Keys map to a `tenant_id` which is
injected into the request context and passed to the MCP boundary.

This is deliberately minimal. Production would use OAuth2 client credentials or
mTLS, with per key rate limits and rotation. Documented in `docs/DESIGN.md`
under known gaps, not silently omitted.

## Caching

Two layers, both behind an interface that allows Redis later.

- Query embedding cache, keyed by normalized query, TTL 1 hour
- Response cache on `/search` and `/ask`, keyed by hash of
  (query, filters, top_k, prompt_version, embed_model_version), TTL 5 minutes

`prompt_version` and `embed_model_version` are in the cache key on purpose. A
prompt or model change must not serve stale answers.

`/agent` is not cached. Tool state changes between calls, so a cached agent
response can be wrong in a way a cached search result cannot.

## Errors

Consistent shape, typed reason codes, never a stack trace.

```json
{"error": {"code": "retrieval_unavailable",
           "message": "Vector store did not respond",
           "request_id": "..."}}
```

| Situation | Status |
|---|---|
| Bad input | 422, FastAPI default |
| Missing or bad key | 401 |
| Nothing relevant retrieved | 200, `confidence: "insufficient"` |
| Vector store down | 503 |
| LLM provider error | 502 |
| MCP tool unreachable | 200 with the failure stated in the trace |

## Tests

- Each endpoint returns its documented schema
- `/ask` on an unanswerable question returns 200 with `insufficient`
- `/agent` trace contains every node that executed
- Missing API key returns 401
- Tenant from the key reaches the MCP boundary and cannot be overridden by the
  request body
- Cache key changes when `prompt_version` changes

## Known gaps

- Static API keys, no rotation, no per key rate limiting
- No streaming responses. `/ask` and `/agent` block until complete
- No pagination on `/search`
