# api

FastAPI surface. Six endpoints, deliberately separated by responsibility: four
that read, two that write.

## Why separated and not one

`/search` is pure retrieval with no LLM. `/ask` is single shot RAG. `/agent`
routes through LangGraph and MCP. They have different latency profiles,
different failure modes, and different costs. Collapsing them would hide which
layer failed, and would make it impossible to benchmark retrieval independently
of generation.

## Why the write endpoints live here at all

The original four were read only, and ingestion was the CLI in `src/rag/demo.py`
and `python -m rag.crawl`. That stops working the moment anything wants to watch
an ingest while the API is up, for one concrete reason: `qdrant.path` runs Qdrant
in process, and an in process collection is single writer. Whichever process
opens it holds it. A second process, a Streamlit front end or a second CLI, gets
a lock error rather than a document.

So `/ingest/url` and `/ingest/file` are not convenience wrappers. They are the
only place the write path can run without stopping the server, which is what
makes `ui/` possible as a pure HTTP client with no pipeline code in it.

The CLI still exists and still works. It is the right tool when the API is down.

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
Request:  {question, filters?, explain?}
Response: {answer, citations: [...], confidence, chunks: [...],
           validation: {...}, explain?: {...}, latency_ms}
```

`chunks` is the full `RetrievedChunk` objects from `src/rag/retrieve/SPEC.md`,
the same shape `/search` returns, not a count. Retrieval that cannot be
inspected cannot be debugged or demonstrated, and a bare number is the one form
that tells you nothing.

`validation` is the `ValidationReport` from `src/rag/prompts/SPEC.md`, present
on every response:

```json
"validation": {"citations_checked": 3, "citations_rejected": 0,
               "repair_attempts": 0, "fell_back": false}
```

It reports whether the citation check fired, which is the system's actual
grounding guarantee. A poisoned citation shows up as `citations_rejected: 1`
rather than as an answer that merely looks fine.

On `confidence == "none"`, returns a stated non answer with the retrieved
chunks, not a generated guess. HTTP 200 with `confidence: "insufficient"`, not
an error status. The request succeeded, the corpus did not cover it.

```bash
curl -X POST localhost:8000/ask \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What cybersecurity incidents were disclosed in 2024?"}'
```

#### The explain block

`explain: true` in the request body adds the assembled context and the strip
log to the response. It is a debug affordance, not a production feature.

```json
"explain": {
  "nonce": "a7f3c1",
  "prompt_version": "rag_answer/v2",
  "task_position": "after_context",
  "rendered_context": "<doc_a7f3c1 id=\"c_8821\" url=\"...\">...</doc_a7f3c1>",
  "stripped": [{"chunk_id": "c_8821", "pattern": "role_marker", "count": 2}]
}
```

`stripped` is the part that matters. It shows the structural defence acting on
this specific request, so injection resistance is something a reviewer observes
rather than something the docs assert.

Three constraints, all load bearing:

- Gated by config `api.explain_enabled`, default false, in addition to the API
  key. A valid key is not authorisation to read prompt internals
- Never returns the system prompt body. `prompt_version` identifies it. An
  endpoint that echoes the prompt is a prompt exfiltration endpoint
- `explain` is part of the response cache key, and a request with
  `explain: true` bypasses the cache on read. Otherwise a cached answer from an
  earlier call returns without the block, or with another request's nonce

A body field rather than a query parameter, because every other input to this
endpoint is in the body and splitting them means two places to validate.

### POST /agent

Routes through the LangGraph graph, which calls MCP tools.

```
Request:  {question}
Response: {answer, citations: [...], confidence, chunks: [...],
           trace: [{node, tool, args, latency_ms, model, prompt_version, note}]}
```

The trace is a first class part of the response, not a debug flag. It is how a
reviewer sees which agent and tool steps were taken. `model` and
`prompt_version` are per step, because the router and the responder are
different models on different prompts and reporting one number for the request
would hide that.

`chunks` is the context the responder saw, the same `RetrievedChunk` shape
`/search` and `/ask` return. On a retry it is the replacement set rather than
both sets, so it is the evidence the answer was actually written from. It was
missing at first, which made the agent the one endpoint whose answer could not
be checked against its own evidence.

```bash
curl -X POST localhost:8000/agent \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Is the SEC filings source up to date?"}'
```

### POST /ingest/url

Runs one URL through fetch, extract, dedup, chunk, embed and store, and reports
each stage.

```
Request:  {url, source_id?, register_domain?}
Response: {ok, source_id, source_url, doc_id, doc_type, title,
           stages: [{name, status, latency_ms, detail, note}],
           chunks_written, vectors_written, skipped_reason,
           chunk_preview: [...], failure?: {stage, reason, detail}, latency_ms}
```

`stages` is the whole point. Six names in pipeline order: `fetch`, `extract`,
`dedup`, `chunk`, `embed`, `store`. `latency_ms` is `null` where nothing was
measured rather than zero, because zero is a measurement and this is not one.
The four pipeline timings come from `IngestResult.stages`, produced inside
`src/rag/index/pipeline.py`. Embedding and storing interleave per batch, so they
are accumulated separately in the loop rather than split by guesswork after.

`status` per stage is `ok`, `skipped` or `failed`. A duplicate is `skipped` at
`dedup`, and the stages after it are absent, because they did not run. Reporting
them as zeroes would claim work that never happened.

`chunk_preview` is a preview: at most 12 chunks, each truncated to 600
characters with `truncated: true` when it was cut. Returning every chunk of a
900 page PDF over JSON is the wrong default.

Domain policy is unchanged and is not negotiable from here:

- An unregistered domain is refused, `reason: "unknown_source"`, before any
  request is made. Seeding a domain is a legal decision, see
  `src/rag/fetch/SPEC.md`
- `register_domain: true` is the caller taking that decision explicitly. It
  registers a source with `allow_unlocker: false` and `max_tier: STEALTH`. It
  does not relax robots.txt, the rate limiter or the circuit breaker, none of
  which this flag can reach
- A blocked fetch returns HTTP 200 with `ok: false` and the reason in the trace.
  The request succeeded, the site refused, and those are different facts

```bash
curl -X POST localhost:8000/ingest/url \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"url": "https://quotes.toscrape.com/page/3/"}'
```

### POST /ingest/file

The same path minus the fetch. Multipart upload, one `file` field. PDF, DOCX,
XLSX, CSV, text or HTML, routed on magic bytes rather than on the declared type.

```
Request:  multipart/form-data, field `file`
Response: the same shape, with an `upload` stage in place of `fetch`
```

An uploaded file has no URL and a citation needs one, so the endpoint mints
`upload://upload.local/<filename>/<content digest>` and attributes the document
to a paused synthetic source named `upload`, never crawled. The digest is in the
path, so the same file uploaded twice resolves to the same `doc_id` and meets
dedup instead of duplicating the corpus.

```bash
curl -X POST localhost:8000/ingest/file \
  -H "X-API-Key: $API_KEY" -F "file=@report.pdf"
```

### GET /ingest/status

Scrape and ingestion state.

```
Query:    ?source_id=... or ?domain=... , omit both for a corpus summary
Response: {sources: [{source_id, status, circuit_state, last_success_at,
                      last_failure_reason, docs_indexed, docs_failed}],
           summary: {total_sources, healthy, degraded, unreachable}}
```

Includes sources that failed and sources currently being retried. Reads the
`source` and `source_state` tables defined in `src/rag/fetch/SPEC.md`, the same
store as the MCP `get_ingest_status` tool. Read only.

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
  (query, filters, top_k, prompt_version, embed_model_version, explain),
  TTL 5 minutes. `explain: true` reads through the cache, since the nonce and
  the strip log are per request and a cached one describes another request

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
- `/ask` returns chunk objects, not a count, and their ids match the cited ids
- `/ask` returns `validation` on every response, including the fallback path
- `explain: true` with `api.explain_enabled` false omits the block, key or no key
- `explain: true` never returns the system prompt body
- `/agent` trace contains every node that executed
- Missing API key returns 401
- `/ingest/url` on an unregistered domain refuses before any request is made
- `register_domain: true` registers a source that still cannot use the unlocker
- A blocked fetch is a `failed` stage in a 200 response, not a 5xx
- An unsupported type stops at `extract` with the earlier stages still `ok`
- A duplicate reports `dedup` as `skipped` and emits no later stages
- Stage latencies come from the pipeline, so a fake pipeline's numbers appear
  verbatim and nothing is invented in the API layer
- The same uploaded bytes always produce the same synthetic url
- A traversal filename cannot escape the synthetic url
- Tenant from the key reaches the MCP boundary and cannot be overridden by the
  request body
- Cache key changes when `prompt_version` changes
- Two `explain: true` calls with the same question return different nonces,
  proving the cache was not serving a stale explain block

## Known gaps

- Static API keys, no rotation, no per key rate limiting
- No streaming responses. `/ask` and `/agent` block until complete
- No pagination on `/search`
- The ingest endpoints write under `index.tenant_id` from config, not under the
  tenant the API key maps to. With one key and one tenant they are the same
  value, so nothing is wrong today, but a second tenant would be able to write
  documents that land in the first tenant's namespace. The write path needs the
  same tenant injection the read path already has
- Ingestion is synchronous. A long PDF holds the request open for the whole
  embed. It should be a job id plus a poll, which is also what would let the UI
  show stages as they happen rather than all at once when the call returns
- No delete endpoint. Qdrant is a derived index that currently only grows.
  Deleting a `document` row cascades to its chunks in Postgres and leaves the
  vectors orphaned. Re-ingesting the same document overwrites its points,
  because chunk ids are deterministic, so search results do not duplicate, but
  removal is not supported
- Any valid key can write to the corpus. There is no separate ingest scope
- `explain` is a single config flag for the whole deployment, not per key or per
  tenant. It stays off outside a demo or a local debug session
