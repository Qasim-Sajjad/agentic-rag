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
Response: {chunks: [...], confidence, k_used,
           steps: [{stage, candidates, latency_ms, note}], latency_ms}
```

`steps` is the retrieval funnel, five stages in order: `embed query`,
`vector search`, `fuse`, `rerank`, `adaptive cut`. `candidates` is how many
chunks left each stage, which is what makes a missing result diagnosable: a
pool of 100 fused to 60, reranked, then cut to 4 by the score floor says where
the chunk went. Each `latency_ms` is the gap since the previous stage, so they
sum to the total rather than each repeating it.

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
Query:    ?background=true to get a job id instead of waiting
Request:  {url, source_id?, register_domain?, allow_unlocker?}
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

`chunk_preview` is a preview: at most 12 chunks, each truncated to 1100
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
- `allow_unlocker: true` is a second, separate decision. Tier 4 is a paid
  service that solves a challenge on the caller's behalf, so it is opt in per
  domain rather than implied by registration. Setting it on a request that also
  registers the domain enables both at once; setting it on a request against a
  domain already registered upgrades that source in place, so the same domain
  never needs deleting and recreating to change its mind. Leaving it unset never
  disables the unlocker on a source that already has it, since that would be a
  request silently downgrading a decision an earlier one made deliberately
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
Query:    ?background=true, as on /ingest/url
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

### GET /ingest/jobs/{job_id}

Progress of a background ingest, and its result once it has one.

```
Response: {job_id, kind, label, status, elapsed_ms,
           progress: [{stage, done, total, detail}],
           result?: <the /ingest/url response>, error?}
```

`status` is `running`, `done` or `failed`. `result` appears only once the job is
`done` and is byte for byte the response the blocking call would have returned,
so a client reads one shape throughout and never needs two code paths.

`progress` exists because a 500 page PDF is minutes of extraction and embedding.
Held open as one HTTP request that is a guaranteed client timeout, and worse, a
caller cannot tell a slow success from a hang. Each stage reports its latest
position: `probe 500/500`, `extract 3/8 ranges`, `embed 640/2500`. A stage is
replaced in place rather than appended, because embedding reports per batch and
a caller wants the position, not hundreds of rows of history. `total` is `0`
where a stage cannot know it in advance, which is indeterminate and not zero
percent.

`failed` means an unexpected exception, and `error` carries its type and
message. A refused fetch is not this: it completes as `done` with `ok: false`
and a typed reason in the trace, the same as on the blocking call.

### GET /ingest/jobs

The last few jobs, newest first, same shape per job.

```
Query:    ?limit=20
Response: {jobs: [ ... ]}
```

In memory and bounded, 50 jobs, oldest evicted. A job describes work in flight,
not a durable record: by the time one is `done` the corpus itself is in Postgres
and Qdrant. A restart losing job history is acceptable, losing a chunk is not.
Recorded under Known gaps.

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
- `allow_unlocker: true` on the same request enables tier 4 on the newly
  registered source
- `allow_unlocker: true` against an already registered source upgrades it in
  place rather than requiring re-registration
- Omitting `allow_unlocker` never disables it on a source that already has it
- A blocked fetch is a `failed` stage in a 200 response, not a 5xx
- An unsupported type stops at `extract` with the earlier stages still `ok`
- A duplicate reports `dedup` as `skipped` and emits no later stages
- Stage latencies come from the pipeline, so a fake pipeline's numbers appear
  verbatim and nothing is invented in the API layer
- `?background=true` returns 202 with a job id and a poll path
- A background job reaches `done` and its `result` is the same trace the
  blocking call returns
- A background job reports the `chunk`, `embed` and `store` stages it passed
- Each stage appears once in `progress`, not once per batch
- An unknown job id is 404, and the job endpoints are behind the same API key
- `background` defaults to false, so the blocking contract is unchanged
- The progress sink reaches extraction and indexing, and an ingest without one
  still runs
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
- Job state is in memory and per process. A restart loses the history of what
  ran, and a second API process would not see the first one's jobs. Neither
  matters while Qdrant in process makes this the only writer there can be, and
  the corpus is durable regardless, but a multi process deployment would need
  the job table in Postgres
- A background job cannot be cancelled. Nothing here kills a fetch or an embed
  in flight, so a job started by mistake runs to completion
- No delete endpoint. Qdrant is a derived index that currently only grows.
  Deleting a `document` row cascades to its chunks in Postgres and leaves the
  vectors orphaned. Re-ingesting the same document overwrites its points,
  because chunk ids are deterministic, so search results do not duplicate, but
  removal is not supported
- Any valid key can write to the corpus. There is no separate ingest scope
- `explain` is a single config flag for the whole deployment, not per key or per
  tenant. It stays off outside a demo or a local debug session
