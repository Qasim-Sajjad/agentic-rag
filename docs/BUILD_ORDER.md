# Build order

Read `CLAUDE.md` first. This file is the sequence, not the conventions.

Build order is not spec reading order. Each phase has a checkpoint. Do not start
the next phase until the current one passes its checkpoint.

## Phase 0: foundation

Files: `pyproject.toml`, `src/rag/config/settings.py`, `src/rag/log.py`,
`tests/fixtures/server.py`

- Pydantic settings loader reading `config/settings.yaml` with env override.
  It lives under `src/` rather than in `config/` so it is importable from an
  installed package and covered by `mypy src`. `config/settings.yaml` is
  gitignored, so the loader falls back to `config/settings.example.yaml`
- structlog setup, with stdlib records routed through the same renderer
- Fixture server with all 8 endpoints from `tests/SPEC.md`, plus `/robots.txt`
  (which `/robots-blocked` is defined against) and `/__stats` and `/__reset`.
  `/__stats` is how a test proves a robots disallowed URL made zero requests

Checkpoint: `pytest tests/integration/test_fixture_server.py` passes. The
fixture server is the first commit because every fetch test depends on it.

## Phase 1: fetch

Spec: `src/rag/fetch/SPEC.md`

Order within the phase:
1. Types: `FetchTier`, `FailureReason`, `FetchResult`, `FetchFailure`
2. `Fetcher` protocol, then `StaticFetcher` (tier 1) alone
3. Robots checker and per domain token bucket
4. Escalation detector: block signatures and emptiness heuristics
5. `BrowserFetcher` (tier 2), `StealthFetcher` (tier 3), `UnlockerFetcher` stub
6. Backoff and requeue
7. Circuit breaker
8. Domain policy cache
9. The `fetch()` orchestrator that ties them together, written last

Checkpoint: all 9 test cases in the fetch spec pass against the fixture server.
No test touches the real network.

## Phase 2: extract

Spec: `src/rag/extract/SPEC.md`

1. `Block`, `Provenance`, `CanonicalDoc`, `DocumentParser` protocol
2. Content type router on headers plus magic bytes
3. `TrafilaturaParser`
4. `PyMuPDF4LLMParser`
5. PDF gates: text layer probe, layout probe
6. `DoclingParser`
7. `VLMOCRParser` stub that raises `NotImplementedError` with a clear message
8. Page range splitting and reassembly with the table merge fixup

Checkpoint: extraction anchors in `evals/anchors/` pass. Router picks correctly
for wrong extensions and magic-bytes-only cases.

## Phase 3: index

Spec: `src/rag/index/SPEC.md`

1. URL canonicalization and the three exact dedup points
2. SimHash with banded index
3. Structure aware chunker with per block type policy
4. Chunk level dedup
5. Metadata assembly, all three classes
6. BGE-M3 embedder with batching and checkpointing
7. Qdrant collection bootstrap and upsert

Checkpoint: end to end ingest of the fixture corpus produces a populated Qdrant
collection. Chunker invariants hold: no table split without a repeated header,
no chunk over max tokens, every chunk has a non empty section path.

## Phase 4: evals harness

Spec: `evals/SPEC.md`

Build the harness before tuning anything. Tuning without it is guessing.

1. `run_eval.py` taking a config, returning metrics
2. Gold set builder with the auto filter step
3. `results.jsonl` writer with config hashing

Checkpoint: harness runs against the current config and appends one row.

## Phase 5: retrieve

Spec: `src/rag/retrieve/SPEC.md`

1. Query embedding, dense plus sparse in one pass
2. Qdrant hybrid query
3. RRF fusion
4. `Reranker` protocol, `MiniLMReranker` default
5. Adaptive k: score floor and elbow detection
6. Confidence assignment

Checkpoint: retrieval correctness tests pass. Harness produces baseline numbers.

## Phase 6: mcp

Spec: `src/rag/mcp/SPEC.md`

1. Tool input and output models
2. FastMCP server with the two tools
3. Boundary scoping: tenant injection, k cap, session budget
4. `client.py` demonstration script

Checkpoint: `python -m rag.mcp.client` lists both tools with schemas, calls
both, prints results.

## Phase 7: agent

Spec: `src/rag/agent/SPEC.md`

1. `AgentState`, `Plan`, `TraceStep`
2. `assess` node first, as a pure function, with its tests
3. Router node
4. Tool executor with MCP client wiring
5. Responder node
6. Graph assembly, loop prevention

Checkpoint: agent branching tests pass. `assess` is tested across all branches
as a pure function.

## Phase 8: prompts

Spec: `src/rag/prompts/SPEC.md`

1. `PromptRegistry` with content hashing
2. Context renderer: nonce, delimiter stripping, sandwiching. It returns
   `RenderedContext` including the strip log, never a bare string. Discarding
   what was stripped is the one thing here that is expensive to retrofit
3. `rag_answer/v1.md` written without injection rules, on purpose
4. Run the injection suite, record failures
5. `rag_answer/v2.md` fixing what broke
6. Structured output validation and the one-repair ladder

Checkpoint: `pytest -k injection` passes all 15 cases plus the benign lookalike.
Both v1 and v2 committed. The diff is a deliverable.

## Phase 9: api

Spec: `src/rag/api/SPEC.md`

Four endpoints, auth dependency, caching, error shapes. `/ask` returns chunk
objects and a `validation` report, plus the `explain` block behind
`api.explain_enabled`. The `api` settings section is added here, since
`extra="forbid"` means the YAML key and the model land together.

Checkpoint: every curl example in the spec runs against a local server.

## Phase 10: README and final pass

- Setup instructions verified from a clean clone
- Every curl example tested
- `docs/DESIGN.md` TODOs filled from `evals/results.jsonl`
- Known gaps list matches reality

## Rules that apply to every phase

- Read the module SPEC before writing its first line
- Types and protocols first, orchestration last
- If the code needs to diverge from the SPEC, update the SPEC in the same commit
- Append to `docs/AI_USAGE.md` at the end of every session
- Do not skip a checkpoint to move faster. A broken phase 2 makes phase 5
  untestable and you will not know why
