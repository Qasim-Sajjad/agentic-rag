# AI tool usage log

Fill this in as you go, not at the end. Reconstructing it after 96 hours
produces exactly the vague account the assessment says it does not want.

One entry per session. Keep it honest and specific.

## Format

```
### YYYY-MM-DD, session N, <module>
Tool: <Claude Code / Copilot / other>
Asked for: <what you actually prompted, one or two lines>
Kept: <what went in unchanged>
Corrected: <what was wrong and what you changed>
Rewrote: <what you threw away and wrote yourself>
```

## Entries

### 2026-XX-XX, session 1, architecture
Tool:
Asked for:
Kept:
Corrected:
Rewrote:

### 2026-08-09, session 2, phase 0 foundation
Tool: Claude Code (Opus 5)
Asked for: Read CLAUDE.md, BUILD_ORDER, ARCHITECTURE and tests/SPEC.md, then
build phase 0 only: pyproject deps, a pydantic-settings loader over
config/settings.yaml with env override, structlog setup, and the fixture server
with all 8 endpoints. Plan and spec ambiguities first, no phase 1.
Kept: The settings model shape (one frozen section per YAML block, extra keys
forbidden), the source ordering that puts env above YAML, the structlog
ProcessorFormatter wiring, the fixture server layout with pages read from
tests/fixtures/pages/, and the hand written minimal PDF with a real text layer.
Corrected: Six things the SPEC left open, decided rather than guessed. The
loader moved from config/settings.py to src/rag/config/settings.py, since
config/ is not importable under a src layout and mypy src would never see it,
BUILD_ORDER updated in the same commit. Fallback to settings.example.yaml added
because config/settings.yaml is gitignored, so a clean clone had no config at
all. /flaky returns 500 rather than 503, because 503 is a block signature in the
fetch SPEC and would trigger escalation instead of the same tier retry the test
asks for. /rate-limited defaults Retry-After to 600, over the 300s cap, so the
bare URL exercises the requeue path, with a query param for the honour path.
/challenge decides by user agent or an x-fixture-tier header, since the spec
requires tier 2 to fail and tier 3 to pass and gives no mechanism, recorded in
known gaps. /robots.txt, /__stats and /__reset added beyond the 8 listed
endpoints, the first because /robots-blocked is defined against it, the other
two because the "zero HTTP requests" assertion and /flaky isolation need them.
Rewrote: Two test failures were mine and were real bugs in the code, not the
tests. The hit counting middleware counted /__stats itself, so a reset assertion
could never pass, fixed by excluding the control paths from the counters, which
is also the honest behaviour. The stdlib bridge test used a uvicorn logger name,
which uvicorn's own dictConfig had already set to propagate: False from the
integration fixture, so it was asserting on the wrong logger.
Verified: ruff check . clean, mypy src clean on 12 files, 42 tests pass.

### 2026-08-10, session 3, phase 1 fetch
Tool: Claude Code (Opus 5)
Asked for: Build phases 1 through 5, committing each phase separately, starting
with fetch against the spec.
Kept: The module split, types and protocols first and the orchestrator last.
Escalation, backoff and circuit transitions are pure functions over data, which
is what let the circuit breaker be tested across every transition with a fake
clock and no database. The `Clock` protocol removed sleeps from tests entirely.
Corrected: Four things the SPEC did not cover and one it got wrong.
`FetchResult` needed a `headers` field, because `cf-mitigated` is a block
signature and `Retry-After` drives the 429 path, and neither can be read after
the fact. `source_state` needed `circuit_first_open_at` and `frontier` needed
`passes` and `last_pass_at`, because the "three reopens in 24 hours" and "two
passes an hour apart" rules are not expressible in the DDL as written. The give
up rule moved to the worker, since it spans scheduling passes and `fetch()`
sees one call.
Rewrote: The escalation rule for 429. The spec lists 429 as a block signature,
so the first implementation escalated to Chromium and then Camoufox after
exhausting tier 1 retries. A test measuring slept time on the fake clock caught
it: 45 seconds of backoff across 9 attempts rather than 15 across 3. A rate
limit is the server asking for less traffic, not a bot wall, and escalating
collects the same 429 more slowly. Rate limiting now stops the ladder at the
tier it happened on. SPEC updated in the same commit.
Verified: ruff clean, mypy clean on 47 files, 105 tests pass including all nine
ladder cases from the spec, with tier 2 Chromium and tier 3 Camoufox launching
for real against the fixture server.

### 2026-08-10, session 4, phases 2 to 5
Tool: Claude Code (Opus 5)
Asked for: Build extract, index, evals and retrieve, one commit per phase.
Kept: The parser registry keyed on resolved MIME, the per page range PDF gates,
the structure aware chunker with its three invariants, RRF fused in code rather
than server side, and adaptive k as pure functions over scores. Fusion and the
adaptive cut are tested against hand computed fixtures with no database.
Corrected: The markdown block splitter treated a blank line after a table as
part of the table, so lists following a table were swallowed into the table
block. Caught by a test asserting list blocks are typed. The chunker had no
answer for a single block larger than the target, so an oversized paragraph
produced a chunk twice the limit; sentence splitting is the fallback, and
overlap now only carries a trailing block small enough to be context rather
than a second copy of the chunk. The chunk insert used a bare parameter inside
a CASE expression, which asyncpg cannot type, so it needed an explicit cast.
Rewrote: The near duplicate test. My first fixture was one sentence repeated
ten times, so changing one word changed every shingle and SimHash correctly
called them different. The test was wrong, not the code, and the fix was a
realistic multi sentence document.
Decided: Phases 4 and 5 were swapped. The eval harness measures retrieval, so
building the harness first would have meant a checkpoint that could not run.
Recorded in DESIGN known gaps.
Verified: ruff clean, mypy clean on 56 files. Ingest and search both run end to
end against Postgres and Qdrant, and `python -m rag.demo ingest-snippet` walks
a local file from routing through to vectors.

### 2026-08-10, session 5, phases 6 to 9
Tool: Claude Code (Opus 5)
Asked for: Build MCP, agent, prompts and API against the specs, one commit per
phase, then the README.
Kept: The MCP tool logic split from the transport, so both tools are tested
without a running server. The renderer returning its strip log, which made the
injection suite assertable on the structural layer rather than only on model
behaviour. `assess` as a pure function, tested across every branch with no
graph and no model. The LLM behind a protocol with a scripted implementation,
which is what let the whole agent be tested with no key and no network.
Corrected: Three things the environment or the SPEC got wrong. The MCP SDK
renamed `FastMCP` to `MCPServer` in 2.0 and `inputSchema` to `input_schema`, so
the SPEC's class name is stale. Phase 8 was built before phase 7, because the
agent imports the prompt registry. The `/ask` explain block needed the cache
key treatment nobody had specced, since a cached answer would otherwise return
another request's nonce.
Rewrote: The agent retry counter. It started in the conditional edge, which
looked right and silently did nothing: LangGraph hands edges a copy of the
state, so the increment was discarded and the loop ran until the fingerprint
check stopped it. A test asserting exactly one retry caught it. The counter
moved into the router node, where the mutation persists. Also swapped the graph
node lambdas for `functools.partial`, because LangGraph checks whether a node
is a coroutine function and a lambda returning a coroutine fails that check.
Verified: ruff clean, mypy clean on 71 files, injection suite green, all four
endpoints tested against the ASGI app with a scripted model.

## Summary for the design doc

Write this at the end, from the entries above. Three or four sentences covering
where AI was used, what needed correction most often, and which decisions were
yours rather than the tool's. Chunking approach, tool scoping, agent boundaries
and failure handling are the decisions being evaluated. Be specific about which
of those you made.
