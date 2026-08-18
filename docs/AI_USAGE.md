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

### 2026-08-11, session 6, streamlit front end plus the ingest endpoints
Tool: Claude Code (Opus 5)
Asked for: A simple Streamlit front end where a URL or an uploaded PDF or DOCX
runs through the pipeline with every step visible, plus ask and agent against
the same corpus. Stated belief in the prompt: the endpoints and backend already
exist, this is only a wrapper.
Kept: Nothing about the framing survived first contact, and that turned out to
be the useful part. The endpoints did not already exist. `/search`, `/ask`,
`/agent` and `/ingest/status` are all read only, and ingestion lived in the
`rag.demo` CLI. So a front end that ran the pipeline itself would have had to
open the Qdrant collection, which `qdrant.path` runs in process and single
writer, meaning it cannot be held by two processes. The wrapper was not possible
as described. What went in unchanged was the decision that follows from that:
put the write path behind `POST /ingest/url` and `POST /ingest/file` so the one
process that already holds the collection is the one that writes, and keep
`ui/app.py` a pure HTTP client that imports nothing from `rag`.
Corrected: Three things I got wrong on the way. First, I designed the stage
trace with six evenly timed stages before checking that `IngestPipeline.ingest`
returns only counts. It cannot report per stage timings it never measured, so
rather than invent them in the API layer I added `StageTiming` to the pipeline
and timed `dedup`, `chunk`, `embed` and `store` where the work happens. Embed
and store interleave per batch, so each accumulates inside the loop. A test now
feeds a fake pipeline fixed numbers and asserts they appear verbatim, which is
what stops a future version from filling gaps with plausible values. Second, I
was about to display chunks by re-running the chunker on the returned document,
copying what `rag.demo` does. That can show chunks that were never stored, so
`IngestResult` now carries the chunks it actually wrote. Third, `latency_ms` on
the upload stage started as `0`. Zero is a measurement and nothing was measured,
so it is `null`.
Rewrote: The domain policy path, twice. My first version let any URL through by
attributing it to a default source, which quietly routed around the rule that
seeding a domain is a legal decision. It now refuses an unregistered domain
before any request is made, and `register_domain: true` is the caller taking
that decision explicitly. Tested that the flag registers a source which still
cannot use the unlocker tier, because the interesting failure is a permission
flag that grants more than its name.
Verified: ruff and mypy clean, 20 new unit tests plus the existing suite, and
the whole thing driven live: an unregistered domain refused in 1 ms, a real
quotes.toscrape.com page fetched at tier 2 and indexed in 83 seconds, the same
URL again stopping at dedup in 972 ms with three stages instead of six, a PDF
upload stopping at dedup, and `/ask` returning three citations with none
rejected. Also caught in passing that the fetch tier is learned per source
rather than per URL, since a static quotes page started at tier 2 because `/js`
had taught the source to. Recorded in the design doc, not fixed.
Not verified: the explain checkbox in the UI. Streamlit's custom checkbox did
not respond to automated clicks, so the block was confirmed over HTTP instead
and its rendering path was never exercised in a browser.

### 2026-08-12, session 7, paid unlocker and VLM OCR
Tool: Claude Code (Opus 5)
Asked for: Wire ScrapingBee as tier 4 with a supplied key, and implement the VLM
OCR stub using Claude Sonnet 5.
Kept: Both interfaces were already the right shape, which is the payoff from
stubbing them rather than omitting them. `UnlockerFetcher` needed a `fetch` body
and a config block, nothing above it changed. OCR needed one real decision that
went the way the earlier design note predicted: send the page range to the API as
a `document` block rather than rasterizing to images here, so there is no image
pipeline and no resolution constant to get wrong, and turn on citations because
the page numbers they return are the only thing that can fill
`Provenance.page`.
Corrected: Four things. The OCR call has to be async and `PdfRouter._parse_range`
was sync, so the router's per range loop became `await`ed. The OCR prompt could
not be an inline string, so it went into the registry as a versioned role like
every other prompt. `extract` must not import from `agent`, so the vision client
is its own protocol in `extract/vision.py` rather than an extension of the
agent's LLM client. And the provider's HTTP status is not the origin's: a
ScrapingBee 200 can wrap a site's 403, so the fetcher reads `Spb-original-status`
and reports that, otherwise every block signature check downstream sees 200 and
concludes the page was fine.
Rewrote: The OCR prompt, after running it. v1 answered a page that was a
photograph with `[Image: a laptop keyboard resting on a marble surface]`, a
model authored sentence entering the corpus as document content and citable as
if the document had said it. v2 forbids describing a page with no text and the
rule is pinned by a test, so the diff between v1 and v2 is the evidence. Also
rewrote the design doc's claim that tier 4 was stubbed, which this session made
false.
Verified: ruff and mypy clean, 410 tests, and both paths against the real
services. Tier 1 on file-examples.com returns 403 with `cf-mitigated: challenge`
and a "just a moment" body; tier 4 returns 200 and the real page. OCR on the
scanned page range of a sample PDF returns no blocks under v2 for the photograph
page, and a forced text page transcribes to six blocks including a structurally
intact Markdown table, all at confidence 0.7.
Found in passing: the `file-examples-pdf` seed URL was returning 404. It had been
recorded as blocked, because a challenge is all the self driven tiers could see.
The paid tier's first useful result was correcting that belief rather than
fetching the file.
Not done: no spend cap on either metered path. Both are billed per call, both are
recorded as gaps rather than guarded, and a per source budget with a hard stop is
the missing control.

### 2026-08-17, session 8, long document performance and background ingest
Tool: Claude Code (Opus 5)
Asked for: A 500 page PDF could not be ingested during a live demo. Find out why,
fix it, and make chunking, indexing and retrieval report their individual steps
rather than one silent stage.
Kept: The measurement approach. Before changing anything I timed the stages
separately on a real report: plain `get_text()` at 6.85 ms per page,
`find_tables()` at 241 ms, `pymupdf4llm.to_markdown()` at 1,289 to 5,863 ms,
which is 11 to 49 minutes for 500 pages and explains the demo exactly.
Corrected: The diagnosis, twice. The first suspect was OCR, and `use_ocr=False`
changed nothing. The actual cause was that `pymupdf-layout` and `rapidocr` being
installed make `pymupdf4llm` take a GNN layout and OCR path on every page,
including pages with a perfectly good text layer, silently overriding the
architecture's own scanned page gate. The public `use_layout(False)` switch gave
8 to 11x for 0.7 percent less text.
Corrected: The claim in DESIGN.md that page ranges ran in parallel. `plan_ranges`
split the document and the caller then awaited each range one at a time, so the
split was bookkeeping. `gather` with a semaphore made it real, and the ranges now
visibly complete out of order.
Rewrote: The synchronous ingest request. A long document held one HTTP request
open for minutes, which is a guaranteed client timeout, and worse, a caller
could not tell a slow success from a hang. `?background=true` returns a job id
and 202, and the progress a caller polls for is the same `Progress` sink the
pipeline reports to, so nothing is measured twice or reported that did not
happen. The `Progress` protocol went at the package root rather than in either
module, because extract and index both report and neither may import the other.
Mine, not the tool's: that `parse` keeps its two argument protocol shape and
`parse_progress` is a separate method found by `getattr`, so adding progress did
not change the contract every other parser implements. Also that a failed fetch
stays a `done` job with `ok: false`, and only an unexpected exception is a
`failed` one: a site refusing is a result, not an error.
Corrected: my own first version of the threading fix. `BGEM3Embedder.embed`
called `self._load()` on the event loop and only the `encode` inside a thread,
so the two gigabyte model load still froze every poll. Caught by running a real
40 page ingest against the server and watching a poll time out at 30 seconds,
not by reading the code, which had looked right to me.
Verified: ruff and mypy clean on 77 files, 433 tests, and a real 40 page PDF
ingested through the running server while polling once a second. Every stage
appeared as it happened: `probe 12/40`, `extract 0/1`, `chunk 80/80`,
`embed 32/80`, `store 80/80`, ending with 80 chunks and 80 vectors. Slowest poll
during the run was 2.4 seconds, against a 30 second timeout before the fix.
Not done: embedding is now the bottleneck and it is hardware, not code. BGE-M3
on this CPU measures about 6 seconds per chunk, so the 500 page document that
failed the interview is still hours of embedding, now visible and attributable
instead of a request that never returns. A GPU or a smaller model is the fix and
neither is wired. `find_tables()` still runs during the probe for a page class
that no longer selects a parser, job state is in memory and per process, and a
job cannot be cancelled. All recorded as gaps.

### 2026-08-18, session 9, embedding swap and range planning
Tool: Claude Code (Opus 5)
Asked for: A 252 page prospectus was still taking an hour. Make it usable, and
match what a hosted assistant does with an uploaded document in two minutes.
Kept: The measurement first, again. bge-small-en-v1.5 at 184 ms per chunk
against BGE-M3 at 8,600 ms, both on this CPU, both timed before any code moved.
Corrected: my own assumption that extraction was still the problem. Threads and
processes were both benchmarked on a real 11 MB PDF and neither helped: parsing
was already 76 ms per page, and 8 processes made it four times slower. The two
real costs were the embedding model and, on the user's document, range planning
splitting 252 pages into 64 ranges because a page with a table was treated as a
different class from a page without, for a distinction that no longer selects a
parser. That document plans 6 ranges now.
Rewrote: the sparse side of hybrid retrieval. A small dense model has no sparse
head, and dropping the lexical half would lose exactly the queries it exists
for: a course code, a part number, a surname. `rag.index.lexical` is a hashed
bag of words with sublinear term frequency, the scoring half of BM25, at
microseconds per chunk. `blake2b` rather than the builtin `hash`, which is
salted per process and would give a document one id at index time and another at
query time.
Mine, not the tool's: that the model choice stays config with BGE-M3 intact
rather than being deleted, and that what the swap gives up is written into
DESIGN section 3 rather than quietly dropped. Also that the query instruction
prefix must not reach the sparse side, since those words would match every
document, which is a test rather than a comment.
Found in passing: changing the embedding model strands the corpus. Postgres
still holds the documents, so re-ingesting the same files is correctly rejected
as duplicate and their vectors never reach the new collection: present, indexed
and invisible. The backfill the index SPEC had described since the first draft
did not exist as code. It does now, and it moved 1,461 chunks in 290 seconds.
Verified: ruff and mypy clean on 79 files, 456 tests, and a 252 page 55 MB PDF
through the running API in 89 seconds end to end, with every stage visible while
it ran. Search after the swap finds an exact table row for a lexical query and
the right passage for a semantic one.
Not done: no eval run on the frozen gold set comparing the two models. The
harness and the backfill both exist now, so the comparison is one command per
model, and the numbers in DESIGN section 3 are throughput, not recall.

### 2026-08-18, session 10, answer rendering and page attribution
Tool: Claude Code (Opus 5)
Asked for: The agent answer showed literal asterisks and ran its numbered list
into one block. Render it properly and make the output less ugly.
Kept: nothing from the previous renderer. It escaped the answer and wrapped it
in paragraph tags, which was the bug: the model writes Markdown, and escaping
Markdown is how `**Regular Undergraduate Admission**` reached the screen with
its asterisks intact.
Corrected: the reason the escaping was there in the first place, rather than
just deleting it. The answer is written from documents this system does not
control, so markup in a document must not become markup on the page. Streamlit
renders Markdown with HTML off, which keeps that guarantee, so the only thing
that had to change was the footnote marker: superscript digits instead of `<sup>`
tags, and nothing needs an exception to the rule any more.
Rewrote: the agent panel as three tabs, answer, path and tools, retrieved
chunks. Reading the answer previously meant scrolling past the graph trace to
reach the chunks it came from. Also the retrieval steps, now a table rather than
a bullet list, and citation sources now show the file name rather than a
synthetic url ending in a content digest.
Found in passing, and this one matters more than the styling: the UI showed
`page 0` on every chunk. `parse_pages` stamped every block in a range with the
range's first page, which was approximately true when a range was a few pages
and became false when I merged ranges to fifty in the previous session. My own
change had quietly degraded citation provenance. `page_chunks=True` returns a
Markdown entry per page with its number attached at the same cost, so pages are
now real and count from one, which is what the OCR path already recorded and
what a PDF reader shows.
Mine, not the tool's: that a chunk preview is returned whole. A chunk cut mid
table is the defect a reviewer opens the preview to find, so truncating it there
would manufacture the defect being looked for.
Verified: ruff, mypy and 458 tests, plus the running UI in a browser: no literal
asterisks, superscript citations in both the prose and the source list, the
retrieval funnel as a table, and a CSV chunk rendering as a real table.
Not done: page numbers are fixed for documents ingested from now on. Anything
already in the corpus keeps the page it was stamped with, because the number
lives in the chunk and the re-embed backfill only replaces vectors.

## Summary for the design doc

Write this at the end, from the entries above. Three or four sentences covering
where AI was used, what needed correction most often, and which decisions were
yours rather than the tool's. Chunking approach, tool scoping, agent boundaries
and failure handling are the decisions being evaluated. Be specific about which
of those you made.
