# Design document

One to two pages. Reasoning behind key decisions. Fill the TODO markers after
the sweeps run. Everything else is already decided and should not drift from
the SPEC files.

## 1. Scraping and resilience strategy

Four tier ladder: curl_cffi TLS impersonation, Chromium, Camoufox, managed
unlocker. Escalation is triggered by observed failure signatures (block status
codes, challenge markers) and emptiness heuristics (SPA root with no text),
never by guessing. The tier that succeeded is written back to a per domain
policy cache with a 7 day TTL, so escalation is discovered once per domain
rather than once per URL. That cache is the difference between six hours and six
days of ingestion.

**Where the line is.** Rendering a page the way a browser would is engineering.
Defeating a specific vendor's protection is not. We write no evasion code and
solve no CAPTCHAs.

Tier 4 draws that line by outsourcing rather than crossing it: a managed
unlocker, ScrapingBee here, does the challenge solving as its business, which
moves the ToS risk to a vendor whose business is that risk and keeps this
codebase free of evasion logic. It is gated twice, and neither gate defaults on:
a provider key must be configured, and the source must set `allow_unlocker`.
Exactly one source has it enabled.

That gating is the point, not a formality. Tiers 1 to 3 change how we present
ourselves and let the site decide. Tier 4 pays someone to decide for us, so it is
opt in per domain and only for domains whose terms permit automated access. It is
not pointed at anything that forbids them.

**What tier 4 actually bought.** On `file-examples.com`, tier 1 returns 403 with
`cf-mitigated: challenge` and a "just a moment" body; tier 4 returns 200 and the
real page. It also corrected a wrong belief: the seed URL that had been recorded
as blocked was returning 404 behind the challenge. The self driven tiers could
only see the challenge, so "blocked" was hiding "this file does not exist".

**Rate limited vs unreachable.** These are never collapsed. A 429 requeues with
a delay and honours Retry-After. Persistent blocking at the highest allowed tier,
twice across passes an hour apart, writes a dead letter entry with reason
`BLOCKED_PERSISTENT`. A source whose circuit reopens three times in 24 hours is
marked unreachable and excluded from scheduling until reset.

**Silent parser breakage.** The real risk is a redesign that yields a 200 with
clean HTML and no useful content. Three defences: a per source extraction yield
metric (mean blocks and characters per document) with alerting on a drop, a
minimum content length gate that dead letters rather than indexing an empty
document, and canary URLs per source whose extraction anchors are asserted in
CI.

## 2. Chunking strategy

Structure aware, operating on typed blocks from `CanonicalDoc`. Recursive
character splitting is a fallback for unknown structure, and we have the
structure.

Tables are never split without repeating the header row. Lists stay with their
parent heading. Every chunk carries its heading path prepended to the embedded
text, which is what makes "we consider this risk material" retrievable.

Size and overlap: 512 target tokens, 0.1 overlap, structure aware. Not swept.
The harness supports the sweep ({256, 512, 1024} x {0, 0.1, 0.2} x {recursive,
structure_aware} at a fixed context token budget rather than fixed k, because at
fixed k larger chunks win by getting more tokens) and nobody has run it. The
committed numbers are one config, not a winner. Stated plainly rather than
implied to be tuned.

What the chunker does measurably produce, on the 1000 chunk corpus: 236 chunks
from 60 book pages, 44 from 40 quote pages, and the rest from six arXiv PDFs
where one paper alone yielded 289 typed blocks. Tables survive as single chunks
and every chunk carries a non empty section path, both asserted in the unit
tests rather than sampled by eye.

## 3. Embedding model

BGE-M3, chosen for an architectural reason rather than a leaderboard position:
it emits dense and sparse vectors from one model, so hybrid retrieval is one
inference pass and one index rather than two systems that can drift apart. The
8192 token context also means oversized table chunks are not silently truncated,
which 512 token models do.

**Measured throughput on the machine this was built on** (CPU only, Intel Core
Ultra 7 155U, no GPU):

| Model | Dims | Sparse | Chunks per second |
|---|---|---|---|
| BGE-M3 | 1024 | yes, 26 terms on a sample chunk | 3.4 on short text, ~0.5 on 512 token chunks |
| bge-small-en-v1.5 | 384 | none | 78.8 |

That is a 20 to 60x gap depending on chunk length, and it is the real cost of
the architectural choice. It is also the reason the default is now the small
model: a 252 page prospectus is roughly 600 chunks, which BGE-M3 embeds in 90
minutes on this CPU. Nobody watches that, and an ingest nobody can finish is not
a retrieval system.

**What the default is now.** `bge-small-en-v1.5` for the dense side, and the
sparse side computed in `src/rag/index/lexical.py`: a hashed bag of words with
sublinear term frequency, the scoring half of BM25 without length
normalisation. Hybrid survives the swap, which matters because the lexical half
is what finds a course code or a part number that no embedding places near the
query. What is given up is real and worth stating: BGE-M3's sparse head weights
a term by its context, this one by its count; the dense side drops from 1024 to
384 dimensions and from an 8192 token window to 512, so a table chunk over 512
tokens is truncated for embedding purposes though never for storage or display.

Measured end to end after the swap, a 252 page 55 MB PDF through the API:
probe 13s, extract 35s, chunk 0.02s, embed 38s, store 0.4s, 89 seconds total.
The same document on BGE-M3 would have spent about an hour in embed alone.

Switching back is `index.embed_model: BAAI/bge-m3`, `embed_dims: 1024` and a
different `qdrant.collection`, because one collection holds one vector width.
The corpus does not need re-scraping to follow: `uv run python -m rag.index.reembed`
streams `chunk.embed_text` out of Postgres into the new collection. That is the
backfill this document has claimed since the first draft, and it is now a
runnable module rather than a paragraph. On a GPU, BGE-M3 is the obvious pick
and this whole trade disappears.

`evals/compare_embeddings.py` runs the head to head on the frozen gold set,
backfilling each model into its own collection from the `chunk` table rather
than re-crawling. It has not been run: re-embedding 1000 chunks twice on CPU is
roughly 25 minutes and the wall clock was spent elsewhere. The command is one
line and the harness appends its rows to `results.jsonl` like any other run.

Qwen3-Embedding-0.6B and text-embedding-3-small were scoped out: the former is
another 1.2 GB download for a comparison the bge-small contrast already makes,
and the latter needs an OpenAI key this project does not use.

MTEB was used to build a shortlist of three, not to pick the winner. It tests
single language text retrieval and does not measure this corpus.

**Domain specific corpora.** For legal or medical, the choice changes. General
embeddings underperform on domain jargon where surface similar terms are
semantically distant. The move is a domain tuned model, or fine tuning on
in domain query and passage pairs, decided by running the same harness on a
domain gold set rather than by assumption.

## 4. MCP and agent design

Two tools. `search_corpus` returns chunks, never a generated answer, so the
responder agent has real work and retrieval stays independently testable.
`get_ingest_status` exists because a `confidence: none` result has two very
different causes, and distinguishing "the corpus does not cover this" from
"that source has been blocked since the 3rd" is the difference between a dead
end and an answer.

Scoped out at the boundary: tenant (injected server side, absent from every
schema), unbounded k (capped at 20), writes (read only surface), arbitrary
filter expressions, and per session call budget.

The router decides using the question alone. It never sees retrieved content,
which makes it immune to injection by construction rather than by instruction.

**Honest answer on MCP.** Routing through MCP gives no capability this
assessment strictly needed. The benefit is architectural: the tool contract is
schema enforced and versioned independently of the agent, the server is process
isolated so a retrieval crash cannot take down the graph, and any MCP client can
use the same corpus without importing this Python. That pays off with multiple
clients. With one, it is overhead chosen deliberately.

## 5. Handling scale

**Ingestion is the bottleneck, not search.** 500K vectors at 1024 dims is about
2 GB, roughly 500 MB quantized, which fits one node.

What keeps ingestion from becoming a multi week job: the domain policy cache so
browser rendering is paid only where needed, page range parallelism so a 1000
page PDF is 20 tasks running four at a time rather than one long lock, layout
model gating so a GNN and an OCR pass do not run on pages that already have a
text layer, every synchronous parse and chunk in a worker thread so a long
document does not stall the event loop, async with a global semaphore plus per
domain token buckets, and a shared browser pool rather than a browser per URL.

Measured on a 60 page report before and after those three changes: 6,009 ms per
page down to 868 ms, a 6.9x difference, which is a 500 page document taking
seven minutes of extraction rather than fifty.

That moves the bottleneck to embedding. BGE-M3 is an XLM-R large, and on this
CPU it measures roughly 6 seconds per 512 token chunk: a 40 page document is 80
chunks and nine minutes, so a 500 page one is hours. That is a hardware
statement, not a code one, and the honest options are a GPU, a smaller model, or
an ONNX or quantized build. What the job endpoint changes is that those hours are
now visible and attributable rather than a request that never returns. Measured
at `embed_max_length` 1024, the current setting; at 512 the same chunk costs 6.0
seconds instead of 8.6, but 512 would truncate the embedding of a table chunk,
which is allowed up to `max_table_tokens` of 2048.

**What breaks first at 1M documents.** Not Qdrant. The browser pool, then cost.
Tier 2 and 3 fetches at 2 to 10 seconds each do not scale linearly on one
machine, and the fix is horizontal workers with a shared queue and a shared
policy cache, not a bigger box.

The two metered paths are next, and they fail on budget rather than throughput:
OCR is billed per page and tier 4 per request, neither has a spend cap, and at a
million documents a scanned minority is the largest line item in the system. A
per source budget with a hard stop is the missing control. Third is the SimHash
banded index, which needs to move out of memory.

**50 clients.** One collection per client is the wrong default. Single
collection, `tenant_id` as a Qdrant tenant key, which co-locates each tenant's
vectors on disk while keeping one index. Separate collections only for a
contractual physical isolation requirement or a different embedding
dimensionality. The source registry is already per source, so per tenant source
sets need no schema change.

**Rough cost per 100K document corpus, extrapolated from measured rates.**
Crawling ran at 9.4 seconds per page end to end on this machine, of which fetch
and extraction were about 0.3 seconds and the rest was CPU embedding. So the
number that matters is the embedding rate, roughly 0.5 chunks per second for
BGE-M3 on 512 token chunks.

At ~3 chunks per document, 100K documents is about 300K chunks, which is 170
hours of single process CPU embedding. That is the wall that makes a GPU
non optional at this scale, not Qdrant and not the crawler. On a GPU the same
work is hours. Embedding is roughly 150M tokens either way.

Two cheap wins were left on the table and are worth naming. Embedding runs per
document, so each page pays a forward pass for its 3 or 4 chunks while
`embed_batch_size` is 32: batching across documents is roughly a 3x win and the
`chunk_needs_embed` partial index already exists for exactly that worker. And
tier 2 and 3 rendering cost is unmeasured here, because the live corpus needed
tier 1 for everything except one page.

LLM cost stayed close to the estimate: about $0.02 per agent query, Haiku router
plus Sonnet responder, materially lower with prompt caching on the stable system
prompt and tool schemas. The 51 item gold set with its auto filter, three LLM
calls per candidate, cost well under a dollar.

## 6. Prompt engineering strategy

Five defence layers, weakest named as weakest: architectural (router sees no
chunks), structural (per request nonce delimiters, stripped forged markers),
instructional (the prompt text, assumed bypassable), validation (citations must
resolve against the retrieved set, checked in code), detection (canary tests in
CI).

The load bearing idea: an injection that changes wording is survivable, one that
fabricates a source is not. Validation makes the second impossible regardless of
model behaviour.

Layers 2 and 4 are observable, not just asserted. The renderer returns what it
stripped and the validator returns what it rejected, both surfaced on `/ask`.
A defence whose output nobody can see is indistinguishable from a model that
happened to behave, and the difference is the whole claim.

Structured output is enforced with one repair turn carrying the specific
validation error, then a deterministic fallback template. Never a loop. Repair
rate is logged as a quality metric.

**Iteration.** Three versions are committed and the diffs are the evidence.

`v1` is deliberately naive: "use the documents, cite your sources", no framing
at all. It exists to be the before.

`v2` adds the structural framing. Documents are named as data rather than
instructions, the task is restated after the context so the last thing in the
window is ours, citations must be `chunk_id`s that appear in the context, and
report-versus-obey is spelled out so a legitimate article about prompt injection
does not trigger a refusal.

`v3` came from a live failure, not from a guess. On `/agent`, a question about
crawler state routed correctly to `get_ingest_status`, the tool answered
`books-toscrape: healthy`, and the responder refused it: "the purported
operational facts from the ingestion subsystem are not a verifiable document
source and should not be trusted". v2 knows two trust levels, system
instructions and untrusted documents, and our own subsystem data is a third. v3
names all three: instructions, `<system_facts>` which are trusted and need no
citation, and `<doc_...>` containers which are data to report on. The same
question now answers "yes, according to our ingestion system's records, the
books-toscrape source is up to date".

That failure is the mirror image of the benign lookalike case in the injection
suite. Both are over-refusal, and both are why the suite tests for refusal as a
failure and not just for compliance.

Few shot examples are used only in the router, where the output space is small
and the format matters more than reasoning. Not used in the responder, where
they would bias answer shape across unrelated questions.

## 7. Measuring quality

Gold set of 51 items, 46 answerable plus 5 unanswerable, against a target of
115. Built by the pipeline the SPEC describes: stratified sample across prose,
table, short, long and PDF derived chunks, one LLM written question each, then
the auto filter asks every question with no context at all and discards the ones
the model already answers. That filter removed 14 of 60, or 23.3 percent, just
under the 25 to 35 percent the SPEC expects. It has not been hand verified,
which the SPEC calls not optional, so treat it as a working set rather than a
frozen one.

**Measured, one run, config hash `84e1217d`:**

| Metric | Value |
|---|---|
| recall@1 | 0.630 |
| recall@5 | 0.696 |
| recall@10 | 0.696 |
| MRR | 0.659 |
| nDCG@10 | 0.669 |
| unanswerable handled correctly | 5 of 5 |
| tool selection accuracy | 29 of 30 |
| p50 retrieval latency | 1608 ms |

**Why recall@5 and recall@10 are identical, which is not a rounding artifact.**
Adaptive k cut to `k_min` of 3 on nearly every query, so the retriever never
returned ten chunks and `recall@10` is really `recall@3` under the wrong label.
The cause is the elbow detector doing its job on cross encoder output: MiniLM
scores drop steeply between rank one and two, the gap exceeds `elbow_delta` of
0.15 immediately, and the cut clamps to the floor. Adaptive k is correct; the
harness is mislabelling what it measured. The fix is for the eval to retrieve at
a fixed k separately from the adaptive path the API uses, and it is not done.

The single routing miss is the deliberately ambiguous "why did that search
return nothing?", where the router chose `answer_directly` and the label
accepted `get_ingest_status` or `search_corpus`. Missing only on an ambiguous
item is a good result and worth more than a rounded 100 percent.

Results append to `evals/results.jsonl` keyed by a config hash covering chunker,
embedding model, retrieval params and prompt versions.

That file is the regression suite. CI runs the current config and fails if
recall@10 drops more than 2 points or injection pass rate drops at all. A prompt
change that quietly weakens injection resistance is caught because prompt
version is part of the config hash and injection pass rate is a tracked column.

Hallucination detection is grounding first: every factual sentence carries a
chunk id, and cited ids are validated against the retrieved set in code, so a
fabricated source cannot leave the system. Wrong tool selection is measured
directly against the 30 item routing set.

TODO: RAGAS is not implemented. See known gaps.

## 8. AI tool usage

See `docs/AI_USAGE.md` for the per session log.

## Known gaps

Every shortcut, stated plainly.

- Tier 4 has no spend cap. ScrapingBee is wired and billable per request; the
  per source `allow_unlocker` gate and a log line per call are the only
  controls. It is enabled on one source, `file-examples-pdf`.
- OCR output is unevaluated. Claude Sonnet 5 reads scanned page ranges and the
  routing gate selects it, but there is no gold transcription set, so accuracy
  on tables and multi column layout is unmeasured. `Block.confidence` is a
  constant 0.7: a VLM returns no calibrated score, and a self reported one would
  be worse than an honest constant.
- OCR is billed per page with no spend cap, the same gap as tier 4.
- Div based HTML table layouts are not reconstructed.
- Contextual retrieval not implemented. Known recall gain, 500K LLM calls.
- No RAGAS answer quality set. Needs reference answers, unstable at this sample
  size. The harness accepts it as an additional metric column.
- API auth is static keys, no rotation, no per key rate limiting.
- MCP auth is a shared secret header, not OAuth.
- No streaming responses. No pagination on `/search`.
- Agent retry broadens filters by a fixed rule, not an LLM decision.
- Injection set is 15 fixed cases with a fixed canary. Measures regression, not
  robustness against novel attack classes.
- The fixture server decides who gets past `/challenge` from the user agent, or
  from an `x-fixture-tier` header. A real interstitial fingerprints far more
  than that. It is enough to test the escalation decision, not evasion.
- `config/models.yaml` is not read by the settings loader yet. Nothing consumes
  model IDs before phase 5.
- Tiers 2 and 3 send the browser's own user agent rather than the honest
  crawler one, since announcing a crawler defeats the point of rendering like a
  browser. They send `X-Crawler-Contact` instead.
- Link discovery is same domain only, from anchors in extracted HTML. No sitemap
  parsing, so a page reachable only from a sitemap is never queued.
- The preferred fetch tier is learned per source, not per URL pattern. One
  JavaScript rendered page raises the starting tier for the whole domain, so
  static pages on that domain then pay for a browser until the policy TTL
  expires. Observed on `quotes-toscrape`, where `/js` taught the source tier 2
  and an ordinary page afterwards started there.
- Postgres is the only storage adapter. There is no in-memory implementation,
  so every fetch test needs a running database.
- Qdrant runs in process by default, which is single process: the API and an
  ingest run cannot hold it at once. `docker compose up -d qdrant` plus
  `path: null` switches to the server. This constraint is why the write path is
  an API endpoint rather than a separate service: see the next four entries.
- Job state is in memory and per process, bounded at 50 jobs. A restart loses
  the record of what ran, and a second API process would not see the first one's
  jobs. Neither matters while Qdrant in process makes this the only writer there
  can be, and the corpus is durable either way, but a multi process deployment
  would need the job table in Postgres.
- A background ingest cannot be cancelled. Nothing kills a fetch or an embed in
  flight, so a job started by mistake runs to completion.
- Progress polls are served from the same process that is doing the work. The
  heavy stages run in worker threads, so a poll is answered rather than blocked,
  but they hold the GIL in bursts: during PDF parsing a poll was measured taking
  up to 8 seconds instead of milliseconds. A process pool, or moving ingestion
  out of the API process entirely, is the real fix, and both are blocked by
  Qdrant in process being single writer.
- Embedding is the dominant cost of an ingest and runs on the CPU. See the scale
  section for the measurement. No GPU path is wired.
- `page_class` no longer selects a parser. COMPLEX_TEXT and SIMPLE_TEXT both
  route to `pymupdf4llm`, which already emits Markdown tables, and Docling
  measured roughly ten times the cost per page for output the chunker treats
  identically. Gate 2's table detection still runs, at roughly 240 ms per page,
  for a distinction that now only appears in the trace.
- `ChunkRepository.save_many` is one INSERT per chunk. Measured at a few percent
  of ingest time on a long document, so not the bottleneck, but it should be a
  multi row INSERT.
- The SimHash near duplicate index lives in memory, so it is empty after a
  restart and only catches near duplicates ingested by the same process. Exact
  duplicates are caught regardless, by content hash in Postgres. This matters
  most for OCR'd documents: a VLM transcribes the same scan slightly differently
  each run, so the content hash differs, the near duplicate check is the only
  thing that would catch it, and a re-ingest after a restart writes a second
  near copy of every chunk. Observed on a faxed consultation: 22 chunks stored
  for a document that has about 14.
- The ingest endpoints write chunks under `index.tenant_id` from config, not
  under the tenant the API key maps to. One key and one tenant makes those the
  same value today, so nothing is currently wrong, but a second tenant could
  write documents into the first tenant's namespace. The read path already
  injects the tenant correctly. The write path does not.
- Any valid API key can write to the corpus. There is no separate ingest scope
  or role.
- There is no delete path anywhere, so Qdrant only grows. Deleting a `document`
  row cascades to its chunks in Postgres and leaves those vectors orphaned in
  the collection. Re-ingesting the same document overwrites its points, because
  chunk ids are deterministic, so search results never duplicate, but genuinely
  removing a document is not supported.
- `ui/` is a demo surface, not a product. No auth beyond the key in the sidebar,
  and one shared Streamlit session state, so two people opening it at once share
  nothing but also confuse each other's reruns. Progress is polled once a second
  rather than pushed: the script blocks in a loop redrawing the stage list, which
  is the only shape Streamlit allows, since a background thread has no session to
  draw into. Results are held in session state so a widget click does not erase
  them, which is also what makes the raw text toggle on a chunk usable.
- Docling is wired for office formats but no PDF page class selects it any more.
  Measured at roughly ten times the cost per page of PyMuPDF4LLM for output the
  chunker treats identically. The gate is still computed and still reported.
- The gold set is 51 items, not the 115 the eval SPEC targets, and it has not
  been hand verified. The SPEC calls that step not optional.
- `recall@10` in `results.jsonl` is bounded by adaptive k, which cuts to 3, so
  it equals `recall@3`. See section 7. The column is honest about what ran and
  dishonest about what it is named.
- The reranker is a poor relevance signal on table heavy documents queried
  conversationally, and the confidence floors cannot fix it. Measured against an
  ingested clinical PDF whose chunks are pipe delimited lab metadata:
  `pap test cytology report` scores 0.682 and `FEDERSPIEL` scores 0.663, both
  correctly `high`, while `what is the patient diagnosis` scores 0.007 and
  `tell me about the user FEDERSPIEL and its clinical information` scores 0.004,
  both wrongly `none`. Two of three deliberately irrelevant queries score 0.0,
  so the floors do separate obvious junk, but a relevant conversational question
  is indistinguishable from an irrelevant one. `ms-marco-MiniLM-L-6-v2` was
  trained on short web queries and this content has almost no prose for it to
  match. The real fix is a better reranker, `bge-reranker-v2-m3` on a GPU, which
  is the config swap the protocol in `src/rag/retrieve/rerank.py` exists to
  allow, or deriving confidence from the dense score rather than the rerank
  score. Lowering the floors further is not a fix: it would admit the junk too.
  `/agent` is unaffected in practice because it rewrites the question into a
  keyword query and retries, which is exactly the case adaptive retry was for.
- Every number in `evals/results.jsonl` predates the reranker squash and the
  floor change, so `k_used`, the confidence labels and the `unanswerable 5/5`
  result are all stale. The eval has not been re-run.
- `evals/compare_embeddings.py` has not been run. Re-embedding the corpus twice
  on CPU is about 25 minutes.
- The corpus is roughly 1000 chunks from three sources. Retrieval behaves
  realistically at that size but the score floor and elbow thresholds were not
  tuned against it.
- Embedding runs per document rather than batched across documents, wasting most
  of a 32 wide batch on every page.
- Phases 4 and 5 were built in the opposite order to `BUILD_ORDER.md`. The
  harness measures retrieval, so retrieval had to exist for the phase 4
  checkpoint to mean anything. Phase 8 was built before phase 7 for the same
  reason: the agent depends on the prompt registry.
- The injection suite measures the structural and validation layers, which are
  deterministic. It does not measure whether a given model follows an injected
  instruction, because that needs a live model. The instructional layer is
  assumed bypassable, which is why it is listed third.
- `/agent` calls the MCP tool implementations in process rather than over the
  MCP transport. The schemas, the tenant injection, the k cap and the session
  budget are the same objects the server exposes, so the boundary is enforced,
  but the process isolation the SPEC describes is not exercised by `/agent`.
- No `langchain-mcp-adapters`. Tool discovery is live in `rag.mcp.client` and
  static in the agent.
- API auth is static keys in config, no rotation, no per key rate limiting.
- The response cache is in memory, so it is per process and empty on restart.
