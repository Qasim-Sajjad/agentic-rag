# index

Takes `CanonicalDoc`, removes duplicates, chunks structurally, embeds, and
upserts to Qdrant with full metadata.

## Deduplication

Exact dedup runs at three points, each saving the cost of the stage after it.

| Point | Key | Saves |
|---|---|---|
| Before fetch | canonical URL hash | the fetch |
| After fetch | raw content bytes hash | extraction |
| After extract | normalized text hash | embedding |

URL canonicalization: lowercase host, strip tracking params
(`utm_*`, `fbclid`, `gclid`), resolve redirects, drop fragments, sort remaining
query params.

**Near dedup: SimHash, after extraction only.** 64 bit, word 5-grams, Hamming
distance threshold 3. Banded index: four 16 bit bands, candidates share at
least one band.

Never run near-dedup on raw HTML. Every page on a site shares nav, footer and
sidebar, which is most of the shingle space. Raw HTML SimHash marks every page
on a domain as a duplicate of every other.

**Chunk level dedup** after chunking. Hash each chunk, drop cross document
duplicates. Catches disclaimers and author blocks that survive extraction, and
directly improves retrieval by keeping five copies of the same paragraph out of
the top 10.

## Chunking

Operate on `Block` objects, not raw text. Recursive character splitting is a
fallback for unknown structure. We have the structure.

| Block type | Policy |
|---|---|
| Paragraph, heading | Pack to `target_tokens` (default 512), hard break at heading boundaries, overlap one block |
| Table | Whole table is one chunk. If oversized, split by rows and repeat the header row in every part |
| List | Keep with parent heading. Never orphan |
| Code | Split at function or class boundaries |

**Every chunk carries its heading path**, prepended to the embedded text:

```
Annual Report 2024 > Risk Factors > Cybersecurity

We consider this risk material because...
```

A chunk saying "we consider this risk material" is useless alone and precise
with its path. This is the single highest value line in the chunker.

Configurable: `target_tokens`, `overlap_ratio`, `max_table_tokens`.
Validated empirically in `notebooks/01_chunking_sweep.ipynb`, not assumed.

## Metadata

Three classes with different operational roles.

**Filterable**, needs a Qdrant payload index:
`doc_type`, `domain`, `source_id`, `published_at`, `language`, `is_table`,
`fetch_tier`, `tenant_id`

**Display**, returned for attribution:
`source_url`, `title`, `section_path`, `page_no`

**Lineage**, never queried, load bearing:
`content_hash`, `chunk_hash`, `extractor_name`, `extractor_version`,
`chunker_version`, `embed_model_version`, `ingested_at`

Lineage fields let you say "re-extract only documents parsed by PyMuPDF4LLM
v0.1 where table count was zero" instead of reprocessing the corpus. They cost
nothing and save a week.

## Embedding

`BGE-M3`. Chosen because it emits dense and sparse vectors from one model. Every
alternative means a second system for the sparse side, with its own index and
its own way to drift out of sync. 8192 token context, so oversized table chunks
are not silently truncated.

Benchmarked against `Qwen3-Embedding-0.6B` in
`notebooks/02_embedding_compare.ipynb`. Qwen3 needs an `Instruct:` prefix on
queries and not on documents. Getting that asymmetry wrong silently destroys
recall and looks like a bad model.

Batching: `embed_batch_size` (default 32), bounded concurrency, retry with
backoff on transient errors. Checkpoint every N batches so a crash at 400K
chunks does not restart from zero.

## Vector store

Qdrant. One collection, `corpus`.

- Dense vector `dense` (1024 dims, cosine) and sparse vector `sparse` on the
  same point, so hybrid fusion is server side.
- `tenant_id` payload index marked as a tenant key, which co-locates each
  tenant's vectors on disk.
- Payload indexes on every filterable field above.

**Do not shard by source.** A general question searches across sources anyway,
so per source collections mean fan out plus a merge you have to write, and
scores stop being comparable. Filter instead. Separate collections only for a
different embedding dimensionality, or a contractual physical isolation
requirement.

Scale note: 500K x 1024 x 4 bytes is about 2 GB, roughly 500 MB quantized. This
fits one node. Search is not the scaling problem in this system. Ingestion is.

## Persistence

Two stores, split by access pattern. Qdrant is a derived index, not the source
of truth.

**`CanonicalDoc` to object storage**, one JSON blob per document at
`docs/{doc_id}.json`. Large, immutable, almost never queried. Needed only to
re-chunk without re-scraping.

**Chunks to Postgres**, text inline. Small rows, queried and joined constantly.
Needed to re-embed without re-extracting.

```sql
CREATE TABLE document (
    doc_id            TEXT PRIMARY KEY,
    source_id         TEXT NOT NULL REFERENCES source(source_id),
    source_url        TEXT NOT NULL,
    title             TEXT,
    published_at      DATE,
    language          TEXT,
    doc_type          TEXT NOT NULL,          -- html | pdf | office
    content_hash      TEXT NOT NULL,
    canonical_doc_key TEXT NOT NULL,          -- object storage key
    fetch_tier        SMALLINT NOT NULL,
    extractor_name    TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX document_content_hash ON document (content_hash);

CREATE TABLE chunk (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    chunk_index         INT NOT NULL,
    text                TEXT NOT NULL,        -- chunk content as extracted
    embed_text          TEXT NOT NULL,        -- section_path prepended, what gets embedded
    section_path        JSONB NOT NULL,
    page_no             INT,
    is_table            BOOLEAN NOT NULL DEFAULT FALSE,
    token_count         INT NOT NULL,
    chunk_hash          TEXT NOT NULL,        -- for cross document chunk dedup
    chunker_version     TEXT NOT NULL,
    embed_model_version TEXT,                 -- NULL until embedded
    embedded_at         TIMESTAMPTZ,
    tenant_id           TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX chunk_by_doc ON chunk (doc_id);
CREATE INDEX chunk_dedup ON chunk (chunk_hash);
CREATE INDEX chunk_needs_embed ON chunk (embed_model_version)
    WHERE embed_model_version IS NULL;
```

`chunk_id` is deterministic:
`sha256(f"{doc_id}:{chunk_index}:{chunker_version}")[:32]`. Re-running the
chunker on the same document produces the same ids, so upserts are idempotent
and a partial ingest can be resumed without duplicating rows.

`embed_text` is stored rather than recomputed. It is what actually goes to the
embedding model, and storing it means a re-embed backfill is a pure read with
no chunker logic in the path.

`chunk_needs_embed` is a partial index. It is what the embedding worker polls,
and it makes "find everything not yet embedded" a fast lookup at 500K rows
instead of a sequential scan.

## What the pipeline reports back

`IngestPipeline.ingest` returns an `IngestResult`. Beyond the counts it carries
two things a caller cannot reconstruct without doing the work twice:

- `chunks`, the chunks actually written. Empty on every skip path, which is the
  honest answer. A caller that wants to display them must not re-run the chunker
  to get them: re-chunking can show chunks that were never stored
- `stages`, a `StageTiming` per measured phase: `dedup`, `chunk`, `embed`,
  `store`. Timed where the work happens, not inferred afterwards. Embedding and
  storing interleave per batch, so each is accumulated separately inside the
  loop. A skip path returns only the stages that ran

This exists so `POST /ingest/url` can report the pipeline honestly, see
`src/rag/api/SPEC.md`. The pipeline itself neither knows nor cares who reads it.

`ingest` also takes an optional `progress` sink, the `Progress` shape defined in
`src/rag/progress.py`. `stages` is what happened, reported at the end; `progress`
is where the work is now, reported as it happens. It reports `dedup` once,
`chunk` with the fresh chunk count, then `embed` and `store` per batch, so a
thousand chunk document is a moving position rather than several silent minutes.
Omitting the sink changes nothing: the default reports nowhere.

Chunking runs in a worker thread. It is pure CPU over every block of the
document, and on the event loop a long document stalls every request the API
process is serving concurrently, which reads as a hung server.

## Demo entry point

Exists for live demonstration and manual verification. Two subcommands, both
running the full path through to a Qdrant upsert.

```bash
python -m rag.demo ingest <url> [--ephemeral] [--max-tier N]
python -m rag.demo ingest-snippet <file> [--url URL] [--content-type TYPE] \
                                        [--tenant TENANT]
```

`ingest` runs one URL through fetch, extract, chunk, embed and upsert, printing
the decision made at each stage. `--ephemeral` registers the domain with an
operator approved `tos_note` rather than requiring a `sources.yaml` edit.

`ingest-snippet` skips fetch and starts at `parse(content, source_url)`, which
is the same entry the fetcher uses, so nothing downstream can tell the two
apart. It must run all the way to upsert. Stopping at chunking would leave
`/ask` with nothing to retrieve, which defeats the reason the command exists.

**Synthetic provenance.** A pasted snippet has none, and three not-null columns
need values:

| Field | Value for a snippet |
|---|---|
| `source_url` | `--url` if given, else `snippet://demo/{sha256(content)[:8]}` |
| `source_id` | `demo`, a row seeded by bootstrap with `status: paused` so the scheduler never crawls it |
| `fetch_tier` | `0`, meaning not fetched |
| `doc_type` | from `--content-type`, defaulting to magic byte detection |
| `tenant_id` | `--tenant`, default `demo` |

`document.source_id` is a foreign key, so the `demo` source row is a
prerequisite, not an afterthought. `status: paused` is what keeps a synthetic
source out of real crawl scheduling.

Pass a real looking `--url` when demonstrating: citations render the URL, and
`snippet://` in the output reads as a placeholder rather than a source.

Prints at each stage: parser chosen and why, block counts by type, chunk count,
one sample chunk with its `section_path`, and points upserted.

## Re-embed

Prerequisite: the `document` and `chunk` tables above, plus `CanonicalDoc` in
object storage.

Given that, swapping models is:

1. Add `dense_v2` as a second named vector on the existing collection
2. Backfill by streaming `SELECT chunk_id, embed_text FROM chunk`, writing
   `embed_model_version` on success. No re-scrape, no re-extract
3. Dual read and compare on the frozen gold set
4. Cut over by config flag, drop `dense_v1`

Cost: 500K chunks at ~400 tokens is about 200M tokens, hours of wall clock.
Re-scraping instead would take weeks and return different content because the
web moved. That asymmetry is the entire reason intermediate artifacts persist.

## Tests

- URL canonicalization: tracking params stripped, params sorted, host lowercased
- SimHash: near duplicate pair inside threshold, unrelated pair outside
- SimHash on raw HTML vs extracted text, demonstrating the boilerplate failure
- Chunker invariants: no table is ever split without a repeated header, no chunk
  exceeds `max_tokens`, every chunk has a non empty `section_path`
- Upsert: idempotent on rerun, correct payload indexes created
- Re-embed: named vector added without touching existing points
- The progress sink reports `dedup`, `chunk`, `embed` and `store`, each stage
  once rather than once per batch

## Known gaps

Contextual retrieval (an LLM generated situating sentence per chunk) is not
implemented. It is a known recall improvement, but it costs 500K LLM calls even
with prompt caching. Noted as future work with the cost stated.

`ChunkRepository.save_many` issues one INSERT per chunk. On a 500 page document
that is thousands of round trips against a local Postgres, measured at a few
percent of ingest time and therefore not the bottleneck, but it should be a
single multi row INSERT or a COPY.
