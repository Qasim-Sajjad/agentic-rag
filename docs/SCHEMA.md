# Postgres schema

All tables in one place. Each is defined in the SPEC of the module that owns it.
This file is the index, not the definition.

| Table | Owner | Defined in | Purpose |
|---|---|---|---|
| `source` | fetch | `src/rag/fetch/SPEC.md` | Crawl policy per source, human edited |
| `source_state` | fetch | `src/rag/fetch/SPEC.md` | Circuit breaker and tier cache, machine written |
| `frontier` | fetch | `src/rag/fetch/SPEC.md` | URL work queue, SKIP LOCKED |
| `dead_letter` | fetch | `src/rag/fetch/SPEC.md` | Typed failures from fetch and extract |
| `document` | index | `src/rag/index/SPEC.md` | One row per ingested document |
| `chunk` | index | `src/rag/index/SPEC.md` | Chunk text and lineage, source of truth for re-embed |

Object storage holds `CanonicalDoc` blobs at `docs/{doc_id}.json`, keyed from
`document.canonical_doc_key`.

Qdrant holds vectors only. It is a derived index and can be rebuilt from
`chunk` without touching the network.

## Write ownership

One module writes each table. Everything else reads.

| Table | Written by | Read by |
|---|---|---|
| `source` | bootstrap, fetch (status only) | fetch, api |
| `source_state` | fetch | fetch, mcp, api |
| `frontier` | fetch scheduler | fetch |
| `dead_letter` | fetch, extract | mcp, api |
| `document` | index | index, api |
| `chunk` | index | index, retrieve |

## Recovery scenarios

What each store lets you rebuild without re-scraping:

| Scenario | Needs | Cost |
|---|---|---|
| Swap embedding model | `chunk` | Hours, backfill only |
| Change chunking strategy | `CanonicalDoc` in object storage | Hours, re-chunk plus re-embed |
| Change extraction parser | Nothing stored, must re-fetch | Days, full re-scrape |
| Rebuild Qdrant | `chunk` | Hours |

Extraction is the only stage where a change forces a re-scrape. That is a
deliberate limit: storing raw fetched bytes for 100K documents would be the
largest store in the system and would only pay off on parser changes.
