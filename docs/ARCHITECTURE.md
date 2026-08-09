# Architecture

Diagram source lives here so it is versioned with the code and updated in the
same commit. Mermaid renders on GitHub and imports into Lucidchart directly.
Export a PNG from Lucidchart for submission.

## 1. Ingestion, fetch through index

```mermaid
flowchart TD
    REG[Source registry<br/>robots, rate budget, allowed tiers] --> Q[Frontier queue<br/>canonical URL, token bucket]
    Q --> PC{Domain policy cache}
    PC -->|preferred tier| T1

    subgraph LADDER [Fetch ladder]
        T1[Tier 1 curl_cffi<br/>50-150ms] -->|block sig or empty| T2[Tier 2 Chromium<br/>2-5s]
        T2 -->|bot challenge| T3[Tier 3 Camoufox<br/>4-10s]
        T3 -->|allow_unlocker| T4[Tier 4 managed unlocker]
        T4 --> DL[Dead letter<br/>typed failure reason]
        T2 -.->|429| RQ[Requeue with delay]
        T1 -.->|5 consecutive fails| CB[Circuit breaker open]
    end

    T1 --> CT
    T2 --> CT
    T3 --> CT
    T4 --> CT
    T1 -.writeback.-> PC

    CT{Content type router<br/>headers + magic bytes}
    CT -->|text/html| HTML[trafilatura]
    CT -->|application/pdf| PDFR{PDF gates<br/>text layer, layout}
    CT -->|office, csv| OFF[Docling]
    CT -->|unknown| DL

    PDFR -->|text, simple| P1[PyMuPDF4LLM]
    PDFR -->|text, tables| P2[Docling]
    PDFR -->|scanned| P3[VLM OCR<br/>stubbed]

    HTML --> CD
    P1 --> CD
    P2 --> CD
    P3 --> CD
    OFF --> CD

    CD[CanonicalDoc<br/>typed blocks, provenance, confidence]
    CD --> DD[Dedup<br/>content hash + SimHash]
    DD --> CH[Structure aware chunking<br/>heading path, tables intact]
    CH --> CDD[Chunk level dedup]
    CDD --> EMB[BGE-M3<br/>dense + sparse]
    EMB --> QD[(Qdrant<br/>one collection, tenant key)]
    CD -.persist.-> OBJ[(Object store + Postgres<br/>source of truth)]
    CDD -.persist.-> OBJ
```

## 2. Query path

```mermaid
flowchart LR
    API[FastAPI] --> RET
    subgraph RET [retrieve]
        QE[Query embed<br/>dense + sparse] --> HY[Qdrant hybrid<br/>pool 50 each side]
        HY --> RRF[RRF fusion]
        RRF --> RR[Cross encoder rerank<br/>MiniLM, pool 25, CPU]
        RR --> AK{Adaptive k<br/>floor + elbow}
        AK -->|above floor| HI[confidence high]
        AK -->|below floor| LO[confidence none]
    end
    HI --> OUT[SearchResult]
    LO --> OUT
```

## 3. MCP and agent layer

```mermaid
flowchart TD
    START[Question in] --> R[Router agent<br/>Haiku, structured output<br/>never sees chunks]
    R -->|no tool needed| RESP
    R -->|search_corpus| TE[MCP tool call]
    R -->|get_ingest_status| TE
    TE --> AS{Assess<br/>deterministic, no LLM}
    AS -->|confidence high| RESP
    AS -->|low and iteration 0| R
    AS -->|tool error| RESP
    AS -->|otherwise| RESP
    RESP[Responder agent<br/>Sonnet, nonce delimited context<br/>validated citations] --> END[Answer + citations + trace]

    TE -.-> MCP
    subgraph MCP [MCP server, own process]
        SC[search_corpus<br/>chunks not answers<br/>top_k capped at 20]
        GS[get_ingest_status<br/>circuit state, failure reason]
    end
    SC -.-> RET2[retrieve module]
    GS -.-> REG2[source registry]
```

## Boundaries worth naming on the diagram

**The narrow waist.** Every extractor emits `CanonicalDoc`. Chunking does not
know which parser ran. Swapping an extractor changes nothing downstream.

**The MCP boundary.** Tenant is injected server side and is absent from every
input schema. `top_k` is capped in the schema. The surface is read only.

**The injection boundary.** The router reads the question only. The responder is
the sole node that touches scraped text. That is why the router is immune to
injection by construction rather than by instruction.

**Source of truth.** Qdrant is a derived index. `CanonicalDoc` and chunks live
in object storage plus Postgres, which is what makes a model swap a backfill
instead of a re-scrape.

## Export

1. Copy a block above into Lucidchart, Import, Mermaid
2. Adjust layout, export PNG to `docs/architecture.png`
3. Update the Mermaid source in the same commit as any code change that alters
   the flow

The diagram, the code and `DESIGN.md` must describe the same system. Keeping the
source in the repo is how that stays true.
