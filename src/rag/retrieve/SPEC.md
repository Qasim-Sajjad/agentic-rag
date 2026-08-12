# retrieve

Query in, ranked chunks out. Pure semantic search and RAG answer generation stay
separate. This module never calls an LLM.

## Contracts

```python
class SearchFilters(BaseModel):
    doc_type: Literal["html", "pdf", "office"] | None = None
    source_id: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_url: str
    section_path: list[str]
    published_at: date | None
    page_no: int | None


class SearchResult(BaseModel):
    chunks: list[RetrievedChunk]
    confidence: Literal["high", "low", "none"]
    k_used: int
    reason: str | None  # set when confidence is not high
```

Entry point: `async def search(query, filters, top_k) -> SearchResult`.

## Pipeline

1. Embed the query with BGE-M3, producing dense and sparse vectors in one pass
2. Qdrant hybrid query, `candidate_pool` (default 50) per side
3. Fuse with Reciprocal Rank Fusion
4. Rerank the top `rerank_pool` (default 25) with the cross encoder
5. Adaptive k cut
6. Assign confidence

## Why hybrid

Dense embeddings fail on exact identifiers, part numbers, rare tokens and proper
nouns. They return similar instead of exact. Lexical fails on paraphrase. A
scraped corpus is full of jargon and identifiers, so it needs both.

Fusion is RRF, `k=60`. No score normalization and no per query tuning. Weighted
score fusion is available behind a config flag but is not the default, because
the weight would need tuning per query class and we have not done that work.

## Reranking

`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, passages truncated to 512 tokens,
all candidates in one forward pass. Budget roughly 150ms at pool 25.

Behind a `Reranker` protocol. `BGEReranker` (bge-reranker-v2-m3) and
`CohereReranker` are implemented but not default. On GPU, bge-reranker-v2-m3 is
the better model and the swap is a config change.

Pad chunks shorter than 50 tokens with their `section_path` before scoring.
Very short passages produce unstable cross encoder scores, and adaptive k keys
off those scores.

## Adaptive k

Static k gives 10 chunks whether the question is "who is the CEO" or "compare
the 2023 and 2024 risk disclosures". Let the retrieval signal decide.

After reranking, cut at whichever comes first:

- Absolute floor: score below `score_floor` (default 0.10)
- Elbow: gap between consecutive scores exceeds `elbow_delta` (default 0.15)

Clamp to `[k_min, k_max]` (defaults 3 and 15).

Two things fall out. Narrow questions get a tight context window. And if the
top reranked score is below `score_floor`, nothing relevant was retrieved, which
is the low confidence branch rather than a silent bad answer.

## Confidence

| Condition | confidence | reason |
|---|---|---|
| Top score >= `score_floor` | `high` | null |
| Top score in `[low_floor, score_floor)` | `low` | "weak match" |
| Top score < `low_floor`, or zero results | `none` | "no relevant documents" |

Scores are compared in the same units the reranker emits, which is 0 to 1. The
cross encoder produces an unbounded logit, so `MiniLMReranker` squashes it
through a logistic first. Skipping that step made the floors meaningless: they
read as probabilities and were compared against logits from roughly -11 to 11,
so `score_floor: 0.3` silently demanded a logit above 0.3 and the same question
phrased as a sentence rather than a keyword returned nothing.

The floors are low because `ms-marco-MiniLM-L-6-v2` was trained on short web
queries. A conversational question scored against table heavy text lands far
below what the model's ranking quality suggests, and the floors have to reflect
the model in use rather than a round number.

`none` returns the candidates it cut to, bounded at `k_min`, not an empty list.
Only zero results is empty. Discarding what was scored made a failed lookup
indistinguishable from a broken one: nothing to inspect in `/search`, nothing in
the `/ask` response, nothing in the agent trace.

The caller decides what to do. This module does not generate a fallback answer,
and it does not decide whether an answer may be generated. `src/rag/api/ask.py`
refuses to generate when confidence is `none`, keyed on the confidence value
rather than on whether the chunk list happens to be empty.

## Caching

Query embedding cache keyed by normalized query text, TTL 1 hour. Full result
cache keyed by hash of (query, filters, top_k), TTL 5 minutes. Both in memory
with an interface that allows Redis later.

## Tests

- Hybrid returns a result that dense alone misses (exact identifier query)
- Hybrid returns a result that sparse alone misses (paraphrase query)
- RRF ordering matches a hand computed expected order on a fixed fixture
- Reranking changes order versus fusion alone on a known case
- Adaptive k returns fewer chunks for a narrow query than a broad one
- Unanswerable query returns `confidence == "none"` with the candidates it
  rejected, bounded at `k_min`. Zero results returns an empty list
- The reranker squashes its logits into 0 to 1, monotonically, so the order is
  unchanged and an extreme logit saturates instead of raising
- Filters are applied, verified by a query that must exclude a known chunk

Retrieval quality is measured in `evals/`, not here. These are correctness
tests, not quality tests.
