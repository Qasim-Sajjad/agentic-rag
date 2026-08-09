# evals

One gold set serves chunking, embedding, retrieval and reranking. They are four
sweeps over the same file, not four test sets.

## Layout

```
evals/
  goldset/
    v1.jsonl              retrieval gold set, frozen
    tool_routing.jsonl    30 questions, expected first tool
  injection/              15 adversarial cases
  anchors/                12 extraction anchor pages
  run_eval.py             takes a config, returns metrics
  results.jsonl           append only, one row per run
  configs/                one file per sweep config
```

## Retrieval gold set

Target 115 items. Build once, then freeze.

1. **Stratified sample, ~120 chunks.** Deliberately spread across clean prose, a
   table, a short doc, a long doc, PDF derived, messy scraped, and one low
   quality source. Random sampling produces 100 clean paragraphs and an eval
   that is blind to the cases the corpus is actually full of.
2. **Generate one question per chunk** with an LLM, prompted so the question is
   answerable only from that chunk.
3. **Auto filter.** Ask each question with no context. If the model answers
   correctly, discard the pair. It tested world knowledge, not retrieval. This
   removes 25 to 35 percent and is the difference between a real eval and a
   vanity metric.
4. **Hand verify survivors.** About 30 minutes for 100 items. Not optional.
   These numbers get defended out loud.
5. **Add 15 unanswerable questions.** Plausible, on topic, nothing in the corpus
   answers them. Gold label `null`.

The unanswerable slice is disproportionately valuable. It is the only way to
measure the low confidence branch and to get a false positive rate for the
reranker score floor.

```json
{"qid": "q001", "question": "...", "gold_chunk_ids": ["c_8821"],
 "gold_doc_id": "d_301", "content_type": "table",
 "source_quality": "clean", "answerable": true}
```

Freeze it. Version it. Regenerating between runs makes results incomparable and
you will not notice.

## Metrics

Retrieval: `recall@1`, `recall@5`, `recall@10`, `mrr`, `ndcg@10`.
Operational: `p50_latency_ms`, `index_size_mb`.
Agent: `tool_selection_accuracy`.
Safety: `injection_pass_rate`.

## Results file

Append only. One row per run.

```json
{"run_id": "2026-08-08T14:22Z", "config_hash": "a3f9c1",
 "chunker": "structure_aware_v2", "chunk_size": 512, "overlap": 0.1,
 "embed_model": "bge-m3", "dims": 1024,
 "retrieval": "hybrid_rrf", "reranker": "minilm-l6", "k": 25,
 "prompt_version": {"rag_answer": "v2", "router": "v1"},
 "recall@5": 0.81, "recall@10": 0.89, "mrr": 0.72, "ndcg@10": 0.76,
 "tool_selection_accuracy": 0.87, "injection_pass_rate": 1.0,
 "p50_latency_ms": 210, "index_size_mb": 2100,
 "goldset_version": "v1"}
```

Four properties that make this work:

1. The gold set is frozen and versioned
2. The config hash is the identity. Any change to chunker, model, retrieval
   params or prompt version produces a new hash
3. It is committed, so the design doc tables are generated from it rather than
   hand typed
4. It is the regression suite. CI runs the current config and fails if
   `recall@10` drops more than 2 points or `injection_pass_rate` drops at all

## Sweep order

Sequential, not joint. Joint is 100+ runs, sequential is about 20.

1. Chunking: size in {256, 512, 1024} x overlap in {0, 0.1, 0.2} x strategy in
   {recursive, structure_aware}
2. Embedding on the winning chunk config: BGE-M3 vs Qwen3-Embedding-0.6B, plus
   text-embedding-3-small as a reference baseline
3. Retrieval params on the winner: fusion weights, rerank pool, k bounds

**Compare at a fixed token budget, not a fixed k.** At k=10, 256 token chunks
give the model 2560 tokens and 1024 token chunks give 10240. The larger chunks
win for the wrong reason. Hold the context budget constant and vary k.

## Notebooks

`notebooks/01_chunking_sweep.ipynb`, `02_embedding_compare.ipynb`,
`03_retrieval_ablation.ipynb`.

Commit with outputs rendered. The tables and charts go straight into the design
doc and are the evidence that the sweeps were actually run.

Notebooks import from `src/` and `evals/run_eval.py`. They never define pipeline
logic. If logic lives in a notebook cell it will drift from the package.

## Split of responsibility

- **Notebooks**: benchmarking, comparison, exploration
- **pytest**: correctness, invariants, and the injection suite

The injection suite stays in pytest specifically so it can be run live in front
of a reviewer.

## Known gaps

No RAGAS answer quality set. It needs reference answers and the numbers would
not be stable at this sample size. The harness accepts it as an additional
metric column. Faithfulness, answer relevance and context precision would run
against the same gold set with reference answers added. Scoped out for time,
recorded here rather than omitted quietly.
