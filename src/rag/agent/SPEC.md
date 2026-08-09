# agent

LangGraph graph with two agent roles. Calls MCP tools, never the retrieval
functions directly.

## Models

Two models, chosen per role. Config in `config/models.yaml`.

| Role | Model | Reason |
|---|---|---|
| Router | `claude-haiku-4-5-20251001` | Classification with structured output. No prose, no untrusted content, low latency in the hot path |
| Responder | `claude-sonnet-5` | Synthesis, citation discipline, injection resistance |
| Eval judge | `claude-opus-5` | Different from the generator, so LLM-as-judge does not grade its own output |

Approximate cost per agent query: router about $0.0015, responder about $0.017.
Roughly $0.02 per query. Prompt caching applies to the system prompt and tool
schemas, which are identical on every request.

## State

```python
class AgentState(TypedDict):
    question: str
    plan: Plan | None
    chunks: list[RetrievedChunk]
    trace: list[TraceStep]
    seen_call_hashes: set[str]
    iteration: int
    confidence: Literal["high", "low", "none"]
    error: str | None
    answer: str | None
    citations: list[Citation]

class TraceStep(BaseModel):
    node: str
    tool: str | None
    args: dict | None
    latency_ms: int
    model: str | None
    prompt_version: str | None
```

**Kept:** normalized chunks capped at `top_k`, the trace, call fingerprints,
iteration count.

**Dropped:** raw MCP response envelopes, discarded after normalization. Full
text of chunks the reranker cut, ids only. Conversation history, because each
`/agent` call is stateless. On retry, the previous iteration's chunks are
replaced rather than accumulated, so the responder never sees two overlapping
context sets.

## Nodes

**router.** LLM with structured output. Reads the question only. Returns a
`Plan`: which tool to call (or none) plus extracted filters.

It never sees retrieved content. That is structural, not incidental. It means
the routing decision cannot be influenced by a poisoned chunk, so the router is
immune to injection by construction rather than by prompt instruction.

**tool_executor.** Calls the MCP tool named in the plan. Catches timeouts and
tool errors, writes `error`, does not raise.

**assess.** Deterministic. No LLM. Reads `confidence`, `error` and `iteration`
and returns the next edge name. Keeping this out of the LLM means the branching
logic is unit testable and cannot be talked out of a decision.

**responder.** Generates the answer with citations from the chunks in state.
The only node that sees untrusted content.

## Edges

From `router`, conditional:
- plan requires `search_corpus` -> `tool_executor`
- plan requires `get_ingest_status` -> `tool_executor`
- plan requires no tool -> `responder`

From `tool_executor` -> `assess`.

From `assess`, conditional:

| Condition | Next |
|---|---|
| `confidence == "high"` | responder |
| `confidence == "low"` and `iteration == 0` | router, retry with broadened filters |
| `error` is set | responder, with error in state |
| anything else | responder, low confidence path |

From `responder` -> END.

The only loop is `assess -> router -> tool_executor -> assess`.

## Loop prevention

Three layers, because one is not enough.

1. **Iteration cap.** `max_iterations` (default 1 retry), checked in the
   conditional edge.
2. **Call fingerprints.** Hash `(tool_name, sorted_args)` into
   `seen_call_hashes`. A repeat is rejected before it reaches MCP. The
   iteration cap alone does not prevent this, because a router will happily
   reissue an identical query.
3. **LangGraph `recursion_limit`**, as the backstop for anything unanticipated.

## Failure handling

- MCP tool raises or times out: `error` is set, `assess` routes to the
  responder, which states plainly that the corpus could not be reached. The
  trace shows the failed call. Never silent, never a fabricated answer.
- Source unavailable: the router may call `get_ingest_status` on a retry, which
  turns an empty result into a specific explanation.
- Responder output fails validation: one repair turn, then a deterministic
  fallback template. See `prompts/SPEC.md`.

## Tool routing eval

`evals/goldset/tool_routing.jsonl`, 30 questions labelled with the expected
first tool: `search_corpus`, `get_ingest_status`, or `answer_directly`.
Ambiguous cases carry a set of acceptable answers.

Produces a tool selection accuracy number. Tracked in `results.jsonl` alongside
retrieval metrics.

## Tests

- Router returns valid structured output for a question of each routing class
- A question needing no tool reaches the responder without a tool call
- Low confidence on the first pass triggers exactly one retry
- An identical repeat tool call is rejected by the fingerprint check
- Tool timeout produces an answer that states the failure, with the trace
  showing it
- `assess` is tested directly as a pure function across all its branches
- Recursion limit is never reached in normal operation

## Known gaps

The retry broadens filters using a fixed rule rather than an LLM decision. A
richer planner would reason about why retrieval failed. Not implemented.
