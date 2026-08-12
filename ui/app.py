"""Streamlit front end for the demo. An HTTP client and nothing else.

Every panel is one call to the FastAPI surface in `src/rag/api`. This file
imports nothing from `rag` on purpose, and the reason is concrete: Qdrant runs
in process and is single writer, so a front end that drove the pipeline itself
could not run while the API server was up.

    uv run streamlit run ui/app.py

The API base url and key come from RAG_API_BASE and RAG_API_KEY, or from the
sidebar. Nothing else here is configurable, because nothing else here decides
anything: the server does.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_KEY = "dev-key"

# A URL that escalates to a browser tier, then embeds a long PDF, is minutes of
# real work. A short timeout here would report a failure the server never had.
INGEST_TIMEOUT = 900.0
QUERY_TIMEOUT = 300.0

UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "html", "htm"]
DOC_TYPES = ["", "html", "pdf", "office", "text"]

STATUS_MARK = {"ok": "🟢", "skipped": "🟡", "failed": "🔴"}

# A chunk id is 32 hex characters. Matched loosely so a shortened one still
# becomes a footnote rather than being left raw in the prose.
CITATION_ID = re.compile(r"\[([0-9a-fA-F]{8,64})\]")

# No colours, so the answer follows whichever theme the reader is using.
ANSWER_CSS = """
<style>
.rag-answer { font-size: 1.06rem; line-height: 1.75; max-width: 46rem; }
.rag-answer p { margin: 0 0 0.95rem 0; }
.rag-answer sup.cite {
  font-size: 0.68em;
  font-weight: 600;
  padding: 0 0.12em;
  opacity: 0.65;
  vertical-align: super;
}
</style>
"""

SINGLE_WRITER_NOTE = (
    "Qdrant runs in process and is single writer, so the API server is the only "
    "process allowed to hold the collection. Stop any crawl or pytest run first."
)


class ApiError(RuntimeError):
    """The server answered, and the answer was a refusal."""


@dataclass(frozen=True)
class Upload:
    name: str
    content: bytes
    content_type: str


@dataclass(frozen=True)
class Api:
    base: str
    key: str

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.key}

    def get(self, path: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base}{path}",
            params=params,
            headers=self._headers(),
            timeout=timeout,
        )
        return _unwrap(response)

    def post(
        self, path: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base}{path}",
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        return _unwrap(response)

    def upload(self, path: str, upload: Upload, timeout: float) -> dict[str, Any]:
        files = {
            "file": (
                upload.name,
                upload.content,
                upload.content_type or "application/octet-stream",
            )
        }
        response = httpx.post(
            f"{self.base}{path}", files=files, headers=self._headers(), timeout=timeout
        )
        return _unwrap(response)


def _unwrap(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 401:
        raise ApiError("the API key was rejected. Check the key in the sidebar")
    if response.status_code >= 400:
        raise ApiError(_error_message(response))
    body: dict[str, Any] = response.json()
    return body


def _error_message(response: httpx.Response) -> str:
    """The API returns a typed reason code and a request id. Show both, so a
    failure here can be found in the server log."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return (
            f"{error.get('code')}: {error.get('message')} "
            f"(request {error.get('request_id')})"
        )
    return f"HTTP {response.status_code}: {str(body)[:300]}"


def _run(call: Callable[[], dict[str, Any]], message: str) -> dict[str, Any] | None:
    try:
        with st.spinner(message):
            return call()
    except ApiError as exc:
        st.error(str(exc))
    except httpx.HTTPError as exc:
        st.error(f"could not reach the API. {exc}")
    return None


def sidebar() -> Api:
    base = os.environ.get("RAG_API_BASE", DEFAULT_BASE)
    key = os.environ.get("RAG_API_KEY", DEFAULT_KEY)
    api = Api(base.rstrip("/"), key)
    return api


def _check(api: Api) -> None:
    body = _run(lambda: api.get("/ingest/status", {}, QUERY_TIMEOUT), "Calling the API")
    if body is None:
        return
    summary = body.get("summary") or {}
    st.sidebar.success(f"connected, {summary.get('total_sources', 0)} sources")


def ingest_tab(api: Api) -> None:
    st.subheader("Run one document through the pipeline")
    st.caption(
        "fetch or upload, extract, dedup, chunk, embed, store. Each stage below "
        "reports what it decided and how long it took, measured server side."
    )
    mode = st.radio(
        "Input", ["URL", "File"], horizontal=True, label_visibility="collapsed"
    )
    if mode == "URL":
        _ingest_url_panel(api)
    else:
        _ingest_file_panel(api)


def _ingest_url_panel(api: Api) -> None:
    with st.form("ingest-url"):
        url = st.text_input("URL", placeholder="https://quotes.toscrape.com/page/3/")
        cols = st.columns(2)
        source_id = cols[0].text_input("source_id, optional")
        register = cols[1].checkbox("register this domain if it is unknown", value=True)
        submitted = st.form_submit_button("Run the pipeline")
    st.caption(
        "An unregistered domain is refused by default, because seeding a domain "
        "is a legal decision. Registering it does not relax robots.txt, the rate "
        "limiter, or the ban on the unlocker tier."
    )
    if not submitted or not url.strip():
        return
    payload: dict[str, Any] = {"url": url.strip(), "register_domain": register}
    if source_id.strip():
        payload["source_id"] = source_id.strip()
    trace = _run(
        lambda: api.post("/ingest/url", payload, INGEST_TIMEOUT),
        "Fetching, extracting, chunking, embedding",
    )
    if trace is not None:
        render_trace(trace)


def _ingest_file_panel(api: Api) -> None:
    chosen = st.file_uploader("PDF, DOCX, text or HTML", type=UPLOAD_TYPES)
    st.caption(
        "No fetch runs. Extraction takes bytes and a url, so an uploaded file "
        "and a scraped page are indistinguishable to everything downstream. The "
        "endpoint mints a synthetic upload:// url from the content digest, which "
        "is what citations then point at."
    )
    if chosen is None or not st.button("Run the pipeline", key="run-file"):
        return
    upload = Upload(chosen.name, chosen.getvalue(), chosen.type or "")
    trace = _run(
        lambda: api.upload("/ingest/file", upload, INGEST_TIMEOUT),
        f"Extracting and indexing {chosen.name}",
    )
    if trace is not None:
        render_trace(trace)


def render_trace(trace: dict[str, Any]) -> None:
    _render_headline(trace)
    _render_failure(trace)
    st.subheader("Pipeline stages")
    for stage in trace.get("stages") or []:
        _render_stage(stage)
    _render_chunks(trace)


def _render_headline(trace: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("chunks written", trace.get("chunks_written", 0))
    cols[1].metric("vectors stored", trace.get("vectors_written", 0))
    cols[2].metric("total ms", trace.get("latency_ms", 0))
    cols[3].metric("doc type", trace.get("doc_type") or "none")
    if trace.get("doc_id"):
        st.caption(
            f"stored under source_id **{trace.get('source_id', '')}** as doc_id "
            f"`{trace['doc_id']}`, url {trace.get('source_url', '')}"
        )
    if trace.get("skipped_reason"):
        st.warning(
            f"Nothing was embedded. {trace['skipped_reason']}. Dedup runs before "
            "embedding, so a repeat ingest costs nothing."
        )


def _render_failure(trace: dict[str, Any]) -> None:
    failure = trace.get("failure")
    if not failure:
        return
    st.error(
        f"Stopped at the {failure.get('stage')} stage. "
        f"Reason {failure.get('reason')}. {failure.get('detail')}"
    )


def _render_stage(stage: dict[str, Any]) -> None:
    mark = STATUS_MARK.get(str(stage.get("status")), "")
    latency = stage.get("latency_ms")
    timing = f"{latency} ms" if latency is not None else "not timed"
    label = f"{mark}  {stage.get('name', '')}  ({timing})"
    with st.expander(label, expanded=stage.get("status") == "failed"):
        if stage.get("note"):
            st.caption(stage["note"])
        if stage.get("detail"):
            st.json(stage["detail"])


def _render_chunks(trace: dict[str, Any]) -> None:
    chunks = trace.get("chunk_preview") or []
    if not chunks:
        return
    st.subheader(f"Chunks written ({len(chunks)} shown)")
    st.caption("A preview. The endpoint caps how many chunks and how many characters.")
    for chunk in chunks:
        path = " > ".join(chunk.get("section_path") or []) or "no section path"
        tokens = chunk.get("token_count")
        label = f"#{chunk.get('chunk_index')}  {tokens} tokens  {path}"
        with st.expander(label):
            _render_chunk_body(chunk)


def _render_chunk_body(chunk: dict[str, Any]) -> None:
    if chunk.get("is_table"):
        st.caption("table chunk, kept whole rather than split mid row")
    if chunk.get("page_no") is not None:
        st.caption(f"page {chunk['page_no']}")
    st.text(chunk.get("text", ""))
    if chunk.get("truncated"):
        st.caption("preview truncated by the endpoint")
    st.code(chunk.get("chunk_id", ""), language="text")


def _filters(source_id: str, doc_type: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if source_id.strip():
        filters["source_id"] = source_id.strip()
    if doc_type:
        filters["doc_type"] = doc_type
    return filters


def _render_answer(answer: str, citations: list[dict[str, Any]]) -> None:
    """Renders the answer readably without touching what the model produced.

    The raw chunk ids stay in the model output on purpose: the citation check
    rejects any id that was not in the retrieved set, and that check is the
    grounding guarantee. Renumbering them to footnotes is a display concern, so
    it happens here and not in the prompt.
    """
    st.markdown(ANSWER_CSS, unsafe_allow_html=True)
    body, order = _footnote(_paragraphs(answer))
    st.markdown(f"<div class='rag-answer'>{body}</div>", unsafe_allow_html=True)
    _render_sources(order, citations)


def _paragraphs(answer: str) -> str:
    """Escaped first. The answer is written from documents this system does not
    control, so a document carrying markup must not become markup here."""
    parts = [part.strip() for part in html.escape(answer).split("\n\n")]
    return "".join(f"<p>{part}</p>" for part in parts if part)


def _footnote(text: str) -> tuple[str, list[str]]:
    """Swap each 32 character id for its footnote number, first use ordering."""
    order: list[str] = []

    def swap(match: re.Match[str]) -> str:
        chunk_id = match.group(1).lower()
        if chunk_id not in order:
            order.append(chunk_id)
        return f"<sup class='cite'>{order.index(chunk_id) + 1}</sup>"

    return CITATION_ID.sub(swap, text), order


def _render_sources(order: list[str], citations: list[dict[str, Any]]) -> None:
    """Numbered to match the footnotes, which is the only reason this is not
    just the citation list the endpoint returned."""
    if not order:
        st.caption("no citations on this answer")
        return
    urls = {
        str(item.get("chunk_id", "")).lower(): item.get("source_url", "")
        for item in citations
    }
    st.markdown("**Sources**")
    for number, chunk_id in enumerate(order, start=1):
        url = urls.get(chunk_id) or "cited id was not in the returned citation list"
        st.markdown(f"{number}. {url}  \n`{chunk_id}`")


def _render_retrieved(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    st.subheader(f"Retrieved chunks ({len(chunks)})")
    for chunk in chunks:
        path = " > ".join(chunk.get("section_path") or []) or "no section path"
        with st.expander(f"{chunk.get('score', 0):.3f}  {path}"):
            st.caption(chunk.get("source_url", ""))
            st.text(chunk.get("text", ""))
            st.code(chunk.get("chunk_id", ""), language="text")


def search_panel(api: Api) -> None:
    st.subheader("Search")
    st.caption(
        "Pure retrieval, no LLM in the path. This is what lets retrieval be "
        "benchmarked and debugged independently of generation."
    )
    with st.form("search"):
        query = st.text_input("Query")
        cols = st.columns(3)
        top_k = cols[0].number_input("top_k", min_value=1, max_value=15, value=8)
        source_id = cols[1].text_input("source_id filter", key="search-source")
        doc_type = cols[2].selectbox("doc_type filter", DOC_TYPES, key="search-type")
        submitted = st.form_submit_button("Search")
    filters = _filters(source_id, str(doc_type))
    _warn_about_filters(filters)
    if not submitted or not query.strip():
        return
    payload: dict[str, Any] = {"query": query.strip(), "top_k": int(top_k)}
    if filters:
        payload["filters"] = filters
    body = _run(lambda: api.post("/search", payload, QUERY_TIMEOUT), "Searching")
    if body is not None:
        _render_search(body, filters)


def _warn_about_filters(filters: dict[str, Any]) -> None:
    """These persist across reruns and tab switches, so a filter set once stays
    set. An invisible filter reads as missing data: `doc_type: html` hides every
    PDF in the corpus and the result looks like a document that never indexed."""
    if not filters:
        return
    stated = ", ".join(f"{key} = {value}" for key, value in sorted(filters.items()))
    st.warning(
        f"Filter active: {stated}. Everything outside it is excluded. Clear the "
        "boxes above to search the whole corpus."
    )


def _render_search(body: dict[str, Any], filters: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("confidence", body.get("confidence", "none"))
    cols[1].metric("k used", body.get("k_used", 0))
    cols[2].metric("latency ms", body.get("latency_ms", 0))
    cols[3].metric("filters", len(filters) or "none")
    st.caption(
        "k_used is chosen per query by the score floor and the elbow rule, which "
        "is why it is often lower than top_k."
    )
    _render_retrieved(body.get("chunks") or [])


def agent_panel(api: Api) -> None:
    st.subheader("Agent")
    st.caption(
        "Routes through LangGraph, which calls the MCP tools. Every node, model, "
        "prompt version and tool call below comes from the response itself, not "
        "from a debug flag. Ask about corpus coverage to see the status tool "
        "fire instead of retrieval, and ask a question the corpus cannot answer "
        "to see it say so rather than guess."
    )
    with st.form("agent"):
        question = st.text_area("Question", height=90, key="agent-question")
        submitted = st.form_submit_button("Send")
    if not submitted or not question.strip():
        return
    body = _run(
        lambda: api.post("/agent", {"question": question.strip()}, QUERY_TIMEOUT),
        "Routing, calling tools, answering",
    )
    if body is not None:
        render_agent(body)


def render_agent(body: dict[str, Any]) -> None:
    steps = body.get("trace") or []
    st.markdown("### Answer")
    _render_answer(body.get("answer", ""), body.get("citations") or [])
    st.divider()
    _render_agent_metrics(body, steps)
    st.markdown("### Path taken")
    st.markdown(_path_line(steps))
    _render_models_and_prompts(steps)
    for index, step in enumerate(steps):
        _render_step(index, step)
    _render_retrieved(body.get("chunks") or [])


def _render_agent_metrics(body: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    cols = st.columns(5)
    cols[0].metric("confidence", body.get("confidence", "none"))
    cols[1].metric("steps", len(steps))
    cols[2].metric("tool calls", sum(1 for step in steps if step.get("tool")))
    cols[3].metric("chunks used", len(body.get("chunks") or []))
    cols[4].metric("total ms", sum(int(step.get("latency_ms") or 0) for step in steps))


def _path_line(steps: list[dict[str, Any]]) -> str:
    """The route through the graph, in one line. A tool step is named by its
    tool, because `tool_executor` twice tells you nothing about what ran."""
    if not steps:
        return "_no steps recorded_"
    hops = [f"`{step.get('tool') or step.get('node', '?')}`" for step in steps]
    return " → ".join(hops)


def _render_models_and_prompts(steps: list[dict[str, Any]]) -> None:
    models = _distinct(steps, "model")
    prompts = _distinct(steps, "prompt_version")
    st.caption(
        f"models: {', '.join(models) or 'none'}  |  "
        f"prompts: {', '.join(prompts) or 'none'}"
    )


def _distinct(steps: list[dict[str, Any]], field: str) -> list[str]:
    """Ordered, deduplicated. A router called twice used one model, not two."""
    seen: list[str] = []
    for step in steps:
        value = step.get(field)
        if value and value not in seen:
            seen.append(str(value))
    return seen


def _render_step(index: int, step: dict[str, Any]) -> None:
    node = str(step.get("node", ""))
    tool = step.get("tool")
    label = (
        f"{index + 1}. {node}"
        f"{'  calls ' + str(tool) if tool else ''}"
        f"  ({step.get('latency_ms', 0)} ms)"
    )
    with st.expander(label, expanded=_is_interesting(step)):
        if step.get("note"):
            st.markdown(f"**result** {step['note']}")
        if step.get("args"):
            st.caption("what this step decided or was given")
            st.json(step["args"])
        _render_step_provenance(step)


def _is_interesting(step: dict[str, Any]) -> bool:
    """Open the steps a reviewer asks about first: a rejected repeat call, and
    the one node that handled untrusted text."""
    note = str(step.get("note") or "")
    return "duplicate" in note or "stripped" in note


def _render_step_provenance(step: dict[str, Any]) -> None:
    model = step.get("model")
    prompt = step.get("prompt_version")
    if not model and not prompt:
        st.caption("no model call, this step is code")
        return
    st.caption(f"model {model or 'none'}, prompt {prompt or 'none'}")


def corpus_panel(api: Api) -> None:
    st.subheader("Ingestion state")
    st.caption(
        "GET /ingest/status, the same store the MCP get_ingest_status tool reads. "
        "This is how the agent can say a source has been blocked since a date "
        "instead of saying it does not know."
    )
    if not st.button("Refresh", key="refresh-status"):
        return
    body = _run(lambda: api.get("/ingest/status", {}, QUERY_TIMEOUT), "Reading state")
    if body is None:
        return
    _render_summary(body.get("summary") or {})
    sources = body.get("sources") or []
    st.dataframe(sources)
    _render_coverage_notes(sources)


def _render_summary(summary: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("sources", summary.get("total_sources", 0))
    cols[1].metric("healthy", summary.get("healthy", 0))
    cols[2].metric("degraded", summary.get("degraded", 0))
    cols[3].metric("unreachable", summary.get("unreachable", 0))


def _render_coverage_notes(sources: list[dict[str, Any]]) -> None:
    st.markdown("**Coverage notes**")
    for source in sources:
        note = source.get("coverage_note") or "no note"
        st.markdown(f"- **{source.get('source_id', '')}**: {note}")


def main() -> None:
    st.set_page_config(page_title="agentic-rag", layout="wide")
    api = sidebar()
    st.title("Agentic RAG")
    # No /ask tab. It is the same retrieval as Search followed by the same
    # generation as Agent, minus the routing, which read as a third answer
    # source rather than as the benchmark baseline it is. The endpoint stays,
    # because it is what isolates a retrieval failure from a routing one.
    tabs = st.tabs(["Ingest", "Agent", "Search", "Corpus"])
    with tabs[0]:
        ingest_tab(api)
    with tabs[1]:
        agent_panel(api)
    with tabs[2]:
        search_panel(api)
    with tabs[3]:
        corpus_panel(api)


main()
