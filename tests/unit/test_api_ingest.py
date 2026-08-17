"""The ingest endpoints, at the seam where the stage trace is assembled.

Every collaborator here is a local fake, so nothing touches the network, a
browser, Postgres or Qdrant. What is worth testing is the refusals: an
unregistered domain, a blocked fetch, a type no parser handles, and a
duplicate. The happy path is the easy half.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.api.ingest import (
    PREVIEW_LIMIT,
    IngestDependencies,
    UploadedFile,
    ingest_upload,
    ingest_url,
    upload_url,
)
from rag.api.models import IngestUrlRequest
from rag.config.settings import get_settings
from rag.extract.protocols import EmptyExtractionError, UnsupportedTypeError
from rag.extract.types import Block, BlockType, CanonicalDoc, DocType
from rag.fetch.service import UnknownSourceError
from rag.fetch.types import (
    FailureReason,
    FetchFailure,
    FetchResult,
    FetchTier,
    Source,
    SourceStatus,
)
from rag.index.pipeline import IngestResult, StageTiming
from rag.index.types import Chunk, ChunkMetadata

HTML = b"<html><body><h1>Quotes</h1><p>A paragraph with enough text.</p></body></html>"
PDF_BYTES = b"%PDF-1.4\n stream of bytes that never reaches a real parser"

QUOTES = Source(
    source_id="quotes",
    domain="quotes.test",
    seed_urls=["https://quotes.test/"],
    tos_note="fixture",
)

FULL_STAGES = (
    StageTiming("dedup", 1),
    StageTiming("chunk", 2),
    StageTiming("embed", 30),
    StageTiming("store", 4),
)


class FakeRegistry:
    def __init__(self, sources=None):
        self.sources = dict(sources or {})
        self.upserted = []

    async def get(self, source_id):
        return self.sources.get(source_id)

    async def by_domain(self, domain):
        for source in self.sources.values():
            if source.domain == domain:
                return source
        return None

    async def upsert(self, source):
        self.upserted.append(source)
        self.sources[source.source_id] = source


class FakeFetch:
    """Returns whatever the test set, or raises it."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def fetch(self, url):
        self.calls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeExtract:
    def __init__(self, doc=None, error=None):
        self.doc = doc
        self.error = error
        self.calls = []

    async def extract(self, content, source_url, content_type, progress=None):
        self.calls.append((source_url, content_type))
        self.progress = progress
        if self.error is not None:
            raise self.error
        return self.doc.model_copy(update={"source_url": source_url})


class FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ingest(self, doc, source_id, fetch_tier=0, progress=None):
        self.calls.append((doc.doc_id, source_id, fetch_tier))
        self.progress = progress
        return self.result


def a_doc():
    return CanonicalDoc(
        doc_id="doc1",
        source_url="https://quotes.test/1",
        title="Quotes",
        blocks=[
            Block(type=BlockType.HEADING, text="Quotes", level=1),
            Block(type=BlockType.PARAGRAPH, text="A paragraph with enough text."),
        ],
        content_hash="hash1",
        extractor_name="trafilatura",
        extractor_version="1.0",
        doc_type=DocType.HTML,
    )


def a_chunk(index=0, tokens=120, text="A paragraph with enough text."):
    return Chunk(
        chunk_id=f"chunk{index}",
        doc_id="doc1",
        chunk_index=index,
        text=text,
        embed_text=f"Quotes > {text}",
        token_count=tokens,
        metadata=ChunkMetadata(
            doc_type=DocType.HTML,
            domain="quotes.test",
            source_id="quotes",
            section_path=["Quotes"],
            chunk_hash=f"hash{index}",
        ),
    )


def a_result(chunks=(), skipped=None, stages=FULL_STAGES):
    return IngestResult(
        doc_id="doc1",
        chunks_written=len(chunks),
        vectors_written=len(chunks),
        skipped_reason=skipped,
        chunks=tuple(chunks),
        stages=tuple(stages),
    )


def a_fetch_result(tier=FetchTier.STATIC, content=HTML, content_type="text/html"):
    return FetchResult(
        url="https://quotes.test/1",
        final_url="https://quotes.test/1",
        status=200,
        content=content,
        content_type=content_type,
        tier_used=tier,
        attempts=1,
        fetched_at=datetime(2026, 8, 11, tzinfo=UTC),
    )


def build_deps(fetch=None, extract=None, pipeline=None, registry=None):
    return IngestDependencies(
        fetch=fetch or FakeFetch(a_fetch_result()),
        extract=extract or FakeExtract(a_doc()),
        pipeline=pipeline or FakePipeline(a_result([a_chunk()])),
        registry=registry if registry is not None else FakeRegistry({"quotes": QUOTES}),
        settings=get_settings(),
    )


def stage_named(response, name):
    return next(stage for stage in response.stages if stage.name == name)


async def test_an_unregistered_domain_is_refused_before_any_request():
    """Seeding a domain is a legal decision, so an unknown one stops here."""
    fetch = FakeFetch(a_fetch_result())
    deps = build_deps(fetch=fetch, registry=FakeRegistry())
    response = await ingest_url(IngestUrlRequest(url="https://unknown.test/x"), deps)
    assert response.ok is False
    assert response.failure.stage == "fetch"
    assert response.failure.reason == "unknown_source"
    assert fetch.calls == []


async def test_a_named_source_that_does_not_exist_is_refused():
    deps = build_deps(registry=FakeRegistry({"quotes": QUOTES}))
    request = IngestUrlRequest(url="https://quotes.test/1", source_id="nope")
    response = await ingest_url(request, deps)
    assert response.ok is False
    assert response.failure.reason == "unknown_source"


async def test_registering_a_domain_does_not_enable_the_unlocker_tier():
    """The flag is permission to register, not permission to bypass."""
    registry = FakeRegistry()
    deps = build_deps(registry=registry)
    request = IngestUrlRequest(url="https://quotes.test/1", register_domain=True)
    response = await ingest_url(request, deps)
    assert response.ok is True
    registered = registry.upserted[0]
    assert registered.allow_unlocker is False
    assert registered.max_tier is FetchTier.STEALTH
    assert registered.domain == "quotes.test"


async def test_registering_a_domain_with_the_unlocker_flag_enables_tier_four():
    """A second, separate decision from `register_domain`. Setting it on the
    same request is what makes the choice visible in the request itself,
    rather than inherited from bare registration."""
    registry = FakeRegistry()
    deps = build_deps(registry=registry)
    request = IngestUrlRequest(
        url="https://quotes.test/1", register_domain=True, allow_unlocker=True
    )
    response = await ingest_url(request, deps)
    assert response.ok is True
    registered = registry.upserted[0]
    assert registered.allow_unlocker is True
    assert registered.max_tier is FetchTier.UNLOCKER


async def test_the_unlocker_flag_upgrades_an_already_registered_source():
    """The same domain can be ingested again later with the flag set, and the
    source that already exists should not need deleting and recreating."""
    registry = FakeRegistry({"quotes": QUOTES})
    deps = build_deps(registry=registry)
    request = IngestUrlRequest(url="https://quotes.test/1", allow_unlocker=True)
    response = await ingest_url(request, deps)
    assert response.ok is True
    upgraded = registry.upserted[0]
    assert upgraded.source_id == "quotes"
    assert upgraded.allow_unlocker is True
    assert upgraded.max_tier is FetchTier.UNLOCKER


async def test_omitting_the_unlocker_flag_never_turns_it_back_off():
    """One directional. A source that already earned tier 4 must not lose it
    because a later request forgot to ask again."""
    registry = FakeRegistry(
        {
            "quotes": QUOTES.model_copy(
                update={"allow_unlocker": True, "max_tier": FetchTier.UNLOCKER}
            )
        }
    )
    deps = build_deps(registry=registry)
    request = IngestUrlRequest(url="https://quotes.test/1")
    response = await ingest_url(request, deps)
    assert response.ok is True
    assert registry.upserted == []


async def test_a_blocked_fetch_is_a_failed_stage_and_not_an_exception():
    """The request succeeded. The site refused. Those are different things."""
    failure = FetchFailure(
        url="https://quotes.test/1",
        reason=FailureReason.BLOCKED_PERSISTENT,
        last_tier=FetchTier.STEALTH,
        attempts=9,
        detail="challenge page at every tier",
    )
    deps = build_deps(fetch=FakeFetch(failure))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is False
    assert response.failure.reason == "blocked_persistent"
    fetch_stage = stage_named(response, "fetch")
    assert fetch_stage.status == "failed"
    assert fetch_stage.detail["attempts"] == 9
    assert fetch_stage.detail["tier"] == int(FetchTier.STEALTH)


async def test_an_unknown_source_raised_by_the_fetch_service_is_caught():
    """The registry check and the service check can disagree under a race."""
    deps = build_deps(fetch=FakeFetch(UnknownSourceError("quotes.test is not seeded")))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is False
    assert response.failure.stage == "fetch"
    assert response.failure.reason == "unknown_source"


async def test_a_type_no_parser_handles_stops_at_extract():
    error = UnsupportedTypeError("application/zip")
    deps = build_deps(extract=FakeExtract(error=error))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is False
    assert response.failure.stage == "extract"
    assert response.failure.reason == "UnsupportedTypeError"
    assert stage_named(response, "extract").status == "failed"
    assert stage_named(response, "fetch").status == "ok"


async def test_an_empty_extraction_stops_at_extract():
    deps = build_deps(
        extract=FakeExtract(error=EmptyExtractionError("no usable pages"))
    )
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is False
    assert response.failure.reason == "EmptyExtractionError"


async def test_the_happy_path_reports_every_stage_in_pipeline_order():
    deps = build_deps()
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is True
    names = [stage.name for stage in response.stages]
    assert names == ["fetch", "extract", "dedup", "chunk", "embed", "store"]
    assert response.chunks_written == 1
    assert response.vectors_written == 1
    assert response.doc_id == "doc1"


async def test_progress_is_reported_while_the_ingest_runs():
    """The trace arrives at the end. Progress is what a caller polling a job
    sees before then, so it has to be reported as the stages happen."""
    seen = []
    deps = build_deps()
    await ingest_url(
        IngestUrlRequest(url="https://quotes.test/1"),
        deps,
        lambda stage, done, total, detail="": seen.append((stage, done, total)),
    )
    assert [row[0] for row in seen] == ["fetch", "fetch"]
    assert seen[-1] == ("fetch", 1, 1)


async def test_the_progress_sink_reaches_extraction_and_indexing():
    """Reported by those modules, not by the API. Passing it down is the whole
    reason `Progress` lives at the package root."""
    deps = build_deps()

    def report(stage, done, total, detail=""):
        return None

    await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps, report)
    assert deps.extract.progress is report
    assert deps.pipeline.progress is report


async def test_an_ingest_without_a_progress_sink_still_runs():
    """`progress` is optional, and the endpoint that waits does not pass one."""
    request = IngestUrlRequest(url="https://quotes.test/1")
    response = await ingest_url(request, build_deps())
    assert response.ok is True


async def test_stage_latencies_come_from_the_pipeline_and_are_not_invented():
    deps = build_deps()
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert stage_named(response, "embed").latency_ms == 30
    assert stage_named(response, "store").latency_ms == 4


async def test_the_fetch_tier_reaches_the_pipeline_as_lineage():
    pipeline = FakePipeline(a_result([a_chunk()]))
    fetch = FakeFetch(a_fetch_result(tier=FetchTier.BROWSER))
    deps = build_deps(fetch=fetch, pipeline=pipeline)
    await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert pipeline.calls == [("doc1", "quotes", int(FetchTier.BROWSER))]


async def test_a_duplicate_reports_dedup_as_skipped_and_never_embeds():
    """Dedup is checked before embedding, so a repeat costs nothing."""
    result = a_result(
        skipped="exact duplicate content hash", stages=(StageTiming("dedup", 2),)
    )
    deps = build_deps(pipeline=FakePipeline(result))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.ok is True
    assert response.chunks_written == 0
    assert response.skipped_reason == "exact duplicate content hash"
    dedup = stage_named(response, "dedup")
    assert dedup.status == "skipped"
    assert [stage.name for stage in response.stages] == ["fetch", "extract", "dedup"]


async def test_a_declared_type_that_disagrees_with_the_bytes_is_named_in_the_trace():
    """Routing on what the caller claimed is how a renamed PDF reaches the
    HTML parser, so a correction is worth stating."""
    fetch = FakeFetch(
        a_fetch_result(content=PDF_BYTES, content_type="text/plain; charset=utf-8")
    )
    deps = build_deps(fetch=fetch)
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    note = stage_named(response, "extract").note
    assert "text/plain" in note
    assert "application/pdf" in note


async def test_the_chunk_preview_is_capped():
    chunks = [a_chunk(index=index) for index in range(PREVIEW_LIMIT + 8)]
    deps = build_deps(pipeline=FakePipeline(a_result(chunks)))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.chunks_written == PREVIEW_LIMIT + 8
    assert len(response.chunk_preview) == PREVIEW_LIMIT


async def test_long_chunk_text_is_marked_as_truncated():
    deps = build_deps(pipeline=FakePipeline(a_result([a_chunk(text="word " * 400)])))
    response = await ingest_url(IngestUrlRequest(url="https://quotes.test/1"), deps)
    assert response.chunk_preview[0].truncated is True


async def test_an_upload_is_not_fetched_and_the_upload_stage_is_not_timed():
    """No `latency_ms` on a stage where nothing was measured. Bytes arrived."""
    fetch = FakeFetch(a_fetch_result())
    deps = build_deps(fetch=fetch)
    upload = UploadedFile("notes.txt", b"a short note", "text/plain")
    response = await ingest_upload(upload, deps)
    assert fetch.calls == []
    assert response.ok is True
    assert stage_named(response, "upload").latency_ms is None


async def test_an_upload_registers_one_paused_synthetic_source():
    registry = FakeRegistry()
    deps = build_deps(registry=registry)
    upload = UploadedFile("notes.txt", b"a short note", "text/plain")
    await ingest_upload(upload, deps)
    await ingest_upload(upload, deps)
    assert len(registry.upserted) == 1
    assert registry.upserted[0].status is SourceStatus.PAUSED


async def test_an_uploaded_file_reaches_extraction_with_its_synthetic_url():
    extract = FakeExtract(a_doc())
    deps = build_deps(extract=extract)
    upload = UploadedFile("notes.txt", b"a short note", "text/plain")
    response = await ingest_upload(upload, deps)
    url, content_type = extract.calls[0]
    assert url.startswith("upload://upload.local/notes.txt/")
    assert content_type == "text/plain"
    assert response.source_url == url


def test_the_same_bytes_always_get_the_same_synthetic_url():
    """Deterministic, so re-uploading a file meets dedup instead of duplicating."""
    assert upload_url("a.pdf", b"same") == upload_url("a.pdf", b"same")
    assert upload_url("a.pdf", b"same") != upload_url("a.pdf", b"different")


def test_a_traversal_filename_cannot_escape_the_synthetic_url():
    url = upload_url("../../etc/passwd", b"payload")
    assert ".." not in url
    assert url.startswith("upload://upload.local/passwd/")


def test_an_empty_filename_still_produces_a_url():
    assert upload_url("", b"payload").startswith("upload://upload.local/file/")
