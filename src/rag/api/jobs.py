"""Background ingest jobs, so a long document does not have to fit in a request.

A 500 page PDF is minutes of extraction and embedding. Held open as one HTTP
request that is a guaranteed client timeout, and the caller cannot tell a slow
success from a hang. A job id plus a poll makes the same work observable: the
request returns immediately, and the progress a caller polls for is the same
`Progress` sink the pipeline already reports to.

In memory on purpose, and bounded. Jobs describe work in flight, not a durable
record: the corpus itself is the durable artefact, and it is in Postgres and
Qdrant by the time a job finishes. A restart losing job history is acceptable,
losing a chunk is not. Recorded as a gap in `src/rag/api/SPEC.md`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rag.api.models import (
    IngestJobStatus,
    IngestTraceResponse,
    ProgressRow,
)
from rag.log import get_logger
from rag.progress import Progress

log = get_logger(__name__)

#: Enough to watch a demo and look back at what just ran, not a history table.
DEFAULT_MAX_JOBS = 50


@dataclass
class ProgressLine:
    """One stage's latest position. Replaced in place, never appended twice.

    Embedding reports per batch, which on a large document is hundreds of calls
    for one stage. A caller wants "embed 640/2500", not 78 rows of history.
    """

    stage: str
    done: int
    total: int
    detail: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class IngestJob:
    job_id: str
    kind: str  # "url" or "file"
    label: str  # the url, or the filename
    status: str = "running"  # running | done | failed
    progress: OrderedDict[str, ProgressLine] = field(default_factory=OrderedDict)
    result: IngestTraceResponse | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    #: Held so the task is not garbage collected mid flight. asyncio keeps only
    #: a weak reference to a bare `create_task` result.
    task: asyncio.Task[None] | None = None

    def record(self, stage: str, done: int, total: int, detail: str = "") -> None:
        """The `Progress` shape. Ordered by first sighting, so reading the dict
        gives the stages in the order the pipeline reached them."""
        self.progress[stage] = ProgressLine(stage, done, total, detail)

    def finish(self, result: IngestTraceResponse) -> None:
        self.result = result
        self.status = "done"
        self.finished_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        """An unexpected exception. A refused fetch is not this: that comes back
        as a normal result with `ok: false` and a reason in the trace."""
        self.error = error
        self.status = "failed"
        self.finished_at = datetime.now(UTC)

    @property
    def elapsed_ms(self) -> int:
        end = self.finished_at or datetime.now(UTC)
        return round((end - self.started_at).total_seconds() * 1000)


class JobStore:
    """Bounded, insertion ordered. The oldest job is evicted, not the newest."""

    def __init__(self, max_jobs: int = DEFAULT_MAX_JOBS) -> None:
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._max_jobs = max_jobs

    def create(self, kind: str, label: str) -> IngestJob:
        job = IngestJob(job_id=uuid.uuid4().hex[:16], kind=kind, label=label)
        self._jobs[job.job_id] = job
        while len(self._jobs) > self._max_jobs:
            evicted, _ = self._jobs.popitem(last=False)
            log.info("ingest job evicted", job_id=evicted)
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[IngestJob]:
        """Newest first, which is the order a caller wants to read them in."""
        return list(reversed(list(self._jobs.values())))[:limit]


#: An ingest that has not started yet, waiting only for somewhere to report to.
IngestWork = Callable[[Progress], Awaitable[IngestTraceResponse]]


def launch(store: JobStore, kind: str, label: str, work: IngestWork) -> IngestJob:
    """Start `work` in the background and hand back the job to poll.

    The task reference is kept on the job. Without it asyncio holds only a weak
    reference and a long ingest can be collected mid flight.
    """
    job = store.create(kind, label)
    job.task = asyncio.create_task(_run(job, work))
    log.info("ingest job started", job_id=job.job_id, kind=kind, label=label)
    return job


async def _run(job: IngestJob, work: IngestWork) -> None:
    """The top of a task, so nothing above it can catch. An escaped exception
    here would leave the job reading `running` forever, which is worse than a
    caller seeing why it failed."""
    try:
        job.finish(await work(job.record))
    except Exception as exc:  # noqa: BLE001
        job.fail(f"{type(exc).__name__}: {exc}")
        log.error("ingest job failed", job_id=job.job_id, error=str(exc))


def status_of(job: IngestJob) -> IngestJobStatus:
    return IngestJobStatus(
        job_id=job.job_id,
        kind=job.kind,
        label=job.label,
        status=job.status,
        elapsed_ms=job.elapsed_ms,
        progress=[
            ProgressRow(
                stage=line.stage, done=line.done, total=line.total, detail=line.detail
            )
            for line in job.progress.values()
        ],
        result=job.result,
        error=job.error,
    )
