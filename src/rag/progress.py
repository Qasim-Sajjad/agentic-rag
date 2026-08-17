"""Progress reporting for long running pipeline stages.

A 500 page PDF is minutes of extraction and embedding. Reporting only the final
stage timings tells a caller what happened after it stopped mattering: what a
user watching a spinner needs is which stage is running and how far through it
is, while it is still running.

Deliberately a plain callable rather than an event bus or a queue. The pipeline
should not know whether the other end is an HTTP poll, a log line, or nothing at
all, and `_silent` means a caller that does not care passes nothing.

Lives at the package root because both `extract` and `index` report progress,
and neither may import the other.
"""

from __future__ import annotations

from typing import Protocol


class Progress(Protocol):
    """Called as a stage advances. Never raises, never blocks.

    `done` and `total` are in whatever unit the stage counts: page ranges,
    chunks, embedding batches. `total` is 0 when the stage cannot know it in
    advance, which a caller renders as indeterminate rather than as 0%.
    """

    def __call__(self, stage: str, done: int, total: int, detail: str = "") -> None: ...


def silent(stage: str, done: int, total: int, detail: str = "") -> None:
    """The default. Costs one call and discards it, so the pipeline needs no
    branch on whether anyone is listening."""
    return None
