# AI tool usage log

Fill this in as you go, not at the end. Reconstructing it after 96 hours
produces exactly the vague account the assessment says it does not want.

One entry per session. Keep it honest and specific.

## Format

```
### YYYY-MM-DD, session N, <module>
Tool: <Claude Code / Copilot / other>
Asked for: <what you actually prompted, one or two lines>
Kept: <what went in unchanged>
Corrected: <what was wrong and what you changed>
Rewrote: <what you threw away and wrote yourself>
```

## Entries

### 2026-XX-XX, session 1, architecture
Tool:
Asked for:
Kept:
Corrected:
Rewrote:

### 2026-08-09, session 2, phase 0 foundation
Tool: Claude Code (Opus 5)
Asked for: Read CLAUDE.md, BUILD_ORDER, ARCHITECTURE and tests/SPEC.md, then
build phase 0 only: pyproject deps, a pydantic-settings loader over
config/settings.yaml with env override, structlog setup, and the fixture server
with all 8 endpoints. Plan and spec ambiguities first, no phase 1.
Kept: The settings model shape (one frozen section per YAML block, extra keys
forbidden), the source ordering that puts env above YAML, the structlog
ProcessorFormatter wiring, the fixture server layout with pages read from
tests/fixtures/pages/, and the hand written minimal PDF with a real text layer.
Corrected: Six things the SPEC left open, decided rather than guessed. The
loader moved from config/settings.py to src/rag/config/settings.py, since
config/ is not importable under a src layout and mypy src would never see it,
BUILD_ORDER updated in the same commit. Fallback to settings.example.yaml added
because config/settings.yaml is gitignored, so a clean clone had no config at
all. /flaky returns 500 rather than 503, because 503 is a block signature in the
fetch SPEC and would trigger escalation instead of the same tier retry the test
asks for. /rate-limited defaults Retry-After to 600, over the 300s cap, so the
bare URL exercises the requeue path, with a query param for the honour path.
/challenge decides by user agent or an x-fixture-tier header, since the spec
requires tier 2 to fail and tier 3 to pass and gives no mechanism, recorded in
known gaps. /robots.txt, /__stats and /__reset added beyond the 8 listed
endpoints, the first because /robots-blocked is defined against it, the other
two because the "zero HTTP requests" assertion and /flaky isolation need them.
Rewrote: Two test failures were mine and were real bugs in the code, not the
tests. The hit counting middleware counted /__stats itself, so a reset assertion
could never pass, fixed by excluding the control paths from the counters, which
is also the honest behaviour. The stdlib bridge test used a uvicorn logger name,
which uvicorn's own dictConfig had already set to propagate: False from the
integration fixture, so it was asserting on the wrong logger.
Verified: ruff check . clean, mypy src clean on 12 files, 42 tests pass.

## Summary for the design doc

Write this at the end, from the entries above. Three or four sentences covering
where AI was used, what needed correction most often, and which decisions were
yours rather than the tool's. Chunking approach, tool scoping, agent boundaries
and failure handling are the decisions being evaluated. Be specific about which
of those you made.
