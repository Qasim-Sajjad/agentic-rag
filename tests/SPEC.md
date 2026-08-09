# tests

Correctness only. Quality measurement lives in `evals/` and `notebooks/`.

## Layout

```
tests/
  unit/            no network, no containers
  integration/     against the local fixture server only
  fixtures/
    server.py      the fixture server
    pages/         saved HTML, PDFs
```

## Fixture server

A small FastAPI app that makes the fetch ladder deterministically testable
offline. This matters twice: it removes flaky network dependence from CI, and it
lets you demonstrate escalation live without depending on a real site.

Endpoints:

| Path | Behaviour |
|---|---|
| `/static` | Plain HTML with real content |
| `/js-only` | Empty SPA root, content injected by script |
| `/rate-limited` | 429 with `Retry-After` |
| `/challenge` | Interstitial with a challenge marker |
| `/flaky` | Fails twice, then succeeds |
| `/always-500` | Permanent server error |
| `/robots-blocked` | Disallowed in the served robots.txt |
| `/doc.pdf` | A PDF served with the correct content type |

## Coverage expectation

Every failure path named in a module SPEC has a test. Happy paths are the easy
half and are not where this system will break.

Priority order if time runs short:

1. Fetch ladder escalation and give up
2. Injection defence suite
3. Chunker invariants, tables never split without a repeated header
4. Content routing on wrong extensions and magic bytes
5. Agent branching, tested on `assess` as a pure function
6. API contract shapes

## Rules

- Unit tests never touch the network
- No sleeps. Fake the clock
- Every test asserts one behaviour. A test needing a section comment is two tests
