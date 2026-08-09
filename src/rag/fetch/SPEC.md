# fetch

Retrieves raw bytes from a URL, escalating through progressively more expensive
strategies only when cheaper ones fail. Gives up on a defined threshold instead
of retrying forever.

## Contracts

```python
class FetchTier(IntEnum):
    STATIC = 1       # curl_cffi with TLS impersonation
    BROWSER = 2      # Playwright + Chromium
    STEALTH = 3      # Playwright + Camoufox
    UNLOCKER = 4     # managed third party API

class FailureReason(StrEnum):
    BLOCKED_PERSISTENT = "blocked_persistent"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ROBOTS_DISALLOWED = "robots_disallowed"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    UNSUPPORTED_TYPE = "unsupported_type"

class FetchResult(BaseModel):
    url: str
    final_url: str
    status: int
    content: bytes
    content_type: str
    tier_used: FetchTier
    attempts: int
    fetched_at: datetime

class FetchFailure(BaseModel):
    url: str
    reason: FailureReason
    last_tier: FetchTier
    attempts: int
    detail: str
```

```python
class Fetcher(Protocol):
    tier: FetchTier
    async def fetch(self, url: str, timeout: float) -> FetchResult: ...
```

Entry point: `async def fetch(url: str) -> FetchResult | FetchFailure`.
It never raises for expected failures. It returns `FetchFailure`.

## Escalation

Start at the tier stored in the domain policy cache, defaulting to `STATIC`.

Escalate to the next tier when any of these fire:

**Block signatures.** Status 403, 429, or 503. Response headers containing
`cf-mitigated`. Body containing a known challenge marker (configurable list,
seeded with "just a moment", `cf_chl_opt`, Datadome and Akamai markers).

**Emptiness heuristics.** Extracted text under `min_text_chars` (default 200).
Body contains an empty SPA root (`<div id="root"></div>`, `__NEXT_DATA__`)
with no rendered text. A `<noscript>` block is the only content.

Do not escalate on 404 or on a clean 200 with real content.

`UNLOCKER` is only attempted when the source registry sets
`allow_unlocker: true` for that domain. Default is false.

## Domain policy cache

Keyed by registrable domain. Fields: `preferred_tier`, `learned_at`,
`ttl_hours` (default 168), `sample_count`.

On a successful fetch at tier N, write N back as `preferred_tier`. On expiry,
reset to `STATIC` so a site that dropped its protection is not permanently
paying for a browser.

This cache is why ingestion is feasible at 100K docs. Escalation is discovered
once per domain, not once per URL.

## Backoff

Per tier: max 3 attempts. Exponential with full jitter, base 2s, cap 60s.

On 429, honour `Retry-After` if present and under `max_retry_after` (default
300s). If it exceeds the cap, do not sleep. Return `RATE_LIMITED` and requeue
the URL with a delayed visibility timestamp.

Never block a worker on a long sleep. Requeue instead.

## Circuit breaker

Per domain. Three states.

- `closed`: normal.
- `open`: after `failure_threshold` consecutive failures (default 5). All
  fetches for that domain return `FetchFailure` immediately without a request.
  Opens for `open_duration` (default 30 min).
- `half_open`: after the duration, allow one probe. Success closes the circuit.
  Failure reopens it with doubled duration, capped at 6 hours.

## Give up

A URL is permanently unreachable when the highest allowed tier has failed all
its attempts twice, across two separate scheduling passes at least one hour
apart. Write a `FetchFailure` to the dead letter store with the reason.

A source is marked unreachable when its circuit has reopened three times in
24 hours. The source registry sets `status: unreachable`. It is excluded from
scheduling until manually reset. `get_ingest_status` surfaces this.

The distinction that matters: rate limiting is temporary and produces a
requeue. Persistent blocking produces a dead letter entry. They are never
collapsed into a generic "failed".

## Legal boundaries

- `robots.txt` fetched once per domain, cached with TTL, parsed with `protego`.
  A disallowed path returns `ROBOTS_DISALLOWED` without a request.
- `Crawl-delay` is honoured. Where absent, the per domain token bucket applies
  a default of 1 request per second.
- The source registry carries a `tos_note` field. Domains whose terms forbid
  automated access are not seeded, and `allow_unlocker` stays false.
- User agent is honest and identifies the crawler with a contact URL.

We do not solve CAPTCHAs and do not attempt to defeat a specific vendor's
protection. Tier 3 exists to render pages the way a normal browser would.
When a site clearly does not want automated access, the source is dropped.

## Concurrency

`asyncio` with a global semaphore plus a per domain token bucket. Browser tiers
use a shared Playwright pool with `browser_pool_size` contexts, acquired for
the duration of one page load. Never one browser per URL.

## Tests

Against `tests/fixtures/server.py`, which serves:
`/static`, `/js-only`, `/rate-limited` (429 with Retry-After),
`/challenge`, `/flaky` (fails twice then succeeds), `/always-500`,
`/robots-blocked`, `/doc.pdf`.

Required cases:
- `/static` resolves at tier 1 without escalating
- `/js-only` escalates to tier 2 and returns rendered text
- `/challenge` escalates to tier 3
- `/rate-limited` requeues rather than sleeping past the cap
- `/flaky` succeeds on the third attempt
- `/always-500` gives up with `SERVER_ERROR` and lands in the dead letter store
- `/robots-blocked` returns `ROBOTS_DISALLOWED` with zero HTTP requests made
- Five consecutive failures open the circuit, and the next call makes no request
- Policy cache: second URL on a tier 2 domain starts at tier 2

## Known gaps

Tier 4 is an interface with a single stub implementation. No paid unlocker is
wired up. Swapping in Zyte or Bright Data is a config change plus one class.
