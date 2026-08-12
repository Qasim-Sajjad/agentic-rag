# fetch

Retrieves raw bytes from a URL, escalating through progressively more expensive
strategies only when cheaper ones fail. Gives up on a defined threshold instead
of retrying forever.

## Contracts

```python
class FetchTier(IntEnum):
    STATIC = 1  # curl_cffi with TLS impersonation
    BROWSER = 2  # Playwright + Chromium
    STEALTH = 3  # Playwright + Camoufox
    UNLOCKER = 4  # managed third party API


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
    headers: dict[str, str] = {}


class FetchFailure(BaseModel):
    url: str
    reason: FailureReason
    last_tier: FetchTier
    attempts: int
    detail: str
    retry_after_seconds: float | None = None
```

`headers` is on the result because two decisions need it and neither can reach
back for it: `cf-mitigated` is a block signature, and 429 handling reads
`Retry-After`. `retry_after_seconds` is set only when a 429 exceeds the sleep
cap, and is what the worker uses as the requeue delay.

```python
class Fetcher(Protocol):
    tier: FetchTier

    async def fetch(self, url: str, timeout: float) -> FetchResult: ...
    async def close(self) -> None: ...
```

A fetcher returns a `FetchResult` for any HTTP response, including 4xx and 5xx,
and raises only when transport failed before a response existed. Deciding what
a status means belongs to the orchestrator, which is what keeps escalation
policy in one module instead of four.

Entry point: `async def fetch(url: str) -> FetchResult | FetchFailure`.
It never raises for expected failures. It returns `FetchFailure`. The one
exception is a URL whose domain is not in the registry, which raises
`UnknownSourceError`: that is a caller bug, not a fetch outcome.

## Source registry

Postgres. Two tables, split because config is human edited and state is machine
written. Splitting them means runtime state can be wiped and rebuilt without
losing crawl policy.

Owned by this module. `mcp` and `api` read from it, neither writes to it.

**One row per source, not per URL.** A source is a domain or a seed with a crawl
policy attached. Individual URLs live in the frontier queue. Putting policy per
URL would duplicate robots and rate config 100,000 times.

```sql
CREATE TABLE source (
    source_id           TEXT PRIMARY KEY,        -- "sec-edgar"
    domain              TEXT NOT NULL,
    seed_urls           JSONB NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
                        -- active | paused | unreachable | retired
    max_tier            SMALLINT NOT NULL DEFAULT 3,
    allow_unlocker      BOOLEAN NOT NULL DEFAULT FALSE,
    requests_per_second REAL NOT NULL DEFAULT 1.0,
    crawl_delay_seconds REAL,                    -- from robots.txt, overrides rps
    robots_txt          TEXT,
    robots_fetched_at   TIMESTAMPTZ,
    tos_note            TEXT,                    -- why max_tier is set as it is
    priority            SMALLINT NOT NULL DEFAULT 5,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE source_state (
    source_id            TEXT PRIMARY KEY REFERENCES source(source_id),
    circuit_state        TEXT NOT NULL DEFAULT 'closed',
    consecutive_failures INT NOT NULL DEFAULT 0,
    circuit_opened_at    TIMESTAMPTZ,
    circuit_open_seconds INT NOT NULL DEFAULT 1800,
    circuit_reopen_count SMALLINT NOT NULL DEFAULT 0,
    circuit_first_open_at TIMESTAMPTZ,   -- start of the 24 hour reopen window
    preferred_tier       SMALLINT NOT NULL DEFAULT 1,
    tier_learned_at      TIMESTAMPTZ,
    last_success_at      TIMESTAMPTZ,
    last_failure_at      TIMESTAMPTZ,
    last_failure_reason  TEXT,
    docs_indexed         INT NOT NULL DEFAULT 0,
    docs_failed          INT NOT NULL DEFAULT 0
);
```

`source_state` is the domain policy cache and the circuit breaker. They live
together because both are per source runtime state written by the fetcher on
every result.

### Seeding is manual, on purpose

Sources are declared in `config/sources.yaml` and loaded by
`python -m rag.fetch.bootstrap`.

```yaml
- source_id: sec-edgar
  domain: sec.gov
  seed_urls: ["https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"]
  max_tier: 1
  requests_per_second: 10
  tos_note: "Public data. Fair access policy requires a declared user agent."
```

URL discovery inside a source is automatic through sitemaps and link following.
Adding a new domain is not. A new domain entering the corpus is a legal
decision, not a crawler decision, and a link on page 40,000 should not be able
to auto enroll a site whose terms forbid automated access.

### Reads and writes

| Field | Written by | Read by |
|---|---|---|
| `source.*` | bootstrap, human edits | scheduler, fetch, api |
| `preferred_tier`, `tier_learned_at` | fetch, on success | fetch |
| `circuit_*`, `consecutive_failures` | fetch, on every result | fetch, mcp, api |
| `docs_indexed`, `docs_failed` | index pipeline | mcp, api |

`status` transitions to `unreachable` when `circuit_reopen_count` reaches 3
within 24 hours. That is the only automatic write to the `source` table.

## Escalation

Start at the tier stored in `source_state.preferred_tier`, defaulting to
`STATIC`.

Escalate to the next tier when any of these fire:

**Block signatures.** Status 403, 429, or 503. Response headers containing
`cf-mitigated`. Body containing a known challenge marker (configurable list,
seeded with "just a moment", `cf_chl_opt`, Datadome and Akamai markers).

**Emptiness heuristics.** Extracted text under `min_text_chars` (default 200).
Body contains an empty SPA root (`<div id="root"></div>`, `__NEXT_DATA__`)
with no rendered text. A `<noscript>` block is the only content. Emptiness is
an HTML judgement only. A PDF has no rendered text to count and is never
escalated for being short.

Do not escalate on 404 or on a clean 200 with real content.

**Do not escalate a 429 either.** It is listed as a block status above because
it is one, but a rate limit is the server saying "slower", not "prove you are a
browser". Escalating collects the same 429 two to ten seconds more slowly. A
429 sleeps or requeues at the tier it happened on, and the ladder stops there.

**Do not escalate a 5xx.** A broken server is broken at every tier. Retry in
place, then give up with `SERVER_ERROR`.

`UNLOCKER` is only attempted when the source registry sets
`allow_unlocker: true` for that domain. Default is false.

### Tier 4 is different in kind

Tiers 1 to 3 change how we present ourselves and let the site decide. Tier 4
pays a third party to get through a challenge on our behalf. That is a different
act, so it is gated twice and neither gate defaults on: a provider key must be
configured, and the source must set `allow_unlocker`.

Provider is ScrapingBee, configured under `fetch.unlocker`. The key comes from
`SCRAPINGBEE_API_KEY` in `.env`, never from a committed file. Swapping providers
is `unlocker.py` plus that config block; nothing above it knows which service
answered.

Two behaviours matter to the ladder:

- **Provider failures are ours, not the site's.** A rejected key or an exhausted
  balance is permanent and stops the ladder (`UnlockerNotConfiguredError`).
  Concurrency limits and provider outages are transient and retry
  (`FetchTransportError`). Retrying an empty balance just spends another request
  to learn the same thing.
- **The origin's status is reported, not the proxy's.** ScrapingBee returns its
  own 200 wrapping whatever the site said, so the fetcher reads
  `Spb-original-status` and reports that. Otherwise every block signature check
  downstream sees a 200 and concludes the page was fine.

Every tier 4 request is billable, so every one is logged at info with its url. A
crawl that quietly escalated should be visible in the log rather than on an
invoice.

## Domain policy cache

Stored in `source_state`, not a separate store. Fields: `preferred_tier`,
`tier_learned_at`. TTL from `settings.fetch.policy_cache_ttl_hours`
(default 168).

On a successful fetch at tier N, write N back as `preferred_tier`. On expiry,
reset to `STATIC` so a site that dropped its protection is not permanently
paying for a browser.

This cache is why ingestion is feasible at 100K docs. Escalation is discovered
once per domain, not once per URL.

## Frontier queue

Postgres table, worked with `SELECT ... FOR UPDATE SKIP LOCKED`. No separate
broker.

```sql
CREATE TABLE frontier (
    url_hash         TEXT PRIMARY KEY,       -- sha256 of canonical URL
    url              TEXT NOT NULL,          -- canonical form
    source_id        TEXT NOT NULL REFERENCES source(source_id),
    status           TEXT NOT NULL DEFAULT 'pending',
                     -- pending | leased | done | dead
    visible_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    priority         SMALLINT NOT NULL DEFAULT 5,
    attempts         SMALLINT NOT NULL DEFAULT 0,
    passes           SMALLINT NOT NULL DEFAULT 0,   -- exhausted scheduling passes
    last_pass_at     TIMESTAMPTZ,
    leased_by        TEXT,
    lease_expires_at TIMESTAMPTZ,
    discovered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX frontier_claimable ON frontier (priority, visible_at)
    WHERE status = 'pending';
```

Claim pattern:

```sql
UPDATE frontier SET status = 'leased', leased_by = $1,
       lease_expires_at = now() + interval '10 minutes', attempts = attempts + 1
WHERE url_hash IN (
    SELECT url_hash FROM frontier
    WHERE status = 'pending' AND visible_at <= now()
    ORDER BY priority, visible_at
    LIMIT $2 FOR UPDATE SKIP LOCKED)
RETURNING url, source_id;
```

`SKIP LOCKED` is what makes this safe across concurrent workers without a
broker. Without it two workers claim the same URL.

`visible_at` is the delayed visibility mechanism. A 429 sets it forward and
returns the row to `pending`, so the worker moves on instead of sleeping.

A sweeper returns rows whose `lease_expires_at` has passed to `pending`, which
recovers work from a crashed worker.

**Why Postgres and not Redis or SQS.** The queue is transactional with the
source registry and the dead letter store, so a claim, a failure write and a
circuit update are one transaction. It is durable by default and adds no
infrastructure to a local demo. The honest limit is a few thousand operations
per second, which is comfortable at 100K documents. At 1M this is the second
thing to move, after the browser pool, and the target would be Redis Streams
or SQS with Postgres kept for state.

## Dead letter store

Postgres. Written by `fetch` and by `extract` for `UNSUPPORTED_TYPE`. Read by
`mcp` and `api` for failure counts.

```sql
CREATE TABLE dead_letter (
    url_hash            TEXT PRIMARY KEY,
    url                 TEXT NOT NULL,
    source_id           TEXT NOT NULL REFERENCES source(source_id),
    reason              TEXT NOT NULL,      -- FailureReason
    stage               TEXT NOT NULL,      -- fetch | extract
    last_tier           SMALLINT,
    http_status         SMALLINT,
    attempts            SMALLINT NOT NULL,
    detail              TEXT,               -- observed MIME, exception, marker
    first_failed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_failed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    retry_eligible_after TIMESTAMPTZ        -- NULL means do not retry
);

CREATE INDEX dead_letter_by_source ON dead_letter (source_id, reason);
```

A dead letter entry is a decision, not a crash. Every row carries the reason
code that produced it, so `/ingest/status` can report "412 blocked, 89
unsupported type, 23 robots disallowed" rather than a single failure count.

`retry_eligible_after` allows a manual replay: set it to a timestamp and the
scheduler will reinsert the row into `frontier`. Nothing does this
automatically, because automatic retry of a permanently blocked URL is exactly
the loop this design exists to prevent.

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

That rule spans passes, so it lives in the worker rather than in `fetch()`.
`frontier.passes` counts exhausted passes and `give_up_pass_gap_hours` sets the
requeue delay between them. `fetch()` itself records the dead letter entry for
a terminal single-call failure; the worker records it for the give up case and
marks the row `dead`.

A source is marked unreachable when its circuit has reopened three times in
24 hours. The source registry sets `status: unreachable`. It is excluded from
scheduling until manually reset. `get_ingest_status` surfaces this.

The distinction that matters: rate limiting is temporary and produces a
requeue. Persistent blocking produces a dead letter entry. They are never
collapsed into a generic "failed".

## Legal boundaries

- `robots.txt` fetched once per domain, cached on the `source` row with a TTL,
  parsed with `protego`. A disallowed path returns `ROBOTS_DISALLOWED` without
  a request to that path. Following RFC 9309, a 4xx on robots.txt means allow
  all and a 5xx means disallow all.
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

## Module layout

```
fetch/
  types.py       enums and models, no behaviour
  protocols.py   Fetcher protocol and transport errors
  escalation.py  what a response means, the only place that decides
  backoff.py     jittered backoff and the Retry-After cap decision
  circuit.py     breaker transitions, pure functions over SourceState
  ratelimit.py   per domain token bucket
  robots.py      fetch, cache and evaluate robots.txt
  registry.py    source and source_state repositories
  frontier.py    queue, claimed with SKIP LOCKED
  deadletter.py  typed give up records
  static.py      tier 1, curl_cffi
  browser.py     tiers 2 and 3, shared browser pool
  unlocker.py    tier 4, ScrapingBee
  service.py     the orchestrator, written last
  worker.py      claims from the frontier, owns the give up rule
  factory.py     wiring
  bootstrap.py   loads config/sources.yaml
```

## Known gaps

- Tier 4 has no spend cap. Nothing counts credits or stops after N requests, so
  a misconfigured crawl over a large registered source could run up a bill. The
  per source `allow_unlocker` gate and the per request log are the only controls.
- Tier 4 is enabled on exactly one source, `file-examples-pdf`, whose host
  challenges all three self driven tiers. It is not a general default and should
  not become one.
- `Spb-original-status` is read defensively, falling back to the proxy's own
  status when absent. A provider that stops sending it would silently degrade
  block detection rather than fail loudly.
- Tier 1 sends the honest crawler user agent. Tiers 2 and 3 send the browser's
  own, because a browser that announces itself as a crawler is not rendering
  the page a browser would get. They carry an `X-Crawler-Contact` header
  instead, which is weaker than a matching user agent.
- No link discovery or sitemap parsing yet. The frontier is filled by
  `bootstrap` from seed URLs. Discovery inside a source arrives with `index`.
