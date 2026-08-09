-- Source registry, frontier queue and dead letter store.
-- Schema is defined in src/rag/fetch/SPEC.md. Keep the two in step.

CREATE TABLE IF NOT EXISTS source (
    source_id           TEXT PRIMARY KEY,
    domain              TEXT NOT NULL,
    seed_urls           JSONB NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    max_tier            SMALLINT NOT NULL DEFAULT 3,
    allow_unlocker      BOOLEAN NOT NULL DEFAULT FALSE,
    requests_per_second REAL NOT NULL DEFAULT 1.0,
    crawl_delay_seconds REAL,
    robots_txt          TEXT,
    robots_fetched_at   TIMESTAMPTZ,
    tos_note            TEXT,
    priority            SMALLINT NOT NULL DEFAULT 5,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS source_domain ON source (domain);

CREATE TABLE IF NOT EXISTS source_state (
    source_id            TEXT PRIMARY KEY REFERENCES source(source_id) ON DELETE CASCADE,
    circuit_state        TEXT NOT NULL DEFAULT 'closed',
    consecutive_failures INT NOT NULL DEFAULT 0,
    circuit_opened_at    TIMESTAMPTZ,
    circuit_open_seconds INT NOT NULL DEFAULT 1800,
    circuit_reopen_count SMALLINT NOT NULL DEFAULT 0,
    circuit_first_open_at TIMESTAMPTZ,
    preferred_tier       SMALLINT NOT NULL DEFAULT 1,
    tier_learned_at      TIMESTAMPTZ,
    last_success_at      TIMESTAMPTZ,
    last_failure_at      TIMESTAMPTZ,
    last_failure_reason  TEXT,
    docs_indexed         INT NOT NULL DEFAULT 0,
    docs_failed          INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS frontier (
    url_hash         TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    source_id        TEXT NOT NULL REFERENCES source(source_id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'pending',
    visible_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    priority         SMALLINT NOT NULL DEFAULT 5,
    attempts         SMALLINT NOT NULL DEFAULT 0,
    passes           SMALLINT NOT NULL DEFAULT 0,
    last_pass_at     TIMESTAMPTZ,
    leased_by        TEXT,
    lease_expires_at TIMESTAMPTZ,
    discovered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS frontier_claimable ON frontier (priority, visible_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS dead_letter (
    url_hash             TEXT PRIMARY KEY,
    url                  TEXT NOT NULL,
    source_id            TEXT NOT NULL REFERENCES source(source_id) ON DELETE CASCADE,
    reason               TEXT NOT NULL,
    stage                TEXT NOT NULL,
    last_tier            SMALLINT,
    http_status          SMALLINT,
    attempts             SMALLINT NOT NULL,
    detail               TEXT,
    first_failed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_failed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    retry_eligible_after TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS dead_letter_by_source ON dead_letter (source_id, reason);
