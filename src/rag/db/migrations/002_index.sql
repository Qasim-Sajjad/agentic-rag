-- Documents and chunks. Schema is defined in src/rag/index/SPEC.md.
-- Qdrant is a derived index. These two tables plus the CanonicalDoc blobs are
-- what make a model swap a backfill instead of a re-scrape.

CREATE TABLE IF NOT EXISTS document (
    doc_id            TEXT PRIMARY KEY,
    source_id         TEXT NOT NULL REFERENCES source(source_id) ON DELETE CASCADE,
    source_url        TEXT NOT NULL,
    title             TEXT,
    published_at      DATE,
    language          TEXT,
    doc_type          TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    canonical_doc_key TEXT NOT NULL,
    fetch_tier        SMALLINT NOT NULL,
    extractor_name    TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS document_content_hash ON document (content_hash);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    chunk_index         INT NOT NULL,
    text                TEXT NOT NULL,
    embed_text          TEXT NOT NULL,
    section_path        JSONB NOT NULL,
    page_no             INT,
    is_table            BOOLEAN NOT NULL DEFAULT FALSE,
    token_count         INT NOT NULL,
    chunk_hash          TEXT NOT NULL,
    chunker_version     TEXT NOT NULL,
    embed_model_version TEXT,
    embedded_at         TIMESTAMPTZ,
    tenant_id           TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunk_by_doc ON chunk (doc_id);
CREATE INDEX IF NOT EXISTS chunk_dedup ON chunk (chunk_hash);
CREATE INDEX IF NOT EXISTS chunk_needs_embed ON chunk (embed_model_version)
    WHERE embed_model_version IS NULL;

-- Near duplicate detection state, so SimHash survives a restart.
CREATE TABLE IF NOT EXISTS document_simhash (
    doc_id    TEXT PRIMARY KEY REFERENCES document(doc_id) ON DELETE CASCADE,
    simhash   BIGINT NOT NULL
);
