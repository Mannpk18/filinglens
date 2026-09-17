-- FilingLens database schema. Run with: psql $DATABASE_URL -f scripts/setup_db.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filing_chunks (
    id               BIGSERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL,
    fiscal_year      INT NOT NULL,
    form             TEXT NOT NULL,             -- '10-K' | '10-Q'
    accession_number TEXT NOT NULL,
    section          TEXT NOT NULL,              -- e.g. 'Item 1A. Risk Factors'
    chunk_index      INT NOT NULL,
    text             TEXT NOT NULL,
    embedding        vector(1024) NOT NULL,       -- matches BAAI/bge-large-en-v1.5 dim
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (accession_number, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_filing_chunks_ticker_year
    ON filing_chunks (ticker, fiscal_year);

-- IVFFlat index for approximate nearest-neighbor search. `lists` should be
-- roughly sqrt(row_count) — 100 is a reasonable default for a portfolio-scale
-- corpus (tens of thousands of chunks); retune once you know your row count.
CREATE INDEX IF NOT EXISTS idx_filing_chunks_embedding
    ON filing_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS eval_runs (
    id               BIGSERIAL PRIMARY KEY,
    run_timestamp    TIMESTAMPTZ NOT NULL DEFAULT now(),
    verifier_enabled BOOLEAN NOT NULL,
    pass_rate        FLOAT NOT NULL,
    numeric_accuracy FLOAT NOT NULL,
    avg_confidence   FLOAT NOT NULL,
    avg_latency_s    FLOAT NOT NULL,
    avg_retries      FLOAT NOT NULL,
    raw_results      JSONB NOT NULL
);
