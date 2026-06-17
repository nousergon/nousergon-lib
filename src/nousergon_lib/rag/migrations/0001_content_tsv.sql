-- Migration 0001: add content_tsv (tsvector) + GIN index to rag.chunks
-- Companion to alpha_engine_lib v0.5.7 schema update for hybrid retrieval.
--
-- Idempotent: ``ADD COLUMN IF NOT EXISTS`` + ``CREATE INDEX IF NOT EXISTS``.
-- Safe to run repeatedly. Re-running is a no-op once both objects exist.
--
-- Usage:
--   psql "$RAG_DATABASE_URL" -f migrations/0001_content_tsv.sql
--
-- What this does:
-- - Adds ``content_tsv`` STORED generated column to ``rag.chunks``.
--   The generated expression is ``to_tsvector('english', content)``;
--   PostgreSQL rewrites the table to populate the new column for every
--   existing row. On the current corpus scale (Neon free tier, low
--   thousands of chunks) this is seconds.
-- - Creates a GIN index on ``content_tsv`` for fast Full-Text Search
--   (FTS) lookups. This is the keyword-side companion to the existing
--   HNSW index on ``embedding``.
--
-- Why STORED rather than VIRTUAL:
-- - PostgreSQL's STORED is the only generated-column flavor supported
--   today. VIRTUAL is reserved for a future major version.
-- - Even when VIRTUAL lands, indexing it would require REFRESH or an
--   IMMUTABLE wrapper expression — STORED is simpler.
--
-- Locking surface:
-- - ``ALTER TABLE … ADD COLUMN GENERATED … STORED`` rewrites the table
--   under an ACCESS EXCLUSIVE lock for the duration of the rewrite. On
--   Neon at the current corpus size this is brief (low single-digit
--   seconds). At ≥1M chunks consider partitioned rollout instead.
-- - ``CREATE INDEX`` (without CONCURRENTLY) holds a SHARE lock that
--   blocks writes but not reads. Same scale caveat applies.

ALTER TABLE rag.chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON rag.chunks USING gin (content_tsv);
