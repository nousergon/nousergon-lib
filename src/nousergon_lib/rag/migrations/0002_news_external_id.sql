-- Migration 0002: news RAG dedup key fix (alpha-engine-config#2957).
--
-- The UNIQUE(ticker, doc_type, filed_date, source) constraint has no
-- per-article identity for news: only the FIRST article per
-- (ticker, source, day) was ever stored, every subsequent distinct
-- article that day silently hit the constraint and was dropped as
-- "already exists".
--
-- Idempotent: ADD COLUMN IF NOT EXISTS + DROP CONSTRAINT IF EXISTS +
-- CREATE INDEX IF NOT EXISTS. Safe to run repeatedly.
--
-- Usage:
--   psql "$RAG_DATABASE_URL" -f migrations/0002_news_external_id.sql
--
-- What this does:
-- - Adds a nullable `external_id` column to rag.documents — a stable
--   per-article identity (news: the aggregator's content fingerprint;
--   every other doc_type: left NULL, behavior unchanged).
-- - Drops the single blanket UNIQUE(ticker, doc_type, filed_date,
--   source) constraint and replaces it with two PARTIAL unique
--   indexes:
--     * non-news doc_types keep the EXACT original 4-column key
--       (external_id deliberately excluded — see schema.sql comment
--       for why including it there would silently weaken the
--       existing guarantee instead of preserving it).
--     * news additionally keys on external_id, so distinct same-day
--       articles for one (ticker, source) now both persist, while
--       re-ingesting the SAME article (same external_id) still dedups.
-- - Existing news rows are NOT backfilled with a real external_id —
--   historical under-coverage from the collapsed key is not
--   recoverable (out of scope per the issue). They keep external_id
--   NULL, which is harmless: NULL never equals NULL under uniqueness
--   semantics, so old rows never block new inserts under the new
--   partial index.
--
-- Locking surface:
-- - ADD COLUMN (nullable, no default) is metadata-only in PostgreSQL —
--   no table rewrite, near-instant even at scale.
-- - DROP CONSTRAINT + CREATE INDEX (non-CONCURRENTLY, matching the
--   existing 0001 migration's precedent) briefly hold locks; on the
--   current corpus size (low thousands of document rows) this is
--   sub-second.
--
-- Constraint name: PostgreSQL auto-names an inline UNIQUE(...) table
-- constraint declared without an explicit CONSTRAINT clause as
-- ``<table>_<col1>_<col2>_..._key`` — verify with:
--   \d rag.documents
-- against your instance before relying on the literal name below if
-- this table was ever manually altered outside schema.sql.

ALTER TABLE rag.documents
    ADD COLUMN IF NOT EXISTS external_id TEXT;

ALTER TABLE rag.documents
    DROP CONSTRAINT IF EXISTS documents_ticker_doc_type_filed_date_source_key;

CREATE UNIQUE INDEX IF NOT EXISTS documents_unique_non_news
    ON rag.documents (ticker, doc_type, filed_date, source)
    WHERE doc_type <> 'news';

CREATE UNIQUE INDEX IF NOT EXISTS documents_unique_news
    ON rag.documents (ticker, doc_type, filed_date, source, external_id)
    WHERE doc_type = 'news';
