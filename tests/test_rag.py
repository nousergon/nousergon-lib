"""Tests for the rag submodule.

The rag submodule consolidates code that used to live in both
alpha-engine-research/rag/ and alpha-engine-data/rag/. These tests verify
that imports work and re-exports resolve correctly. Live database
operations are out of scope here — those are integration-tested in the
consumer repos against a real Neon pgvector instance.
"""

from __future__ import annotations

import importlib


def test_top_level_imports_resolve():
    """All advertised re-exports should be importable from the top level."""
    from nousergon_lib.rag import (
        coerce_embedding,
        document_exists,
        embed_texts,
        get_connection,
        ingest_document,
        is_available,
        retrieve,
    )

    # Verify the re-exports are callables (or at minimum, attributes — we
    # don't invoke them here because that requires a live database)
    for name, obj in [
        ("coerce_embedding", coerce_embedding),
        ("get_connection", get_connection),
        ("is_available", is_available),
        ("embed_texts", embed_texts),
        ("retrieve", retrieve),
        ("ingest_document", ingest_document),
        ("document_exists", document_exists),
    ]:
        assert callable(obj), f"{name} should be callable"


def test_submodules_importable():
    """Each submodule of nousergon_lib.rag should import cleanly."""
    for sub in ("db", "embeddings", "retrieval"):
        mod = importlib.import_module(f"nousergon_lib.rag.{sub}")
        assert mod is not None


def test_schema_sql_packaged():
    """schema.sql ships as package data so consumers can locate it."""
    import importlib.resources as ir

    files = ir.files("nousergon_lib.rag")
    schema_path = files / "schema.sql"
    assert schema_path.is_file(), "schema.sql should be packaged with nousergon_lib.rag"

    content = schema_path.read_text()
    assert "CREATE" in content.upper(), "schema.sql should contain DDL"


def test_schema_sql_declares_hybrid_retrieval_surface():
    """Hybrid retrieval (PR 1 of the BM25 + vector arc) requires
    ``content_tsv`` + a GIN index on it. Pin both in schema.sql so a
    future schema rewrite that drops them fails here instead of
    silently regressing the keyword-side of retrieval to a sequential
    scan.
    """
    import importlib.resources as ir

    schema = (ir.files("nousergon_lib.rag") / "schema.sql").read_text()
    assert "content_tsv" in schema, (
        "schema.sql missing content_tsv generated column for hybrid retrieval"
    )
    assert "to_tsvector('english', content)" in schema, (
        "content_tsv must use the english FTS config (matches Voyage's "
        "single-language English embeddings)"
    )
    assert "GENERATED ALWAYS" in schema and "STORED" in schema, (
        "content_tsv must be a STORED generated column so existing rows "
        "auto-populate from content"
    )
    assert "USING gin (content_tsv)" in schema, (
        "GIN index on content_tsv missing — keyword retrieval would fall "
        "back to a sequential scan"
    )


def test_migration_0001_packaged_and_idempotent():
    """0001_content_tsv.sql ships as package data and uses idempotent
    DDL so re-runs against an already-migrated DB are no-ops.
    """
    import importlib.resources as ir

    files = ir.files("nousergon_lib.rag")
    migration = files / "migrations" / "0001_content_tsv.sql"
    assert migration.is_file(), (
        "migrations/0001_content_tsv.sql should ship as package data "
        "(check pyproject.toml::tool.setuptools.package-data)"
    )

    content = migration.read_text()
    # Idempotency markers — re-running the migration must be a no-op.
    assert "ADD COLUMN IF NOT EXISTS content_tsv" in content, (
        "migration must use ADD COLUMN IF NOT EXISTS for idempotency"
    )
    assert "CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin" in content, (
        "migration must use CREATE INDEX IF NOT EXISTS for idempotency"
    )


def test_schema_sql_declares_news_dedup_key():
    """config#2957: 'news' must dedup on external_id, not just
    (ticker, doc_type, filed_date, source) — else same-day articles for
    one (ticker, source) collapse onto a single row. Pin the partial
    unique indexes in schema.sql so a future schema rewrite that drops
    them fails here instead of silently reintroducing the collapse.
    """
    import importlib.resources as ir

    schema = (ir.files("nousergon_lib.rag") / "schema.sql").read_text()
    assert "external_id" in schema, (
        "schema.sql missing external_id column on rag.documents"
    )
    assert "documents_unique_non_news" in schema and "documents_unique_news" in schema, (
        "schema.sql missing the partial unique indexes that replace the "
        "blanket UNIQUE(ticker, doc_type, filed_date, source) constraint"
    )
    assert "WHERE doc_type <> 'news'" in schema, (
        "non-news partial index must exclude external_id from its key so "
        "NULL-vs-NULL doesn't silently stop catching non-news duplicates"
    )
    assert "WHERE doc_type = 'news'" in schema, (
        "news partial index must key on external_id for per-article dedup"
    )


def test_migration_0002_packaged_and_idempotent():
    """0002_news_external_id.sql ships as package data and uses
    idempotent DDL so re-runs against an already-migrated DB are no-ops.
    """
    import importlib.resources as ir

    files = ir.files("nousergon_lib.rag")
    migration = files / "migrations" / "0002_news_external_id.sql"
    assert migration.is_file(), (
        "migrations/0002_news_external_id.sql should ship as package data "
        "(check pyproject.toml::tool.setuptools.package-data)"
    )

    content = migration.read_text()
    assert "ADD COLUMN IF NOT EXISTS external_id" in content, (
        "migration must use ADD COLUMN IF NOT EXISTS for idempotency"
    )
    assert "DROP CONSTRAINT IF EXISTS" in content, (
        "migration must use DROP CONSTRAINT IF EXISTS for idempotency"
    )
    assert "CREATE UNIQUE INDEX IF NOT EXISTS documents_unique_non_news" in content, (
        "migration must use CREATE INDEX IF NOT EXISTS for idempotency"
    )
    assert "CREATE UNIQUE INDEX IF NOT EXISTS documents_unique_news" in content, (
        "migration must use CREATE INDEX IF NOT EXISTS for idempotency"
    )


def test_document_exists_accepts_external_id():
    """Signature guard: document_exists/ingest_document must accept the
    new external_id kwarg (config#2957) so news ingestion can pass a
    per-article identity. Live DB behavior is integration-tested in the
    consumer repo (nousergon-data) via injected fakes."""
    import inspect

    from nousergon_lib.rag import document_exists, ingest_document

    assert "external_id" in inspect.signature(document_exists).parameters
    assert "external_id" in inspect.signature(ingest_document).parameters


def test_is_available_safe_when_db_unreachable(monkeypatch):
    """is_available() must never raise — it's a probe, not an assertion."""
    from nousergon_lib.rag import is_available

    # Force RAG_DATABASE_URL to a guaranteed-unreachable target. The probe
    # should swallow the connection error and return False.
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://nope:nope@localhost:1/nope")
    result = is_available()
    assert result is False


def test_no_bare_rag_imports_in_lib():
    """Inside the lib, every `rag.*` import must be relative or fully qualified.

    The v0.3.0 RAG consolidation moved code from consumer-side `rag/` packages
    into `nousergon_lib.rag`, but four deferred imports inside retrieval.py
    were left as bare `from rag.X import ...`. They worked when called from a
    consumer that had its own top-level `rag/` package on sys.path, but blew
    up on the spot orchestrator (alpha-engine-data) where the package was
    already migrated out, only firing when the dedup branch was hit during a
    real ingestion run. Catch the class statically — walk every module file
    in the rag submodule and assert no `^\\s*(from|import)\\s+rag\\.` lines.
    """
    import importlib.resources as ir
    import re

    pattern = re.compile(r"^\s*(from|import)\s+rag\.", re.MULTILINE)
    rag_files = ir.files("nousergon_lib.rag")
    offenders: list[str] = []
    for entry in rag_files.iterdir():
        if entry.name.endswith(".py"):
            text = entry.read_text()
            if pattern.search(text):
                offenders.append(entry.name)
    assert not offenders, (
        f"Bare `rag.*` imports found in {offenders}; use relative imports "
        "(`from .db import …`) inside the lib package"
    )
