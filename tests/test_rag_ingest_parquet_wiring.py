"""ingest_document -> parquet_mirror wiring (config#2958).

Neon is mocked (this repo's established fake-cursor pattern, see
test_rag_retrieval_hybrid.py); the parquet mirror call itself is asserted
via patch, not exercised end to end (that's test_rag_parquet_mirror.py's
job) — this file only proves ingest_document calls it with the right
arguments, and that it never blocks/fails the Neon path.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch


def _mock_connection(returned_doc_id="11111111-1111-1111-1111-111111111111"):
    cur = MagicMock()
    cur.fetchone.return_value = (returned_doc_id,)
    cur.mogrify.side_effect = lambda sql, args: b"mogrified"
    cm = MagicMock()
    cm.__enter__ = lambda self: cur
    cm.__exit__ = lambda self, *a: None

    conn_cm = MagicMock()
    conn_cm.__enter__ = lambda self: MagicMock(cursor=lambda: cm)
    conn_cm.__exit__ = lambda self, *a: None
    return conn_cm


def test_ingest_document_calls_parquet_mirror_by_default():
    from nousergon_lib.rag.retrieval import ingest_document

    chunks = [{"content": "hello", "section_label": None, "embedding": [0.1] * 512}]

    with patch("nousergon_lib.rag.retrieval.document_exists", return_value=False), \
         patch("nousergon_lib.rag.db.get_connection", return_value=_mock_connection()), \
         patch("nousergon_lib.rag.parquet_mirror.mirror_document_to_parquet") as mirror_mock:
        doc_id = ingest_document(
            ticker="AAPL", sector="Technology", doc_type="10-K", source="sec_edgar",
            filed_date=date(2026, 7, 1), title="t", url="u", chunks=chunks,
        )

    assert doc_id == "11111111-1111-1111-1111-111111111111"
    mirror_mock.assert_called_once()
    _, kwargs = mirror_mock.call_args
    assert kwargs["document_id"] == doc_id
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["chunks"] == chunks


def test_ingest_document_skips_mirror_when_disabled():
    from nousergon_lib.rag.retrieval import ingest_document

    chunks = [{"content": "hello", "section_label": None, "embedding": [0.1] * 512}]

    with patch("nousergon_lib.rag.retrieval.document_exists", return_value=False), \
         patch("nousergon_lib.rag.db.get_connection", return_value=_mock_connection()), \
         patch("nousergon_lib.rag.parquet_mirror.mirror_document_to_parquet") as mirror_mock:
        ingest_document(
            ticker="AAPL", sector="Technology", doc_type="10-K", source="sec_edgar",
            filed_date=date(2026, 7, 1), title="t", url="u", chunks=chunks,
            mirror_to_parquet=False,
        )

    mirror_mock.assert_not_called()


def test_ingest_document_mirror_failure_does_not_fail_ingest():
    """A mirror-side exception must never surface to the caller — Neon's
    insert already committed by the time the mirror runs."""
    from nousergon_lib.rag.retrieval import ingest_document

    chunks = [{"content": "hello", "section_label": None, "embedding": [0.1] * 512}]

    with patch("nousergon_lib.rag.retrieval.document_exists", return_value=False), \
         patch("nousergon_lib.rag.db.get_connection", return_value=_mock_connection()), \
         patch(
             "nousergon_lib.rag.parquet_mirror.mirror_document_to_parquet",
             side_effect=RuntimeError("boom"),
         ):
        # mirror_document_to_parquet itself never raises (see its own
        # tests) — but ingest_document's call site shouldn't propagate
        # even a bug there either, since it's a derived/optimization tier.
        doc_id = ingest_document(
            ticker="AAPL", sector="Technology", doc_type="10-K", source="sec_edgar",
            filed_date=date(2026, 7, 1), title="t", url="u", chunks=chunks,
        )

    assert doc_id == "11111111-1111-1111-1111-111111111111"
