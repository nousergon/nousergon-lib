"""``RetrievalResult.as_source()`` — a retrieved chunk as a citable prompt
source for the ``llm_trace.v1`` record (alpha-engine-config-I12312).

The dict must be accepted by ``krepis.llm_trace.SourceBlock.from_mapping``
without this library importing krepis, and must carry the chunk id, the
parent document id and the URL that the Think Tank used to strip before
rendering.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from nousergon_lib.rag.retrieval import (
    RetrievalMethod,
    RetrievalResult,
    _blend,
    retrieve,
)

# The exact key set ``krepis.llm_trace.SourceBlock.from_mapping`` accepts
# (krepis PR 247). An unknown key raises there, so the dict must stay inside
# it; ``test_from_mapping_accepts_it`` checks the real thing once a krepis
# carrying ``llm_trace`` is installed.
FROM_MAPPING_KEYS = {
    "cid", "kind", "source_id", "text", "document_id", "uri",
    "version", "retrieval", "meta",
}


def _hybrid_result(**overrides) -> RetrievalResult:
    fields: dict = {
        "content": "Revenue grew 12% on subscription renewals.",
        "ticker": "QLYS",
        "doc_type": "8-K",
        "filed_date": date(2026, 9, 4),
        "section_label": "Item 2.02",
        "similarity": 0.81,
        "chunk_id": "8d0c1f2e-0000-4000-8000-000000000001",
        "retrieval_method": "hybrid",
        "vector_score": 0.77,
        "keyword_score": 3.2,
        "combined_score": 0.81,
        "document_id": "41aa0000-0000-4000-8000-000000000002",
        "url": "https://www.sec.gov/Archives/edgar/data/1107843/x.htm",
    }
    fields.update(overrides)
    return RetrievalResult(**fields)


class TestAsSource:
    def test_shape(self) -> None:
        src = _hybrid_result().as_source("c1", rank=1, query="QLYS revenue")
        assert set(src) == FROM_MAPPING_KEYS
        assert src["cid"] == "c1"
        assert src["kind"] == "rag_chunk"
        assert src["source_id"] == "rag.chunks/8d0c1f2e-0000-4000-8000-000000000001"
        assert src["document_id"] == "rag.documents/41aa0000-0000-4000-8000-000000000002"
        assert src["uri"] == "https://www.sec.gov/Archives/edgar/data/1107843/x.htm"
        assert src["version"] == "filed:2026-09-04"
        assert src["text"] == "Revenue grew 12% on subscription renewals."
        assert src["retrieval"] == {
            "method": "hybrid",
            "rank": 1,
            "score": 0.81,
            "vector_score": 0.77,
            "keyword_score": 3.2,
            "combined_score": 0.81,
            "query_sha256": hashlib.sha256(b"QLYS revenue").hexdigest(),
        }
        assert src["meta"] == {
            "ticker": "QLYS",
            "doc_type": "8-K",
            "filed_date": "2026-09-04",
            "section_label": "Item 2.02",
        }

    def test_text_override_is_what_gets_hashed(self) -> None:
        # The caller injects a truncated excerpt; the source must carry the
        # injected text, not the longer original.
        r = _hybrid_result(content="x" * 3000)
        assert r.as_source("c1", text=r.content[:1500])["text"] == "x" * 1500

    def test_plain_json(self) -> None:
        src = _hybrid_result(rerank_score=0.9, rerank_method="cross_encoder").as_source(
            "c2", rank=2, retrieved_at="2026-10-14T12:00:00Z"
        )
        # Deterministic, JSON-serializable, no dates or UUID objects inside.
        assert json.loads(json.dumps(src, sort_keys=True)) == src
        assert src["retrieval"]["rerank_score"] == 0.9
        assert src["retrieval"]["rerank_method"] == "cross_encoder"
        assert src["retrieval"]["retrieved_at"] == "2026-10-14T12:00:00Z"

    def test_absent_optionals_are_omitted_not_invented(self) -> None:
        r = _hybrid_result(
            document_id=None, url=None, section_label=None,
            retrieval_method="vector", keyword_score=None, combined_score=None,
        )
        src = r.as_source("c1")
        assert src["document_id"] is None
        assert src["uri"] is None
        assert "section_label" not in src["meta"]
        assert "rank" not in src["retrieval"]
        assert "query_sha256" not in src["retrieval"]
        assert src["retrieval"]["method"] == "vector"

    def test_missing_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            _hybrid_result(chunk_id=None).as_source("c1")

    def test_from_mapping_accepts_it(self) -> None:
        llm_trace = pytest.importorskip(
            "krepis.llm_trace",
            reason="needs a krepis release carrying llm_trace (krepis PR 247)",
        )
        block = llm_trace.SourceBlock.from_mapping(
            _hybrid_result().as_source("c1", rank=1)
        )
        rendered = llm_trace.render_sources([block])
        assert rendered.sources[0].block.source_id.startswith("rag.chunks/")


class TestIdsSurviveRetrieval:
    def test_blend_carries_document_id_and_url(self) -> None:
        v = _hybrid_result(retrieval_method="vector", keyword_score=None, combined_score=None)
        k = _hybrid_result(
            retrieval_method="keyword", vector_score=None, combined_score=None,
            document_id=None, url=None,
        )
        out = _blend([v], [k], vector_weight=0.7, top_k=5)
        assert out[0].document_id == v.document_id
        assert out[0].url == v.url

    @pytest.mark.parametrize("method", ["vector", "keyword"])
    def test_sql_selects_and_maps_document_id_and_url(
        self, method: RetrievalMethod
    ) -> None:
        captured: dict = {}
        # psycopg2 returns a UUID column as uuid.UUID, not str.
        doc_uuid = uuid.UUID("41aa0000-0000-4000-8000-000000000002")
        row = (
            "chunk-1", "text", "QLYS", "10-K", date(2026, 2, 1), "MD&A", 0.5,
            doc_uuid, "https://example.test/doc",
        )

        def fake_cursor():
            cur = MagicMock()
            cur.execute.side_effect = lambda sql, params: captured.update(sql=sql)
            cur.fetchall.return_value = [row]
            cm = MagicMock()
            cm.__enter__ = lambda self: cur
            cm.__exit__ = lambda self, *a: None
            return cm

        conn_cm = MagicMock()
        conn_cm.__enter__ = lambda self: MagicMock(cursor=fake_cursor)
        conn_cm.__exit__ = lambda self, *a: None

        with patch("nousergon_lib.rag.embeddings.embed_query", return_value=[0.1] * 512), \
             patch("nousergon_lib.rag.db.get_connection", return_value=conn_cm):
            results = retrieve("query", method=method, top_k=5)

        assert "d.id, d.url" in captured["sql"]
        assert results[0].document_id == "41aa0000-0000-4000-8000-000000000002"
        assert results[0].url == "https://example.test/doc"
        assert results[0].as_source("c1")["document_id"] == (
            "rag.documents/41aa0000-0000-4000-8000-000000000002"
        )
