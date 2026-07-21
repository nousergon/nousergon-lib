"""Tests for nousergon_lib.rag.batch_tier (config#2958).

Covers: parquet mirror write + round trip via moto S3, the "tier not
configured" / "write failed" no-raise contracts, and LocalCorpusIndex's
exact brute-force query path (hnswlib isn't installed in this test env,
so the fallback path is what's actually exercised — see
test_local_corpus_index_matches_expected_neighbor for the correctness
assertion that must hold regardless of which backend serves it).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from moto import mock_aws

from nousergon_lib.rag import batch_tier

BUCKET = "test-rag-parquet-bucket"


def _chunks(n=3):
    rng = np.random.default_rng(42)
    return [
        {
            "content": f"chunk {i} content",
            "section_label": "Risk Factors" if i == 0 else None,
            "embedding": rng.random(8).astype(np.float32),
        }
        for i in range(n)
    ]


def test_mirror_disabled_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("RAG_PARQUET_BUCKET", raising=False)
    ok = batch_tier.mirror_chunks_to_parquet(
        "doc-1", "AAPL", "Technology", "10-K", "sec_edgar",
        date(2026, 1, 15), "AAPL 10-K", "http://example.com", _chunks(),
    )
    assert ok is False


@mock_aws
def test_mirror_and_load_round_trip(monkeypatch):
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    monkeypatch.setenv("RAG_PARQUET_BUCKET", BUCKET)
    monkeypatch.setenv("RAG_PARQUET_PREFIX", "rag_corpus_parquet")

    chunks = _chunks(3)
    ok = batch_tier.mirror_chunks_to_parquet(
        "doc-1", "AAPL", "Technology", "10-K", "sec_edgar",
        date(2026, 1, 15), "AAPL 10-K", "http://example.com", chunks,
    )
    assert ok is True

    key = batch_tier.parquet_key("doc-1", "10-K", date(2026, 1, 15))
    assert key == "rag_corpus_parquet/doc_type=10-K/date=2026-01-15/doc-1.parquet"
    listed = s3.list_objects_v2(Bucket=BUCKET, Prefix="rag_corpus_parquet/")
    assert listed["KeyCount"] == 1
    assert listed["Contents"][0]["Key"] == key

    index = batch_tier.load_local_corpus(doc_types=["10-K"])
    assert len(index) == 3


def test_mirror_key_partitioned_by_doc_type_and_date():
    key = batch_tier.parquet_key("doc-9", "news", date(2026, 3, 1), prefix="my_prefix")
    assert key == "my_prefix/doc_type=news/date=2026-03-01/doc-9.parquet"


@mock_aws
def test_mirror_write_failure_does_not_raise(monkeypatch):
    # Bucket deliberately NOT created — put_object will fail against moto.
    monkeypatch.setenv("RAG_PARQUET_BUCKET", "nonexistent-bucket-xyz")
    ok = batch_tier.mirror_chunks_to_parquet(
        "doc-1", "AAPL", "Technology", "10-K", "sec_edgar",
        date(2026, 1, 15), "AAPL 10-K", "http://example.com", _chunks(),
    )
    assert ok is False


@mock_aws
def test_load_local_corpus_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("RAG_PARQUET_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="RAG_PARQUET_BUCKET"):
        batch_tier.load_local_corpus()


@mock_aws
def test_load_local_corpus_filters_by_ticker_and_min_date(monkeypatch):
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    monkeypatch.setenv("RAG_PARQUET_BUCKET", BUCKET)

    batch_tier.mirror_chunks_to_parquet(
        "doc-aapl", "AAPL", None, "10-K", "sec_edgar",
        date(2026, 1, 1), None, None, _chunks(2),
    )
    batch_tier.mirror_chunks_to_parquet(
        "doc-msft", "MSFT", None, "10-K", "sec_edgar",
        date(2026, 6, 1), None, None, _chunks(2),
    )

    only_aapl = batch_tier.load_local_corpus(doc_types=["10-K"], tickers=["AAPL"])
    assert len(only_aapl) == 2
    assert all(r["ticker"] == "AAPL" for r in only_aapl._rows)

    only_recent = batch_tier.load_local_corpus(doc_types=["10-K"], min_date=date(2026, 3, 1))
    assert len(only_recent) == 2
    assert all(r["ticker"] == "MSFT" for r in only_recent._rows)


def test_local_corpus_index_query_returns_top_k_by_similarity():
    rows = [
        {"content": "c0", "ticker": "AAPL", "doc_type": "10-K", "filed_date": date(2026, 1, 1),
         "section_label": None, "document_id": "d0", "chunk_index": 0},
        {"content": "c1", "ticker": "AAPL", "doc_type": "10-K", "filed_date": date(2026, 1, 1),
         "section_label": None, "document_id": "d0", "chunk_index": 1},
        {"content": "c2", "ticker": "AAPL", "doc_type": "10-K", "filed_date": date(2026, 1, 1),
         "section_label": None, "document_id": "d0", "chunk_index": 2},
    ]
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.99, 0.01],
    ], dtype=np.float32)
    index = batch_tier.LocalCorpusIndex(rows, embeddings)

    results = index.query(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    # Nearest should be the exact match (c0), second nearest the near-match (c2).
    assert results[0].content == "c0"
    assert results[1].content == "c2"
    assert results[0].similarity > results[1].similarity > 0.5


def test_local_corpus_index_empty():
    index = batch_tier.LocalCorpusIndex([], np.zeros((0, 0), dtype=np.float32))
    assert len(index) == 0
    assert index.query(np.array([1.0, 0.0], dtype=np.float32)) == []


class _FakeHnswIndex:
    """Minimal stand-in for hnswlib.Index's API surface used by
    LocalCorpusIndex, so the hnswlib-accelerated path is exercised even
    on hosts (like CI) where the real C-extension isn't installed."""

    def __init__(self, space, dim):
        self.dim = dim
        self._embeddings = None

    def init_index(self, max_elements, ef_construction, M):
        pass

    def add_items(self, embeddings, labels):
        self._embeddings = embeddings

    def set_ef(self, ef):
        pass

    def knn_query(self, query, k):
        norms = np.linalg.norm(self._embeddings, axis=1) * (np.linalg.norm(query) or 1e-12)
        sims = (self._embeddings @ query) / norms
        order = np.argsort(-sims)[:k]
        return np.array([order]), np.array([1.0 - sims[order]])


def test_local_corpus_index_uses_hnswlib_when_available(monkeypatch):
    fake_hnswlib = type("FakeHnswlibModule", (), {"Index": _FakeHnswIndex})
    monkeypatch.setattr(batch_tier, "hnswlib", fake_hnswlib)
    monkeypatch.setattr(batch_tier, "_HAVE_HNSWLIB", True)

    rows = [
        {"content": "c0", "ticker": "AAPL", "doc_type": "10-K", "filed_date": date(2026, 1, 1),
         "section_label": None, "document_id": "d0", "chunk_index": 0},
        {"content": "c1", "ticker": "AAPL", "doc_type": "10-K", "filed_date": date(2026, 1, 1),
         "section_label": None, "document_id": "d0", "chunk_index": 1},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = batch_tier.LocalCorpusIndex(rows, embeddings)
    assert index._hnsw_index is not None

    results = index.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].content == "c0"
