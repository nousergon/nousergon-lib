"""Tests for the local (in-process) ANN batch tier (config#2958).

Builds a small parquet corpus via the mirror module, then reads it back
through local_ann and verifies both the plain-DataFrame path (what
filing_change_detection-style groupby consumers use) and the HNSW index
path (what nearest-neighbor batch consumers use) work end to end against
a moto-mocked S3 bucket — no live Neon or live S3 involved.
"""

from __future__ import annotations

import sys
from datetime import date

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.rag.parquet_mirror import mirror_document_to_parquet

BUCKET = "alpha-engine-research"

skip_py39_no_hnswlib = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="hnswlib native extension crashes with SIGILL on py3.9 GitHub Actions runner (wheel compiled with incompatible CPU instructions)",
)


@pytest.fixture
def corpus(s3_client=None):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)

        # Deliberately non-colinear embeddings: AAPL points "all positive",
        # MSFT points "half positive / half negative" — colinear vectors
        # (e.g. both [0.1]*512 and [0.3]*512) have cosine similarity 1.0
        # regardless of scale, which would make a nearest-neighbor test
        # order-dependent noise rather than a real assertion.
        embeddings = {
            "AAPL": [1.0] * 512,
            "MSFT": [1.0] * 256 + [-1.0] * 256,
        }
        for i, ticker in enumerate(["AAPL", "MSFT"]):
            mirror_document_to_parquet(
                document_id=f"doc-{i}",
                ticker=ticker,
                sector="Technology",
                doc_type="10-K",
                source="sec_edgar",
                filed_date=date(2026, 7, 1),
                title=f"{ticker} 10-K",
                url=None,
                chunks=[
                    {"content": f"{ticker} chunk", "section_label": None,
                     "embedding": embeddings[ticker]},
                ],
                s3_client=client,
            )
        yield client


def test_list_parquet_keys_filters_by_doc_type(corpus):
    from nousergon_lib.rag.local_ann import list_parquet_keys

    keys = list_parquet_keys(doc_type="10-K", s3_client=corpus)
    assert len(keys) == 2
    assert all("doc_type=10-K" in k for k in keys)

    assert list_parquet_keys(doc_type="news", s3_client=corpus) == []


def test_list_parquet_keys_filters_by_filed_date(corpus):
    from nousergon_lib.rag.local_ann import list_parquet_keys

    assert len(list_parquet_keys(filed_date_gte=date(2026, 7, 1), s3_client=corpus)) == 2
    assert list_parquet_keys(filed_date_gte=date(2026, 7, 2), s3_client=corpus) == []


def test_load_corpus_dataframe_concatenates_all_keys(corpus):
    from nousergon_lib.rag.local_ann import list_parquet_keys, load_corpus_dataframe

    keys = list_parquet_keys(s3_client=corpus)
    df = load_corpus_dataframe(keys, s3_client=corpus)
    assert len(df) == 2
    assert set(df["ticker"]) == {"AAPL", "MSFT"}


def test_load_corpus_dataframe_empty_keys_returns_empty_frame():
    from nousergon_lib.rag.local_ann import load_corpus_dataframe

    df = load_corpus_dataframe([])
    assert len(df) == 0
    assert "embedding" in df.columns


@skip_py39_no_hnswlib
def test_build_local_ann_index_and_query(corpus):
    from nousergon_lib.rag.local_ann import (
        build_local_ann_index,
        list_parquet_keys,
        load_corpus_dataframe,
    )

    keys = list_parquet_keys(s3_client=corpus)
    df = load_corpus_dataframe(keys, s3_client=corpus)
    index = build_local_ann_index(df)

    results = index.query([1.0] * 512, k=1)
    assert len(results) == 1
    assert results[0]["ticker"] == "AAPL"
    assert results[0]["cosine_similarity"] > 0.99


@skip_py39_no_hnswlib
def test_build_local_ann_index_handles_empty_corpus():
    from nousergon_lib.rag.local_ann import build_local_ann_index, load_corpus_dataframe

    df = load_corpus_dataframe([])
    index = build_local_ann_index(df)
    assert index.rows.empty


