"""Tests for the S3 parquet batch-tier mirror (config#2958).

Uses moto to stand in for S3 — no live bucket needed. Live Neon writes
are out of scope here (matches test_rag.py's stated scope boundary).
"""

from __future__ import annotations

import io
from datetime import date

import boto3
import pandas as pd
import pytest
from moto import mock_aws

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3_client():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _sample_chunks(n=2):
    return [
        {"content": f"chunk {i}", "section_label": "MD&A", "embedding": [0.01 * i] * 512}
        for i in range(n)
    ]


def test_parquet_mirror_key_is_hive_partitioned():
    from nousergon_lib.rag.parquet_mirror import parquet_mirror_key

    key = parquet_mirror_key("10-K", date(2026, 7, 1), "doc-123")
    assert key == "rag/parquet/doc_type=10-K/filed_date=2026-07-01/doc-123.parquet"


def test_mirror_document_writes_one_row_per_chunk(s3_client):
    from nousergon_lib.rag.parquet_mirror import mirror_document_to_parquet

    key = mirror_document_to_parquet(
        document_id="doc-1",
        ticker="AAPL",
        sector="Technology",
        doc_type="10-K",
        source="sec_edgar",
        filed_date=date(2026, 7, 1),
        title="Apple 10-K",
        url="https://example.com",
        chunks=_sample_chunks(3),
        s3_client=s3_client,
    )

    assert key == "rag/parquet/doc_type=10-K/filed_date=2026-07-01/doc-1.parquet"
    obj = s3_client.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    assert len(df) == 3
    assert list(df["chunk_index"]) == [0, 1, 2]
    assert df["ticker"].unique().tolist() == ["AAPL"]
    assert len(df["embedding"].iloc[0]) == 512


def test_mirror_document_empty_chunks_is_noop(s3_client):
    from nousergon_lib.rag.parquet_mirror import mirror_document_to_parquet

    key = mirror_document_to_parquet(
        document_id="doc-2", ticker="AAPL", sector=None, doc_type="10-K",
        source="sec_edgar", filed_date=date(2026, 7, 1), title=None, url=None,
        chunks=[], s3_client=s3_client,
    )
    assert key is None


def test_mirror_document_failure_is_logged_not_raised(s3_client, caplog):
    """A bucket that doesn't exist must degrade to a logged error, never raise —
    Neon is already durable by the time ingest_document calls this."""
    from nousergon_lib.rag.parquet_mirror import mirror_document_to_parquet

    key = mirror_document_to_parquet(
        document_id="doc-3", ticker="AAPL", sector=None, doc_type="10-K",
        source="sec_edgar", filed_date=date(2026, 7, 1), title=None, url=None,
        chunks=_sample_chunks(1), s3_client=s3_client, bucket="does-not-exist",
    )
    assert key is None
    assert "Parquet mirror failed" in caplog.text
