"""S3 parquet mirror of the RAG corpus — batch-read tier (config#2958).

``rag.documents`` + ``rag.chunks`` live only in Neon today; every batch
consumer (``filing_change_detection``, evals, future backtests) that needs
to scan the corpus does so by querying Neon directly, which is the exact
shape of the 2026-07-16 egress lockout (config-I2780/I2938 audit) — Neon's
free-tier 5GB/month transfer quota is a binding constraint, not a
theoretical one.

This module gives ingestion a second, append-only write target: one
parquet file per ingested document, partitioned Hive-style by
``doc_type=.../filed_date=...`` under ``s3://<bucket>/rag/parquet/``.
Batch consumers read this partition set (see :mod:`nousergon_lib.rag.local_ann`)
instead of touching Neon at all — Neon remains solely the live low-latency
retrieval path (pgvector HNSW + tsvector hybrid search), unaffected by any
of this.

The mirror is intentionally best-effort relative to the Neon write: Neon is
the system of record (:func:`nousergon_lib.rag.retrieval.ingest_document`
already committed the document there by the time this is called), so a
mirror failure must never roll back or fail an otherwise-successful
ingest — it's logged loudly and left for the next backfill pass
(``rebuild_parquet_mirror`` — not yet needed since nothing has been
written any other way yet) to reconcile. Silent failure is not
acceptable either, hence the ``logger.error`` with full context on any
write exception.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_PREFIX = "rag/parquet"


def parquet_mirror_key(
    doc_type: str,
    filed_date: date,
    document_id: str,
    prefix: str = DEFAULT_PREFIX,
) -> str:
    """Hive-style partition key for one document's chunk mirror.

    Partitioned by ``doc_type``/``filed_date`` (not ``ticker``) because
    batch consumers scan by document class and recency, not by symbol —
    matching ``filing_change_detection``'s own ``date_rank``-windowed
    access pattern.
    """
    return (
        f"{prefix}/doc_type={doc_type}/filed_date={filed_date.isoformat()}/"
        f"{document_id}.parquet"
    )


def mirror_document_to_parquet(
    document_id: str,
    ticker: str,
    sector: str | None,
    doc_type: str,
    source: str,
    filed_date: date,
    title: str | None,
    url: str | None,
    chunks: list[dict],
    *,
    s3_client: Any = None,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
) -> str | None:
    """Write one document + its chunks to the S3 parquet batch tier.

    One row per chunk (document metadata denormalized onto every row —
    the corpus is small enough per-document, and downstream batch reads
    want a flat table without a join). Returns the S3 key on success,
    None on failure (logged, never raised — see module docstring;
    includes the ``[rag-parquet]`` extra's pandas/pyarrow not being
    installed — callers that don't want that dependency at all should
    pass ``mirror_to_parquet=False`` to ``ingest_document`` instead).
    """
    if not chunks:
        return None

    try:
        import boto3
        import pandas as pd

        client = s3_client if s3_client is not None else boto3.client("s3")

        rows = [
            {
                "document_id": document_id,
                "ticker": ticker,
                "sector": sector,
                "doc_type": doc_type,
                "source": source,
                "filed_date": filed_date.isoformat(),
                "title": title,
                "url": url,
                "chunk_index": i,
                "content": chunk["content"],
                "section_label": chunk.get("section_label"),
                "embedding": [float(x) for x in chunk["embedding"]],
            }
            for i, chunk in enumerate(chunks)
        ]
        df = pd.DataFrame(rows)

        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)

        key = parquet_mirror_key(doc_type, filed_date, document_id, prefix=prefix)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        return key
    except Exception:
        logger.error(
            "Parquet mirror failed for document %s (%s %s %s) — Neon is "
            "already durable, this only degrades batch-tier freshness "
            "until the next mirror attempt or backfill",
            document_id, ticker, doc_type, filed_date,
            exc_info=True,
        )
        return None
