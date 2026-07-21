"""S3 parquet mirror + local ANN reader for batch RAG consumers (config#2958).

Why this exists: ``rag.chunks``/``rag.documents`` in Neon are the ONLY
storage tier for the corpus today — every batch/analytical consumer that
needs the embeddings has no option but a Neon round trip, and the corpus
grows every day (config-I2943 daily-delta ingestion). Neon's free-tier
5GB/month data-transfer quota is the binding constraint that caused a
hard connect lockout for every RAG consumer on 2026-07-16
(config-I2780/I2781) — a single unbounded ``SELECT c.embedding`` was the
proximate cause there, but the structural fix is giving batch/analytical
workloads (evals, backtests, ad-hoc corpus analysis — anything that isn't
the live low-latency agent-retrieval path ``retrieval.retrieve()``
serves) a read path that never touches Neon at all.

Two halves:

- :func:`mirror_chunks_to_parquet` — called once per ingested document
  (wired into :func:`nousergon_lib.rag.retrieval.ingest_document`) to
  write that document's chunks + embeddings to a small, partitioned
  parquet file on S3. Append-only: one file per document, named by
  document UUID, so concurrent ingestion across pipelines never needs
  read-modify-write locking. Best-effort — a mirror failure is logged
  LOUD (``logger.error``, not swallowed) but never raised, so an S3
  hiccup can't fail an otherwise-successful Neon ingest; the Neon row is
  the source of truth, the parquet tier is a derived, backfillable copy.
- :func:`load_local_corpus` / :class:`LocalCorpusIndex` — for batch
  consumers: lists + reads the parquet tier (optionally scoped by
  doc_type/date-range partition prefix, further filtered by ticker
  client-side), builds an in-memory nearest-neighbor index, and exposes
  a ``retrieve()``-shaped query method. Uses ``hnswlib`` when installed
  (optional, C-extension, not always buildable — see the try/except
  below) and falls back to an exact numpy brute-force cosine top-k
  otherwise; both return identical result shapes, so callers never need
  to know which backend served a given batch run. Brute-force is O(n) per
  query, which is fine at the partition sizes a single doc_type/date
  scope produces (thousands, not millions, of chunks) — this is a batch
  tool, not the live retrieval path.

Requires the ``rag-batch-tier`` extra (``pyarrow`` always; ``hnswlib``
optional, attempted at import time only).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)

try:
    import hnswlib  # type: ignore[import-not-found]

    _HAVE_HNSWLIB = True
except ImportError:  # pragma: no cover — exercised by the fallback test only
    hnswlib = None
    _HAVE_HNSWLIB = False


_BUCKET_ENV = "RAG_PARQUET_BUCKET"
_PREFIX_ENV = "RAG_PARQUET_PREFIX"
_DEFAULT_PREFIX = "rag_corpus_parquet"

_PARQUET_COLUMNS = [
    "document_id", "ticker", "sector", "doc_type", "source", "filed_date",
    "title", "url", "chunk_index", "content", "section_label", "embedding",
]


def _tier_config() -> tuple[str | None, str]:
    return os.environ.get(_BUCKET_ENV), os.environ.get(_PREFIX_ENV, _DEFAULT_PREFIX)


def parquet_key(document_id: str, doc_type: str, filed_date: date, prefix: str | None = None) -> str:
    """The partitioned S3 key for one document's chunk mirror.

    Partitioned by doc_type/date (per config#2958's deliverable 1 spec) so
    a batch consumer scoped to one doc_type/date-range can list just that
    prefix instead of the whole tier.
    """
    _, default_prefix = _tier_config()
    prefix = prefix or default_prefix
    return f"{prefix}/doc_type={doc_type}/date={filed_date.isoformat()}/{document_id}.parquet"


def mirror_chunks_to_parquet(
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
    bucket: str | None = None,
    prefix: str | None = None,
) -> bool:
    """Best-effort mirror of one document's chunks to the S3 parquet tier.

    Returns True on a successful write, False on any failure (including
    the tier being unconfigured — ``RAG_PARQUET_BUCKET`` unset is a valid
    "mirror disabled" state, not an error, so it also returns False
    silently rather than logging). Never raises: called from the hot
    ingest path, and a derived mirror's failure must never fail the
    primary Neon write it follows.
    """
    env_bucket, env_prefix = _tier_config()
    bucket = bucket or env_bucket
    prefix = prefix or env_prefix
    if not bucket:
        return False

    try:
        import boto3
        import pyarrow as pa
        import pyarrow.parquet as pq
        import io

        n = len(chunks)
        table = pa.table({
            "document_id": pa.array([document_id] * n, type=pa.string()),
            "ticker": pa.array([ticker] * n, type=pa.string()),
            "sector": pa.array([sector] * n, type=pa.string()),
            "doc_type": pa.array([doc_type] * n, type=pa.string()),
            "source": pa.array([source] * n, type=pa.string()),
            "filed_date": pa.array([filed_date] * n, type=pa.date32()),
            "title": pa.array([title] * n, type=pa.string()),
            "url": pa.array([url] * n, type=pa.string()),
            "chunk_index": pa.array(list(range(n)), type=pa.int32()),
            "content": pa.array([c["content"] for c in chunks], type=pa.string()),
            "section_label": pa.array([c.get("section_label") for c in chunks], type=pa.string()),
            "embedding": pa.array(
                [np.asarray(c["embedding"], dtype=np.float32).tolist() for c in chunks],
                type=pa.list_(pa.float32()),
            ),
        })

        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)

        key = parquet_key(document_id, doc_type, filed_date, prefix)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        logger.info("Mirrored %d chunks for document %s to s3://%s/%s", n, document_id, bucket, key)
        return True
    except Exception:
        logger.error(
            "S3 parquet mirror failed for document %s (doc_type=%s, filed_date=%s) — "
            "Neon ingest already succeeded, this is a derived-copy gap only",
            document_id, doc_type, filed_date, exc_info=True,
        )
        return False


@dataclass
class BatchRetrievalResult:
    content: str
    ticker: str
    doc_type: str
    filed_date: date
    section_label: str | None
    similarity: float
    document_id: str
    chunk_index: int


class LocalCorpusIndex:
    """An in-memory nearest-neighbor index over a loaded parquet-tier slice.

    Prefer building via :func:`load_local_corpus` rather than constructing
    directly.
    """

    def __init__(self, rows: list[dict], embeddings: np.ndarray):
        self._rows = rows
        self._embeddings = embeddings  # shape (n, dim), NOT assumed normalized
        self._hnsw_index = None
        if _HAVE_HNSWLIB and len(rows) > 0:
            dim = embeddings.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(rows), ef_construction=200, M=16)
            index.add_items(embeddings, np.arange(len(rows)))
            index.set_ef(max(50, min(len(rows), 200)))
            self._hnsw_index = index

    def __len__(self) -> int:
        return len(self._rows)

    def query(self, query_embedding, top_k: int = 10) -> list[BatchRetrievalResult]:
        if len(self._rows) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)

        if self._hnsw_index is not None:
            labels, distances = self._hnsw_index.knn_query(q, k=min(top_k, len(self._rows)))
            order = list(zip(labels[0].tolist(), (1.0 - distances[0]).tolist()))
        else:
            # Exact brute-force cosine similarity — fine at batch-partition
            # scale (thousands of rows), see module docstring.
            norms = np.linalg.norm(self._embeddings, axis=1) * (np.linalg.norm(q) or 1e-12)
            norms[norms == 0] = 1e-12
            sims = (self._embeddings @ q) / norms
            k = min(top_k, len(self._rows))
            top_idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
            order = sorted(((int(i), float(sims[i])) for i in top_idx), key=lambda t: -t[1])

        results = []
        for idx, score in order:
            row = self._rows[idx]
            results.append(BatchRetrievalResult(
                content=row["content"],
                ticker=row["ticker"],
                doc_type=row["doc_type"],
                filed_date=row["filed_date"],
                section_label=row.get("section_label"),
                similarity=score,
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
            ))
        return results


def _list_parquet_keys(s3_client, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


def load_local_corpus(
    doc_types: list[str] | None = None,
    tickers: list[str] | None = None,
    min_date: date | None = None,
    *,
    bucket: str | None = None,
    prefix: str | None = None,
) -> LocalCorpusIndex:
    """Load a slice of the S3 parquet tier into an in-memory ANN index.

    Zero Neon reads. ``doc_types`` narrows the S3 listing itself (one
    ``list_objects_v2`` prefix per doc_type, since the tier is partitioned
    doc_type/date/); ``tickers``/``min_date`` filter client-side after
    each parquet file is read (partitioning isn't ticker-scoped, so this
    can't be pushed into the S3 prefix). Omitting ``doc_types`` scans the
    whole tier — fine for a nightly batch job, likely too broad for an
    interactive one.
    """
    import boto3
    import pyarrow.parquet as pq
    import io

    env_bucket, env_prefix = _tier_config()
    bucket = bucket or env_bucket
    prefix = prefix or env_prefix
    if not bucket:
        raise RuntimeError(
            f"{_BUCKET_ENV} is not set — the S3 parquet tier is not configured"
        )

    s3 = boto3.client("s3")
    partitions = [f"{prefix}/doc_type={dt}/" for dt in doc_types] if doc_types else [f"{prefix}/"]

    rows: list[dict] = []
    for partition_prefix in partitions:
        for key in _list_parquet_keys(s3, bucket, partition_prefix):
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            table = pq.read_table(io.BytesIO(body))
            batch = table.to_pylist()
            for row in batch:
                if tickers and row["ticker"] not in tickers:
                    continue
                if min_date and row["filed_date"] < min_date:
                    continue
                rows.append(row)

    logger.info("Loaded %d chunks from S3 parquet tier (bucket=%s, prefix=%s)", len(rows), bucket, prefix)

    if rows:
        embeddings = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    return LocalCorpusIndex(rows, embeddings)
