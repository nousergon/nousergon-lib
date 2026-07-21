"""Local (in-process) ANN index over the S3 parquet batch tier (config#2958).

Pairs with :mod:`nousergon_lib.rag.parquet_mirror`: that module writes the
corpus to S3 as partitioned parquet at ingest time; this module reads it
back and builds an ephemeral HNSW index (hnswlib) on whatever box runs the
batch job (the canary/predictor spot box, an eval run, a backtest). No
Neon connection is opened by either half — this is the "zero Neon egress
for batch workloads" path the RAG storage-tiering design calls for.

Batch jobs whose access pattern is a groupby/aggregate (e.g.
``filing_change_detection``'s per-ticker centroid) don't need the ANN
index at all — :func:`load_corpus_dataframe` alone (a plain pandas
DataFrame) is enough; pandas groupby replaces the Neon-side ``AVG(...)
GROUP BY``. The :class:`LocalANNIndex` is for consumers that need
nearest-neighbor *search* (evals judging retrieval quality, backtests
replaying historical retrieval) rather than aggregation.

Not a replacement for Neon's pgvector HNSW index on the live retrieval
path — this is a batch/offline structure, rebuilt per-run from whatever
parquet partitions the caller selects, with no incremental-update or
concurrent-write story. See parquet_mirror.py's module docstring for why
that's an acceptable, deliberate scope boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from .parquet_mirror import DEFAULT_BUCKET, DEFAULT_PREFIX

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # must match rag.chunks.embedding vector(512) — voyage-3-lite


def list_parquet_keys(
    *,
    doc_type: str | None = None,
    filed_date_gte: date | None = None,
    s3_client: Any = None,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
) -> list[str]:
    """List mirror object keys under the Hive-partitioned prefix.

    ``doc_type`` narrows to a single partition directory (cheap prefix
    list). ``filed_date_gte`` is applied client-side against the
    ``filed_date=YYYY-MM-DD`` segment of each key, since S3 list
    operations can't range-filter a partition value — fine at this
    corpus's key-count scale (~tens of thousands of objects, one LIST
    call per 1000 keys).
    """
    import boto3

    client = s3_client if s3_client is not None else boto3.client("s3")
    list_prefix = f"{prefix}/doc_type={doc_type}/" if doc_type else f"{prefix}/"

    keys: list[str] = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": list_prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            if filed_date_gte is not None:
                marker = "filed_date="
                start = key.find(marker)
                if start == -1:
                    continue
                key_date = key[start + len(marker):start + len(marker) + 10]
                if key_date < filed_date_gte.isoformat():
                    continue
            keys.append(key)
        if not resp.get("IsTruncated"):
            break
        continuation_token = resp.get("NextContinuationToken")

    return keys


def load_corpus_dataframe(
    keys: list[str],
    *,
    s3_client: Any = None,
    bucket: str = DEFAULT_BUCKET,
):
    """Read and concatenate the given parquet mirror objects into one DataFrame.

    Empty ``keys`` returns an empty (but correctly-columned) DataFrame so
    callers can groupby/concat without a special-case branch.
    """
    import io

    import pandas as pd

    columns = [
        "document_id", "ticker", "sector", "doc_type", "source",
        "filed_date", "title", "url", "chunk_index", "content",
        "section_label", "embedding",
    ]
    if not keys:
        # pd.Index(...), not the bare list: pandas' inline stubs type `columns`
        # as Axes, and a bare list[str] fails structural matching against the
        # SequenceNotStr protocol on `.index()`'s parameter type (pyright).
        return pd.DataFrame(columns=pd.Index(columns))

    import boto3

    client = s3_client if s3_client is not None else boto3.client("s3")

    frames = []
    for key in keys:
        obj = client.get_object(Bucket=bucket, Key=key)
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow"))

    return pd.concat(frames, ignore_index=True)


@dataclass
class LocalANNIndex:
    """A built HNSW index plus the row metadata it was built from.

    ``index.knn_query(vector, k=k)`` returns hnswlib-internal integer
    labels; use :meth:`query` instead, which resolves labels back to the
    source DataFrame rows.
    """

    index: Any
    rows: Any  # pandas.DataFrame, same row order as the labels used to build the index

    def query(self, embedding, k: int = 10) -> list[dict]:
        import numpy as np

        labels, distances = self.index.knn_query(np.asarray(embedding, dtype=np.float32), k=k)
        results = []
        for label, distance in zip(labels[0], distances[0]):
            row = self.rows.iloc[int(label)].to_dict()
            row["cosine_similarity"] = 1.0 - float(distance)  # hnswlib "cosine" space returns 1 - cos_sim
            results.append(row)
        return results


def build_local_ann_index(rows, *, ef_construction: int = 200, m: int = 16) -> LocalANNIndex:
    """Build an in-memory cosine HNSW index over a corpus DataFrame's embeddings.

    ``rows`` is typically the output of :func:`load_corpus_dataframe`. Its
    ``embedding`` column entries must all be ``EMBEDDING_DIM``-length
    sequences (as parquet stores them — see parquet_mirror.py).
    """
    import hnswlib
    import numpy as np

    n = len(rows)
    index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
    index.init_index(max_elements=max(n, 1), ef_construction=ef_construction, M=m)
    if n:
        vectors = np.stack(rows["embedding"].to_numpy()).astype(np.float32)
        index.add_items(vectors, np.arange(n))
        index.set_ef(max(ef_construction, 2 * m))

    return LocalANNIndex(index=index, rows=rows.reset_index(drop=True))
