"""Semantic retrieval over RAG document store.

Three retrieval methods:

- ``"vector"`` — pgvector cosine similarity via the HNSW index on
  ``rag.chunks.embedding``. Strong on conceptual / abstract queries
  ("describe the company's competitive moat"); weaker on exact-match
  surfaces like ticker symbols, filing types, named entities, dollar
  amounts.
- ``"keyword"`` — PostgreSQL Full-Text Search (FTS) via the GIN index
  on ``rag.chunks.content_tsv`` with ``ts_rank_cd`` ranking. Strong
  on exact-match surfaces; weaker on paraphrase / conceptual queries.
- ``"hybrid"`` — union of vector top_k + keyword top_k, min-max
  normalize each side within the candidate set, blend via
  ``score = vector_weight * v_norm + (1 - vector_weight) * k_norm``,
  return overall top_k. Default ``vector_weight=0.7``.

All three preserve the existing metadata pre-filters (ticker,
doc_type, min_date) — pre-filtering happens before candidate
generation regardless of method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

logger = logging.getLogger(__name__)


RetrievalMethod = Literal["vector", "keyword", "hybrid"]


@dataclass
class RetrievalResult:
    content: str
    ticker: str
    doc_type: str
    filed_date: date
    section_label: str | None
    similarity: float  # the field used for sort ordering — populated for every method
    chunk_id: str | None = None
    retrieval_method: RetrievalMethod = "vector"
    vector_score: float | None = None    # cosine similarity, [-1, 1]; None if not retrieved via vector
    keyword_score: float | None = None   # ts_rank_cd, [0, ∞); None if not retrieved via keyword
    combined_score: float | None = None  # blended score in hybrid mode; None for non-hybrid
    rerank_score: float | None = None    # cross-encoder score; None if rerank wasn't run
    rerank_method: str | None = None     # "cross_encoder" / None — disambiguates which reranker stamped this


def retrieve(
    query: str,
    tickers: list[str] | None = None,
    doc_types: list[str] | None = None,
    min_date: date | None = None,
    top_k: int = 10,
    *,
    method: RetrievalMethod = "vector",
    vector_weight: float = 0.7,
    rerank: str | None = None,
    rerank_input_n: int = 30,
) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks for a natural language query.

    Args:
        query: Natural language search query.
        tickers: Filter to these stock symbols (optional).
        doc_types: Filter to these doc types, e.g. ['10-K', '10-Q'] (optional).
        min_date: Only return documents filed on or after this date (optional).
        top_k: Maximum number of results to return.
        method: ``"vector"`` (default for back-compat — existing callers see
            unchanged behavior), ``"keyword"``, or ``"hybrid"``. New callers
            should pass ``method="hybrid"`` explicitly.
        vector_weight: Hybrid blend weight on the vector side, in [0, 1].
            ``vector_weight=1.0`` ≡ pure vector; ``0.0`` ≡ pure keyword.
            Ignored for non-hybrid methods.
        rerank: When set, run a reranker over the retrieved candidates
            before truncating to ``top_k``. Supported values:
            ``"cross_encoder"`` (local BAAI bge-reranker-v2-m3 — no
            API cost). ``None`` (default) preserves the pre-rerank
            behavior — back-compat path for callers not yet wired to
            reranking. ``"llm_judge"`` was removed v0.34.0 (see
            ``rerank`` module docstring for the no-lift finding).
        rerank_input_n: When ``rerank`` is set, retrieve this many
            candidates from the underlying method before passing the
            pool to the reranker. Larger pools give the reranker more
            room to find precision; default of 30 matches the standard
            production RAG pattern (retrieve-50 / rerank-to-10 is the
            published baseline, scaled down here for the typical
            ticker-pre-filtered query that returns fewer hits to begin
            with). Ignored when ``rerank`` is ``None``.

    Returns:
        List of RetrievalResult sorted by similarity descending. For hybrid
        results, ``similarity`` carries the blended score and the per-side
        components are exposed via ``vector_score`` / ``keyword_score``.
        When rerank is set, results are reordered by ``rerank_score``
        (highest first) and the score + method are stamped onto each
        result.
    """
    if method not in ("vector", "keyword", "hybrid"):
        raise ValueError(f"unknown method: {method!r}")
    if not 0.0 <= vector_weight <= 1.0:
        raise ValueError(f"vector_weight must be in [0,1]; got {vector_weight}")
    if rerank is not None and rerank_input_n < top_k:
        raise ValueError(
            f"rerank_input_n ({rerank_input_n}) must be >= top_k ({top_k}); "
            "otherwise the rerank pool can't yield enough survivors."
        )

    # When reranking, fetch a wider candidate pool from the underlying
    # method so the reranker has room to surface precision wins.
    retrieve_k = rerank_input_n if rerank is not None else top_k

    if method == "vector":
        results = _vector_search(query, tickers, doc_types, min_date, retrieve_k)
    elif method == "keyword":
        results = _keyword_search(query, tickers, doc_types, min_date, retrieve_k)
    else:
        # Hybrid — retrieve retrieve_k from each side, blend the union, return retrieve_k overall.
        v = _vector_search(query, tickers, doc_types, min_date, retrieve_k)
        k = _keyword_search(query, tickers, doc_types, min_date, retrieve_k)
        results = _blend(v, k, vector_weight=vector_weight, top_k=retrieve_k)

    n_pre_rerank = len(results)

    if rerank is not None and results:
        # Imported lazily so a bare ``from nousergon_lib.rag import
        # retrieve`` keeps the sentence-transformers / torch install
        # optional. The reranker is registered + memoized at module
        # scope so repeat calls within a Lambda container share the
        # model handle + the in-process score cache.
        from .rerank import get_reranker
        reranker = get_reranker(rerank)
        results = reranker.rerank(query, results, top_k=top_k)

    logger.info(
        "RAG retrieve: query=%r method=%s tickers=%s top_k=%d rerank=%s "
        "rerank_input_n=%d → %d candidates → %d results",
        query[:60], method, tickers, top_k, rerank,
        rerank_input_n if rerank is not None else 0,
        n_pre_rerank, len(results),
    )
    return results


# ── Vector path ─────────────────────────────────────────────────────────────


def _vector_search(
    query: str,
    tickers: list[str] | None,
    doc_types: list[str] | None,
    min_date: date | None,
    top_k: int,
) -> list[RetrievalResult]:
    """pgvector cosine top-K via the HNSW index on ``embedding``."""
    from .db import get_connection
    from .embeddings import embed_query

    query_vec = embed_query(query)
    where, params = _build_metadata_where(tickers, doc_types, min_date)
    # First param is the SELECT-clause vector for the similarity score; then any
    # metadata-filter params; then the ORDER BY vector + LIMIT.
    select_params: list = [str(query_vec)]
    order_params: list = [str(query_vec), top_k]
    # S608 false positive: `where` is built by _build_metadata_where, which
    # only ever composes fixed column/operator strings ("d.ticker = ANY(%s)"
    # etc.) — every actual value flows through the %s placeholders passed to
    # cur.execute below, never string-interpolated. ruff has no data-flow
    # analysis for this rule and flags any f-string touching a `sql = f"""`
    # block regardless (see pyproject.toml's S603 note for the same class of
    # ruff limitation).
    sql = f"""
        SELECT c.id, c.content, d.ticker, d.doc_type, d.filed_date, c.section_label,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM rag.chunks c
        JOIN rag.documents d ON c.document_id = d.id
        {where}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """  # noqa: S608
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, select_params + params + order_params)
            rows = cur.fetchall()

    return [
        RetrievalResult(
            chunk_id=str(row[0]),
            content=row[1],
            ticker=row[2],
            doc_type=row[3],
            filed_date=row[4],
            section_label=row[5],
            similarity=round(float(row[6]), 4),
            retrieval_method="vector",
            vector_score=round(float(row[6]), 4),
        )
        for row in rows
    ]


# ── Keyword path ────────────────────────────────────────────────────────────


def _keyword_search(
    query: str,
    tickers: list[str] | None,
    doc_types: list[str] | None,
    min_date: date | None,
    top_k: int,
) -> list[RetrievalResult]:
    """PostgreSQL Full-Text Search top-K via ``ts_rank_cd`` over the GIN
    index on ``content_tsv``. May return fewer than ``top_k`` rows when
    the corpus has fewer matches.

    OR-relaxed tsquery: ``plainto_tsquery`` defaults to AND-of-all-terms,
    which zeros out natural-language queries like "ABBV competitive moat
    in immunology" — no chunk contains all four stemmed terms even
    though each term individually has 100s-1000s of hits. We rewrite
    the query to OR-of-terms so any chunk with at least one term is a
    candidate, then rely on ``ts_rank_cd`` to surface chunks with the
    most overlap + best proximity at the top. Standard RAG candidate-
    generation pattern (recall at the gate, precision via the ranker).
    Verified empirically against the prod corpus — AND mode returned
    0-3 hits per typical qual-analyst query; OR + ts_rank_cd returns
    1k-16k candidates with the genuinely-relevant chunks ranked at top.
    """
    from .db import get_connection

    # OR-relaxed tsquery via PG's own parser:
    #   plainto_tsquery handles tokenize + stem + stopword removal,
    #   then replace its default '&' with '|' for OR semantics.
    or_tsquery = (
        "to_tsquery('english', "
        "replace(plainto_tsquery('english', %s)::text, ' & ', ' | '))"
    )

    where, params = _build_metadata_where(tickers, doc_types, min_date)
    fts_clause = f"c.content_tsv @@ {or_tsquery}"
    if where:
        where = f"{where} AND {fts_clause}"
    else:
        where = f"WHERE {fts_clause}"
    rank_params: list = [query]   # ts_rank_cd query in SELECT
    fts_params: list = [query]    # OR-tsquery in WHERE
    order_params: list = [top_k]
    # S608 false positive: same as _vector_search above — `where` and
    # `or_tsquery` are fixed SQL-fragment strings (column names, PG function
    # calls, %s placeholders), never user data. Actual values flow through
    # rank_params/fts_params/order_params to cur.execute below.
    sql = f"""
        SELECT c.id, c.content, d.ticker, d.doc_type, d.filed_date, c.section_label,
               ts_rank_cd(c.content_tsv, {or_tsquery}) AS rank
        FROM rag.chunks c
        JOIN rag.documents d ON c.document_id = d.id
        {where}
        ORDER BY rank DESC
        LIMIT %s
    """  # noqa: S608
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, rank_params + params + fts_params + order_params)
            rows = cur.fetchall()

    return [
        RetrievalResult(
            chunk_id=str(row[0]),
            content=row[1],
            ticker=row[2],
            doc_type=row[3],
            filed_date=row[4],
            section_label=row[5],
            similarity=round(float(row[6]), 4),
            retrieval_method="keyword",
            keyword_score=round(float(row[6]), 4),
        )
        for row in rows
    ]


# ── Hybrid blender ──────────────────────────────────────────────────────────


def _blend(
    vector_results: list[RetrievalResult],
    keyword_results: list[RetrievalResult],
    *,
    vector_weight: float,
    top_k: int,
) -> list[RetrievalResult]:
    """Union the two candidate lists by ``chunk_id``, min-max normalize
    each side within the union, and return the top-K by blended score.

    Missing-side handling: a chunk that appears only in the vector list
    has ``keyword_score=None`` pre-blend → 0.0 after normalization. The
    weighted blend then leans entirely on the vector side for that chunk
    (and vice versa). This is the standard convention; the eval harness
    in PR 4 will expose whether it's the right choice or whether RRF
    (Reciprocal Rank Fusion) better fits the corpus.

    Pure function: no DB / network. Easy to test.
    """
    # Index by chunk_id; first-seen wins for content/metadata (the two SQL
    # paths return the same row data from rag.chunks for a given chunk_id,
    # so this is a no-op identity).
    by_id: dict[str, RetrievalResult] = {}
    v_scores: dict[str, float] = {}
    k_scores: dict[str, float] = {}

    for r in vector_results:
        if r.chunk_id is None:
            continue
        by_id[r.chunk_id] = r
        if r.vector_score is not None:
            v_scores[r.chunk_id] = r.vector_score

    for r in keyword_results:
        if r.chunk_id is None:
            continue
        # Don't overwrite vector-side row data; just carry the keyword score.
        by_id.setdefault(r.chunk_id, r)
        if r.keyword_score is not None:
            k_scores[r.chunk_id] = r.keyword_score

    if not by_id:
        return []

    v_norm = _minmax_normalize(v_scores)
    k_norm = _minmax_normalize(k_scores)

    blended: list[RetrievalResult] = []
    for cid, base in by_id.items():
        vn = v_norm.get(cid, 0.0)  # missing-side floor
        kn = k_norm.get(cid, 0.0)
        combined = vector_weight * vn + (1.0 - vector_weight) * kn
        blended.append(
            RetrievalResult(
                chunk_id=cid,
                content=base.content,
                ticker=base.ticker,
                doc_type=base.doc_type,
                filed_date=base.filed_date,
                section_label=base.section_label,
                similarity=round(combined, 4),
                retrieval_method="hybrid",
                vector_score=v_scores.get(cid),
                keyword_score=k_scores.get(cid),
                combined_score=round(combined, 4),
            )
        )

    blended.sort(key=lambda r: r.similarity, reverse=True)
    return blended[:top_k]


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale a score map into [0, 1].

    Edge cases: empty map → empty map. All-equal scores (incl. single
    element) → every score normalizes to 1.0 (treat as uniform-best,
    not uniform-worst — preserves their candidacy under the blend).
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Metadata WHERE builder (shared by vector + keyword paths) ───────────────


def _build_metadata_where(
    tickers: list[str] | None,
    doc_types: list[str] | None,
    min_date: date | None,
) -> tuple[str, list]:
    """Build the metadata pre-filter WHERE clause + param list.

    Returns ('', []) when no filters apply. Caller composes additional
    conditions (e.g. FTS @@) on top of the returned WHERE string.
    """
    conditions: list[str] = []
    params: list = []
    if tickers:
        conditions.append("d.ticker = ANY(%s)")
        params.append(tickers)
    if doc_types:
        conditions.append("d.doc_type = ANY(%s)")
        params.append(doc_types)
    if min_date:
        conditions.append("d.filed_date >= %s")
        params.append(min_date)
    if not conditions:
        return "", []
    return "WHERE " + " AND ".join(conditions), params


# ── Ingestion helpers (unchanged from v0.5.7) ───────────────────────────────


def document_exists(
    ticker: str,
    doc_type: str,
    filed_date: date,
    source: str,
    external_id: str | None = None,
) -> bool:
    """Check if a document has already been ingested (dedup).

    ``external_id`` is the per-article identity news ingestion must pass
    (config#2957) — without it, distinct same-day articles for one
    (ticker, source) collapse onto a single existing row. Every other
    doc_type omits it; the query keys on the original 4-column shape.
    """
    from .db import execute_query

    if external_id is not None:
        rows = execute_query(
            "SELECT 1 FROM rag.documents WHERE ticker=%s AND doc_type=%s "
            "AND filed_date=%s AND source=%s AND external_id=%s LIMIT 1",
            (ticker, doc_type, filed_date, source, external_id),
        )
    else:
        rows = execute_query(
            "SELECT 1 FROM rag.documents WHERE ticker=%s AND doc_type=%s AND filed_date=%s AND source=%s LIMIT 1",
            (ticker, doc_type, filed_date, source),
        )
    return len(rows) > 0


def ingest_document(
    ticker: str,
    sector: str | None,
    doc_type: str,
    source: str,
    filed_date: date,
    title: str | None,
    url: str | None,
    chunks: list[dict],
    external_id: str | None = None,
    mirror_to_parquet: bool = True,
) -> str | None:
    """Ingest a document and its embedded chunks into the RAG store.

    Args:
        ticker: Stock symbol.
        sector: GICS sector (optional).
        doc_type: '10-K', '10-Q', 'earnings_transcript', 'thesis', 'news'.
        source: 'sec_edgar', 'fmp', 'alpha_engine'.
        filed_date: Date the document was filed/published.
        title: Document title (optional).
        url: Source URL (optional).
        chunks: List of dicts with keys: content, section_label, embedding.
        external_id: Stable per-article identity, REQUIRED for reliable
            dedup when doc_type='news' (config#2957) — omit for every
            other doc_type.
        mirror_to_parquet: also write the document + chunks to the S3
            parquet batch tier (config#2958) after the Neon insert
            commits. Best-effort — a mirror failure is logged, not
            raised (Neon already has the durable copy by that point);
            see ``parquet_mirror.py``'s module docstring. Set False for
            callers that intentionally don't want the [rag-parquet]
            extra's pandas/pyarrow dependency pulled at import time
            (mirroring is a plain top-level ``import pandas`` inside the
            helper, so this only matters if that import would fail).

    Returns:
        Document UUID on success, None on failure.
    """
    from .db import get_connection

    if document_exists(ticker, doc_type, filed_date, source, external_id):
        logger.debug("Skipping duplicate: %s %s %s", ticker, doc_type, filed_date)
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO rag.documents (ticker, sector, doc_type, source, filed_date, title, url, external_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (ticker, sector, doc_type, source, filed_date, title, url, external_id),
            )
            row = cur.fetchone()
            if row is None:
                # INSERT ... RETURNING id not returning a row would mean
                # the insert silently didn't happen — a driver/DB-level
                # anomaly, not a normal "no match" case (unlike a SELECT).
                raise RuntimeError(
                    f"INSERT INTO rag.documents returned no row for "
                    f"{ticker} {doc_type} {filed_date}"
                )
            doc_id = row[0]

            chunk_params = [
                (doc_id, i, c["content"], c.get("section_label"), str(c["embedding"]))
                for i, c in enumerate(chunks)
            ]
            from psycopg2.extras import execute_batch
            execute_batch(
                cur,
                """INSERT INTO rag.chunks (document_id, chunk_index, content, section_label, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)""",
                chunk_params,
                page_size=100,
            )

    logger.info("Ingested %s %s %s: %d chunks", ticker, doc_type, filed_date, len(chunks))

    if mirror_to_parquet:
        # mirror_document_to_parquet already catches and logs everything
        # internally (see its module docstring); this call-site guard is
        # defense-in-depth so ingest_document's own "never raises for the
        # mirror" contract holds even against a bug in that internal
        # handling, not just its currently-intended behavior.
        try:
            from .parquet_mirror import mirror_document_to_parquet

            mirror_document_to_parquet(
                document_id=str(doc_id),
                ticker=ticker,
                sector=sector,
                doc_type=doc_type,
                source=source,
                filed_date=filed_date,
                title=title,
                url=url,
                chunks=chunks,
            )
        except Exception:
            logger.error(
                "Parquet mirror call site raised for document %s — Neon "
                "insert already committed, ingest is unaffected",
                doc_id, exc_info=True,
            )

    return str(doc_id)
