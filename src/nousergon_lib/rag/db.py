"""Neon PostgreSQL connection management for RAG.

Uses psycopg2 with connection pooling suitable for Lambda (short-lived
connections via Neon's built-in pgbouncer pooler).

Requires: RAG_DATABASE_URL environment variable (Neon pooled connection string).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def coerce_embedding(value) -> NDArray[np.float32]:
    """Normalize a pgvector ``vector`` column read to a float32 ndarray.

    THE chokepoint for the representation-fragile guarantee that
    :func:`get_connection` aspires to via pgvector's ``register_vector``.
    pgvector's read representation has flip-flopped across builds — a numpy
    ndarray in most versions, a plain ``list`` in some, a naked
    ``pgvector.Vector`` (which has NO numpy interop: no
    ``__array__``/``__len__``/``__iter__``) in the build that resolved on the
    weekly data spot, and a raw bracketed *string* when the codec never
    registered at all. Any consumer that SELECTs a ``vector`` column and does
    the natural ``np.array(value, dtype=np.float32)`` therefore crashes with
    ``TypeError: float() argument must be ... not 'Vector'`` the moment the
    build shifts under it (the 2026-07-11 weekly-freshness break at RAG Step
    8/9, filing change detection). Route every embedding read through here so
    that class of crash cannot recur regardless of which representation the
    active pgvector/psycopg2 build hands back.

    Accepts a numpy ndarray, a ``pgvector.Vector`` (normalized via its
    documented ``.to_numpy()``), or a plain sequence of floats (``list`` /
    ``tuple``).

    FAIL-LOUD, by design: a ``str``/``bytes`` value means the pgvector codec
    silently failed to register on the connection, so the DB handed back the
    stringified vector. We refuse to parse it — a silently mis-parsed (or
    single-element-coerced) embedding would corrupt every downstream cosine
    computation without a trace. The caller must see the unregistered-codec
    failure and fix the connection path, not have it papered over here.
    """
    if isinstance(value, (str, bytes, bytearray)):
        snippet = repr(value)[:48]
        raise TypeError(
            f"coerce_embedding received a raw {type(value).__name__} "
            f"({snippet}...): the pgvector codec did not register on this "
            "connection, so a stringified vector was returned. Refusing to "
            "silently parse it — read vectors through "
            "nousergon_lib.rag.get_connection (which registers the codec) with "
            "pgvector installed. This is fail-loud on purpose (config#2221)."
        )
    if hasattr(value, "to_numpy"):  # pgvector.Vector — not numpy-coercible
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)

_DATABASE_URL: str | None = None


def _get_url() -> str:
    global _DATABASE_URL
    if _DATABASE_URL is None:
        _DATABASE_URL = os.environ.get("RAG_DATABASE_URL")
        if not _DATABASE_URL:
            raise RuntimeError("RAG_DATABASE_URL not set — cannot connect to vector DB")
    return _DATABASE_URL


@contextmanager
def get_connection():
    """Context manager for a database connection.

    Opens a new connection per call (Neon pooler handles connection reuse
    server-side). Commits on success, rolls back on exception.

    Vector-column reads: this registers pgvector's psycopg2 codec so ``vector``
    columns *usually* deserialize to numpy arrays — but that representation is
    codec-and-build-dependent and has flip-flopped across pgvector versions
    (ndarray / list / naked ``pgvector.Vector`` / raw string). Do NOT rely on
    the codec's output type. The ENFORCED guarantee lives in
    :func:`coerce_embedding`: normalize every embedding you read through it and
    the representation-fragility can't reach your arithmetic (config#2221).
    """
    conn = psycopg2.connect(_get_url())
    # Best-effort: register pgvector's psycopg2 codec so SELECTs on `vector`
    # columns deserialize to numpy arrays where the resolved build supports it.
    # This is NOT the guarantee — the codec's read representation is build-
    # dependent (see get_connection docstring / config#2221); coerce_embedding
    # is the enforced normalization. Must run per-connection because psycopg2
    # scopes type adapters to the connection.
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_query(sql: str, params: tuple | list = ()) -> list[dict]:
    """Execute a SELECT query and return results as list of dicts."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def execute_insert(sql: str, params: tuple | list = ()) -> None:
    """Execute an INSERT/UPDATE statement."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def execute_batch(sql: str, params_list: list[tuple]) -> None:
    """Execute a batch of INSERT statements efficiently."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list, page_size=100)


def is_available() -> bool:
    """Check if the RAG database is reachable. Never raises.

    NOTE (2026-04-14): currently has zero callers inside alpha-engine-data.
    The ingestion pipelines call ``get_connection()`` directly, which
    hard-fails on connect errors (correct behavior while the system is
    unstable). Kept in the module in case retrieval-side consumers want
    a non-raising probe; flag for deletion if still unused after the
    cross-repo audit completes.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        logger.warning("RAG database unavailable: %s", e)
        return False
