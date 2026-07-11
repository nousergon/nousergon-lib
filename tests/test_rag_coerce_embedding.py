"""Tests for ``nousergon_lib.rag.coerce_embedding`` — the pgvector
representation-normalization chokepoint (config#2221).

Lifts the guarantee that used to live at a single call site
(``alpha-engine-data/rag/pipelines/filing_change_detection.py::_embedding_to_f32``,
shipped in nousergon-data PR #747 as the call-site fix for the 2026-07-11
weekly-freshness break) up to the owned ``nousergon_lib.rag`` chokepoint, so
ANY consumer that reads a ``vector`` column normalizes identically and no
future consumer doing the natural ``np.array(...)`` can reintroduce the
``TypeError: float() ... not 'Vector'`` crash.

pgvector's read representation is build-dependent — ndarray / list / naked
``pgvector.Vector`` / raw string — so this pins that ALL of them normalize to a
float32 ndarray, and that a raw string (unregistered codec) stays FAIL-LOUD.
"""

from __future__ import annotations

import numpy as np
import pytest

from nousergon_lib.rag import coerce_embedding


class _VectorLike:
    """Mimics ``pgvector.Vector``: exposes ``.to_numpy()`` and, crucially,
    NO ``__array__``/``__len__``/``__iter__`` — so a naive
    ``np.array(obj, dtype=np.float32)`` raises exactly as prod did on the
    weekly data spot (2026-07-11)."""

    def __init__(self, values):
        self._values = list(values)

    def to_numpy(self):
        return np.array(self._values, dtype=np.float32)


def test_naive_cast_reproduces_the_regression():
    # Lock the failure mode the chokepoint defends against: a to_numpy-only
    # object is NOT coercible by np.array directly.
    with pytest.raises(TypeError):
        np.array(_VectorLike([1.0, 2.0, 3.0]), dtype=np.float32)


def test_vector_like_object_is_coerced():
    out = coerce_embedding(_VectorLike([1.0, 2.0, 3.0]))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.float32))


def test_ndarray_passthrough_downcasts_to_float32():
    # The register_vector "happy path" — value already an ndarray (float64).
    out = coerce_embedding(np.array([1.5, 2.5], dtype=np.float64))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1.5, 2.5], dtype=np.float32))


def test_list_is_coerced():
    out = coerce_embedding([0.1, 0.2, 0.3])
    assert out.dtype == np.float32
    assert out.shape == (3,)


def test_tuple_is_coerced():
    out = coerce_embedding((0.1, 0.2, 0.3))
    assert out.dtype == np.float32
    assert out.shape == (3,)


def test_raw_string_fails_loud():
    # A raw bracketed string means the codec silently didn't register — must
    # surface, never be silently re-parsed.
    with pytest.raises(TypeError):
        coerce_embedding("[1,2,3]")


def test_single_numeric_string_still_fails_loud():
    # np.asarray("1.5", dtype=float32) would SUCCEED (array(1.5)) — the exact
    # silent-mis-parse the explicit str guard exists to prevent. A stringified
    # scalar must not sneak through as a 0-d "embedding".
    with pytest.raises(TypeError):
        coerce_embedding("1.5")


def test_bytes_fails_loud():
    with pytest.raises(TypeError):
        coerce_embedding(b"[1,2,3]")


def test_real_pgvector_vector_if_available():
    # When pgvector is installed (it is via the [rag] extra), lock the exact
    # prod type, not just the stub.
    pgvector = pytest.importorskip("pgvector")
    Vector = pgvector.Vector
    out = coerce_embedding(Vector([4.0, 5.0, 6.0]))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([4, 5, 6], dtype=np.float32))


def test_exported_from_rag_namespace():
    import nousergon_lib.rag as rag

    assert "coerce_embedding" in rag.__all__
    assert rag.coerce_embedding is coerce_embedding
