"""Tests for the hybrid-retrieval primitive (PR 2 of 5 in the
BM25 + vector arc).

Three layers exercised:

1. ``_minmax_normalize`` — pure score-normalization helper. Edge cases:
   empty, single-element, all-equal scores.
2. ``_blend`` — pure union/normalize/blend function. Verified without
   any DB. Covers single-side, both-sides, missing-side floor, weight
   extremes (1.0 → pure vector, 0.0 → pure keyword), top_k truncation.
3. ``retrieve(method=…)`` — dispatch + SQL shape. Live DB is out of
   scope here (consumer-side integration tests own that surface);
   we mock psycopg2 cursors and assert the right SQL fragments fire.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from nousergon_lib.rag.retrieval import (
    RetrievalResult,
    _blend,
    _minmax_normalize,
    retrieve,
)

# ── _minmax_normalize ───────────────────────────────────────────────────────


class TestMinMaxNormalize:
    def test_empty_returns_empty(self) -> None:
        assert _minmax_normalize({}) == {}

    def test_single_element_normalizes_to_one(self) -> None:
        # All-equal (incl. n=1) → uniform-best, not uniform-worst — so the
        # element keeps full candidacy in the blended ranking.
        assert _minmax_normalize({"a": 0.42}) == {"a": 1.0}

    def test_all_equal_scores_normalize_to_one(self) -> None:
        out = _minmax_normalize({"a": 0.5, "b": 0.5, "c": 0.5})
        assert out == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_spread_normalizes_to_unit_interval(self) -> None:
        out = _minmax_normalize({"lo": 0.1, "mid": 0.4, "hi": 0.9})
        assert out["lo"] == pytest.approx(0.0)
        assert out["hi"] == pytest.approx(1.0)
        assert out["mid"] == pytest.approx((0.4 - 0.1) / (0.9 - 0.1))

    def test_negative_values_handled(self) -> None:
        # Vector cosine can be negative; normalization should still
        # bracket [0, 1] across the candidate set.
        out = _minmax_normalize({"a": -0.2, "b": 0.6})
        assert out == {"a": 0.0, "b": 1.0}


# ── _blend ──────────────────────────────────────────────────────────────────


def _result(chunk_id: str, *, vec: float | None = None, kw: float | None = None) -> RetrievalResult:
    """Test fixture for a RetrievalResult with one or both side-scores."""
    return RetrievalResult(
        content=f"content for {chunk_id}",
        ticker="TEST",
        doc_type="10-K",
        filed_date=date(2025, 1, 1),
        section_label=None,
        similarity=vec if vec is not None else (kw or 0.0),
        chunk_id=chunk_id,
        retrieval_method=("vector" if vec is not None else "keyword"),
        vector_score=vec,
        keyword_score=kw,
    )


class TestBlend:
    def test_empty_inputs_return_empty(self) -> None:
        assert _blend([], [], vector_weight=0.7, top_k=10) == []

    def test_single_side_only_vector(self) -> None:
        # Only vector results; keyword side empty. After blending,
        # candidates should still rank in vector order with combined_score
        # equal to vector_weight (since v_norm=1 for top, 0 for bottom).
        v = [_result("a", vec=0.9), _result("b", vec=0.5), _result("c", vec=0.1)]
        out = _blend(v, [], vector_weight=0.7, top_k=10)
        assert [r.chunk_id for r in out] == ["a", "b", "c"]
        # Top should be 0.7 (vector_weight × 1.0); bottom 0.0; mid ~0.7×(0.4/0.8)=0.35
        assert out[0].combined_score == pytest.approx(0.7)
        assert out[2].combined_score == pytest.approx(0.0)
        assert all(r.retrieval_method == "hybrid" for r in out)

    def test_single_side_only_keyword(self) -> None:
        k = [_result("a", kw=0.8), _result("b", kw=0.2)]
        out = _blend([], k, vector_weight=0.7, top_k=10)
        assert [r.chunk_id for r in out] == ["a", "b"]
        # Top → keyword_weight = 0.3; bottom 0.0
        assert out[0].combined_score == pytest.approx(0.3)
        assert out[1].combined_score == pytest.approx(0.0)

    def test_both_sides_full_overlap(self) -> None:
        # Same candidates appear in both sides with same ordering →
        # blended ordering matches both. Scores compose linearly.
        v = [_result("a", vec=0.9), _result("b", vec=0.5)]
        k = [_result("a", kw=0.8), _result("b", kw=0.2)]
        out = _blend(v, k, vector_weight=0.7, top_k=10)
        # 'a' normalizes to v=1, k=1 → combined = 0.7+0.3 = 1.0
        # 'b' normalizes to v=0, k=0 → combined = 0.0
        assert out[0].chunk_id == "a"
        assert out[0].combined_score == pytest.approx(1.0)
        assert out[1].combined_score == pytest.approx(0.0)

    def test_disjoint_sides_get_floored_on_missing_side(self) -> None:
        # 'a' only in vector, 'b' only in keyword. Missing side → floor 0.0
        # post-normalize; the present side normalizes to 1.0 (single elem).
        # 'a' → 0.7×1.0 + 0.3×0.0 = 0.7
        # 'b' → 0.7×0.0 + 0.3×1.0 = 0.3
        v = [_result("a", vec=0.9)]
        k = [_result("b", kw=0.5)]
        out = _blend(v, k, vector_weight=0.7, top_k=10)
        assert [r.chunk_id for r in out] == ["a", "b"]
        assert out[0].combined_score == pytest.approx(0.7)
        assert out[1].combined_score == pytest.approx(0.3)

    def test_vector_weight_one_equals_pure_vector(self) -> None:
        v = [_result("a", vec=0.9), _result("b", vec=0.5)]
        k = [_result("c", kw=0.8)]  # would-be heavy keyword but ignored
        out = _blend(v, k, vector_weight=1.0, top_k=10)
        # 'a' v_norm=1, k missing → 1.0 × 1.0 + 0 × 0 = 1.0
        # 'b' v_norm=0, k missing → 0.0
        # 'c' v missing, k_norm=1 → 1.0 × 0.0 + 0.0 × 1.0 = 0.0
        scores = {r.chunk_id: r.combined_score for r in out}
        assert scores["a"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.0)
        assert scores["c"] == pytest.approx(0.0)

    def test_vector_weight_zero_equals_pure_keyword(self) -> None:
        v = [_result("a", vec=0.9)]
        k = [_result("b", kw=0.8), _result("c", kw=0.2)]
        out = _blend(v, k, vector_weight=0.0, top_k=10)
        scores = {r.chunk_id: r.combined_score for r in out}
        assert scores["b"] == pytest.approx(1.0)  # k_norm=1, weight=1.0
        assert scores["c"] == pytest.approx(0.0)
        assert scores["a"] == pytest.approx(0.0)  # vector ignored

    def test_top_k_truncation(self) -> None:
        v = [_result(f"v{i}", vec=1.0 - i * 0.1) for i in range(5)]
        k = [_result(f"k{i}", kw=1.0 - i * 0.1) for i in range(5)]
        out = _blend(v, k, vector_weight=0.5, top_k=3)
        assert len(out) == 3

    def test_blended_results_carry_both_side_scores(self) -> None:
        v = [_result("a", vec=0.9)]
        k = [_result("a", kw=0.8)]
        out = _blend(v, k, vector_weight=0.7, top_k=10)
        assert out[0].vector_score == 0.9
        assert out[0].keyword_score == 0.8
        assert out[0].combined_score is not None
        assert out[0].retrieval_method == "hybrid"

    def test_results_with_no_chunk_id_are_skipped(self) -> None:
        # Defensive: a RetrievalResult missing chunk_id can't be unioned;
        # blender silently drops it rather than crashing.
        bad = RetrievalResult(
            content="x", ticker="X", doc_type="10-K",
            filed_date=date(2025, 1, 1), section_label=None,
            similarity=0.5, chunk_id=None, vector_score=0.5,
        )
        out = _blend([bad], [], vector_weight=0.7, top_k=10)
        assert out == []


# ── retrieve(method=…) dispatch ─────────────────────────────────────────────


class TestRetrieveDispatch:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            retrieve("query", method="invalid")  # type: ignore[arg-type]

    def test_vector_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="vector_weight"):
            retrieve("query", method="hybrid", vector_weight=1.5)
        with pytest.raises(ValueError, match="vector_weight"):
            retrieve("query", method="hybrid", vector_weight=-0.1)

    def test_default_method_is_vector_for_back_compat(self) -> None:
        # PR 2 keeps default = "vector" so existing callers (qual_tools.py)
        # see no behavior change. PR 3 explicitly opts qual_tools into
        # method="hybrid".
        with patch("nousergon_lib.rag.retrieval._vector_search") as v_mock, \
             patch("nousergon_lib.rag.retrieval._keyword_search") as k_mock:
            v_mock.return_value = []
            retrieve("query")
        v_mock.assert_called_once()
        k_mock.assert_not_called()

    def test_keyword_method_dispatches_to_keyword_only(self) -> None:
        with patch("nousergon_lib.rag.retrieval._vector_search") as v_mock, \
             patch("nousergon_lib.rag.retrieval._keyword_search") as k_mock:
            k_mock.return_value = []
            retrieve("query", method="keyword")
        v_mock.assert_not_called()
        k_mock.assert_called_once()

    def test_hybrid_method_calls_both_paths_then_blend(self) -> None:
        with patch("nousergon_lib.rag.retrieval._vector_search") as v_mock, \
             patch("nousergon_lib.rag.retrieval._keyword_search") as k_mock:
            v_mock.return_value = [_result("a", vec=0.9)]
            k_mock.return_value = [_result("a", kw=0.8)]
            out = retrieve("query", method="hybrid", vector_weight=0.7)
        v_mock.assert_called_once()
        k_mock.assert_called_once()
        assert len(out) == 1
        assert out[0].retrieval_method == "hybrid"
        assert out[0].vector_score == 0.9
        assert out[0].keyword_score == 0.8


# ── SQL fragment shape ──────────────────────────────────────────────────────


class TestSQLShape:
    """Verify the SQL fired by each path uses the right index-friendly
    fragments. We're not running the SQL — just asserting the strings.
    """

    def test_vector_path_uses_pgvector_cosine(self) -> None:
        captured: dict = {}

        def fake_cursor():
            cur = MagicMock()
            cur.execute.side_effect = lambda sql, params: captured.update(sql=sql, params=params)
            cur.fetchall.return_value = []
            cm = MagicMock()
            cm.__enter__ = lambda self: cur
            cm.__exit__ = lambda self, *a: None
            return cm

        conn_cm = MagicMock()
        conn_cm.__enter__ = lambda self: MagicMock(cursor=fake_cursor)
        conn_cm.__exit__ = lambda self, *a: None

        with patch("nousergon_lib.rag.retrieval.embed_query", create=True), \
             patch("nousergon_lib.rag.embeddings.embed_query", return_value=[0.1] * 512), \
             patch("nousergon_lib.rag.db.get_connection", return_value=conn_cm):
            retrieve("query", method="vector", top_k=5)

        assert "<=>" in captured["sql"], "vector path must use pgvector <=> cosine operator"
        assert "embedding" in captured["sql"]
        # No FTS fragments leaked into the vector-only SQL.
        assert "ts_rank_cd" not in captured["sql"]
        assert "plainto_tsquery" not in captured["sql"]

    def test_keyword_path_uses_ts_rank_cd_and_gin_predicate(self) -> None:
        captured: dict = {}

        def fake_cursor():
            cur = MagicMock()
            cur.execute.side_effect = lambda sql, params: captured.update(sql=sql, params=params)
            cur.fetchall.return_value = []
            cm = MagicMock()
            cm.__enter__ = lambda self: cur
            cm.__exit__ = lambda self, *a: None
            return cm

        conn_cm = MagicMock()
        conn_cm.__enter__ = lambda self: MagicMock(cursor=fake_cursor)
        conn_cm.__exit__ = lambda self, *a: None

        with patch("nousergon_lib.rag.db.get_connection", return_value=conn_cm):
            retrieve("research and development", method="keyword", top_k=5)

        # Score fragment + GIN-friendly @@ predicate + 'english' config.
        assert "ts_rank_cd" in captured["sql"]
        assert "plainto_tsquery('english'" in captured["sql"]
        assert "content_tsv @@" in captured["sql"]
        # OR-relaxed tsquery — plainto_tsquery's default & semantics
        # rewritten to | so natural-language queries don't zero out.
        # See _keyword_search docstring.
        assert "to_tsquery('english'" in captured["sql"]
        assert "' & ', ' | '" in captured["sql"]
        # No vector fragments.
        assert "<=>" not in captured["sql"]
