"""Tests for the RAG rerank primitive (alpha-engine-lib v0.11.0+).

Covers:

1. ``RerankCache`` — key derivation, hit/miss/eviction.
2. ``CrossEncoderReranker`` — reorders by predict() output; cache short-
   circuits repeat scoring; passthrough when candidates empty. Real
   BAAI model load is mocked via the ``_model`` slot so tests don't
   download 600MB of weights.
3. ``retrieve(rerank=...)`` — fetches ``rerank_input_n`` from the
   underlying method, passes through to the reranker, truncates to
   ``top_k``; rerank=None preserves legacy behavior; invalid
   ``rerank_input_n < top_k`` raises.

``LLMJudgeReranker`` (formerly tested here) was removed v0.34.0. See
the ``rerank`` module docstring for the no-lift finding +
institutional rerank-revisit path (domain-finetune CE on retrieval
triples, not LLM-judge).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from alpha_engine_lib.rag.rerank import (
    CrossEncoderReranker,
    RerankCache,
    _RERANKER_REGISTRY,
    get_reranker,
)
from alpha_engine_lib.rag.retrieval import RetrievalResult, retrieve


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_reranker_registry():
    """Reset the module-level reranker registry between tests."""
    _RERANKER_REGISTRY.clear()
    yield
    _RERANKER_REGISTRY.clear()


def _make_result(content: str, chunk_id: str, similarity: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        ticker="AAPL",
        doc_type="10-K",
        filed_date=date(2026, 1, 1),
        section_label="Item 1A",
        similarity=similarity,
        chunk_id=chunk_id,
        retrieval_method="hybrid",
    )


# ── RerankCache ─────────────────────────────────────────────────────────────


class TestRerankCache:
    def test_key_stable_across_calls(self) -> None:
        c = RerankCache()
        k1 = c.make_key("competitive moat", "chunk-42")
        k2 = c.make_key("competitive moat", "chunk-42")
        assert k1 == k2

    def test_key_disambiguates_query_and_chunk(self) -> None:
        c = RerankCache()
        assert c.make_key("a", "1") != c.make_key("a", "2")
        assert c.make_key("a", "1") != c.make_key("b", "1")

    def test_get_miss_returns_none(self) -> None:
        c = RerankCache()
        assert c.get("missing") is None

    def test_put_then_get_returns_value(self) -> None:
        c = RerankCache()
        c.put("k", 0.9)
        assert c.get("k") == 0.9

    def test_lru_evicts_oldest(self) -> None:
        c = RerankCache(maxsize=2)
        c.put("a", 1.0)
        c.put("b", 2.0)
        c.put("c", 3.0)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2.0
        assert c.get("c") == 3.0

    def test_get_moves_to_end_protecting_from_eviction(self) -> None:
        c = RerankCache(maxsize=2)
        c.put("a", 1.0)
        c.put("b", 2.0)
        c.get("a")        # touch — promote to MRU
        c.put("c", 3.0)   # evicts "b" (now LRU), not "a"
        assert c.get("a") == 1.0
        assert c.get("b") is None
        assert c.get("c") == 3.0


# ── CrossEncoderReranker ────────────────────────────────────────────────────


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    Predict returns a score list controlled by ``scores_by_content`` so
    tests can pin the expected reorder. Also counts predict calls so
    cache-hit tests can assert the model wasn't re-queried.
    """

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self.scores_by_content = scores_by_content
        self.predict_calls = 0

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.predict_calls += 1
        return [self.scores_by_content[content] for _query, content in pairs]


class TestCrossEncoderReranker:
    def test_reorders_by_score_descending(self) -> None:
        fake_model = _FakeCrossEncoder({"low": 0.1, "mid": 0.5, "high": 0.9})
        reranker = CrossEncoderReranker(_model=fake_model)
        candidates = [
            _make_result("low", "c1"),
            _make_result("mid", "c2"),
            _make_result("high", "c3"),
        ]
        out = reranker.rerank("query", candidates, top_k=3)
        assert [r.content for r in out] == ["high", "mid", "low"]
        assert out[0].rerank_score == pytest.approx(0.9)
        assert out[0].rerank_method == "cross_encoder"

    def test_truncates_to_top_k(self) -> None:
        fake_model = _FakeCrossEncoder({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4})
        reranker = CrossEncoderReranker(_model=fake_model)
        candidates = [_make_result(c, f"id_{c}") for c in "abcd"]
        out = reranker.rerank("query", candidates, top_k=2)
        assert [r.content for r in out] == ["d", "c"]
        assert len(out) == 2

    def test_cache_hit_skips_model_call(self) -> None:
        fake_model = _FakeCrossEncoder({"x": 0.5, "y": 0.7})
        reranker = CrossEncoderReranker(_model=fake_model)
        candidates = [_make_result("x", "cx"), _make_result("y", "cy")]
        reranker.rerank("query", candidates, top_k=2)
        first_call_count = fake_model.predict_calls
        # Second call with the same (query, chunk_id) pairs — all hits.
        reranker.rerank("query", candidates, top_k=2)
        assert fake_model.predict_calls == first_call_count

    def test_partial_cache_hits_only_score_new(self) -> None:
        fake_model = _FakeCrossEncoder({"x": 0.5, "y": 0.7, "z": 0.3})
        reranker = CrossEncoderReranker(_model=fake_model)
        # Warm cache with x, y.
        reranker.rerank("query", [_make_result("x", "cx"), _make_result("y", "cy")], top_k=2)
        # New call adds z — only the new pair should be predicted.
        out = reranker.rerank(
            "query",
            [_make_result("x", "cx"), _make_result("y", "cy"), _make_result("z", "cz")],
            top_k=3,
        )
        # predict was called twice total (once warmup, once for z-only).
        assert fake_model.predict_calls == 2
        assert [r.content for r in out] == ["y", "x", "z"]

    def test_empty_candidates_returns_empty(self) -> None:
        reranker = CrossEncoderReranker(_model=_FakeCrossEncoder({}))
        assert reranker.rerank("query", [], top_k=5) == []

    def test_missing_sentence_transformers_raises_with_hint(self) -> None:
        reranker = CrossEncoderReranker()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match=r"alpha-engine-lib\[rerank\]"):
                reranker._ensure_model()


# ── retrieve(rerank=...) integration ────────────────────────────────────────


class _ScriptedReranker:
    """Reranker that reverses the candidate order — a deterministic
    contrast with the underlying retrieval ordering so tests can assert
    rerank actually ran."""

    name = "scripted"

    def rerank(self, query, candidates, top_k):
        out = list(reversed(candidates))[:top_k]
        for i, r in enumerate(out):
            r.rerank_score = float(len(candidates) - i)
            r.rerank_method = self.name
        return out


class TestRetrieveWithRerank:
    def test_rerank_none_preserves_legacy_behavior(self) -> None:
        with patch("alpha_engine_lib.rag.retrieval._vector_search") as mock_vec:
            mock_vec.return_value = [_make_result("a", "1")]
            results = retrieve("q", top_k=1, method="vector")
            mock_vec.assert_called_once_with("q", None, None, None, 1)
            assert results[0].rerank_method is None

    def test_rerank_set_fetches_input_n_then_truncates_to_top_k(self) -> None:
        # Build 10 candidates from the underlying retriever; rerank reverses + truncates to 3.
        candidates = [_make_result(f"doc-{i}", f"id-{i}", similarity=0.1 * i) for i in range(10)]
        _RERANKER_REGISTRY["scripted"] = _ScriptedReranker()
        with patch("alpha_engine_lib.rag.retrieval._vector_search") as mock_vec:
            mock_vec.return_value = candidates
            results = retrieve(
                "q", top_k=3, method="vector",
                rerank="scripted", rerank_input_n=10,
            )
            # _vector_search called with the wider rerank_input_n, not top_k.
            mock_vec.assert_called_once_with("q", None, None, None, 10)
            # Reranker reversed the order then truncated to 3.
            assert [r.content for r in results] == ["doc-9", "doc-8", "doc-7"]
            assert all(r.rerank_method == "scripted" for r in results)

    def test_rerank_input_n_below_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="rerank_input_n.*top_k"):
            retrieve("q", top_k=10, method="vector", rerank="cross_encoder", rerank_input_n=5)

    def test_unknown_reranker_raises(self) -> None:
        with patch("alpha_engine_lib.rag.retrieval._vector_search") as mock_vec:
            mock_vec.return_value = [_make_result("a", "1")]
            with pytest.raises(ValueError, match="Unknown reranker"):
                retrieve("q", top_k=1, method="vector", rerank="bogus", rerank_input_n=1)


# ── get_reranker registry ───────────────────────────────────────────────────


class TestGetReranker:
    def test_memoizes_instance(self) -> None:
        _RERANKER_REGISTRY["x"] = _ScriptedReranker()
        a = get_reranker("x")
        b = get_reranker("x")
        assert a is b

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reranker"):
            get_reranker("nonsense")
