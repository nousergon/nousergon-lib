"""RAG reranking — reorder retrieval candidates under a joint query+doc model.

Reranking sits between candidate generation (`retrieve(method="hybrid", ...)`)
and LLM consumption. Hybrid retrieval over a wide candidate pool (e.g. top-30)
gives high recall; rerank then provides precision by scoring each
``(query, document)`` pair jointly under a model that's purpose-built for
relevance ranking.

**One implementation shipped:** :class:`CrossEncoderReranker` — local
BAAI ``bge-reranker-v2-m3`` (or any cross-encoder loadable via
``sentence-transformers``). Zero external API surface, deterministic,
~100-300ms latency on CPU at top-50. The institutional/SOTA rerank
pattern for production RAG is domain-finetuned cross-encoders;
general-purpose CE models (like our bundled BAAI default) are tier-2
SOTA, dominant for general-domain RAG but expected to regress on
specialized corpora until finetuned on domain-labeled (query, doc,
relevance) pairs.

**``LLMJudgeReranker`` removed v0.34.0** (2026-05-25). The class
fired one Haiku call per (query, doc) pair — a tier-5 SOTA approach
useful for novel rubrics that lack training labels, not for general
relevance reranking. Empirical eval on the SEC-filings RAG corpus
(2026-05-12, EXPERIMENTS.md) measured -14.2% recall@10 vs the hybrid
w=0.7 baseline. Removed per ``[[preference_llm_calls_confined_to_research_module]]``
+ the no-lift finding. Re-attempting LLM-judge rerank in the future
goes inside alpha-engine-research (where LLM calls belong); the
institutional rerank-revisit path is domain-finetune the CE model
on operator-labeled retrieval triples.

The :class:`RerankCache` (LRU, keyed by ``sha256(query) + chunk_id``)
is process-local — no cross-run persistence, because query embeddings
drift with corpus updates and rerank scores are cheap to recompute.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# Cap how many ``(query, doc)`` pairs the in-process cache retains so a
# long-running Lambda container doesn't grow unbounded. 1024 entries is
# ~8 queries × top-50 reranks × 2x slack — plenty of headroom for the
# 6-sector × ~25-ticker research run's qual-tool burst.
_DEFAULT_CACHE_MAXSIZE = 1024


# ── Cache ───────────────────────────────────────────────────────────────────


class RerankCache:
    """Process-local LRU cache for rerank scores keyed by ``(query, chunk_id)``.

    Keeps a tight cap on memory (``maxsize`` entries, eviction in
    insertion order) so a hot Lambda container that processes many
    distinct queries doesn't accumulate unbounded state. Lifetime is
    the container — no cross-invocation persistence (the
    ``RAG_RERANK_CACHE_TTL`` knob is intentionally absent because
    Lambda /tmp + the implied IO cost would exceed the cost of the
    rerank itself for typical query volumes).
    """

    def __init__(self, maxsize: int = _DEFAULT_CACHE_MAXSIZE) -> None:
        self._store: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def make_key(query: str, chunk_id: str | None) -> str:
        # chunk_id can be None for results that didn't carry a primary key
        # back from the retriever (legacy ``vector_score-only`` paths); fall
        # back to hashing the content snippet plus the doc tuple so we
        # still get a stable key per ``(query, doc)`` pair.
        suffix = chunk_id if chunk_id is not None else "no_chunk_id"
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return f"{digest}:{suffix}"

    def get(self, key: str) -> float | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, score: float) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = score
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


# ── Reranker protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Reranker(Protocol):
    """Score-and-reorder a candidate list under a joint query+doc model.

    Implementations may consult a cache, but the protocol surface is
    pure: take a query + candidate list, return the same candidates
    reordered (and optionally truncated to ``top_k``) with
    per-result ``rerank_score`` populated.
    """

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


# ── Cross-encoder (local model) ─────────────────────────────────────────────


@dataclass
class CrossEncoderReranker:
    """Local cross-encoder reranker.

    Default model is BAAI ``bge-reranker-v2-m3``: a multilingual
    cross-encoder published 2024 at ~600MB on disk, ~100-300ms latency
    per query at top-50 on CPU. Any sentence-transformers
    :class:`CrossEncoder`-compatible model can be substituted via
    ``model_name``.

    The underlying ``sentence-transformers`` install is gated behind
    the ``[rerank]`` extra so callers that only use vector/hybrid
    retrieval don't pay the ~2GB torch + transformers + model-download
    install cost. Importing this module does NOT load the model;
    initialization happens lazily on the first :meth:`rerank` call so
    a non-rerank import path stays cheap.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    cache: RerankCache = field(default_factory=RerankCache)
    name: str = "cross_encoder"
    # When unset, defer model load until first rerank() call. Tests
    # patch this directly with a callable returning predict-able scores
    # to exercise the score-aware reorder path without paying the
    # ~600MB model download.
    _model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            # Imported lazily so a bare ``from nousergon_lib.rag import
            # retrieve`` stays cheap on consumers that don't rerank.
            # [rerank] extra deliberately not installed in CI (heavy
            # torch+transformers dep), matches test.yml's install-skip
            # precedent (sentence-transformers / flow-doctor are both
            # deferred until first call / mocked in tests).
            from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install 'nousergon-lib[rerank]'"
            ) from exc
        logger.info("Loading cross-encoder model: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        uncached_pairs: list[tuple[int, str]] = []
        scores: list[float | None] = [None] * len(candidates)
        for idx, cand in enumerate(candidates):
            key = self.cache.make_key(query, cand.chunk_id)
            cached = self.cache.get(key)
            if cached is not None:
                scores[idx] = cached
            else:
                uncached_pairs.append((idx, cand.content))

        if uncached_pairs:
            model = self._ensure_model()
            pair_inputs = [[query, content] for _, content in uncached_pairs]
            # ``predict`` returns one logit per pair; higher = more relevant.
            # Type cast through ``list(map(float, ...))`` keeps tests
            # happy when a numpy array is returned by the real model and
            # when a plain list is returned by the test fake.
            raw = model.predict(pair_inputs)  # type: ignore[attr-defined]
            fresh_scores = list(map(float, raw))
            for (idx, _content), score in zip(uncached_pairs, fresh_scores):
                scores[idx] = score
                self.cache.put(
                    self.cache.make_key(query, candidates[idx].chunk_id),
                    score,
                )

        return _attach_and_sort(candidates, scores, self.name, top_k)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _attach_and_sort(
    candidates: list[RetrievalResult],
    scores: list[float | None],
    method_name: str,
    top_k: int,
) -> list[RetrievalResult]:
    """Stamp ``rerank_score`` + ``rerank_method`` on each result and sort.

    ``RetrievalResult`` is a dataclass — set the fields directly. If the
    score list contains ``None`` for any candidate (shouldn't happen
    under correct caller flow, but defensive), those candidates sort to
    the tail so we don't drop them silently.
    """
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
    out: list[RetrievalResult] = []
    for cand, score in paired[:top_k]:
        cand.rerank_score = score  # type: ignore[attr-defined]
        cand.rerank_method = method_name  # type: ignore[attr-defined]
        out.append(cand)
    return out


# ── Factory for the retrieve() integration ──────────────────────────────────


# Module-level registry of named reranker instances. Lazily populated
# the first time :func:`get_reranker` resolves a given name, then
# memoized so subsequent retrieve(rerank="cross_encoder", ...) calls
# share the same cache + model handle within the Lambda container.
_RERANKER_REGISTRY: dict[str, Reranker] = {}


def get_reranker(name: str) -> Reranker:
    """Resolve a named reranker, constructing + caching on first use.

    Supported names: ``"cross_encoder"`` (local BAAI bge-reranker-v2-m3
    via sentence-transformers). Tests register fakes by writing
    directly to :data:`_RERANKER_REGISTRY` before the
    ``retrieve(rerank=...)`` call.

    ``"llm_judge"`` was removed v0.34.0 — see module docstring for the
    no-lift finding + the institutional rerank-revisit path
    (domain-finetune the CE model, not LLM-judge).
    """
    if name in _RERANKER_REGISTRY:
        return _RERANKER_REGISTRY[name]
    if name == "cross_encoder":
        instance: Reranker = CrossEncoderReranker()
    else:
        raise ValueError(
            f"Unknown reranker {name!r}; supported: 'cross_encoder'"
        )
    _RERANKER_REGISTRY[name] = instance
    return instance
