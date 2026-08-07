"""The decision set — one definition of *which names the system is deciding on*.

``rag-corpus-policy.md`` §2.1 states the rule and names the constant:

    the **scanner focus list** (top-N by the scanner's own ranking —
    ``ATTRACTIVENESS_FEED_TOP_N``, today 60) — the names passed forward to
    research and prediction

That constant existed in the policy and in **no code anywhere in the fleet**
(measured 2026-08-07). Each consumer picked its own 60, and they were not the
same 60.

WHAT WENT WRONG WITHOUT IT (alpha-engine-config-I6630)
------------------------------------------------------
The scanner publishes several 60-wide cuts, and they come from two different
rankings:

* ``scanner_candidates`` — the **momentum gate** cut: ``tech_score`` top-60
  (RSI / MACD / MA50 / MA200 / 20d momentum, equally weighted) plus the most
  oversold-by-RSI names. No fundamentals.
* ``attractiveness_top_60`` — the top 60 of the **6-pillar attractiveness**
  rank over the whole ~900-name board (quality / value / momentum / growth /
  stewardship / defensiveness).

The predictor consumes ``attractiveness_top_20`` (alpha-engine-config-I4983,
Brian 2026-07-28) and Think Tank's gap-fill window is the attractiveness
top-60. The RAG corpus scoped itself to ``scanner_candidates``. Measured on
the live 2026-08-07 membership artifact: the corpus's 60 and the predictor's
20 overlapped on **2 names**, so 18 of 20 scored names carried no fresh
evidence and the vendor spend went to names no arm decides on.

The funnel was never a funnel. This module makes it one, by construction:

    attractiveness rank over the full board
      → top ``ATTRACTIVENESS_FEED_TOP_N`` (60)   the feed / evidence set
        → top ``PREDICTOR_CUT_TOP_N`` (20)       the scored set

``assert_cut_nests`` is the invariant that would have caught it. It reads the
membership artifact alone — no ranking re-derivation, no producer knowledge.

WHAT THIS MODULE IS NOT
-----------------------
Not an IO layer. It holds no bucket, key, or client: consumers resolve the
membership artifact however their context sources it (S3 pointer, dated key,
fixture) and pass the parsed dict in. Mirrors ``nousergon_lib.universe``'s
posture for exactly the same reason — the predicate is shared, the plumbing
is not.

Not the owner of *which* cut the predictor consumes. That is published by the
producer in the artifact's ``predictor_universe_cut`` field
(``crucible-research/scoring/universe_membership.py``), so changing it stays a
producer-side, versioned, reviewable decision. :func:`predictor_cut_name`
reads that field rather than asserting a value.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ATTRACTIVENESS_FEED_TOP_N",
    "PREDICTOR_CUT_TOP_N",
    "FEED_CUT_NAME",
    "DecisionSetContractError",
    "attractiveness_cut_name",
    "predictor_cut_name",
    "cut_tickers",
    "assert_cut_nests",
]


ATTRACTIVENESS_FEED_TOP_N = 60
"""Width of the decision set fed forward to research, evidence ingestion and
the challenger arms (``rag-corpus-policy.md`` §2.1).

60 is count-matched to the scanner's ``momentum_top_n`` so the gate cut and
the rank cut stay directly comparable as champion/challenger arms
(``champion-challenger-policy.md`` §4 — an arm's win must not be confounded
between selection rule and breadth), and to Think Tank's
``thinktank/run.py::GAP_FILL_TOP_N``.
"""

PREDICTOR_CUT_TOP_N = 20
"""Width of the cut actually scored by the predictor — the head of the feed
set. Count-matched to Think Tank's ``CHALLENGER_TOP_N`` so champion and
challenger submit at equal breadth."""

FEED_CUT_NAME = f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}"
"""Cut name in ``universe_membership/{date}/membership.json::cuts`` carrying
the decision set. Derived, never a second literal."""


class DecisionSetContractError(RuntimeError):
    """A membership artifact violates the decision-set contract.

    Deliberately an error rather than a warning: every failure this class
    covers is silent at the point it happens and surfaces only as degraded
    predictions or a vendor bill weeks later.
    """


def attractiveness_cut_name(n: int) -> str:
    """Cut name for the top-``n`` attractiveness rank cut.

    The producer emits these for ``_RANK_CUTS`` (20, 25, 60). Consumers that
    need a width this module does not name should slice the artifact's
    ``ranks`` table rather than inventing a cut name the producer never wrote.
    """
    return f"attractiveness_top_{int(n)}"


def predictor_cut_name(membership: dict[str, Any]) -> str:
    """The cut the predictor scores, as published by the producer.

    Reads ``predictor_universe_cut`` from the artifact. Raises rather than
    defaulting: a consumer guessing this value is how a cut change becomes
    invisible to half the fleet.
    """
    name = membership.get("predictor_universe_cut")
    if not isinstance(name, str) or not name:
        raise DecisionSetContractError(
            "membership artifact carries no 'predictor_universe_cut' "
            f"(got {name!r}). The producer names the scored cut in the "
            "artifact so consumers never hardcode it; its absence means the "
            "artifact predates the contract or was written by something that "
            "is not the scanner."
        )
    return name


def cut_tickers(membership: dict[str, Any], cut_name: str) -> list[str]:
    """Uppercased, whitespace-stripped tickers of ``cut_name``.

    Raises :class:`DecisionSetContractError` when the cut is absent or empty —
    never returns a partial or widened set. The available cut names are named
    in the message, because the failure this replaces ("it returned nothing
    and the pipeline carried on") is diagnosable only with them.
    """
    cuts = membership.get("cuts") or {}
    entry = cuts.get(cut_name) or {}
    # Normalise BEFORE filtering: a whitespace-only entry is truthy, so
    # filtering first lets "  " through and it lands downstream as "".
    normalised = (str(t).strip().upper() for t in (entry.get("tickers") or []))
    tickers = [t for t in normalised if t]
    if not tickers:
        raise DecisionSetContractError(
            f"membership artifact carries no non-empty cuts.{cut_name}. "
            f"Available cuts: {sorted(cuts.keys())}."
        )
    return tickers


def assert_cut_nests(
    membership: dict[str, Any],
    *,
    inner: str,
    outer: str,
) -> None:
    """Assert every ticker of cut ``inner`` is also in cut ``outer``.

    The funnel invariant. ``assert_cut_nests(m, inner="attractiveness_top_20",
    outer="attractiveness_top_60")`` holds by construction when both are cuts
    of one ranking, and fails loudly the moment the two are computed from
    different rankings — which is exactly the state
    alpha-engine-config-I6630 found live and no test could see.

    Raises :class:`DecisionSetContractError` naming the escaping tickers.
    """
    inner_set = set(cut_tickers(membership, inner))
    outer_set = set(cut_tickers(membership, outer))
    escaped = sorted(inner_set - outer_set)
    if escaped:
        raise DecisionSetContractError(
            f"cut {inner!r} is not nested inside cut {outer!r}: "
            f"{len(escaped)} of {len(inner_set)} ticker(s) escape "
            f"({escaped}). Two cuts that are meant to be a funnel are being "
            f"computed from different rankings — see "
            f"alpha-engine-config-I6630."
        )
