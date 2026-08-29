"""The FILLING arms' selection rule — one implementation, two consumers.

Normative sources: ``champion-challenger-policy.md`` §3–4 (an arm is measured
on the same basis it is served) and ``shared-code-policy.md`` (the
second-adoption trigger). Tracked as ``alpha-engine-config-I9338``.

WHY THIS MODULE EXISTS
----------------------
``alpha-engine-config-I9307`` required the champion arm to write its picks to a
comparable artifact so it could be scored on the same basis as its challengers.
The shadow is written in **crucible-research** (``producers/filling_arms.py``),
because the executor synthesizes picks only for the arm that is *currently*
champion — an executor-side capture would go dark on the incumbent the moment
the pointer moved (§3).

The consequence was that the selection rule existed in two repos:

* ``crucible-research/producers/filling_arms.py`` — ``rank_by_alpha``,
  ``rank_to_score``;
* ``crucible-executor/executor/champion.py`` — ``_rank_to_score`` and the
  ``sort_values("predicted_alpha", ascending=False)`` in each filling-arm
  handler.

WHY IT MATTERS EVEN THOUGH THE RULE IS THREE LINES
--------------------------------------------------
Small is what makes the drift risk insidious. A divergence between the two
copies would not crash anything — it would produce a shadow that scores a rule
the executor does not serve, i.e. **a track record that means something other
than what it claims**. That is worse than the defect ``-I9307`` closed, because
the artifact would still look healthy.

WHAT IS AND IS NOT HERE
-----------------------
PURE. Zero I/O, zero logging, no third-party dependency — importable from a
Lambda without pulling pandas, the same constraint the rest of
:mod:`nousergon_lib.arena` holds itself to.

The *pool loaders* stay with their consumers: which S3 artifact an arm's pool
comes from is a per-arm fact, and lifting it would drag S3 into a pure package
to share nothing. Only the RULE is shared, because only the rule is the same.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "DEFAULT_SCORE_CEILING",
    "DEFAULT_SCORE_FLOOR",
    "rank_by_alpha",
    "rank_to_score",
]

#: The score band the selection-producer slot maps rank onto. Declared here
#: because both consumers must agree on it for their scores to be comparable,
#: but every call site takes them as arguments — the executor reads them from
#: its private ``risk.yaml`` (``champion_score_floor`` /
#: ``champion_score_ceiling``), and a library constant must never be the thing
#: that silently overrides an operator's configured value.
DEFAULT_SCORE_FLOOR = 60.0
DEFAULT_SCORE_CEILING = 95.0


def rank_to_score(rank_fraction: float, floor: float, ceiling: float) -> float:
    """Map a within-pool rank fraction in ``[0, 1]`` (0 = best) onto
    ``[floor, ceiling]``: best rank -> ``ceiling``, worst rank -> ``floor``.

    **Monotone by construction**, which is the load-bearing property: the
    transform fixes the RENDERING of an arm's ranking and can never change the
    ranking itself. Downstream the score feeds the executor's
    ``min_score_to_enter`` gate, so arms whose native score scales differ
    (a 0-100 technical composite, a subjective analyst rating, a rank band) are
    put on one scale here rather than being gated at different effective
    thresholds — which would make the slot a comparison of score scales rather
    than of selection rules (``champion-challenger-policy.md`` §4).

    ``rank_fraction`` is clamped rather than rejected: a caller computing
    ``rank / (pool_size - 1)`` on a one-name pool legitimately produces 0.0,
    and floating error at the ends must not become an exception on a serving
    path.

    Raises ``ValueError`` when the band is empty or inverted — that is a
    misconfiguration, not an edge case, and a silently inverted band would
    rank every arm backwards while looking healthy.
    """
    if ceiling <= floor:
        raise ValueError(f"score ceiling ({ceiling}) must exceed score floor ({floor})")
    rank_fraction = min(max(float(rank_fraction), 0.0), 1.0)
    return ceiling - rank_fraction * (ceiling - floor)


def rank_by_alpha(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """``(ticker, alpha)`` pairs sorted by alpha DESCENDING, ties broken by
    ticker ASCENDING.

    **The tie-break is explicit, and that is the point.** Two names carrying
    the same alpha must land in the same order on every machine and in every
    process, or the shadow the leaderboard scores and the set the executor
    serves can differ on a tie — and the recorded track record becomes
    unverifiable, which is the only asset this loop has
    (``champion-challenger-policy.md`` §3.1). Python's sort is stable, so
    relying on input order would make the result a property of whatever
    produced the rows (a DataFrame's row order, a dict's insertion order)
    rather than of the rule.
    """
    return sorted(rows, key=lambda row: (-row[1], row[0]))
