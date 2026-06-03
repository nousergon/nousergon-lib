"""Performance attribution — Brinson-Fachler decomposition + Cariño linking.

Pure stdlib, data-source-agnostic (takes plain group weight/return dicts, not a
broker client), so it's unit-testable in isolation and reusable unchanged across
front ends. This *explains* a portfolio's active return vs a benchmark — where
the over/under-performance came from — without ever prescribing a trade.

**Single period (Brinson-Fachler).** For each group *i* (typically a sector),
decompose the active return ``R_p − R_b`` into:

  - **Allocation** ``(w_p,i − w_b,i) · (r_b,i − R_b)`` — did over/under-weighting a
    group (vs its benchmark weight) help, given how that group did vs the whole
    benchmark? (The Fachler refinement subtracts the total benchmark return
    ``R_b`` so allocation rewards over-weighting *out-performing* groups.)
  - **Selection** ``w_b,i · (r_p,i − r_b,i)`` — did picks within a group beat the
    group's benchmark, at benchmark weight?
  - **Interaction** ``(w_p,i − w_b,i) · (r_p,i − r_b,i)`` — the cross term.

The three effects summed over all groups equal the arithmetic active return.

**Multi-period (Cariño linking).** Arithmetic single-period effects don't simply
add across periods because returns compound geometrically. Cariño (1999) scales
each period's effects by ``k_t / k`` so the linked effects sum *exactly* to the
geometric cumulative active return — the institutional standard for chaining a
Brinson attribution through time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Returns within ~1e-12 of each other are treated as equal for the linking-limit
# branches (where the divided-difference coefficient hits its L'Hôpital limit).
_EPS = 1e-12


@dataclass(frozen=True)
class GroupAttribution:
    """Per-group Brinson-Fachler effects (all in return-fraction units)."""

    group: str
    allocation: float
    selection: float
    interaction: float

    @property
    def total(self) -> float:
        return self.allocation + self.selection + self.interaction


@dataclass(frozen=True)
class BrinsonResult:
    """A single- or linked-period attribution: per-group effects + totals.

    ``portfolio_return`` / ``benchmark_return`` are the (cumulative, for a linked
    result) totals; ``active_return`` is their difference and equals
    ``total_effect`` up to floating-point error.
    """

    groups: list[GroupAttribution] = field(default_factory=list)
    allocation: float = 0.0
    selection: float = 0.0
    interaction: float = 0.0
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0

    @property
    def active_return(self) -> float:
        return self.portfolio_return - self.benchmark_return

    @property
    def total_effect(self) -> float:
        return self.allocation + self.selection + self.interaction


def brinson_fachler(
    weights_p: dict[str, float],
    returns_p: dict[str, float],
    weights_b: dict[str, float],
    returns_b: dict[str, float],
) -> BrinsonResult:
    """Single-period Brinson-Fachler attribution over the union of groups.

    Args are group → weight / group → return maps for the portfolio (``_p``) and
    benchmark (``_b``). Weights are fractions of their respective totals. The
    group sets need not match: a group the benchmark doesn't hold defaults its
    benchmark return to the overall benchmark return ``R_b`` (neutral allocation
    baseline); a group the portfolio doesn't hold defaults its portfolio return
    to that group's benchmark return (zero selection). Missing weights default to
    0.

    Totals: ``R_p = Σ w_p,i·r_p,i``, ``R_b = Σ w_b,i·r_b,i`` over the groups given
    — so for the decomposition to tie to the true portfolio/benchmark returns,
    the weights should each sum to ~1 across the groups passed in.
    """
    groups = sorted(set(weights_p) | set(returns_p) | set(weights_b) | set(returns_b))

    r_b_total = sum(weights_b.get(g, 0.0) * returns_b.get(g, 0.0) for g in groups)
    r_p_total = sum(weights_p.get(g, 0.0) * returns_p.get(g, 0.0) for g in groups)

    per_group: list[GroupAttribution] = []
    for g in groups:
        wp = weights_p.get(g, 0.0)
        wb = weights_b.get(g, 0.0)
        # Benchmark return for a group the benchmark doesn't hold → the overall
        # benchmark return (so its allocation baseline is neutral). Portfolio
        # return for a group the portfolio doesn't hold → that group's benchmark
        # return (so selection is zero — you can't pick within what you don't own).
        rb = returns_b.get(g, r_b_total)
        rp = returns_p.get(g, rb)
        allocation = (wp - wb) * (rb - r_b_total)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        per_group.append(GroupAttribution(g, allocation, selection, interaction))

    return BrinsonResult(
        groups=per_group,
        allocation=sum(a.allocation for a in per_group),
        selection=sum(a.selection for a in per_group),
        interaction=sum(a.interaction for a in per_group),
        portfolio_return=r_p_total,
        benchmark_return=r_b_total,
    )


def _carino_coefficient(r_p: float, r_b: float) -> float:
    """Cariño linking coefficient ``(ln(1+r_p) − ln(1+r_b)) / (r_p − r_b)``.

    At ``r_p == r_b`` this is the L'Hôpital limit ``1 / (1 + r)``. Requires
    ``1 + r > 0`` for both (a ≤ −100% period return has no log).
    """
    if 1.0 + r_p <= 0.0 or 1.0 + r_b <= 0.0:
        raise ValueError("Cariño linking requires period returns > -100%")
    if abs(r_p - r_b) < _EPS:
        return 1.0 / (1.0 + r_b)
    return (math.log(1.0 + r_p) - math.log(1.0 + r_b)) / (r_p - r_b)


def link_periods(periods: list[BrinsonResult]) -> BrinsonResult:
    """Cariño-link a sequence of single-period attributions into one result.

    Each period's effects are scaled by ``k_t / k`` so the linked per-group and
    total effects sum exactly to the **geometric** cumulative active return
    ``(∏(1+r_p,t) − 1) − (∏(1+r_b,t) − 1)``. Group identities are matched by name
    across periods (a group absent in a period contributes 0 that period).

    Single period → returned unchanged. Empty → an all-zero result. Raises
    ``ValueError`` if any period return is ≤ −100% (no log).
    """
    if not periods:
        return BrinsonResult()
    if len(periods) == 1:
        return periods[0]

    cum_p = math.prod(1.0 + p.portfolio_return for p in periods) - 1.0
    cum_b = math.prod(1.0 + p.benchmark_return for p in periods) - 1.0
    k_overall = _carino_coefficient(cum_p, cum_b)

    # Accumulate scaled effects per group and for the totals.
    alloc: dict[str, float] = {}
    select: dict[str, float] = {}
    interact: dict[str, float] = {}
    for p in periods:
        k_t = _carino_coefficient(p.portfolio_return, p.benchmark_return)
        scale = k_t / k_overall
        for ga in p.groups:
            alloc[ga.group] = alloc.get(ga.group, 0.0) + ga.allocation * scale
            select[ga.group] = select.get(ga.group, 0.0) + ga.selection * scale
            interact[ga.group] = interact.get(ga.group, 0.0) + ga.interaction * scale

    per_group = [
        GroupAttribution(g, alloc.get(g, 0.0), select.get(g, 0.0), interact.get(g, 0.0))
        for g in sorted(alloc.keys() | select.keys() | interact.keys())
    ]
    return BrinsonResult(
        groups=per_group,
        allocation=sum(a.allocation for a in per_group),
        selection=sum(a.selection for a in per_group),
        interaction=sum(a.interaction for a in per_group),
        portfolio_return=cum_p,
        benchmark_return=cum_b,
    )
