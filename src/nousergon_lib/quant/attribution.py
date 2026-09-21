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

**Security-level (name-level) attribution.** The Brinson decomposition above
groups by sector/GICS group. ``active_weights``, ``security_contributions``,
``reconcile`` and ``top_drivers`` below do the same active-return explanation
at the individual-security level — "which *names*, not which sectors, drove
the active return" — using the simpler contribution identity (no
allocation/selection/interaction split, since there's no group to allocate
across at the leaf level):

  - **Active weight** ``w_p,i − w_b,i``
  - **Active contribution** ``(w_p,i − w_b,i) · r_i`` — the security's share of
    active return, at that security's *single* return (portfolio and
    benchmark are assumed to hold the same security at the same realized
    return; there is no portfolio-only vs. benchmark-only return per name).

Summed over all securities, the active contributions equal the active return
only when weights are exhaustive and consistent with the return series used
to compute ``portfolio_return``/``benchmark_return`` — ``reconcile`` makes
that check explicit and returns the residual rather than assuming it's zero.

**Units — same convention as ``brinson_fachler`` above.** Weights are
fractions of portfolio/benchmark total (0.0121, not 1.21); returns are
return-fractions (0.05 for +5%, not 5.0); contributions (``w · r``) are
therefore also return-fractions. Callers passing percent-scaled inputs will
silently get contributions scaled by 100x — this is the single likeliest
defect in a caller of this module.
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


# --------------------------------------------------------------------------
# Security-level (name-level) attribution.
# --------------------------------------------------------------------------

#: Classification of a security's role in the active bet.
#:
#: - ``"not_held"``   — the portfolio holds ~0 weight in this name (regardless
#:   of the benchmark's weight).
#: - ``"overweight"``  — held, and the portfolio's weight exceeds the
#:   benchmark's (positive active weight).
#: - ``"underweight"`` — held, and the portfolio's weight is below the
#:   benchmark's (negative active weight).
#: - ``"held"``        — held, and the active weight is ~0 (portfolio and
#:   benchmark weight this name equally — no bet either way).
#:
#: Mutually exclusive and exhaustive; ``_EPS`` (1e-12) is the zero tolerance.


@dataclass(frozen=True)
class SecurityContribution:
    """Per-security active-return contribution (all in return-fraction units).

    ``port_weight`` / ``bench_weight`` are fractions of their respective
    totals (0 if the security isn't held on that side); ``ret`` is the
    security's realized return over the period, shared by both sides (see
    module docstring — there is no separate portfolio-only vs.
    benchmark-only return per name at this granularity).
    """

    symbol: str
    port_weight: float
    bench_weight: float
    ret: float
    port_contribution: float
    bench_contribution: float
    active_contribution: float
    classification: str

    @property
    def active_weight(self) -> float:
        return self.port_weight - self.bench_weight


@dataclass(frozen=True)
class SecurityContributionResult:
    """Result of :func:`security_contributions`.

    ``missing`` lists symbols that appear in ``port_weights`` or
    ``bench_weights`` but have no entry in the ``returns`` map passed in —
    an explicit, sorted collection the caller MUST handle rather than a
    silent zero contribution. A symbol with a weight but no return is a data
    gap, not a zero-return security.
    """

    contributions: list[SecurityContribution] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def active_weights(
    port_weights: dict[str, float],
    bench_weights: dict[str, float],
) -> dict[str, float]:
    """Per-symbol active weight ``w_p − w_b`` over the union of symbols.

    Weights are fractions (see module docstring). A symbol absent from
    either map is treated as 0.0 weight on that side — e.g. a symbol only in
    ``bench_weights`` gets a negative active weight (fully underweight), and
    vice versa.
    """
    symbols = set(port_weights) | set(bench_weights)
    return {s: port_weights.get(s, 0.0) - bench_weights.get(s, 0.0) for s in symbols}


def security_contributions(
    port_weights: dict[str, float],
    bench_weights: dict[str, float],
    returns: dict[str, float],
) -> SecurityContributionResult:
    """Per-security active-return contributions over the union of symbols.

    For each symbol in ``port_weights | bench_weights`` that also has an
    entry in ``returns``:

      - ``port_contribution = w_p · r``
      - ``bench_contribution = w_b · r``
      - ``active_contribution = (w_p − w_b) · r``

    A symbol present in ``port_weights`` or ``bench_weights`` but absent
    from ``returns`` is **not** silently given a zero contribution — it is
    omitted from ``contributions`` and listed (sorted) in
    ``SecurityContributionResult.missing``, which the caller must check.
    Fail loud: a caller that ignores ``missing`` and sums
    ``active_contribution`` is silently understating the active return by
    whatever weight the missing symbols carried, which ``reconcile`` below
    would then surface as a residual outside tolerance.

    Returns are shared by both sides — see module docstring for why there is
    no separate ``returns_p`` / ``returns_b``.
    """
    symbols = sorted(set(port_weights) | set(bench_weights))
    missing = [s for s in symbols if s not in returns]

    contributions: list[SecurityContribution] = []
    for s in symbols:
        if s not in returns:
            continue
        wp = port_weights.get(s, 0.0)
        wb = bench_weights.get(s, 0.0)
        r = returns[s]
        active_w = wp - wb

        if abs(wp) < _EPS:
            classification = "not_held"
        elif active_w > _EPS:
            classification = "overweight"
        elif active_w < -_EPS:
            classification = "underweight"
        else:
            classification = "held"

        contributions.append(
            SecurityContribution(
                symbol=s,
                port_weight=wp,
                bench_weight=wb,
                ret=r,
                port_contribution=wp * r,
                bench_contribution=wb * r,
                active_contribution=active_w * r,
                classification=classification,
            )
        )

    return SecurityContributionResult(contributions=contributions, missing=missing)


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of :func:`reconcile` — the correctness gate for security-level attribution.

    ``residual = (portfolio_return − benchmark_return) − Σ active_contribution``.
    A caller presenting "top drivers" of active return MUST check
    ``within_tolerance`` first and refuse to present plausible-looking
    drivers when it's ``False`` — a residual outside tolerance means the
    contributions don't actually explain the stated active return (stale
    weights, a missing symbol not routed through ``missing``, percent/
    fraction unit mismatch, etc.), and presenting them anyway would be
    showing a story, not the reconciled truth.
    """

    residual: float
    within_tolerance: bool
    tolerance: float


def reconcile(
    contributions: list[SecurityContribution],
    portfolio_return: float,
    benchmark_return: float,
    tolerance: float = 1e-9,
) -> ReconciliationResult:
    """Check that security-level active contributions explain the active return.

    Returns the residual ``(portfolio_return − benchmark_return) −
    Σ active_contribution`` as a first-class value (never swallowed) plus
    whether ``abs(residual) <= tolerance`` (inclusive boundary). ``tolerance``
    is in the same return-fraction units as everything else in this module —
    pass a value that reflects real floating-point/rounding slack for your
    inputs, not business-significance (that judgment belongs to the caller
    deciding what to do with a residual outside tolerance).
    """
    active_sum = sum(c.active_contribution for c in contributions)
    residual = (portfolio_return - benchmark_return) - active_sum
    return ReconciliationResult(
        residual=residual,
        within_tolerance=abs(residual) <= tolerance,
        tolerance=tolerance,
    )


def top_drivers(contributions: list[SecurityContribution], n: int) -> list[SecurityContribution]:
    """The ``n`` largest ``contributions`` by absolute active contribution.

    Ties (equal ``abs(active_contribution)``, including two zero
    contributions) break by ascending ``symbol`` so the output is
    deterministic regardless of input order. ``n <= 0`` returns ``[]``;
    ``n`` larger than ``len(contributions)`` returns all of them, sorted.
    """
    ordered = sorted(contributions, key=lambda c: (-abs(c.active_contribution), c.symbol))
    return ordered[: max(n, 0)]
