"""Return-math engine — XIRR (money-weighted) and time-weighted return.

Pure, dependency-light (stdlib only), and data-source-agnostic: every function
takes plain value/cash-flow inputs so it's unit-testable in isolation and reused
unchanged regardless of front end. These measure *what happened* to a portfolio
— descriptive, not advisory.

Two returns, two questions (both institutional/GIPS-relevant):
  - **MWR / XIRR** — the *investor's* actual return, sensitive to contribution/
    withdrawal timing (an internal rate of return over dated cash flows).
  - **TWR** — the *manager/strategy* return, neutralizing cash-flow timing by
    chaining sub-period returns geometrically. The GIPS standard for comparing
    against a benchmark.

Sign convention (Excel-XIRR compatible): a cash flow is signed from the
**investor's** perspective — money *into* the portfolio is negative (it left your
pocket), money *out* (withdrawals, and the terminal market value treated as a
final liquidating inflow) is positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class CashFlow:
    """A dated external cash flow, signed from the investor's perspective.

    ``amount < 0`` = money invested into the portfolio (left your pocket);
    ``amount > 0`` = money returned to you (withdrawal, or the terminal value as a
    final positive flow). This matches Excel/Sheets ``XIRR``.
    """

    when: date
    amount: float


@dataclass(frozen=True)
class ValuationPoint:
    """Portfolio market value at a date, with any external flow on that date.

    ``value`` is the portfolio's market value at ``when`` **before** ``flow`` is
    applied; ``flow`` is the external cash flow on that date signed from the
    *portfolio's* perspective (``> 0`` = contribution into the portfolio,
    ``< 0`` = withdrawal). Used for time-weighted return, which needs a valuation
    at each flow date to isolate market performance from cash movements.
    """

    when: date
    value: float
    flow: float = 0.0


def _year_fraction(d0: date, d1: date) -> float:
    """Actual/365 year fraction between two dates."""
    return (d1 - d0).days / _DAYS_PER_YEAR


def _npv(rate: float, flows: list[CashFlow], t0: date) -> float:
    """Net present value of dated flows at an annual ``rate`` (actual/365)."""
    return sum(cf.amount / (1.0 + rate) ** _year_fraction(t0, cf.when) for cf in flows)


def _npv_derivative(rate: float, flows: list[CashFlow], t0: date) -> float:
    """d(NPV)/d(rate) — for Newton's method."""
    deriv = 0.0
    for cf in flows:
        yf = _year_fraction(t0, cf.when)
        if yf == 0.0:
            continue
        deriv -= yf * cf.amount / (1.0 + rate) ** (yf + 1.0)
    return deriv


def xirr(
    flows: list[CashFlow],
    *,
    guess: float = 0.1,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float | None:
    """Money-weighted annualized return (IRR over irregular dated cash flows).

    Solves for the annual rate ``r`` where the actual/365-discounted NPV of
    ``flows`` is zero. Newton's method with a bisection fallback for robustness.
    Returns None when there's no sign change (no IRR exists) or it fails to
    converge — callers should treat None as "not computable", never as 0.

    The flow list must include the terminal portfolio value as a final positive
    flow (the "if you liquidated today" inflow); otherwise the IRR is undefined.
    """
    if len(flows) < 2:
        return None
    flows = sorted(flows, key=lambda cf: cf.when)
    # An IRR requires at least one inflow and one outflow.
    if not (any(cf.amount > 0 for cf in flows) and any(cf.amount < 0 for cf in flows)):
        return None
    t0 = flows[0].when

    # Newton's method.
    rate = guess
    for _ in range(max_iter):
        try:
            value = _npv(rate, flows, t0)
        except (OverflowError, ZeroDivisionError):
            break
        if abs(value) < tol:
            return rate
        deriv = _npv_derivative(rate, flows, t0)
        if deriv == 0.0:
            break
        new_rate = rate - value / deriv
        if new_rate <= -1.0:  # keep (1+r) positive so discounting stays defined
            new_rate = (rate - 1.0) / 2.0
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate

    # Bisection fallback over a wide, sane bracket.
    lo, hi = -0.9999, 100.0
    f_lo = _npv(lo, flows, t0)
    f_hi = _npv(hi, flows, t0)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = _npv(mid, flows, t0)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def time_weighted_return(points: list[ValuationPoint]) -> float | None:
    """Time-weighted return — geometric chain of sub-period returns.

    ``points`` must be sorted ascending and cover the full window: each carries
    the portfolio value *before* its external ``flow``. Between consecutive
    points the sub-period return is::

        r = end_value / (begin_value + begin_flow) - 1

    i.e. the next point's pre-flow value over the prior point's value *after* its
    flow (capital actually at work over the sub-period). Returns the geometric
    product minus 1, neutralizing cash-flow timing. None if < 2 points or a
    sub-period starts from non-positive capital.
    """
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: p.when)
    growth = 1.0
    for begin, end in zip(pts, pts[1:], strict=False):
        invested = begin.value + begin.flow
        if invested <= 0:
            return None
        growth *= end.value / invested
    return growth - 1.0


def cumulative_return(begin_value: float, end_value: float, net_contributions: float = 0.0) -> float | None:
    """Simple cumulative return adjusted for net external contributions.

    ``(end - net_contributions) / begin - 1``. For a clean window with no flows,
    that's just ``end/begin - 1``. None if ``begin_value`` is non-positive.
    """
    if begin_value <= 0:
        return None
    return (end_value - net_contributions) / begin_value - 1.0


def annualize(total_return: float, days: float) -> float | None:
    """Annualize a cumulative ``total_return`` observed over ``days`` (actual/365).

    ``(1 + total_return) ** (365/days) - 1``. None if ``days <= 0`` or the base is
    non-positive (a ≤ -100% return can't be annualized meaningfully).
    """
    if days <= 0:
        return None
    base = 1.0 + total_return
    if base <= 0:
        return None
    return base ** (_DAYS_PER_YEAR / days) - 1.0
