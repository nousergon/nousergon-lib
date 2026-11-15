"""risk_matched_benchmark — alpha vs comparable-risk baselines.

Reframes "did you beat the market?" as "given how much risk you took, did you
outperform the dumb version of taking that risk?" Two complementary baselines:

1. **Equal-weight high-vol subset** — top vol-quartile of the population,
   equal-weight, rolled weekly. The within-universe selection grade: "given
   you're fishing in volatile waters, did you pick the good fish?"

2. **Beta-matched SPY** — synthetic position of size (portfolio_beta × SPY)
   constructed each rebalance window. The pure-alpha grade: "did you
   outperform the dumb version of being this exposed?"

Both produce a daily return series for the benchmark alongside the input
portfolio's daily return series; the difference = excess return that *can't* be
explained by the risk taken. Information Ratio is computed against each
benchmark independently.

Pure-compute. Operates on price + portfolio-return series; no I/O.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Iterable
from typing import Protocol, TypedDict, TypeVar, cast, overload

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]  # pandas ships no stubs and pandas-stubs is not a dependency
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


# ── Typed views of the pandas surface used here ─────────────────────────────
#
# pandas ships no stubs (pandas-stubs is not a dependency), so pyright infers
# many DataFrame/Series/Index members from pandas' implementation and they
# come out partially unknown. Each Protocol below names exactly the calls this
# module makes, with the shapes they return for the inputs it is given; the
# pandas objects are cast to them at the boundary. Runtime behaviour is
# unchanged — ``cast`` is a no-op.

_RowsT_co = TypeVar("_RowsT_co", covariant=True)


class _PositionalRows(Protocol[_RowsT_co]):
    """``.iloc`` sliced by a positional row range (same container kind back)."""

    def __getitem__(self, key: slice, /) -> _RowsT_co: ...


class _TradingDayIndex(Protocol):
    """A unique, sorted ``pd.DatetimeIndex`` of trading days.

    Both indexes this is applied to are unique (a price history, or an
    inner-joined + ``dropna``-ed pair of daily series), so ``get_loc`` on one
    of the index's own elements always resolves to a scalar position — its
    wider ``int | slice | ndarray`` return only applies to non-unique or
    partial-match lookups. Boolean-mask indexing returns another
    DatetimeIndex; scalar indexing returns a Timestamp.
    """

    def __len__(self) -> int: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __ge__(self, other: np.datetime64, /) -> NDArray[np.bool_]: ...
    @overload
    def __getitem__(self, key: int, /) -> pd.Timestamp: ...
    @overload
    def __getitem__(self, key: NDArray[np.bool_], /) -> _TradingDayIndex: ...
    def get_loc(self, key: pd.Timestamp, /) -> int: ...


class _DatetimeAccessor(Protocol):
    """``Series.dt`` on a datetime64 Series."""

    def to_period(self, freq: str) -> _PeriodSeries: ...


class _PeriodAccessor(Protocol):
    """``Series.dt`` on a Period Series."""

    @property
    def start_time(self) -> _DatetimeSeries: ...


class _DatetimeSeries(Protocol):
    """A tz-naive datetime64 Series; ``unique`` returns a datetime64 ndarray."""

    @property
    def dt(self) -> _DatetimeAccessor: ...
    def unique(self) -> NDArray[np.datetime64]: ...


class _PeriodSeries(Protocol):
    @property
    def dt(self) -> _PeriodAccessor: ...


class _TickerSeries(Protocol):
    """A Series indexed by ticker (per-ticker vol, or a boolean mask of it)."""

    @property
    def empty(self) -> bool: ...
    @property
    def index(self) -> _TickerLabels: ...
    def dropna(self) -> _TickerSeries: ...
    def quantile(self, q: float) -> float: ...
    def __ge__(self, other: float, /) -> _TickerSeries: ...
    def __getitem__(self, key: _TickerSeries, /) -> _TickerSeries: ...


class _TickerLabels(Protocol):
    def tolist(self) -> list[Hashable]: ...


class _ReturnsFrame(Protocol):
    """Daily returns: DatetimeIndex rows, ticker columns.

    Selecting by a LIST of column labels always returns a DataFrame (never
    the Series of a scalar-label lookup).
    """

    @property
    def iloc(self) -> _PositionalRows[_ReturnsFrame]: ...
    def __getitem__(self, key: list[Hashable], /) -> _ReturnsFrame: ...
    def std(self, *, ddof: int) -> _TickerSeries: ...
    def mean(self, *, axis: int) -> pd.Series: ...


class _PriceFrame(Protocol):
    """Daily close prices: DatetimeIndex rows, ticker columns."""

    @property
    def index(self) -> pd.Index: ...
    def pct_change(self) -> _ReturnsFrame: ...


class _AlignedColumn(Protocol):
    """One float64 column of the ``port``/``spy`` aligned frame."""

    @property
    def iloc(self) -> _PositionalRows[_AlignedColumn]: ...
    def to_numpy(self) -> NDArray[np.float64]: ...
    def __mul__(self, other: float, /) -> pd.Series: ...


class _AlignedFrame(Protocol):
    """Two uniquely-named float64 columns, so a name lookup is always a Series."""

    @property
    def iloc(self) -> _PositionalRows[_AlignedFrame]: ...
    def __getitem__(self, key: str, /) -> _AlignedColumn: ...


class _SegmentSeries(Protocol):
    """``pd.concat`` of a list of Series: a Series."""

    def sort_index(self) -> pd.Series: ...


def _period_starts(index: pd.Index, rebalance_freq: str) -> Iterable[np.datetime64]:
    """Unique ``rebalance_freq`` period-start timestamps covering ``index``."""
    as_series = cast(_DatetimeSeries, pd.Series(index))
    return as_series.dt.to_period(rebalance_freq).dt.start_time.unique()


class BenchmarkResult(TypedDict, total=False):
    status: str
    n_days: int
    benchmark_total_return: float
    portfolio_total_return: float
    excess_return: float                # portfolio - benchmark, total
    excess_daily_mean: float            # mean daily excess
    excess_daily_std: float
    information_ratio: float            # annualized excess / std of excess
    benchmark_daily_returns: pd.Series
    excess_daily_returns: pd.Series
    label: str                          # "ew_high_vol" | "beta_matched_spy"


# ── Equal-weight high-vol subset ────────────────────────────────────────────


def construct_ew_high_vol_benchmark(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
    vol_quantile: float = 0.75,
    vol_lookback_days: int = 60,
    rebalance_freq: str = "W-MON",
) -> pd.Series:
    """Build daily return series for an equal-weight high-vol subset.

    Each rebalance date: rank tickers in ``universe`` by trailing
    realized vol over ``vol_lookback_days``, take the top quartile (or
    whatever ``vol_quantile`` selects), equal-weight them, hold until
    next rebalance.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices. DatetimeIndex (trading days). Columns =
        tickers.
    universe : list[str] | None
        Tickers to select from. Default = all columns of ``prices``.
        Pass the actual decision universe (the population picked from)
        for a fair counterfactual.
    vol_quantile : float
        Vol-percentile threshold. Default 0.75 = top quartile. Use 0.5
        for top half, 0.9 for top decile, etc.
    vol_lookback_days : int
        Trailing window for realized vol estimation. Default 60 ≈ 3
        months — long enough to be stable, short enough to reflect
        regime shifts.
    rebalance_freq : str
        Pandas offset alias. Default ``"W-MON"`` (every Monday). Use
        ``"M"`` for monthly, ``"D"`` for daily (warning: high turnover).

    Returns
    -------
    pd.Series of daily simple returns indexed by trading day. NaN-free
    (rebalance days that produce zero-member subsets are dropped).
    """
    if not (0.0 < vol_quantile < 1.0):
        raise ValueError(f"vol_quantile must be in (0, 1), got {vol_quantile}")
    if vol_lookback_days < 5:
        raise ValueError(f"vol_lookback_days must be >= 5, got {vol_lookback_days}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")

    # Column labels are Hashable; the cast only names the element type of
    # pandas' (unstubbed) Index iteration.
    cols: list[Hashable] = (
        list(cast("Iterable[Hashable]", prices.columns))
        if universe is None
        else [t for t in universe if t in prices.columns]
    )
    if not cols:
        raise ValueError("universe is empty after intersecting with prices.columns")

    # A list of column labels always selects a DataFrame.
    sub = cast(_PriceFrame, prices[cols])
    daily_returns = sub.pct_change()
    # `sub.index` was validated as a DatetimeIndex above (a price history:
    # unique trading days).
    sub_index = cast(_TradingDayIndex, sub.index)

    # Rebalance dates: first trading day on or after each period boundary.
    # `to_period(rebalance_freq)` then `.start_time` gives the period boundary;
    # we snap to the next trading day in the price index.
    period_starts = _period_starts(sub.index, rebalance_freq)
    rebalance_dates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ps in period_starts:
        candidates = sub_index[sub_index >= ps]
        if len(candidates) > 0:
            d = candidates[0]
            if d not in seen and d in sub_index:
                rebalance_dates.append(d)
                seen.add(d)

    benchmark_segments: list[pd.Series] = []
    for i, rd in enumerate(rebalance_dates):
        # Compute trailing vol on returns up to rd. `rd` is one of
        # `sub_index`'s own elements, so get_loc resolves to a scalar int
        # (see _TradingDayIndex).
        rd_pos = sub_index.get_loc(rd)
        if rd_pos < vol_lookback_days:
            continue  # not enough history yet
        lookback = daily_returns.iloc[rd_pos - vol_lookback_days : rd_pos]
        vol = lookback.std(ddof=1)  # per-ticker realized vol
        vol = vol.dropna()
        if vol.empty:
            continue
        threshold = vol.quantile(vol_quantile)
        selected = vol[vol >= threshold].index.tolist()
        if not selected:
            continue

        # Hold from rd until next rebalance (exclusive) or end of data.
        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        end_pos = (
            sub_index.get_loc(next_rd)
            if next_rd is not None
            else len(sub_index)
        )
        # Daily returns over [rd+1, next_rd] — first day post-rebalance
        # is the first day the held basket starts compounding. `selected`
        # is a list of column labels, so this is always column-subset
        # selection returning a DataFrame.
        selected_returns = daily_returns[selected]
        segment = selected_returns.iloc[rd_pos + 1 : end_pos].mean(axis=1)
        benchmark_segments.append(segment)

    if not benchmark_segments:
        return pd.Series(dtype=np.float64, name="ew_high_vol")
    out = cast(_SegmentSeries, pd.concat(benchmark_segments)).sort_index()
    out.name = "ew_high_vol"
    return out.dropna()


# ── Beta-matched SPY ────────────────────────────────────────────────────────


def construct_beta_matched_spy_benchmark(
    portfolio_daily_returns: pd.Series,
    spy_daily_returns: pd.Series,
    beta_lookback_days: int = 60,
    rebalance_freq: str = "W-MON",
) -> pd.Series:
    """Build daily return series for a beta-matched SPY position.

    Each rebalance date: estimate the portfolio's trailing realized
    beta to SPY (OLS slope of portfolio returns regressed on SPY
    returns over ``beta_lookback_days``). The benchmark return on each
    subsequent day is ``beta * SPY_return``. Beta is held constant
    between rebalances.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Indexed by trading day. The portfolio whose alpha we're
        evaluating.
    spy_daily_returns : pd.Series
        SPY's daily simple returns aligned (or alignable) to the same
        trading-day index.
    beta_lookback_days : int
        Trailing window for beta estimation. Default 60 days.
    rebalance_freq : str
        Pandas offset alias for beta re-estimation cadence.

    Returns
    -------
    pd.Series of daily benchmark returns indexed by trading day. The
    first ``beta_lookback_days`` of the portfolio history are dropped
    (no trailing data to estimate beta on).
    """
    if beta_lookback_days < 5:
        raise ValueError(f"beta_lookback_days must be >= 5, got {beta_lookback_days}")

    aligned = pd.concat(
        [portfolio_daily_returns.rename("port"), spy_daily_returns.rename("spy")],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=np.float64, name="beta_matched_spy")

    # `aligned.index` is datetime-like at runtime (both inputs are "indexed
    # by trading day" per the docstring — the `.dt` accessor in
    # _period_starts already depends on that) and unique (inner-joined +
    # deduped above).
    aligned_index = cast(_TradingDayIndex, aligned.index)
    aligned_frame = cast(_AlignedFrame, aligned)
    period_starts = _period_starts(aligned.index, rebalance_freq)
    rebalance_dates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ps in period_starts:
        candidates = aligned_index[aligned_index >= ps]
        if len(candidates) > 0:
            d = candidates[0]
            if d not in seen:
                rebalance_dates.append(d)
                seen.add(d)

    bench_segments: list[pd.Series] = []
    for i, rd in enumerate(rebalance_dates):
        # rd is one of aligned_index's own elements, so get_loc resolves to
        # a scalar int (see _TradingDayIndex).
        rd_pos = aligned_index.get_loc(rd)
        if rd_pos < beta_lookback_days:
            continue
        window = aligned_frame.iloc[rd_pos - beta_lookback_days : rd_pos]
        port = window["port"].to_numpy()
        spy = window["spy"].to_numpy()
        spy_var = float(np.var(spy, ddof=1))
        if spy_var == 0.0:
            beta = 0.0
        else:
            beta = float(np.cov(port, spy, ddof=1)[0, 1] / spy_var)

        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        end_pos = (
            aligned_index.get_loc(next_rd)
            if next_rd is not None
            else len(aligned_index)
        )
        segment = aligned_frame["spy"].iloc[rd_pos + 1 : end_pos] * beta
        bench_segments.append(segment)

    if not bench_segments:
        return pd.Series(dtype=np.float64, name="beta_matched_spy")
    out = cast(_SegmentSeries, pd.concat(bench_segments)).sort_index()
    out.name = "beta_matched_spy"
    return out.dropna()


# ── Alpha + Information Ratio against any benchmark ─────────────────────────


def compute_alpha_vs_benchmark(
    portfolio_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    label: str = "benchmark",
) -> BenchmarkResult:
    """Compute total + per-day excess return + Information Ratio.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Daily simple returns of the portfolio being evaluated.
    benchmark_daily_returns : pd.Series
        Daily simple returns of the comparison benchmark.
    label : str
        Stamped onto the result for downstream identification (e.g.
        "ew_high_vol", "beta_matched_spy", "spy", "qqq").

    Returns
    -------
    BenchmarkResult dict (see TypedDict at top).
    """
    aligned = pd.concat(
        [portfolio_daily_returns.rename("port"),
         benchmark_daily_returns.rename("bench")],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return {"status": "insufficient_data", "n_days": 0, "label": label}

    # aligned was built from two uniquely-named Series concatenated
    # column-wise, so selecting by name always yields a Series (never
    # the DataFrame branch of __getitem__'s duplicate-column overload).
    port = cast("pd.Series", aligned["port"])
    bench = cast("pd.Series", aligned["bench"])
    excess = port - bench
    # Total returns via geometric compounding.
    port_total = float((1.0 + port).prod() - 1.0)
    bench_total = float((1.0 + bench).prod() - 1.0)

    excess_mean = float(excess.mean())
    excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    # Use a small floor (1e-12) instead of `> 0.0` — a constant-excess
    # series has std ≈ 1e-17 due to float-mean residual, which would
    # otherwise produce a nonsense ~1e16 IR.
    if excess_std > 1e-12:
        ir = excess_mean / excess_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
    else:
        ir = 0.0

    return {
        "status": "ok",
        "n_days": int(len(aligned)),
        "benchmark_total_return": bench_total,
        "portfolio_total_return": port_total,
        "excess_return": port_total - bench_total,
        "excess_daily_mean": excess_mean,
        "excess_daily_std": excess_std,
        "information_ratio": float(ir),
        "benchmark_daily_returns": bench,
        "excess_daily_returns": excess,
        "label": label,
    }
