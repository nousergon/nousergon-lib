"""expectancy — hit rate × win/loss ratio decomposition.

The single most diagnostic breakdown for distinguishing skilled vs unskilled
risk-taking:

  - **Selection skill**: high hit rate with symmetric W/L ratio (~1.0) — picks
    winners more often than losers, magnitudes about equal.
  - **Convexity skill**: moderate hit rate with W/L ratio > 1.5 — rides winners
    and cuts losers; total expectancy positive even with <50% hit rate.
  - **No skill (just YOLO into vol)**: declining hit rate with no compensating
    W/L improvement → expectancy ≤ 0.

Formula:
    expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss

where avg_win is the mean of positive returns (or alpha) and avg_loss is the
mean *magnitude* of negative returns. Reports both expectancy and the
decomposition components so consumers can see WHICH dimension is failing.

Pure-compute. Operates on a returns array; no I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExpectancyResult(TypedDict, total=False):
    status: str
    n: int
    hit_rate: float
    avg_win: float | None
    avg_loss: float  # magnitude (positive number)
    win_loss_ratio: float | None
    expectancy: float
    expectancy_per_unit_loss: float | None  # expectancy / avg_loss; the "R-multiple" expectancy


def compute_expectancy(
    returns: pd.Series | NDArray[np.generic],
    threshold: float = 0.0,
    min_samples: int = 10,
) -> ExpectancyResult:
    """Compute expectancy decomposition over a return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Per-trade or per-pick returns (or alphas). NaN dropped.
    threshold : float
        Win/loss boundary. Default 0 → wins are positive returns. Set
        non-zero to compute relative to a benchmark return (e.g.
        threshold = SPY_return for "did we beat SPY?" expectancy).
    min_samples : int
        Minimum non-NaN samples required to compute. Returns
        status=insufficient_data below this floor. Default 10.

    Returns
    -------
    ExpectancyResult dict with:
        status: "ok" | "insufficient_data" | "no_wins" | "no_losses"
        n: sample size
        hit_rate: fraction of trades where return > threshold
        avg_win: mean of returns > threshold (None if no wins)
        avg_loss: mean magnitude of returns <= threshold (None if no losses)
        win_loss_ratio: avg_win / avg_loss (None if either is missing/zero)
        expectancy: hit_rate * avg_win - (1 - hit_rate) * avg_loss
        expectancy_per_unit_loss: expectancy / avg_loss (R-multiple form)

    Notes
    -----
    The R-multiple form (expectancy / avg_loss) is the "expectancy per unit of
    risk taken" — useful for comparing across regimes where absolute return
    levels shift but the ratio of skilled-edge-to-typical-loss is the
    invariant signal of skill.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < min_samples:
        return {"status": "insufficient_data", "n": n}

    excess = arr - threshold
    wins = excess[excess > 0]
    losses = excess[excess <= 0]

    hit_rate = float(wins.size) / n

    if wins.size == 0:
        return {
            "status": "no_wins",
            "n": n,
            "hit_rate": 0.0,
            "avg_win": None,
            "avg_loss": float(-losses.mean()) if losses.size else 0.0,
            "win_loss_ratio": None,
            "expectancy": float(excess.mean()),
            "expectancy_per_unit_loss": None,
        }

    if losses.size == 0:
        return {
            "status": "no_losses",
            "n": n,
            "hit_rate": 1.0,
            "avg_win": float(wins.mean()),
            "avg_loss": 0.0,
            "win_loss_ratio": None,
            "expectancy": float(excess.mean()),
            "expectancy_per_unit_loss": None,
        }

    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())  # report as positive magnitude
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else None
    expectancy = hit_rate * avg_win - (1.0 - hit_rate) * avg_loss
    epul = expectancy / avg_loss if avg_loss > 0 else None

    return {
        "status": "ok",
        "n": n,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "expectancy": expectancy,
        "expectancy_per_unit_loss": epul,
    }


class _GroupableFrame(Protocol):
    def groupby(self, by: str) -> Iterable[tuple[object, pd.DataFrame]]: ...


def compute_expectancy_by_group(
    df: pd.DataFrame,
    return_col: str,
    group_col: str,
    threshold: float = 0.0,
    min_samples: int = 10,
) -> dict[str, ExpectancyResult]:
    """Stratify expectancy by a grouping column (team_id, conviction, sector, ...).

    Returns
    -------
    dict[group_value, ExpectancyResult]
        One result per group. Groups below ``min_samples`` get
        status=insufficient_data entries.
    """
    if return_col not in df.columns:
        raise KeyError(f"return_col {return_col!r} not in dataframe")
    if group_col not in df.columns:
        raise KeyError(f"group_col {group_col!r} not in dataframe")
    out: dict[str, ExpectancyResult] = {}
    # pandas ships no stubs here; describe the group iterator at this boundary.
    for group_value, sub in cast(_GroupableFrame, df).groupby(group_col):
        # return_col is plain str (not a literal), so pyright can't
        # statically rule out DataFrame.__getitem__'s duplicate-column
        # -label overload; it was confirmed present in df.columns above,
        # so this is always single-column Series selection.
        out[str(group_value)] = compute_expectancy(
            cast("pd.Series", sub[return_col]), threshold=threshold, min_samples=min_samples,
        )
    return out
