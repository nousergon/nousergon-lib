"""calibration — Expected Calibration Error (ECE) for probabilistic predictions.

The one canonical ECE implementation for the fleet, consumed by BOTH the
producer (alpha-engine-predictor fits its isotonic/Platt calibrator and reports
``ece_before``/``ece_after`` from this) AND the monitor (alpha-engine-backtester
re-measures production calibration against realized outcomes). Sharing one impl
is the point: a train-time ECE and a production-time ECE are only comparable —
and "calibration drift" only means anything — if both bin the *same quantity the
same way*.

**What ECE measures.** Given predicted probabilities ``p`` and binary outcomes
``y``, bin the predictions and, within each bin, compare the mean predicted
probability to the empirical frequency of ``y=1``. ECE is the sample-weighted
mean absolute gap. A well-calibrated model has ECE≈0: when it says 0.7, the
event happens ~70% of the time.

**Critical contract — pass a PROBABILITY, not a margin.** ``predicted_probs``
must be on the same scale as the thing ``actual_binary`` counts. For the
predictor that means pass ``p_up`` (calibrated P(direction=UP)) against
``1[realized_alpha > 0]`` — NOT ``prediction_confidence`` (``|p_up-0.5|*2``, a
margin on [0,1]). Binning a margin against a hit-rate compares two different
scales and manufactures a structural ECE of ~0.2-0.25 even for a perfectly
calibrated model (for a calibrated model ``P(correct) = 0.5 + margin/2``, so the
gap ``|hit_rate - margin|`` never vanishes). That exact scale-mismatch produced
months of false ``calibration_breakdown`` retrain alerts after the predictor
flipped its confidence convention (2026-05-12); this module is the fix.

Pure numpy, no I/O. ``actual_binary`` is coerced to {0,1}; non-finite pairs in
either array are dropped before binning.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class _BinRecord(TypedDict, total=False):
    range: list[float]   # [lo, hi)
    n: int
    mean_pred: float     # mean predicted probability in the bin
    hit_rate: float      # empirical P(y=1) in the bin


class ECEResult(TypedDict, total=False):
    status: str          # "ok" | "insufficient_data" | "no_data"
    n: int               # samples contributing to ece (after min_bin_n drops)
    n_total: int         # valid (prob, label) pairs after NaN filtering
    ece: float | None  # sample-weighted mean |mean_pred - hit_rate|; None if no usable bin
    n_bins_used: int
    bins: list[_BinRecord]
    dropped_bins: list[_BinRecord]


def expected_calibration_error(
    predicted_probs: np.ndarray | list[float],
    actual_binary: np.ndarray | list[float],
    *,
    n_bins: int = 10,
    bin_edges: np.ndarray | list[float] | None = None,
    min_bin_n: int = 0,
    min_samples: int = 1,
) -> ECEResult:
    """Compute Expected Calibration Error of ``predicted_probs`` vs ``actual_binary``.

    Parameters
    ----------
    predicted_probs : array-like
        Predicted probabilities in [0, 1]. MUST be a probability on the same
        scale as ``actual_binary`` (see module docstring — do not pass a
        margin / confidence-distance).
    actual_binary : array-like
        Realized binary outcomes. Coerced to {0, 1} (``> 0`` → 1). NaN/inf in
        either array drops that pair.
    n_bins : int
        Number of equal-width bins over [0, 1]. Ignored when ``bin_edges`` is
        given. Default 10 (standard ECE; matches the predictor's calibrator).
    bin_edges : array-like, optional
        Explicit monotonically-increasing bin edges. The last bin is
        right-closed so probability 1.0 lands in it. Use the default
        equal-width [0,1] bins unless a consumer needs to match a legacy edge
        set.
    min_bin_n : int
        Bins with fewer than this many samples are recorded in ``dropped_bins``
        and excluded from the ECE sum — ECE is noise-dominated in sparse bins.
        Default 0 (keep every non-empty bin).
    min_samples : int
        Minimum valid pairs (after NaN filtering) required to compute ECE.
        Below this, returns ``status="insufficient_data"``. Default 1.

    Returns
    -------
    ECEResult
        ``status`` is ``"no_data"`` (no finite pairs), ``"insufficient_data"``
        (fewer than ``min_samples``), or ``"ok"``. On ``"ok"``, ``ece`` is the
        sample-weighted mean absolute calibration gap, or ``None`` if every bin
        was dropped by ``min_bin_n``.
    """
    p = np.asarray(predicted_probs, dtype=np.float64).ravel()
    y = np.asarray(actual_binary, dtype=np.float64).ravel()
    if p.size != y.size:
        raise ValueError(
            f"predicted_probs (n={p.size}) and actual_binary (n={y.size}) "
            "must be same length"
        )

    valid = np.isfinite(p) & np.isfinite(y)
    p = p[valid]
    y = (y[valid] > 0).astype(np.float64)
    n_total = int(p.size)

    if n_total == 0:
        return {"status": "no_data", "n": 0, "n_total": 0, "ece": None}
    if n_total < min_samples:
        return {"status": "insufficient_data", "n": 0, "n_total": n_total, "ece": None}

    if bin_edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.asarray(bin_edges, dtype=np.float64).ravel()
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be monotonically increasing with >=2 edges")

    bins: list[_BinRecord] = []
    dropped: list[_BinRecord] = []
    weighted_gap = 0.0
    used_n = 0

    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p < hi)
        if i == edges.size - 2:  # right-close the final bin so p == hi lands in it
            mask = mask | (p == hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        mean_pred = float(p[mask].mean())
        hit_rate = float(y[mask].mean())
        record: _BinRecord = {
            "range": [round(lo, 4), round(hi, 4)],
            "n": n_bin,
            "mean_pred": round(mean_pred, 4),
            "hit_rate": round(hit_rate, 4),
        }

        if n_bin < min_bin_n:
            record["dropped_reason"] = f"n<{min_bin_n}"  # type: ignore[typeddict-unknown-key]
            dropped.append(record)
            continue

        bins.append(record)
        weighted_gap += abs(mean_pred - hit_rate) * n_bin
        used_n += n_bin

    # Headline ECE is returned UNROUNDED — it's a primitive; callers round for
    # display/artifacts. (Per-bin mean_pred/hit_rate are rounded for readability.)
    ece = (weighted_gap / used_n) if used_n > 0 else None

    return {
        "status": "ok",
        "n": used_n,
        "n_total": n_total,
        "ece": ece,
        "n_bins_used": len(bins),
        "bins": bins,
        "dropped_bins": dropped,
    }
