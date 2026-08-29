"""Anytime-valid confidence sequences — the evidence bar that replaces minimum-week gates.

**The problem this removes.** The fleet looks at every slot once a week and
promotes on what it sees. A fixed-sample test at a nominal 5% is valid for
ONE look; taken 52 times a year the family-wise false-promotion rate drifts
toward ~40%. The fleet's defence was the opposite error — ``thin_evidence``
floors and minimum-week gates, which do not control error at all and which
deadlocked every promotion in the fleet (exactly one predictor challenger
has ever won a promotion, 2026-07-17).

A **confidence sequence** is a sequence of intervals ``CI_t`` satisfying
``P(∃t: μ ∉ CI_t) ≤ α`` — coverage holds *uniformly over time*, so it may be
inspected at every cycle, and a decision may be taken at any stopping time,
without inflating the error rate. It subsumes minimum-evidence floors
naturally: the interval is very wide at week 1 (no lead is supported) and
narrows as evidence accrues. **So the floors are removed rather than
retuned** — the mechanism that made them necessary is gone.

**Method.** The normal-mixture (conjugate-mixture) boundary of Robbins,
restated as Howard, Ramdas, McAuliffe & Sekhon (2021), *Time-uniform
Chernoff bounds via nonnegative supermartingales*, §3.5 — for observations
with sub-Gaussian scale ``σ`` the boundary on the centred partial sum is

    u(t) = sqrt( 2 (V_t + ρ) · ln( sqrt((V_t + ρ)/ρ) / α ) ),   V_t = t·σ²

and the interval on the mean is ``mean ± u(t)/t``. ``ρ`` tunes where the
boundary is tightest; it is set from ``opt_n``, the look count the slot
expects to need, and the bound stays valid for every other ``t`` — ``ρ`` is
a tuning constant, never a stopping rule.

**Scale must be declared, not discovered.** ``variance_mode="declared"``
(the default) derives ``σ`` from ``clip``: per-date differences are clipped
to ``[-clip, +clip]``, and a variable bounded on a range of width ``2c`` is
sub-Gaussian with scale ``c`` (Hoeffding). That makes validity checkable
from configuration alone. ``variance_mode="empirical"`` uses the running
sample standard deviation, which is tighter and is what most practitioners
use, but it is an approximation — the plug-in estimate is not itself
time-uniform — so it is opt-in and the returned bound says which was used.

Nothing here is a multiple-comparison correction across ARMS. Multiplicity
across arms stays where it already is: DSR deflated for ``n_trials`` and
CSCV-PBO (``nousergon_lib.quant.stats.dsr`` / ``.pbo``,
champion-challenger-policy.md §5.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ConfSeqBound",
    "confidence_sequence",
    "DEFAULT_ALPHA",
    "DEFAULT_OPT_N",
]

#: Two-sided nominal level. ``supported`` is a one-sided read of it and is
#: therefore conservative at ``α/2``.
DEFAULT_ALPHA = 0.05

#: Look count the mixture boundary is tuned to be tightest at. 26 ≈ half a
#: year of weekly cycles: tight enough to promote inside two quarters, loose
#: enough that a slot running for years is not badly served.
DEFAULT_OPT_N = 26


@dataclass(frozen=True)
class ConfSeqBound:
    """A time-uniform interval on the mean of the paired per-date difference.

    ``supported`` is the operative field: the lead is supported only when the
    whole interval sits above zero. It is deliberately a property of the
    INTERVAL, not a p-value, so a caller cannot accumulate looks against it.
    """

    mean: float
    lower: float
    upper: float
    radius: float
    n: int
    alpha: float
    sigma: float
    variance_mode: str
    method: str
    n_clipped: int

    @property
    def supported(self) -> bool:
        """True when the mean's whole interval is strictly above zero."""
        return self.lower > 0.0

    @property
    def supported_negative(self) -> bool:
        """True when the whole interval is strictly below zero."""
        return self.upper < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "radius": self.radius,
            "n": self.n,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "variance_mode": self.variance_mode,
            "method": self.method,
            "n_clipped": self.n_clipped,
            "supported": self.supported,
        }


def _sample_sigma(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def confidence_sequence(
    diffs: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
    clip: float | None = None,
    variance_mode: str = "declared",
    opt_n: int = DEFAULT_OPT_N,
) -> ConfSeqBound:
    """Time-uniform interval on the mean of ``diffs``.

    ``diffs`` are the per-date paired differences from
    :func:`nousergon_lib.arena.window.pair_on_common_window`. Raises on an
    empty input — an unmeasurable comparison is the CALLER's ``unmeasurable``
    status with a reason (§7.2), never a silently wide interval that reads
    as a legitimate "not supported".
    """
    n = len(diffs)
    if n == 0:
        raise ValueError(
            "confidence_sequence on an empty window; the caller must record "
            "`unmeasurable` with a reason (champion-challenger-policy.md §7.2)"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    if opt_n < 1:
        raise ValueError(f"opt_n must be >= 1; got {opt_n!r}")
    if variance_mode not in ("declared", "empirical"):
        raise ValueError(
            "variance_mode must be 'declared' or 'empirical'; got "
            f"{variance_mode!r}"
        )

    n_clipped = 0
    values = []
    if clip is not None:
        if clip <= 0.0:
            raise ValueError(f"clip must be > 0; got {clip!r}")
        for value in diffs:
            if value > clip:
                values.append(clip)
                n_clipped += 1
            elif value < -clip:
                values.append(-clip)
                n_clipped += 1
            else:
                values.append(float(value))
    else:
        values = [float(v) for v in diffs]

    mean = sum(values) / n

    if variance_mode == "declared":
        if clip is None:
            raise ValueError(
                "variance_mode='declared' requires an explicit `clip`: the "
                "sub-Gaussian scale is derived from the declared bound, so "
                "without it the interval's validity rests on nothing"
            )
        sigma = float(clip)
    else:
        sigma = _sample_sigma(values, mean)
        if sigma <= 0.0:
            # A degenerate sample (all identical, or n == 1) carries no
            # information about scale. Fall back to the declared bound rather
            # than emitting a zero-width interval, which would support every
            # non-zero lead on a single observation.
            if clip is None:
                raise ValueError(
                    "variance_mode='empirical' produced a zero scale from "
                    f"{n} observation(s) and no `clip` fallback was declared; "
                    "a zero-width interval would support any lead"
                )
            sigma = float(clip)

    # Robbins normal-mixture boundary, tuned to be tightest near opt_n.
    rho = sigma * sigma * float(opt_n)
    v_t = float(n) * sigma * sigma
    inner = math.sqrt((v_t + rho) / rho) / alpha
    boundary = math.sqrt(2.0 * (v_t + rho) * math.log(inner))
    radius = boundary / float(n)

    return ConfSeqBound(
        mean=mean,
        lower=mean - radius,
        upper=mean + radius,
        radius=radius,
        n=n,
        alpha=alpha,
        sigma=sigma,
        variance_mode=variance_mode,
        method="robbins-normal-mixture",
        n_clipped=n_clipped,
    )
