"""Transaction-cost / tradeability — the institutional square-root impact model.

The SOTA market-impact model (Almgren-Chriss / Kissell "square-root law" /
Kyle's λ) on top of a half-spread + commission floor. Per-side cost in basis
points::

    per_side_bps(notional, adv_dollar) = half_spread_bps
        + impact_coef_bps * (sigma / ref_sigma) * sqrt(notional / adv_dollar)
        + commission_bps

i.e. ``cost ≈ half_spread + c·σ·√(Q/ADV) + commission`` — cost scales with
volatility (σ) and the *square root* of the participation rate (Q/ADV).

This is the ONE shared cost engine for the fleet (ARCHITECTURE §15 / §43): the
backtester turns gross per-horizon alpha into NET with it (``horizon_net_alpha``),
and the live research universe board lifts it into a per-name **tradeability**
score emitted alongside (but never blended into — §43) attractiveness. Keeping
the math here means consumers read one definition instead of re-deriving it.

Design:
- **Pure + config-driven.** No I/O, no logging — fully unit-testable; callers
  own coverage logging (e.g. how many names lacked ADV).
- **Volatility scaling is OPTIONAL and backward-compatible.** With ``sigma`` /
  ``ref_sigma`` omitted (or non-positive) the σ term collapses to 1.0 and the
  model is exactly the σ-agnostic √-impact law — so existing callers are
  unchanged. When supplied, ``ref_sigma`` is the volatility at which the
  calibrated ``impact_coef_bps`` holds (e.g. the cross-sectional MEDIAN σ), so
  the median-volatility name reproduces the σ-agnostic cost and more/less
  volatile names cost proportionally more/less — the true Almgren-Chriss form
  with a parameter-free, self-calibrating reference.
- **ADV-absent fallback.** When average-daily-dollar-volume is missing/≤0 the
  impact term drops to 0 (half-spread + commission only) rather than erroring —
  the conservative degrade, not a silent zero.
- **Defaults** calibrated for liquid large-cap US equities on an IBKR-paper book
  (overridable via the ``transaction_cost`` config block): half-spread ~2.5bps
  (≈5bps full spread), impact ~10bps at 100% participation, commission ~0.5bps
  ($0.005/share ≈ <1bp on large-caps).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# Institutional defaults — large-cap US equities / IBKR-paper calibration.
_DEFAULT_HALF_SPREAD_BPS = 2.5
_DEFAULT_IMPACT_COEF_BPS = 10.0
_DEFAULT_COMMISSION_BPS = 0.5
_DEFAULT_MIN_COST_BPS = 0.0


def _sigma_scale(sigma: float | None, ref_sigma: float | None) -> float:
    """Multiplicative volatility factor for the impact term. Collapses to 1.0
    (σ-agnostic) unless BOTH sigma and ref_sigma are finite and positive."""
    if sigma is None or ref_sigma is None:
        return 1.0
    try:
        s = float(sigma)
        r = float(ref_sigma)
    except (TypeError, ValueError):
        return 1.0
    if not (s > 0 and r > 0) or s != s or r != r:  # non-positive / NaN
        return 1.0
    return s / r


@dataclass(frozen=True)
class TransactionCostModel:
    """Per-side equity transaction cost via the square-root impact law."""

    half_spread_bps: float = _DEFAULT_HALF_SPREAD_BPS
    impact_coef_bps: float = _DEFAULT_IMPACT_COEF_BPS
    commission_bps: float = _DEFAULT_COMMISSION_BPS
    min_cost_bps: float = _DEFAULT_MIN_COST_BPS

    @classmethod
    def from_config(cls, config: dict | None) -> TransactionCostModel:
        """Build from the optional ``transaction_cost`` block of the config;
        absent keys fall back to the institutional defaults."""
        cfg = ((config or {}).get("transaction_cost") or {}) if config else {}
        return cls(
            half_spread_bps=float(cfg.get("half_spread_bps", _DEFAULT_HALF_SPREAD_BPS)),
            impact_coef_bps=float(cfg.get("impact_coef_bps", _DEFAULT_IMPACT_COEF_BPS)),
            commission_bps=float(cfg.get("commission_bps", _DEFAULT_COMMISSION_BPS)),
            min_cost_bps=float(cfg.get("min_cost_bps", _DEFAULT_MIN_COST_BPS)),
        )

    def per_side_bps(
        self,
        notional: float,
        adv_dollar: float | None,
        *,
        sigma: float | None = None,
        ref_sigma: float | None = None,
    ) -> float:
        """Per-side cost (bps) to trade ``notional`` dollars of a name whose
        average daily dollar volume is ``adv_dollar``. ADV missing/≤0 → the
        √-impact term drops to 0 (half-spread + commission only). ``sigma`` /
        ``ref_sigma`` optionally scale the impact term by σ/ref_σ (see module
        docstring); omitted → σ-agnostic."""
        notional = abs(float(notional))
        impact_bps = 0.0
        if adv_dollar is not None and adv_dollar > 0 and notional > 0:
            participation = notional / float(adv_dollar)
            impact_bps = (
                self.impact_coef_bps
                * _sigma_scale(sigma, ref_sigma)
                * math.sqrt(participation)
            )
        bps = self.half_spread_bps + impact_bps + self.commission_bps
        return max(bps, self.min_cost_bps)

    def round_trip_bps(
        self,
        notional: float,
        adv_dollar: float | None,
        *,
        sigma: float | None = None,
        ref_sigma: float | None = None,
    ) -> float:
        """Round-trip (enter + exit) cost in bps at a reference ``notional`` —
        the headline ``expected_cost_bps`` for a tradeability score. A buy then
        a sell is two same-side applications, so this is ``2 × per_side_bps``."""
        return 2.0 * self.per_side_bps(
            notional, adv_dollar, sigma=sigma, ref_sigma=ref_sigma
        )

    def cost_for_turnover(
        self,
        turnover_notional: float,
        adv_dollar: float | None,
        *,
        sigma: float | None = None,
        ref_sigma: float | None = None,
    ) -> float:
        """Dollar cost of trading ``turnover_notional`` dollars (ONE side) of a
        name. Each rebalance's per-name |Δweight|·book_notional IS one side, so
        the caller applies this per (rebalance, name); a full buy-then-sell cycle
        is naturally two applications."""
        notional = abs(float(turnover_notional))
        if notional <= 0:
            return 0.0
        return (
            notional
            * self.per_side_bps(notional, adv_dollar, sigma=sigma, ref_sigma=ref_sigma)
            / 1e4
        )


def tradeability_percentiles(
    cost_bps_by_name: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Map per-name expected cost (bps) → a 0-100 cross-sectional tradeability
    score where **lower cost ⇒ HIGHER score** (100 = the cheapest/most tradeable
    name in the cross-section). Uses average-rank percentile (ties share their
    mean rank), matching pandas ``rank(pct=True)*100`` on ``-cost``.

    Names with a ``None`` cost (e.g. no price/ADV coverage) get a ``None`` score
    — a coverage gap, never a fabricated rank — and are excluded from the ranked
    population so they do not distort the live names' percentiles.
    """
    import bisect

    ranked = {k: float(v) for k, v in cost_bps_by_name.items() if v is not None}
    out: dict[str, float | None] = dict.fromkeys(cost_bps_by_name)
    if not ranked:
        return out
    # Rank on -cost so cheapest → highest percentile.
    arr = sorted(-c for c in ranked.values())
    n = len(arr)
    for k, c in ranked.items():
        x = -c
        lo = bisect.bisect_left(arr, x)
        hi = bisect.bisect_right(arr, x)
        avg_rank = (lo + 1 + hi) / 2.0  # 1-indexed average rank
        out[k] = round(avg_rank / n * 100, 2)
    return out
