"""Cross-sectional 6-pillar attractiveness composite (Grinold-Kahn z-blend).

Mirrors the institutional method in ``crucible-research`` ``scoring/universe_board.py``
(schema v3): sector-neutral pillar percentiles → per-pillar cross-sectional winsorized
z-scores → coverage-renormalized weighted blend → terminal cross-sectional percentile
(0–100). Pure stdlib — no S3, no pandas — so Metron, Research, and the backtester can
share byte-identical numbers for the same factor-profile inputs.
"""

from __future__ import annotations

from typing import Any, cast

PILLAR_ORDER: tuple[str, ...] = (
    "quality",
    "value",
    "momentum",
    "growth",
    "stewardship",
    "defensiveness",
)

PILLAR_TO_FACTOR_KEY: dict[str, str] = {
    "quality": "quality_score",
    "value": "value_score",
    "momentum": "momentum_score",
    "growth": "growth_score",
    "stewardship": "stewardship_score",
    "defensiveness": "low_vol_score",
}

DEFAULT_PILLAR_WEIGHTS: dict[str, float] = {
    p: 1.0 / len(PILLAR_ORDER) for p in PILLAR_ORDER
}

_ZSCORE_CLIP = 3.0


def _num(v: object) -> float | None:
    if v is None:
        return None
    try:
        # v is deliberately `object` — this coerces arbitrary upstream
        # JSON/dict values (str, int, bool, Decimal, ...); the
        # try/except (TypeError, ValueError) below IS the type-safety
        # mechanism for non-ConvertibleToFloat input at runtime, so the
        # cast just tells pyright to let the runtime check do its job.
        f = float(cast(Any, v))
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def _avg_rank_pct(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional percentile (0–100) via average-rank — matches pandas rank(pct=True)×100."""
    import bisect

    if not values:
        return {}
    arr = sorted(values.values())
    n = len(arr)
    out: dict[str, float] = {}
    for k, x in values.items():
        lo = bisect.bisect_left(arr, x)
        hi = bisect.bisect_right(arr, x)
        avg_rank = (lo + 1 + hi) / 2.0
        out[k] = round(avg_rank / n * 100, 2)
    return out


def _zscore(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    z = (value - mean) / std
    return max(-_ZSCORE_CLIP, min(_ZSCORE_CLIP, z))


def normalize_pillar_weights(raw: dict[str, float] | None) -> dict[str, float]:
    """Normalize pillar weights to sum 1.0; negative / missing → 0; empty → equal weights."""
    if not raw:
        return dict(DEFAULT_PILLAR_WEIGHTS)
    parsed = {p: max(0.0, _num(raw.get(p)) or 0.0) for p in PILLAR_ORDER}
    total = sum(parsed.values())
    if total <= 0:
        return dict(DEFAULT_PILLAR_WEIGHTS)
    return {p: round(w / total, 6) for p, w in parsed.items()}


def compute_cross_sectional_attractiveness(
    pillar_scores_by_ticker: dict[str, dict[str, float | None]],
    pillar_weights: dict[str, float],
) -> dict[str, dict]:
    """Blend sector-neutral pillar percentiles into cross-sectional attractiveness scores.

    Returns ``{ticker: {attractiveness_raw, attractiveness_score, pillar_contributions}}``.
    """
    weights = normalize_pillar_weights(pillar_weights)

    pillar_values: dict[str, dict[str, float]] = {p: {} for p in PILLAR_ORDER}
    for ticker, scores in pillar_scores_by_ticker.items():
        for p in PILLAR_ORDER:
            v = scores.get(p)
            if v is not None:
                pillar_values[p][ticker] = v
    pillar_stats = {p: _mean_std(list(v.values())) for p, v in pillar_values.items() if v}

    blends: dict[str, float] = {}
    out: dict[str, dict] = {}
    for ticker, scores in pillar_scores_by_ticker.items():
        contribs: dict[str, tuple[float, float]] = {}
        num = 0.0
        wsum = 0.0
        for p in PILLAR_ORDER:
            v = scores.get(p)
            w = weights.get(p, 0.0)
            if v is None or w <= 0 or p not in pillar_stats:
                continue
            mean, std = pillar_stats[p]
            z = _zscore(v, mean, std)
            num += w * z
            wsum += w
            contribs[p] = (w, z)
        rec = {"attractiveness_raw": None, "attractiveness_score": None, "pillar_contributions": {}}
        if wsum > 0:
            blend = num / wsum
            blends[ticker] = blend
            rec["attractiveness_raw"] = round(blend, 4)
            rec["pillar_contributions"] = {p: round(w * z / wsum, 4) for p, (w, z) in contribs.items()}
        out[ticker] = rec

    pct = _avg_rank_pct(blends)
    for ticker in out:
        out[ticker]["attractiveness_score"] = pct.get(ticker)
    return out


def attractiveness_from_factor_profiles(
    factor_profiles: dict[str, dict],
    *,
    pillar_weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Compute attractiveness from ``{ticker: {quality_score, value_score, …}}`` profiles."""
    weights = normalize_pillar_weights(pillar_weights)
    pillar_scores_by_ticker = {
        ticker: {
            pillar: _num(profile.get(PILLAR_TO_FACTOR_KEY[pillar]))
            for pillar in PILLAR_ORDER
        }
        for ticker, profile in (factor_profiles or {}).items()
        if isinstance(profile, dict)
    }
    return compute_cross_sectional_attractiveness(pillar_scores_by_ticker, weights)
