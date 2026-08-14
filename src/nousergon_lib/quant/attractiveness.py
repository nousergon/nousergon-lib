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


def _zscore(value: float, mean: float, std: float) -> float | None:
    """Winsorized cross-sectional z-score, or ``None`` where it is UNDEFINED.

    Returns ``None`` when ``std <= 0`` — a single observation for this pillar,
    or every ticker carrying the identical percentile. There is no defined
    z-score against a zero-dispersion cohort, and ``0.0`` is EXACTLY the value
    a genuinely at-the-mean ticker produces, so returning it would make "there
    was nothing to compare against" indistinguishable from "this name sits at
    its cohort mean" (config-I7272). Worse than the ambiguity: the fabricated
    ``0.0`` was then VOTED into the weighted blend, diluting every measured
    pillar toward neutral with a number nobody measured.

    The caller DROPS an undefined leg and lets the surviving weights
    renormalize — the same coverage-renormalization already applied to a
    pillar that is simply missing — and a ticker with no surviving leg is
    EXCLUDED from the terminal percentile rather than ranked. This matches the
    convention adopted at the two sibling sites in the same arc:
    ``regime/composite.py::_zscore`` (crucible-predictor PR490) and
    ``scoring/attractiveness_trajectory.py::_zmap`` (crucible-research PR628).
    """
    if std <= 0:
        return None
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

    Returns ``{ticker: {attractiveness_raw, attractiveness_score,
    pillar_contributions, undefined_pillars}}``.

    Thin wrapper over :func:`compute_cross_sectional_attractiveness_with_coverage`
    for callers that do not publish the coverage report. ONE implementation, two
    renderings — the excluded count can never disagree with the scores it
    describes. Prefer the coverage variant on any path that writes an artifact
    or renders a surface: an exclusion nobody can see is the same failure as the
    fabricated ``0.0`` this replaced (config-I7272, ``principles.md`` §2.7).
    """
    return compute_cross_sectional_attractiveness_with_coverage(
        pillar_scores_by_ticker, pillar_weights
    )[0]


def compute_cross_sectional_attractiveness_with_coverage(
    pillar_scores_by_ticker: dict[str, dict[str, float | None]],
    pillar_weights: dict[str, float],
) -> tuple[dict[str, dict], dict]:
    """As :func:`compute_cross_sectional_attractiveness`, plus the coverage report.

    Returns ``(scores, coverage)``. ``coverage`` is emitted UNCONDITIONALLY —
    a healthy cross-section publishes an explicit zero rather than nothing, so
    a reader can never mistake "no exclusions" for "nobody looked":

    ``n_tickers``
        Size of the cross-section offered.
    ``n_scored``
        Tickers that received a terminal percentile.
    ``n_excluded_undefined``
        Tickers EXCLUDED from the ranking because no pillar leg survived. These
        carry ``attractiveness_score: None`` and hold no rank position at all —
        a fabricated ``0.0`` would rank them against measured names, and a
        ``None`` sorted to one end would still rank them, silently and
        systematically (which in a ranking is the worse of the two).
    ``excluded_tickers``
        The MEMBERS of that count, sorted. A count published without its members
        is unactionable.
    ``degenerate_pillars``
        ``{pillar: n_tickers}`` whose leg was dropped as undefined. Non-empty
        here with ``n_excluded_undefined == 0`` is the common, benign case: the
        pillar had no dispersion but other pillars carried the name.
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
    degenerate_pillars: dict[str, int] = {}
    for ticker, scores in pillar_scores_by_ticker.items():
        contribs: dict[str, tuple[float, float]] = {}
        undefined: list[str] = []
        num = 0.0
        wsum = 0.0
        for p in PILLAR_ORDER:
            v = scores.get(p)
            w = weights.get(p, 0.0)
            if v is None or w <= 0 or p not in pillar_stats:
                continue
            mean, std = pillar_stats[p]
            z = _zscore(v, mean, std)
            if z is None:
                # config-I7272: the z is UNDEFINED (zero cross-sectional
                # dispersion). DROP the leg — `wsum` deliberately does not
                # advance, so the surviving pillars renormalize — rather than
                # vote a fabricated 0.0 that would drag the blend toward
                # neutral. Recorded on the row and counted in the coverage
                # report so the drop is inspectable, never silent.
                undefined.append(p)
                degenerate_pillars[p] = degenerate_pillars.get(p, 0) + 1
                continue
            num += w * z
            wsum += w
            contribs[p] = (w, z)
        rec = {
            "attractiveness_raw": None,
            "attractiveness_score": None,
            "pillar_contributions": {},
            "undefined_pillars": undefined,
        }
        if wsum > 0:
            blend = num / wsum
            blends[ticker] = blend
            rec["attractiveness_raw"] = round(blend, 4)
            rec["pillar_contributions"] = {p: round(w * z / wsum, 4) for p, (w, z) in contribs.items()}
        out[ticker] = rec

    # Tickers absent from `blends` are EXCLUDED from the ranking, not scored:
    # `_avg_rank_pct` never sees them, so they take no percentile position.
    pct = _avg_rank_pct(blends)
    for ticker in out:
        out[ticker]["attractiveness_score"] = pct.get(ticker)

    excluded = sorted(t for t in out if out[t]["attractiveness_score"] is None)
    coverage = {
        "n_tickers": len(out),
        "n_scored": len(out) - len(excluded),
        "n_excluded_undefined": len(excluded),
        "excluded_tickers": excluded,
        "degenerate_pillars": dict(sorted(degenerate_pillars.items())),
    }
    return out, coverage


def attractiveness_from_factor_profiles(
    factor_profiles: dict[str, dict],
    *,
    pillar_weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Compute attractiveness from ``{ticker: {quality_score, value_score, …}}`` profiles."""
    return attractiveness_from_factor_profiles_with_coverage(
        factor_profiles, pillar_weights=pillar_weights
    )[0]


def attractiveness_from_factor_profiles_with_coverage(
    factor_profiles: dict[str, dict],
    *,
    pillar_weights: dict[str, float] | None = None,
) -> tuple[dict[str, dict], dict]:
    """As :func:`attractiveness_from_factor_profiles`, plus the coverage report.

    See :func:`compute_cross_sectional_attractiveness_with_coverage` for the
    report's shape and why it is emitted even when nothing was excluded.
    """
    weights = normalize_pillar_weights(pillar_weights)
    pillar_scores_by_ticker = {
        ticker: {
            pillar: _num(profile.get(PILLAR_TO_FACTOR_KEY[pillar]))
            for pillar in PILLAR_ORDER
        }
        for ticker, profile in (factor_profiles or {}).items()
        if isinstance(profile, dict)
    }
    return compute_cross_sectional_attractiveness_with_coverage(
        pillar_scores_by_ticker, weights
    )
