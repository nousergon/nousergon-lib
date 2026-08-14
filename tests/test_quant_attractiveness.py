"""Cross-sectional 6-pillar attractiveness composite."""

from __future__ import annotations

from nousergon_lib.quant.attractiveness import (
    DEFAULT_PILLAR_WEIGHTS,
    PILLAR_ORDER,
    attractiveness_from_factor_profiles,
    compute_cross_sectional_attractiveness,
    normalize_pillar_weights,
)


def _profiles(n: int) -> dict[str, dict]:
    return {
        f"T{i}": {
            "quality_score": float(10 + i),
            "value_score": float(20 + i),
            "momentum_score": float(30 + i),
            "growth_score": float(40 + i),
            "stewardship_score": float(50 + i),
            "low_vol_score": float(60 + i),
        }
        for i in range(n)
    }


def test_equal_weights_sum_to_one():
    assert abs(sum(DEFAULT_PILLAR_WEIGHTS.values()) - 1.0) < 1e-9


def test_normalize_pillar_weights_falls_back_to_equal():
    assert normalize_pillar_weights(None) == DEFAULT_PILLAR_WEIGHTS
    assert normalize_pillar_weights({}) == DEFAULT_PILLAR_WEIGHTS


def test_dispersion_restored_vs_mean_of_percentiles():
    profiles = _profiles(20)
    out = attractiveness_from_factor_profiles(profiles)
    scores = [out[t]["attractiveness_score"] for t in profiles]
    assert min(scores) < 30
    assert max(scores) > 70


def test_pillar_contributions_sum_to_raw_blend():
    profiles = {
        "AAPL": {
            "quality_score": 90.0,
            "value_score": 30.0,
            "momentum_score": 85.0,
            "growth_score": 80.0,
            "stewardship_score": 70.0,
            "low_vol_score": 60.0,
        },
        "MSFT": {
            "quality_score": 60.0,
            "value_score": 50.0,
            "momentum_score": 55.0,
            "growth_score": 45.0,
            "stewardship_score": 40.0,
            "low_vol_score": 35.0,
        },
    }
    out = attractiveness_from_factor_profiles(profiles)
    aapl = out["AAPL"]
    assert aapl["attractiveness_raw"] is not None
    assert abs(sum(aapl["pillar_contributions"].values()) - aapl["attractiveness_raw"]) < 1e-3


def test_missing_pillars_renormalize_weights():
    pillar_scores = {
        "AAPL": {p: 80.0 if p == "quality" else None for p in PILLAR_ORDER},
        "MSFT": {p: 40.0 if p == "quality" else None for p in PILLAR_ORDER},
    }
    out = compute_cross_sectional_attractiveness(pillar_scores, DEFAULT_PILLAR_WEIGHTS)
    assert out["AAPL"]["attractiveness_score"] == 100.0
    assert out["MSFT"]["attractiveness_score"] == 50.0


def test_no_usable_pillars_returns_null_score():
    out = compute_cross_sectional_attractiveness(
        {"ZZZ": dict.fromkeys(PILLAR_ORDER)},
        DEFAULT_PILLAR_WEIGHTS,
    )
    assert out["ZZZ"]["attractiveness_score"] is None


# ── config-I7272: an undefined z-score is UNDEFINED, never a fabricated 0.0 ──
#
# `_zscore` used to return a finite 0.0 whenever the cross-sectional std was
# <= 0 (a single observation for that pillar, or every ticker carrying the
# identical percentile). 0.0 is EXACTLY the value a genuinely at-the-mean
# ticker produces, so "there was nothing to compare against" and "this name
# sits at its cohort mean" were indistinguishable — and worse, the fabricated
# 0.0 was VOTED into the weighted blend, diluting every other pillar toward
# neutral with a number nobody measured.
#
# These tests are the acceptance battery for the fix. Each one FAILS against
# the pre-fix implementation; see the PR body for the recorded output.


def _degenerate_pillar_scores() -> dict[str, dict[str, float | None]]:
    """Two names: `quality` disperses, `value` is CONSTANT (undefined z)."""
    return {
        "AAA": {"quality": 90.0, "value": 50.0,
                **{p: None for p in PILLAR_ORDER if p not in ("quality", "value")}},
        "BBB": {"quality": 10.0, "value": 50.0,
                **{p: None for p in PILLAR_ORDER if p not in ("quality", "value")}},
    }


def test_zscore_reports_undefined_rather_than_a_fabricated_zero():
    """REPRESENTABLE + DISTINGUISHABLE. A zero-std cross-section has no defined
    z-score; the function must say so instead of returning the one value a real
    at-the-mean observation also produces."""
    from nousergon_lib.quant.attractiveness import _zscore

    assert _zscore(50.0, 50.0, 0.0) is None, "zero std must be UNDEFINED, not 0.0"
    assert _zscore(80.0, 50.0, 0.0) is None, "zero std is undefined regardless of value"
    assert _zscore(-1.0, 50.0, -0.5) is None, "negative std is undefined"
    # DISTINGUISHABLE: a genuinely at-the-mean observation against a REAL
    # dispersion still returns a measured 0.0. The two cases must not collide.
    assert _zscore(50.0, 50.0, 10.0) == 0.0


def test_undefined_pillar_is_dropped_from_the_blend_not_voted_as_zero():
    """A degenerate pillar must EXCLUDE itself from the weighted blend and let
    the surviving weights renormalize — not cast a fabricated 0.0 vote that
    drags the blend toward neutral."""
    out = compute_cross_sectional_attractiveness(
        _degenerate_pillar_scores(), DEFAULT_PILLAR_WEIGHTS
    )
    # `value` contributed nothing at all — no contribution term, and named as
    # undefined so the drop is inspectable rather than silent.
    for ticker in ("AAA", "BBB"):
        assert "value" not in out[ticker]["pillar_contributions"]
        assert out[ticker]["undefined_pillars"] == ["value"]
    # With `value` dropped, `quality` is the ONLY surviving leg, so the blend
    # equals that pillar's own (clipped) z — undiluted. Pre-fix this was
    # halved by the fabricated zero.
    assert out["AAA"]["attractiveness_raw"] == 1.0
    assert out["BBB"]["attractiveness_raw"] == -1.0


def test_a_wholly_undefined_ticker_is_excluded_from_the_ranking():
    """EXCLUDED, not ranked. A name whose every pillar is undefined has no
    measured position — it must not occupy one. A fabricated 0.0 would rank it
    against real scores; a None sorted to an end would still rank it, silently
    and systematically."""
    scores = {
        t: {"quality": 50.0, **{p: None for p in PILLAR_ORDER if p != "quality"}}
        for t in ("AAA", "BBB", "CCC")
    }
    out = compute_cross_sectional_attractiveness(scores, DEFAULT_PILLAR_WEIGHTS)
    for ticker in scores:
        assert out[ticker]["attractiveness_raw"] is None
        # The terminal percentile IS the ranking. An excluded name gets no
        # percentile at all, rather than the 100.0 the pre-fix code handed a
        # cross-section of fabricated zeros.
        assert out[ticker]["attractiveness_score"] is None


def test_the_excluded_count_is_published_even_when_it_is_zero():
    """An exclusion nobody can see is the same failure as a fabricated zero
    (principles.md 2.7). The coverage report is emitted UNCONDITIONALLY — a
    healthy run publishes an explicit zero, never silence."""
    from nousergon_lib.quant.attractiveness import (
        compute_cross_sectional_attractiveness_with_coverage,
    )

    # Healthy cross-section: nothing excluded, and the count still appears.
    _, coverage = compute_cross_sectional_attractiveness_with_coverage(
        {t: dict(zip(PILLAR_ORDER, [float(10 + i * 7)] * len(PILLAR_ORDER)))
         for i, t in enumerate(("AAA", "BBB", "CCC"))},
        DEFAULT_PILLAR_WEIGHTS,
    )
    assert coverage["n_excluded_undefined"] == 0
    assert coverage["excluded_tickers"] == []
    assert coverage["degenerate_pillars"] == {}
    assert coverage["n_tickers"] == 3
    assert coverage["n_scored"] == 3

    # Degenerate cross-section: the count is non-zero and NAMES its members —
    # a number published without its members is unactionable.
    _, coverage = compute_cross_sectional_attractiveness_with_coverage(
        _degenerate_pillar_scores(), DEFAULT_PILLAR_WEIGHTS
    )
    assert coverage["degenerate_pillars"] == {"value": 2}
    assert coverage["n_excluded_undefined"] == 0  # both still scored via `quality`
    assert coverage["n_scored"] == 2

    scores = {
        t: {"quality": 50.0, **{p: None for p in PILLAR_ORDER if p != "quality"}}
        for t in ("AAA", "BBB")
    }
    _, coverage = compute_cross_sectional_attractiveness_with_coverage(
        scores, DEFAULT_PILLAR_WEIGHTS
    )
    assert coverage["n_excluded_undefined"] == 2
    assert coverage["excluded_tickers"] == ["AAA", "BBB"]
    assert coverage["n_scored"] == 0


def test_the_plain_entrypoint_stays_a_thin_wrapper_over_the_coverage_one():
    """One implementation, two renderings — the count can never disagree with
    the scores it describes."""
    from nousergon_lib.quant.attractiveness import (
        compute_cross_sectional_attractiveness_with_coverage,
    )

    scores = _degenerate_pillar_scores()
    plain = compute_cross_sectional_attractiveness(scores, DEFAULT_PILLAR_WEIGHTS)
    detailed, _ = compute_cross_sectional_attractiveness_with_coverage(
        scores, DEFAULT_PILLAR_WEIGHTS
    )
    assert plain == detailed
