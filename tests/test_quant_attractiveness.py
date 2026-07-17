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
