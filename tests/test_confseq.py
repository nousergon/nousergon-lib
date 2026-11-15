"""Direct tests for ``nousergon_lib.arena.confseq`` with literal expected values.

Radii below were checked against the Robbins normal-mixture formula
u(t) = sqrt(2 (V+rho) ln(sqrt((V+rho)/rho)/alpha)), V = n*sigma^2, rho = opt_n*sigma^2.
"""

from __future__ import annotations

import pytest

from nousergon_lib.arena.confseq import ConfSeqBound, confidence_sequence


def test_single_observation_radius_matches_the_closed_form() -> None:
    bound = confidence_sequence([0.01], clip=1.0)
    assert bound.n == 1
    assert bound.sigma == 1.0
    assert bound.radius == pytest.approx(12.75886090634475)
    assert bound.mean == pytest.approx(0.01)
    assert bound.lower == pytest.approx(0.01 - 12.75886090634475)
    assert bound.upper == pytest.approx(0.01 + 12.75886090634475)
    assert bound.method == "robbins-normal-mixture"
    assert bound.variance_mode == "declared"
    assert bound.alpha == 0.05


def test_alpha_and_opt_n_change_the_radius() -> None:
    bound = confidence_sequence([0.5] * 4, alpha=0.1, clip=1.0, opt_n=4)
    assert bound.radius == pytest.approx(1.6276236307187293)
    assert bound.alpha == 0.1


def test_interval_is_wide_early_and_narrows_with_evidence() -> None:
    short = confidence_sequence([0.03] * 3, clip=0.05)
    long = confidence_sequence([0.03] * 300, clip=0.05)
    assert short.radius > long.radius
    # Mutation guarded: dropping the 1/n in radius makes `supported` flip early.
    assert not short.supported
    assert long.supported
    assert not long.supported_negative


def test_negative_lead_is_supported_negative_only() -> None:
    bound = confidence_sequence([-0.03] * 300, clip=0.05)
    assert bound.supported_negative
    assert not bound.supported


def test_interval_straddling_zero_supports_neither_direction() -> None:
    bound = confidence_sequence([0.01, -0.01] * 5, clip=0.05)
    assert bound.mean == pytest.approx(0.0)
    assert not bound.supported
    assert not bound.supported_negative


def test_clip_winsorises_both_tails_and_counts_them() -> None:
    bound = confidence_sequence([1.0, -1.0, 3.0], clip=2.0)
    # values become [1, -1, 2]
    assert bound.n_clipped == 1
    assert bound.mean == pytest.approx(2.0 / 3.0)
    assert bound.radius == pytest.approx(8.867399752479523)
    both = confidence_sequence([9.0, -9.0, 0.0], clip=2.0)
    assert both.n_clipped == 2
    assert both.mean == pytest.approx(0.0)


def test_values_exactly_at_the_clip_are_not_counted_as_clipped() -> None:
    bound = confidence_sequence([2.0, -2.0], clip=2.0)
    assert bound.n_clipped == 0


def test_empirical_mode_uses_sample_standard_deviation() -> None:
    bound = confidence_sequence([1.0, 3.0], variance_mode="empirical")
    assert bound.sigma == pytest.approx(1.4142135623730951)
    assert bound.radius == pytest.approx(9.21509713837374)
    assert bound.variance_mode == "empirical"


def test_empirical_degenerate_sample_falls_back_to_declared_clip() -> None:
    bound = confidence_sequence([1.0, 1.0], variance_mode="empirical", clip=0.5)
    assert bound.sigma == 0.5
    single = confidence_sequence([1.0], variance_mode="empirical", clip=0.5)
    assert single.sigma == 0.5


def test_empirical_degenerate_sample_without_clip_raises() -> None:
    # Mutation guarded: returning a zero-width interval would support any lead.
    with pytest.raises(ValueError, match="zero scale"):
        confidence_sequence([1.0, 1.0], variance_mode="empirical")


def test_empty_window_raises_instead_of_returning_a_wide_interval() -> None:
    with pytest.raises(ValueError, match="empty window"):
        confidence_sequence([], clip=0.05)


def test_declared_mode_requires_clip() -> None:
    with pytest.raises(ValueError, match="requires an explicit `clip`"):
        confidence_sequence([0.01, 0.02])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_outside_open_unit_interval_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        confidence_sequence([0.01], alpha=alpha, clip=0.05)


def test_opt_n_below_one_raises() -> None:
    with pytest.raises(ValueError, match="opt_n must be >= 1"):
        confidence_sequence([0.01], clip=0.05, opt_n=0)


@pytest.mark.parametrize("clip", [0.0, -0.05])
def test_non_positive_clip_raises(clip: float) -> None:
    with pytest.raises(ValueError, match="clip must be > 0"):
        confidence_sequence([0.01], clip=clip)


def test_unknown_variance_mode_raises() -> None:
    with pytest.raises(ValueError, match="variance_mode must be"):
        confidence_sequence([0.01], clip=0.05, variance_mode="bootstrap")


def test_to_dict_carries_every_field_and_derived_supported() -> None:
    payload = confidence_sequence([0.03] * 300, clip=0.05).to_dict()
    assert set(payload) == {
        "mean", "lower", "upper", "radius", "n", "alpha", "sigma",
        "variance_mode", "method", "n_clipped", "supported",
    }  # fmt: skip
    assert payload["supported"] is True
    assert payload["n"] == 300


def test_from_dict_roundtrip_and_supported_is_recomputed() -> None:
    bound = confidence_sequence([0.01, 0.02, -0.01], clip=0.05)
    assert ConfSeqBound.from_dict(bound.to_dict()) == bound
    tampered = {**bound.to_dict(), "supported": True}
    assert not ConfSeqBound.from_dict(tampered).supported


def test_from_dict_missing_field_raises_keyerror() -> None:
    payload = confidence_sequence([0.01], clip=0.05).to_dict()
    del payload["sigma"]
    with pytest.raises(KeyError):
        ConfSeqBound.from_dict(payload)
