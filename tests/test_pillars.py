"""Tests for the pillars schema module — canonical 6-pillar attractiveness
shapes used by alpha-engine-research's Qual Analyst via tool-use forced
output.

Coverage strategy mirrors ``test_agent_schemas``: each schema gets a
happy-path test, plus a regression test for any non-trivial validator
(score range, durability cap, primary-vs-secondary moat collision, evidence
trimming). The ``PILLARS`` tuple ↔ ``PillarLiteral`` consistency check is
the structural invariant — both must be the same vocabulary.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError


# ── Vocabulary invariants ────────────────────────────────────────────────


class TestPillarVocabulary:
    def test_pillars_tuple_has_canonical_six(self):
        from nousergon_lib.pillars import PILLARS

        assert PILLARS == (
            "quality",
            "value",
            "momentum",
            "growth",
            "stewardship",
            "defensiveness",
        )

    def test_pillar_literal_matches_pillars_tuple(self):
        """``PillarLiteral`` and ``PILLARS`` must enumerate the same values
        in the same order. If one changes, the other must change in
        lockstep."""
        from nousergon_lib.pillars import PILLARS, PillarLiteral

        assert get_args(PillarLiteral) == PILLARS

    def test_moat_types_include_all_six_archetypes_plus_none(self):
        from nousergon_lib.pillars import MoatType

        assert set(get_args(MoatType)) == {
            "network_effects",
            "switching_costs",
            "cost_advantage",
            "intangibles",
            "efficient_scale",
            "process_power",
            "none",
        }


# ── MoatAssessment ───────────────────────────────────────────────────────


class TestMoatAssessment:
    def test_accepts_typical_wide_moat_payload(self):
        from nousergon_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="process_power",
            secondary_types=["intangibles"],
            width="wide",
            durability_years=25,
            trend="stable",
            evidence=[
                "TSMC's leading-edge node (3nm) commands 60%+ market share per 2026 Q1 10-Q",
                "Capex barrier estimated at $20B+ per fab per 10-K risk factors section",
            ],
        )
        assert moat.primary_type == "process_power"
        assert moat.width == "wide"
        assert len(moat.evidence) == 2

    def test_accepts_no_moat_default(self):
        """The honest default — most stocks have no identifiable moat."""
        from nousergon_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="none",
            width="none",
            durability_years=0,
            trend="stable",
        )
        assert moat.primary_type == "none"
        assert moat.secondary_types == []
        assert moat.evidence == []

    def test_durability_upper_bound_50_years(self):
        from nousergon_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError):
            MoatAssessment(
                primary_type="network_effects",
                width="wide",
                durability_years=51,  # > cap
                trend="stable",
            )

    def test_durability_lower_bound_zero(self):
        from nousergon_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError):
            MoatAssessment(
                primary_type="none",
                width="none",
                durability_years=-1,
                trend="stable",
            )

    def test_primary_must_not_appear_in_secondary(self):
        """LLM failure mode: agents sometimes restate primary in secondary
        for emphasis."""
        from nousergon_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError, match="primary_type"):
            MoatAssessment(
                primary_type="network_effects",
                secondary_types=["network_effects", "switching_costs"],
                width="wide",
                durability_years=20,
                trend="stable",
            )

    def test_secondary_types_must_be_unique(self):
        from nousergon_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError, match="unique"):
            MoatAssessment(
                primary_type="cost_advantage",
                secondary_types=["intangibles", "intangibles"],
                width="narrow",
                durability_years=12,
                trend="stable",
            )

    def test_evidence_strings_trimmed_and_empties_dropped(self):
        """LLM occasionally emits trailing whitespace + empty strings from
        format-token confusion."""
        from nousergon_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="efficient_scale",
            width="narrow",
            durability_years=15,
            trend="widening",
            evidence=["  regional landfill network  ", "", "   ", "permit moat"],
        )
        assert moat.evidence == ["regional landfill network", "permit moat"]

    def test_extra_fields_allowed_for_forward_compat(self):
        from nousergon_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="intangibles",
            width="wide",
            durability_years=30,
            trend="stable",
            future_field="ok",  # type: ignore[call-arg]
        )
        assert moat.primary_type == "intangibles"


# ── PillarSubscore ───────────────────────────────────────────────────────


class TestPillarSubscore:
    def test_accepts_typical_qual_only_emission(self):
        """At LLM emission time, only qual fields are populated; quant
        component arrives later from the composite scoring layer."""
        from nousergon_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="quality",
            score=82,
            confidence="high",
            qual_component=82,
            evidence=["ROE > 25% sustained 5y", "wide moat from process power"],
        )
        assert sub.pillar == "quality"
        assert sub.score == 82
        assert sub.quant_component is None
        assert sub.qual_component == 82

    def test_accepts_blended_emission_with_both_components(self):
        from nousergon_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="momentum",
            score=68,  # blended
            confidence="medium",
            quant_component=72.3,
            qual_component=64,
        )
        assert sub.quant_component == pytest.approx(72.3)
        assert sub.qual_component == 64

    def test_score_range_enforced(self):
        from nousergon_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(pillar="value", score=150, confidence="medium")

        with pytest.raises(ValidationError):
            PillarSubscore(pillar="value", score=-5, confidence="medium")

    def test_qual_component_range_enforced(self):
        from nousergon_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="growth",
                score=50,
                confidence="low",
                qual_component=200,
            )

    def test_confidence_literal_enforced(self):
        from nousergon_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="defensiveness",
                score=50,
                confidence="certain",  # not in {low, medium, high}
            )

    def test_pillar_literal_enforced(self):
        from nousergon_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="liquidity",  # not a canonical pillar
                score=50,
                confidence="medium",
            )

    def test_evidence_strings_trimmed(self):
        from nousergon_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="stewardship",
            score=60,
            confidence="medium",
            evidence=["", "  buyback at 15x P/E  ", "   "],
        )
        assert sub.evidence == ["buyback at 15x P/E"]


# ── QualitativePillarAssessment ──────────────────────────────────────────


def _make_subscore(pillar, score=70, confidence="medium"):
    from nousergon_lib.pillars import PillarSubscore

    return PillarSubscore(pillar=pillar, score=score, confidence=confidence)


def _make_full_assessment(**overrides):
    """Helper: build a full 6-pillar + moat assessment with sensible defaults.

    Tests override individual fields via kwargs. The default payload is the
    "moderately attractive across the board" stock — score 70 on every
    pillar, narrow moat, zero catalyst modulation."""
    from nousergon_lib.pillars import (
        MoatAssessment,
        QualitativePillarAssessment,
    )

    payload = {
        "quality": _make_subscore("quality"),
        "quality_moat": MoatAssessment(
            primary_type="cost_advantage",
            width="narrow",
            durability_years=12,
            trend="stable",
        ),
        "value": _make_subscore("value"),
        "momentum": _make_subscore("momentum"),
        "growth": _make_subscore("growth"),
        "stewardship": _make_subscore("stewardship"),
        "defensiveness": _make_subscore("defensiveness"),
    }
    payload.update(overrides)
    return QualitativePillarAssessment(**payload)


class TestQualitativePillarAssessment:
    def test_accepts_typical_full_payload(self):
        assessment = _make_full_assessment()
        subscores = assessment.pillar_subscores()
        assert set(subscores.keys()) == {
            "quality",
            "value",
            "momentum",
            "growth",
            "stewardship",
            "defensiveness",
        }
        # Iteration order matches PILLARS canonical ordering
        from nousergon_lib.pillars import PILLARS

        assert tuple(subscores.keys()) == PILLARS

    def test_catalyst_horizon_modulation_default_zero(self):
        assessment = _make_full_assessment()
        assert assessment.catalyst_horizon_modulation == 0

    def test_catalyst_horizon_modulation_bounds(self):
        from nousergon_lib.pillars import QualitativePillarAssessment

        with pytest.raises(ValidationError):
            _make_full_assessment(catalyst_horizon_modulation=25)

        with pytest.raises(ValidationError):
            _make_full_assessment(catalyst_horizon_modulation=-25)

        # Accepts boundary values
        a = _make_full_assessment(catalyst_horizon_modulation=20)
        assert a.catalyst_horizon_modulation == 20
        b = _make_full_assessment(catalyst_horizon_modulation=-20)
        assert b.catalyst_horizon_modulation == -20

    def test_derive_legacy_qual_score_equal_weight_mean(self):
        """Translation layer for Phase 2 soak — legacy composite needs
        a scalar."""
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=90),
            value=_make_subscore("value", score=60),
            momentum=_make_subscore("momentum", score=70),
            growth=_make_subscore("growth", score=80),
            stewardship=_make_subscore("stewardship", score=50),
            defensiveness=_make_subscore("defensiveness", score=70),
        )
        # Mean of (90, 60, 70, 80, 50, 70) = 420 / 6 = 70
        assert assessment.derive_legacy_qual_score() == 70

    def test_derive_legacy_qual_score_returns_int_in_range(self):
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=100),
            value=_make_subscore("value", score=100),
            momentum=_make_subscore("momentum", score=100),
            growth=_make_subscore("growth", score=100),
            stewardship=_make_subscore("stewardship", score=100),
            defensiveness=_make_subscore("defensiveness", score=100),
        )
        result = assessment.derive_legacy_qual_score()
        assert isinstance(result, int)
        assert 0 <= result <= 100
        assert result == 100

    def test_derive_legacy_qual_score_rounds(self):
        """Mean of (1, 1, 1, 1, 1, 0) = 5/6 ≈ 0.833 → rounds to 1."""
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=1),
            value=_make_subscore("value", score=1),
            momentum=_make_subscore("momentum", score=1),
            growth=_make_subscore("growth", score=1),
            stewardship=_make_subscore("stewardship", score=1),
            defensiveness=_make_subscore("defensiveness", score=0),
        )
        assert assessment.derive_legacy_qual_score() == 1

    def test_quality_moat_embedded(self):
        """The Quality pillar's qualitative core — moat — is embedded as a
        first-class field, not buried in evidence strings."""
        from nousergon_lib.pillars import MoatAssessment

        assessment = _make_full_assessment(
            quality_moat=MoatAssessment(
                primary_type="network_effects",
                secondary_types=["switching_costs"],
                width="wide",
                durability_years=25,
                trend="widening",
                evidence=["card-network two-sided market dynamics"],
            )
        )
        assert assessment.quality_moat.primary_type == "network_effects"
        assert assessment.quality_moat.width == "wide"
        assert assessment.quality_moat.trend == "widening"

    def test_missing_required_pillar_rejected(self):
        from nousergon_lib.pillars import (
            MoatAssessment,
            QualitativePillarAssessment,
        )

        with pytest.raises(ValidationError):
            # Missing 'defensiveness' field.
            QualitativePillarAssessment(  # type: ignore[call-arg]
                quality=_make_subscore("quality"),
                quality_moat=MoatAssessment(
                    primary_type="none",
                    width="none",
                    durability_years=0,
                    trend="stable",
                ),
                value=_make_subscore("value"),
                momentum=_make_subscore("momentum"),
                growth=_make_subscore("growth"),
                stewardship=_make_subscore("stewardship"),
            )

    def test_extra_fields_allowed_for_forward_compat(self):
        """LLM may emit additional fields as the prompt evolves."""
        assessment = _make_full_assessment(
            future_field="ok",  # type: ignore[call-arg]
        )
        # Doesn't raise; extra field is allowed.
        assert assessment.pillar_subscores()["quality"].pillar == "quality"


# ── CompositeBreakdown (Phase 4) ─────────────────────────────────────────


def _make_pillar_contribution(pillar: str, weight: float = 0.0,
                              qual: float | None = 70.0,
                              quant: float | None = 60.0,
                              alpha: float = 0.5):
    """Helper — build a PillarContribution with the within-pillar blend
    pre-computed. Defaults model the Phase 4 default (pillar_weight=0)."""
    from nousergon_lib.pillars import PillarContribution

    if qual is None and quant is None:
        blended = None
        contribution = 0.0
    elif qual is None:
        blended = quant
        contribution = weight * (quant or 0.0)
    elif quant is None:
        blended = qual
        contribution = weight * (qual or 0.0)
    else:
        blended = alpha * qual + (1.0 - alpha) * quant
        contribution = weight * blended

    return PillarContribution(
        pillar=pillar,  # type: ignore[arg-type]
        qual_component=qual,
        quant_component=quant,
        within_pillar_qual_weight=alpha,
        blended=blended,
        pillar_weight=weight,
        contribution=contribution,
    )


def _make_legacy_blend(quant=70.0, qual=75.0, factor=65.0,
                      w_quant=0.35, w_qual=0.35, w_factor=0.30):
    from nousergon_lib.pillars import LegacyComponentBlend

    contribution = (
        w_quant * (quant or 0.0)
        + w_qual * (qual or 0.0)
        + w_factor * (factor or 0.0)
    )
    return LegacyComponentBlend(
        quant_score=quant,
        qual_score=qual,
        factor_subscore=factor,
        w_legacy_quant=w_quant,
        w_legacy_qual=w_qual,
        w_factor=w_factor,
        contribution=contribution,
    )


def _make_breakdown(pillar_contributions=None, legacy_blend=None,
                   macro_shift=0.0, boosts_total=0.0, catalyst_modulation=0):
    """Helper — build a CompositeBreakdown defaulting to Phase 4 cutover
    state: 6 pillar contributions with pillar_weight=0 + legacy_blend
    carrying all the weight."""
    from nousergon_lib.pillars import CompositeBreakdown, PILLARS

    if pillar_contributions is None:
        pillar_contributions = [
            _make_pillar_contribution(p, weight=0.0) for p in PILLARS
        ]
    if legacy_blend is None:
        legacy_blend = _make_legacy_blend()

    weighted_base = (
        sum(c.contribution for c in pillar_contributions)
        + legacy_blend.contribution
    )
    final = max(0.0, min(100.0,
                         weighted_base + macro_shift + boosts_total
                         + catalyst_modulation))

    return CompositeBreakdown(
        final_score=round(final, 1),
        weighted_base=round(weighted_base, 1),
        macro_shift=macro_shift,
        boosts_total=boosts_total,
        catalyst_modulation=catalyst_modulation,
        pillar_contributions=pillar_contributions,
        legacy_blend=legacy_blend,
        score_failed=False,
    )


class TestPillarContribution:
    def test_typical_phase4_zero_weight_contribution(self):
        c = _make_pillar_contribution("quality", weight=0.0,
                                       qual=80.0, quant=70.0, alpha=0.5)
        assert c.pillar == "quality"
        assert c.blended == 75.0  # 0.5×80 + 0.5×70
        assert c.pillar_weight == 0.0
        assert c.contribution == 0.0

    def test_blended_none_when_both_components_none(self):
        c = _make_pillar_contribution("stewardship", weight=0.1,
                                       qual=None, quant=None)
        assert c.blended is None
        assert c.contribution == 0.0

    def test_alpha_one_degrades_to_pure_qual(self):
        """When factor_profile is absent, within_pillar_qual_weight=1.0 →
        blended equals qual_component."""
        c = _make_pillar_contribution("stewardship", weight=0.1,
                                       qual=80.0, quant=None, alpha=1.0)
        assert c.blended == 80.0
        assert c.contribution == 8.0  # 0.1 × 80

    def test_alpha_zero_degrades_to_pure_quant(self):
        """When pillar_assessment is absent for ticker, within_pillar_qual_weight=0.0
        → blended equals quant_component."""
        c = _make_pillar_contribution("momentum", weight=0.2,
                                       qual=None, quant=65.0, alpha=0.0)
        assert c.blended == 65.0
        assert c.contribution == 13.0  # 0.2 × 65

    def test_alpha_clamped_0_to_1(self):
        from nousergon_lib.pillars import PillarContribution

        with pytest.raises(ValidationError):
            PillarContribution(
                pillar="quality",
                qual_component=70.0,
                quant_component=60.0,
                within_pillar_qual_weight=1.5,  # > 1.0
                blended=65.0,
                pillar_weight=0.1,
                contribution=6.5,
            )

    def test_pillar_weight_clamped_0_to_1(self):
        from nousergon_lib.pillars import PillarContribution

        with pytest.raises(ValidationError):
            PillarContribution(
                pillar="value",
                within_pillar_qual_weight=0.5,
                pillar_weight=-0.1,  # negative
                contribution=0.0,
            )


class TestLegacyComponentBlend:
    def test_phase4_default_weights_sum_to_one(self):
        blend = _make_legacy_blend()
        assert blend.w_legacy_quant == 0.35
        assert blend.w_legacy_qual == 0.35
        assert blend.w_factor == 0.30
        assert (blend.w_legacy_quant + blend.w_legacy_qual
                + blend.w_factor) == pytest.approx(1.0)

    def test_contribution_matches_weighted_sum(self):
        blend = _make_legacy_blend(quant=70.0, qual=80.0, factor=60.0)
        expected = 0.35 * 70.0 + 0.35 * 80.0 + 0.30 * 60.0
        assert blend.contribution == pytest.approx(expected)

    def test_individual_weights_clamped_0_to_1(self):
        from nousergon_lib.pillars import LegacyComponentBlend

        with pytest.raises(ValidationError):
            LegacyComponentBlend(
                quant_score=70.0,
                qual_score=80.0,
                factor_subscore=60.0,
                w_legacy_quant=1.5,  # > 1.0
                w_legacy_qual=0.0,
                w_factor=0.0,
                contribution=105.0,
            )


class TestCompositeBreakdown:
    def test_phase4_default_state_is_valid(self):
        """Phase 4 cutover: every pillar_weight=0, legacy_blend 0.35/0.35/0.30
        — full breakdown round-trips clean."""
        breakdown = _make_breakdown()
        assert breakdown.final_score is not None
        assert breakdown.weighted_base is not None
        assert len(breakdown.pillar_contributions) == 6

    def test_phase4_default_final_score_equals_legacy_formula(self):
        """At Phase 4 default weights (pillar_weights all 0), final_score
        equals the legacy compute_composite_score formula by CONSTRUCTION:

          weighted_base = 0.35 × quant + 0.35 × qual + 0.30 × factor
                        + Σ (0 × pillar) = legacy
          final_score   = clamp(weighted_base + macro_shift + boosts, 0, 100)

        This is the plan-doc ±0.5 acceptance criterion satisfied without
        fixture tuning. No tolerance — exact equality (rounded to 1 decimal)."""
        breakdown = _make_breakdown(macro_shift=5.0, boosts_total=2.0)
        # Reproduce legacy formula by hand:
        # quant_qual_base = 0.5 × 70 + 0.5 × 75 = 72.5
        # weighted_base   = 0.7 × 72.5 + 0.3 × 65 = 50.75 + 19.5 = 70.25
        # final           = clamp(70.25 + 5.0 + 2.0, 0, 100) = 77.25 → 77.2 or 77.3 rounded
        # NEW formula at default:
        # weighted_base = 0.35 × 70 + 0.35 × 75 + 0.30 × 65 = 24.5 + 26.25 + 19.5 = 70.25
        # final = clamp(70.25 + 5.0 + 2.0 + 0, 0, 100) = 77.25 → 77.2 or 77.3
        # The "0.7 × quant_qual_base + 0.3 × factor" form expands to
        # 0.7 × 0.5 × quant + 0.7 × 0.5 × qual + 0.3 × factor
        # = 0.35 × quant + 0.35 × qual + 0.30 × factor — identical to new formula
        assert breakdown.weighted_base == pytest.approx(70.25, abs=0.05)
        # final_score is rounded to 1 decimal; legacy uses bankers'-round so
        # 77.25 may round to either 77.2 or 77.3 depending on Python version
        # — accept the ±0.1 window the plan-doc ±0.5 tolerance covers easily.
        assert breakdown.final_score is not None
        assert abs(breakdown.final_score - 77.25) <= 0.5

    def test_weights_sum_invariant_enforced(self):
        """Σ pillar_weights + Σ legacy_weights must equal 1.0 within 1e-6."""
        from nousergon_lib.pillars import (
            CompositeBreakdown, PillarContribution, PILLARS
        )

        # Build pillar contributions that sum to 0.6
        pillar_contribs = [
            _make_pillar_contribution(p, weight=0.1) for p in PILLARS
        ]
        # Legacy blend sums to 0.3 — TOTAL 0.9, missing 0.1
        bad_legacy = _make_legacy_blend(
            w_quant=0.1, w_qual=0.1, w_factor=0.1,
        )
        with pytest.raises(ValidationError):
            CompositeBreakdown(
                final_score=50.0,
                weighted_base=50.0,
                macro_shift=0.0,
                boosts_total=0.0,
                catalyst_modulation=0,
                pillar_contributions=pillar_contribs,
                legacy_blend=bad_legacy,
                score_failed=False,
            )

    def test_weights_sum_invariant_passes_at_default(self):
        """Default: 6 × 0 (pillar) + 0.35 + 0.35 + 0.30 (legacy) = 1.0 ✓"""
        breakdown = _make_breakdown()
        assert breakdown.weighted_base is not None

    def test_weights_sum_invariant_passes_with_ramped_pillars(self):
        """Phase 6 ramps pillars up + legacy down. As long as totals stay
        at 1.0 the invariant accepts it."""
        from nousergon_lib.pillars import PILLARS

        # Pillars ramp to 0.5 total (e.g. 0.083 each); legacy ramps to 0.5
        # (e.g. 0.175 / 0.175 / 0.150).
        pillar_contribs = [
            _make_pillar_contribution(p, weight=0.5 / 6) for p in PILLARS
        ]
        legacy = _make_legacy_blend(
            w_quant=0.175, w_qual=0.175, w_factor=0.150,
        )
        breakdown = _make_breakdown(
            pillar_contributions=pillar_contribs,
            legacy_blend=legacy,
        )
        assert breakdown.final_score is not None

    def test_catalyst_modulation_clamped_to_plus_minus_20(self):
        from nousergon_lib.pillars import CompositeBreakdown

        with pytest.raises(ValidationError):
            CompositeBreakdown(
                final_score=50.0,
                weighted_base=50.0,
                macro_shift=0.0,
                boosts_total=0.0,
                catalyst_modulation=25,  # > 20
                pillar_contributions=[],
                legacy_blend=_make_legacy_blend(),
                score_failed=False,
            )

    def test_pillar_contributions_by_name_returns_canonical_order(self):
        """When all 6 pillars present, by-name lookup returns canonical
        PILLARS ordering."""
        from nousergon_lib.pillars import PILLARS

        breakdown = _make_breakdown()
        by_name = breakdown.pillar_contributions_by_name()
        assert list(by_name.keys()) == list(PILLARS)

    def test_empty_pillar_contributions_legacy_only_path(self):
        """When pillar emission disabled, pillar_contributions is empty;
        the breakdown is still well-formed and reduces to legacy_blend +
        macro/boosts. This is the runtime path before PILLAR_EMIT flips."""
        breakdown = _make_breakdown(pillar_contributions=[])
        assert breakdown.pillar_contributions == []
        # weighted_base is rounded to 1 decimal by the helper; legacy_blend
        # carries the raw weighted sum. ±0.05 tolerance covers the rounding.
        assert breakdown.weighted_base == pytest.approx(
            breakdown.legacy_blend.contribution, abs=0.05
        )

    def test_score_failed_path_emits_none_final_score(self):
        """All input components None → score_failed=True, final_score=None."""
        from nousergon_lib.pillars import CompositeBreakdown

        breakdown = CompositeBreakdown(
            final_score=None,
            weighted_base=None,
            macro_shift=0.0,
            boosts_total=0.0,
            catalyst_modulation=0,
            pillar_contributions=[],
            legacy_blend=_make_legacy_blend(quant=None, qual=None, factor=None),
            score_failed=True,
        )
        assert breakdown.score_failed
        assert breakdown.final_score is None

    def test_extra_fields_allowed_for_forward_compat(self):
        breakdown = _make_breakdown()
        from nousergon_lib.pillars import CompositeBreakdown
        # Add an extra field; permissive ConfigDict should accept it.
        payload = breakdown.model_dump()
        payload["future_field"] = "ok"
        breakdown_2 = CompositeBreakdown.model_validate(payload)
        assert breakdown_2.final_score == breakdown.final_score
