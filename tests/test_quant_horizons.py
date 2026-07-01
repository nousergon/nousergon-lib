"""Tests for nousergon_lib.quant.horizons — the evaluation-horizon chokepoint (config#1483).

Pure stdlib module; no extra required. Pins the wide-column naming convention
(ground truth: the live score_performance table), the canonical-label guardrails
(primary fail-loud, diagnostics graceful-empty), and the config-mapping loader.
"""

from __future__ import annotations

import pytest

from nousergon_lib.quant import horizons
from nousergon_lib.quant.horizons import (
    DEFAULT_POLICY,
    DIAGNOSTIC_HORIZONS,
    PRIMARY_HORIZON,
    HorizonPolicy,
    LabelDefinition,
    PrimaryHorizonMissing,
    outcome_columns,
)


class TestDefaults:
    def test_primary_is_21(self):
        assert PRIMARY_HORIZON == 21

    def test_diagnostics_are_a_tuple_not_a_list(self):
        # Immutable so a caller can't mutate the module-level default in place.
        assert isinstance(DIAGNOSTIC_HORIZONS, tuple)
        assert 5 in DIAGNOSTIC_HORIZONS

    def test_default_policy_matches_module_defaults(self):
        assert DEFAULT_POLICY.primary_horizon == PRIMARY_HORIZON
        assert DEFAULT_POLICY.diagnostic_horizons == tuple(sorted(DIAGNOSTIC_HORIZONS))

    def test_all_horizons_primary_first(self):
        assert horizons.all_horizons()[0] == PRIMARY_HORIZON
        assert set(horizons.all_horizons()) == {PRIMARY_HORIZON, *DIAGNOSTIC_HORIZONS}

    def test_canonical_label_is_log_spy_sector(self):
        label = horizons.canonical_label()
        assert (label.domain, label.relative_to, label.neutralization) == (
            "log",
            "spy",
            "sector",
        )


class TestOutcomeColumns:
    def test_naming_matches_score_performance_ground_truth(self):
        # Verified against the live score_performance table (research.db).
        c = outcome_columns(21)
        assert c.price == "price_21d"
        assert c.stock_return == "return_21d"
        assert c.spy_return == "spy_21d_return"
        assert c.beat_spy == "beat_spy_21d"
        assert c.eval_date == "eval_date_21d"
        assert c.log_alpha == "log_alpha_21d"
        assert c.horizon_days == 21

    def test_diagnostic_horizon_columns(self):
        c = outcome_columns(5)
        assert (c.beat_spy, c.stock_return, c.spy_return) == (
            "beat_spy_5d",
            "return_5d",
            "spy_5d_return",
        )

    def test_nonpositive_horizon_raises(self):
        with pytest.raises(ValueError):
            outcome_columns(0)
        with pytest.raises(ValueError):
            outcome_columns(-3)


class TestAbsorbedBacktesterConstants:
    """The embryonic weight_optimizer constants must be reproducible from here."""

    def test_short_and_long_outcome(self):
        # _SHORT_OUTCOME = "beat_spy_5d", _LONG_OUTCOME = "beat_spy_21d"
        assert outcome_columns(5).beat_spy == "beat_spy_5d"
        assert outcome_columns(PRIMARY_HORIZON).beat_spy == "beat_spy_21d"

    def test_resolved_gate_column(self):
        # _RESOLVED_OUTCOME = _LONG_OUTCOME (primary's beat_spy)
        assert DEFAULT_POLICY.resolved_gate_column() == "beat_spy_21d"

    def test_skill_target_map(self):
        # _SKILL_TARGET = {"beat_spy_5d": "return_5d", "beat_spy_21d": "log_alpha_21d"}
        assert DEFAULT_POLICY.skill_target_column(5) == "return_5d"
        assert DEFAULT_POLICY.skill_target_column(21) == "log_alpha_21d"
        assert horizons.skill_target_column(21) == "log_alpha_21d"


class TestPrimaryGuardrails:
    def test_is_primary(self):
        assert horizons.is_primary(21) is True
        assert horizons.is_primary(5) is False

    def test_require_primary_present_passes_when_present(self):
        DEFAULT_POLICY.require_primary_present([5, 21])  # no raise

    def test_require_primary_present_fails_loud_when_absent(self):
        with pytest.raises(PrimaryHorizonMissing):
            DEFAULT_POLICY.require_primary_present([5, 10])

    def test_diagnostic_absence_is_tolerated(self):
        # Only the primary is fail-loud; a produced set with just the primary
        # (no diagnostics) is fine — diagnostics are graceful-empty.
        DEFAULT_POLICY.require_primary_present([21])  # no raise


class TestHorizonPolicy:
    def test_primary_cannot_also_be_diagnostic(self):
        with pytest.raises(ValueError):
            HorizonPolicy(primary_horizon=21, diagnostic_horizons=(5, 21))

    def test_diagnostics_normalized_sorted_deduped(self):
        p = HorizonPolicy(primary_horizon=21, diagnostic_horizons=(10, 5, 5))
        assert p.diagnostic_horizons == (5, 10)

    def test_nonpositive_primary_raises(self):
        with pytest.raises(ValueError):
            HorizonPolicy(primary_horizon=0)

    def test_from_mapping_full(self):
        p = HorizonPolicy.from_mapping(
            {
                "primary_horizon": 21,
                "diagnostic_horizons": [5, 10],
                "label_definition": {
                    "domain": "log",
                    "relative_to": "spy",
                    "neutralization": "sector",
                },
            }
        )
        assert p.primary_horizon == 21
        assert p.diagnostic_horizons == (5, 10)
        assert p.label == LabelDefinition()

    def test_from_mapping_defaults_when_partial(self):
        p = HorizonPolicy.from_mapping({"primary_horizon": 63})
        assert p.primary_horizon == 63
        assert p.diagnostic_horizons == tuple(sorted(DIAGNOSTIC_HORIZONS))
        assert p.label == LabelDefinition()

    def test_from_mapping_unknown_key_raises(self):
        # A typo'd config key that silently no-ops is the failure mode this
        # EPIC exists to kill — so it must raise, not be ignored.
        with pytest.raises(ValueError):
            HorizonPolicy.from_mapping({"primary_horzon": 21})

    def test_from_mapping_unknown_label_key_raises(self):
        with pytest.raises(ValueError):
            HorizonPolicy.from_mapping({"label_definition": {"domain": "log", "typo": 1}})

    def test_round_trips_through_as_dict(self):
        p = HorizonPolicy(primary_horizon=21, diagnostic_horizons=(5, 10))
        assert HorizonPolicy.from_mapping(p.as_dict()) == p

    def test_with_overrides_revalidates(self):
        p = DEFAULT_POLICY.with_overrides(primary_horizon=63)
        assert p.primary_horizon == 63
        # override that violates an invariant still raises
        with pytest.raises(ValueError):
            DEFAULT_POLICY.with_overrides(diagnostic_horizons=(21,))

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_POLICY.primary_horizon = 5  # type: ignore[misc]
