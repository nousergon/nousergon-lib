"""A slot declares WHICH statistic the pointer ranks on
(``ArenaConfig.promote_statistic``, alpha-engine-config-I11403).

Before this the engine ranked on ``window.mean_diff``, unconditionally. That is
correct for a slot whose arms are count-matched and WRONG for one whose arms
carry different widths ON PURPOSE: mean alpha per name declines with depth
whenever a ranking carries any signal, so a raw mean hands the narrowest arm a
win it did not earn. The information ratio prices that concentration
(IR ~= IC x sqrt(breadth)).

crucible-research's `research` slot is the first consumer: it runs
`attractiveness_60` against `attractiveness_20` on the SAME ranking precisely to
measure what depth costs, which only means anything if the comparison is not
already decided by breadth.

WHY IR IS CONFINED TO ``promote_evidence="point"``. The anytime-valid sequence
is a bound on the MEAN of bounded per-date differences
(champion-challenger-policy.md §5.0). It says nothing about a difference of two
ratios, and applying it there would report a coverage guarantee the
construction does not have. The config REFUSES the combination rather than
quietly producing a bound nobody should trust; the sequence is still computed
and emitted on ``mean_diff`` for every comparison, so the evidence it would
have required stays on the record.
"""

from __future__ import annotations

import pytest

from nousergon_lib.arena import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    STATISTIC_INFORMATION_RATIO,
    STATISTIC_MEAN_DIFF,
    ArenaConfig,
    PairedWindow,
    promotion_statistic,
)
from nousergon_lib.arena.engine import ArenaConfigError


def _config(**kw) -> ArenaConfig:
    base = {
        "slot": "research",
        "slot_kind": "producer",
        "benchmark": "population",
        "promote_evidence": EVIDENCE_POINT,
    }
    base.update(kw)
    return ArenaConfig(**base)


def _window(scores_a, scores_b) -> PairedWindow:
    dates = tuple(f"2026-09-{i:02d}" for i in range(1, len(scores_a) + 1))
    return PairedWindow(
        arm_a="a",
        arm_b="b",
        dates=dates,
        diffs=tuple(x - y for x, y in zip(scores_a, scores_b)),
        scores_a=tuple(scores_a),
        scores_b=tuple(scores_b),
    )


class TestTheDefaultIsUnchanged:
    def test_a_config_that_names_nothing_ranks_on_the_mean_difference(self):
        """Every slot's behaviour before this field existed. A default that
        moved would silently re-decide four live slots."""
        cfg = _config()
        assert cfg.promote_statistic == STATISTIC_MEAN_DIFF
        w = _window([0.02, 0.03, 0.01], [0.01, 0.01, 0.01])
        assert promotion_statistic(cfg, w) == pytest.approx(w.mean_diff)


class TestTheInformationRatio:
    def test_it_is_each_arms_own_ratio_not_the_ratio_of_the_difference(self):
        """A difference of ratios, not a ratio of differences. The second would
        be a different quantity and would not price concentration at all."""
        w = _window([0.02, 0.04, 0.06], [0.01, 0.01, 0.01])
        assert w.information_ratio_a == pytest.approx(0.04 / 0.02)
        # b is constant: no dispersion, so no ratio.
        assert w.information_ratio_b is None

    def test_a_narrower_arm_does_not_win_on_a_bigger_mean_alone(self):
        """The whole reason the field exists. `a` has the larger mean and the
        far larger dispersion; on the mean it wins, on the ratio it loses."""
        wide = [0.010, 0.011, 0.009, 0.010]
        narrow = [0.035, -0.020, 0.045, -0.010]
        w = _window(narrow, wide)
        assert w.mean_diff > 0, "the narrow arm leads on the raw mean"
        assert w.ir_diff < 0, "and loses once its dispersion is priced"
        assert promotion_statistic(_config(), w) > 0
        assert promotion_statistic(
            _config(promote_statistic=STATISTIC_INFORMATION_RATIO), w
        ) < 0

    def test_one_date_is_not_estimable_and_is_not_a_loss(self):
        """An arm whose ratio cannot be formed has not lost the comparison, it
        has not been in one. None, never a signed default."""
        w = _window([0.02], [0.01])
        assert w.information_ratio_a is None
        assert w.ir_diff is None
        assert promotion_statistic(
            _config(promote_statistic=STATISTIC_INFORMATION_RATIO), w
        ) is None

    def test_zero_dispersion_is_none_not_an_infinity(self):
        """A constant series has no risk to divide by. Returning an infinity
        would rank it above every arm that took real risk."""
        w = _window([0.01, 0.01, 0.01], [0.02, 0.01, 0.03])
        assert w.information_ratio_a is None
        assert w.ir_diff is None

    def test_it_is_not_annualised(self):
        """Annualising assumes independent periods, and a slot's cohort dates
        routinely overlap — scaling by sqrt(periods) would report a ratio
        inflated by the overlap rather than by skill."""
        w = _window([0.02, 0.04, 0.06], [0.0, 0.0, 0.0])
        # mean 0.04, sample sd 0.02 -> exactly 2.0, with no sqrt(252) anywhere.
        assert w.information_ratio_a == pytest.approx(2.0)

    def test_both_statistics_are_emitted_on_every_window(self):
        """The statistic that did NOT decide stays on the record beside the one
        that did — §11: the artifact must explain the decision without a reader
        re-deriving the alternative."""
        d = _window([0.02, 0.04, 0.06], [0.01, 0.02, 0.01]).to_dict()
        assert d["mean_diff"] is not None
        assert d["information_ratio_a"] is not None
        assert d["ir_diff"] is not None


class TestTheConfigRefusesTheUnsoundCombination:
    def test_information_ratio_with_the_anytime_valid_sequence_is_refused(self):
        with pytest.raises(ArenaConfigError, match="difference of two ratios"):
            _config(
                promote_statistic=STATISTIC_INFORMATION_RATIO,
                promote_evidence=EVIDENCE_ANYTIME_VALID,
            )

    def test_an_unknown_statistic_is_refused(self):
        with pytest.raises(ArenaConfigError, match="promote_statistic must be"):
            _config(promote_statistic="sharpe")

    def test_information_ratio_with_point_evidence_is_accepted(self):
        cfg = _config(promote_statistic=STATISTIC_INFORMATION_RATIO)
        assert cfg.promote_statistic == STATISTIC_INFORMATION_RATIO


class TestItSurvivesTheRoundTrip:
    def test_the_statistic_is_recorded_on_the_cycle_config(self):
        """A reader reconstructing why a pointer moved needs the statistic that
        was in force, not today's."""
        cfg = _config(promote_statistic=STATISTIC_INFORMATION_RATIO)
        assert cfg.to_dict()["promote_statistic"] == STATISTIC_INFORMATION_RATIO

    def test_a_cycle_recorded_before_this_field_reconstructs_as_mean_diff(self):
        """Back-compat, stated: a cycle written before the field existed was
        decided on the mean paired difference, and reconstructing it as
        anything else would misreport why that pointer moved."""
        data = dict(_config().to_dict())
        del data["promote_statistic"]
        back = ArenaConfig.from_dict(
            data, slot="research", slot_kind="producer", benchmark="population"
        )
        assert back.promote_statistic == STATISTIC_MEAN_DIFF

    def test_the_round_trip_preserves_a_declared_statistic(self):
        cfg = _config(promote_statistic=STATISTIC_INFORMATION_RATIO)
        back = ArenaConfig.from_dict(
            cfg.to_dict(), slot="research", slot_kind="producer", benchmark="population"
        )
        assert back == cfg
