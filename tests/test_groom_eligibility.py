"""Tests for nousergon_lib.groom_eligibility (config#1933)."""

from __future__ import annotations

import pytest

from nousergon_lib.groom_eligibility import (
    BUNDLED_FILTERS,
    SlotDecision,
    TIER_MODELS,
    VALID_ISSUE_FILTERS,
    decide_slot,
    filter_for_tiers,
    filter_tiers,
    is_actionable,
    is_gate_excluded,
    tier_of,
)


class TestTierOf:
    def test_unlabeled_defaults_to_mid(self):
        assert tier_of([]) == "mid"
        assert tier_of(["P2", "area:groom"]) == "mid"

    def test_explicit_tiers(self):
        assert tier_of(["complexity:low"]) == "low"
        assert tier_of(["complexity:high"]) == "high"

    def test_high_wins_over_low_on_conflict(self):
        # Mirrors driver semantics: HIGH checked before LOW.
        assert tier_of(["complexity:high", "complexity:low"]) == "high"

    def test_excluded(self):
        assert tier_of(["complexity:ultra"]) is None
        assert tier_of(["in-progress"]) is None
        assert tier_of(["do-not-groom", "complexity:low"]) is None


class TestGateExclusion:
    def test_hard_gates_always_excluded(self):
        assert is_gate_excluded(["gate:operator"])
        assert is_gate_excluded(["gate:decision", "gate-due"])  # gate-due doesn't lift HARD

    def test_soft_gates_excluded_unless_due(self):
        assert is_gate_excluded(["gate:date"])
        assert not is_gate_excluded(["gate:date", "gate-due"])

    def test_actionable_composes(self):
        assert is_actionable(["complexity:low"]) == "low"
        assert is_actionable(["complexity:low", "gate:operator"]) is None
        assert is_actionable(["complexity:ultra"]) is None


class TestFilterGrammar:
    def test_round_trip_single(self):
        assert filter_for_tiers(["mid"]) == "mid-only"
        assert filter_tiers("mid-only") == ("mid",)

    def test_round_trip_bundles(self):
        for f in BUNDLED_FILTERS:
            assert filter_for_tiers(filter_tiers(f)) == f

    def test_bundle_ordering_highest_first(self):
        assert filter_for_tiers(["low", "high", "mid"]) == "high+mid+low"

    def test_default_alias_and_reverify(self):
        assert filter_tiers("default") == ("mid",)
        assert filter_tiers("gated-reverify") == ()

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError):
            filter_tiers("nope")

    def test_valid_set_contents(self):
        assert "gated-reverify" in VALID_ISSUE_FILTERS  # the PR683 drift lesson
        assert "high+mid+low" in VALID_ISSUE_FILTERS


class TestDecideSlot:
    def test_all_tiers_above_floor_each_slot_runs_own_tier(self):
        # Brian's 8/9/10 example: every slot launches, each works ONLY its
        # own tier (no lower tier is below floor, so nothing bundles).
        counts = {"low": 8, "mid": 9, "high": 10}
        for slot, expected_filter, expected_model in [
            ("low", "low-only", TIER_MODELS["low"]),
            ("mid", "mid-only", TIER_MODELS["mid"]),
            ("high", "high-only", TIER_MODELS["high"]),
        ]:
            d = decide_slot(slot, counts)
            assert d.launch and d.issue_filter == expected_filter
            assert d.model == expected_model

    def test_starving_low_bundles_into_mid_slot(self):
        # Brian's example: low=6 (< floor) rides the mid slot.
        d = decide_slot("mid", {"low": 6, "mid": 9, "high": 0})
        assert d.launch
        assert d.issue_filter == "mid+low"
        assert d.model == TIER_MODELS["mid"]

    def test_thin_everything_bundles_at_high_slot_on_cheapest_adequate_model(self):
        # 1 low + 3 mid + 0 high at the Opus slot: queue is 4 < floor -> skip
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 0})
        assert not d.launch
        # ...but with 5 high it launches, and the model is Opus (high present)
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 5})
        assert d.launch and d.issue_filter == "high+mid+low"
        assert d.model == TIER_MODELS["high"]

    def test_model_is_highest_present_not_slot(self):
        # Opus slot, no high issues, bundle of starving low+mid -> Sonnet.
        d = decide_slot("high", {"low": 5, "mid": 6, "high": 0})
        assert d.launch  # 11 >= floor
        assert d.issue_filter == "mid+low"
        assert d.model == TIER_MODELS["mid"]  # never Opus without high issues

    def test_light_queue_skips_with_zero_spend(self):
        d = decide_slot("low", {"low": 6, "mid": 40, "high": 3})
        assert not d.launch
        assert "deferred upward" in d.reason

    def test_p0_escape_valve(self):
        d = decide_slot("low", {"low": 2, "mid": 0, "high": 0}, has_actionable_p0=True)
        assert d.launch and "P0" in d.reason

    def test_age_escape_valve(self):
        d = decide_slot("mid", {"low": 0, "mid": 3, "high": 0},
                        oldest_wait_hours={"mid": 80.0})
        assert d.launch and "waited" in d.reason
        # under the threshold -> still skips
        d = decide_slot("mid", {"low": 0, "mid": 3, "high": 0},
                        oldest_wait_hours={"mid": 24.0})
        assert not d.launch

    def test_higher_tiers_never_bundle_down(self):
        # Low slot with a starving high queue: high must NOT ride Haiku.
        d = decide_slot("low", {"low": 9, "mid": 0, "high": 3})
        assert d.launch and d.issue_filter == "low-only"
        assert d.model == TIER_MODELS["low"]

    def test_empty_queue(self):
        d = decide_slot("high", {"low": 0, "mid": 0, "high": 0})
        assert not d.launch and "empty" in d.reason

    def test_record_shape(self):
        rec = decide_slot("mid", {"low": 0, "mid": 9, "high": 0}).as_record()
        assert set(rec) == {"launch", "tiers", "issue_filter", "model", "reason"}


class TestDecideTrigger:
    def _launches(self, counts, **kw):
        from nousergon_lib.groom_eligibility import decide_trigger
        return decide_trigger(counts, **kw)

    def test_brians_8_9_10_all_three_spin_up_same_trigger(self):
        ls = [l for l in self._launches({"low": 8, "mid": 9, "high": 10}) if l.launch]
        assert [(l.issue_filter, l.model) for l in sorted(ls, key=lambda x: x.issue_filter)] == [
            ("high-only", "claude-opus-4-8"),
            ("low-only", "claude-haiku-4-5"),
            ("mid-only", "claude-sonnet-5"),
        ]

    def test_thin_low_attaches_to_nearest_standalone_above(self):
        ls = [l for l in self._launches({"low": 6, "mid": 9, "high": 10}) if l.launch]
        filters = {l.issue_filter for l in ls}
        assert filters == {"mid+low", "high-only"}  # low rides mid, not high

    def test_leftover_thin_pool_launches_at_highest_model_when_over_floor(self):
        ls = [l for l in self._launches({"low": 4, "mid": 5, "high": 2}) if l.launch]
        assert len(ls) == 1
        assert ls[0].issue_filter == "high+mid+low"
        assert ls[0].model == "claude-opus-4-8"  # high present in pool

    def test_thin_pool_under_floor_skips_with_reason(self):
        ls = self._launches({"low": 1, "mid": 2, "high": 1})
        assert len(ls) == 1 and not ls[0].launch
        assert "deferred" in ls[0].reason

    def test_thin_pool_p0_valve(self):
        ls = [l for l in self._launches({"low": 1, "mid": 2, "high": 0}, p0_tiers=["mid"]) if l.launch]
        assert len(ls) == 1 and ls[0].model == "claude-sonnet-5"  # no high -> Sonnet

    def test_thin_pool_age_valve(self):
        ls = [l for l in self._launches({"low": 3, "mid": 0, "high": 0},
                                        oldest_wait_hours={"low": 96.0}) if l.launch]
        assert len(ls) == 1 and ls[0].model == "claude-haiku-4-5"

    def test_high_never_rides_below_opus(self):
        # standalone low, thin high: high must NOT attach downward.
        ls = [l for l in self._launches({"low": 9, "mid": 0, "high": 2}) if l.launch]
        by_filter = {l.issue_filter: l for l in ls}
        assert "low-only" in by_filter
        assert all("high" not in f or l.model == "claude-opus-4-8" for f, l in by_filter.items())

    def test_empty_backlog_no_launches(self):
        assert all(not l.launch for l in self._launches({"low": 0, "mid": 0, "high": 0}))


class TestFreshSkip:
    def test_recent_engagement_no_activity_skips(self):
        from nousergon_lib.groom_eligibility import fresh_skip_active
        now = 1_000_000.0
        assert fresh_skip_active(now - 3600, now - 3600, now)

    def test_new_activity_readmits(self):
        from nousergon_lib.groom_eligibility import fresh_skip_active
        now = 1_000_000.0
        assert not fresh_skip_active(now - 3600, now - 60, now)

    def test_old_engagement_expires(self):
        from nousergon_lib.groom_eligibility import fresh_skip_active
        now = 1_000_000.0
        assert not fresh_skip_active(now - 80 * 3600, now - 80 * 3600, now)


class TestFreshSkipConstantsContract:
    """config#2038: these three constants are the SSoT both groom consumers
    (groom_driver.py on-box, contract-tested against this module; the
    scheduled-groom-dispatcher Lambda, imported directly) must use — pins the
    values so a future edit here can't silently re-drift one consumer from
    the other the way FRESH_SKIP_SLACK_SEC (900 vs the driver's 1800) and the
    3-vs-4-day lookback did."""

    def test_slack_matches_driver_value(self):
        from nousergon_lib.groom_eligibility import FRESH_SKIP_SLACK_SEC
        assert FRESH_SKIP_SLACK_SEC == 1800.0

    def test_lookback_days_covers_the_72h_window(self):
        from nousergon_lib.groom_eligibility import (
            ENGAGEMENT_LOOKBACK_DAYS,
            FRESH_SKIP_HOURS,
        )
        # A run starting just before UTC midnight (FRESH_SKIP_HOURS/24) days
        # ago must still fall inside the scanned date-bucket range.
        assert ENGAGEMENT_LOOKBACK_DAYS >= (FRESH_SKIP_HOURS / 24.0) + 1

    def test_engaged_dispositions_matches_driver_value(self):
        from nousergon_lib.groom_eligibility import ENGAGED_DISPOSITIONS
        assert ENGAGED_DISPOSITIONS == ("closed", "pr_opened", "commented", "labeled")
