"""Tests for nousergon_lib.groom_eligibility (config#1933)."""

from __future__ import annotations

import pytest

from nousergon_lib.groom_eligibility import (
    BUNDLED_FILTERS,
    CI_EXPECTED_RED_LABEL,
    GATE_SOFT_EXCLUDE_LABELS,
    TIER_MODELS,
    VALID_ISSUE_FILTERS,
    decide_slot,
    expected_red_labels_for_checks,
    filter_for_tiers,
    filter_tiers,
    is_actionable,
    is_gate_excluded,
    RULING_PENDING_LABEL,
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

    def test_flap_breaker_stall_labels_excluded(self):
        # config#2146 / alpha-engine-config#688 (2026-07-11): a flap-broken
        # issue routed to the human Decision Queue must never re-enter
        # machine grooming, even after a gate-due re-admission.
        assert tier_of(["groom:stalled"]) is None
        assert tier_of(["triage:session"]) is None
        assert tier_of(["groom:stalled", "triage:session", "gate:weekly-sf", "gate-due"]) is None


class TestGateExclusion:
    def test_hard_gates_always_excluded(self):
        assert is_gate_excluded(["gate:operator"])
        assert is_gate_excluded(["gate:decision", "gate-due"])  # gate-due doesn't lift HARD

    def test_soft_gates_excluded_unless_due(self):
        assert is_gate_excluded(["gate:date"])
        assert not is_gate_excluded(["gate:date", "gate-due"])

    def test_sf_gates_are_soft_excluded_unless_due(self):
        # gate:live-run split by named pipeline (config#2057, 2026-07-09) —
        # each behaves exactly like gate:date: soft-excluded unless gate-due.
        for label in ("gate:weekly-sf", "gate:preopen-sf", "gate:postclose-sf"):
            assert label in GATE_SOFT_EXCLUDE_LABELS
            assert is_gate_excluded([label])
            assert not is_gate_excluded([label, "gate-due"])
        assert "gate:live-run" not in GATE_SOFT_EXCLUDE_LABELS

    def test_actionable_composes(self):
        assert is_actionable(["complexity:low"]) == "low"
        assert is_actionable(["complexity:low", "gate:operator"]) is None
        assert is_actionable(["complexity:ultra"]) is None

    def test_ruling_pending_lifts_soft_exclusion_not_hard(self):
        # config#3199: an operator ruling awaiting execution overrides the
        # SOFT gate exclusion — executing the ruling is what resolves the
        # remaining gate label — but never a HARD exclude (a re-escalated
        # gate:decision item is human-owned again, marker or not).
        assert RULING_PENDING_LABEL == "ruling:pending-exec"
        assert not is_gate_excluded(["gate:weekly-sf", RULING_PENDING_LABEL])
        assert not is_gate_excluded(["gate:date", RULING_PENDING_LABEL])
        assert is_gate_excluded(["gate:decision", RULING_PENDING_LABEL])
        assert is_gate_excluded(["gate:operator", RULING_PENDING_LABEL])
        assert is_actionable(["complexity:low", "gate:weekly-sf",
                              RULING_PENDING_LABEL]) == "low"

    def test_milestone_gate_is_soft_excluded_unless_due(self):
        # config#2519: event-driven gate — never gets gate-due in practice
        # (gate_milestone_sweep.py auto-clears directly), but the SOFT
        # exclusion semantics (excluded unless gate-due) still apply for
        # consistency with the other auto-clearing gate classes.
        assert "gate:milestone" in GATE_SOFT_EXCLUDE_LABELS
        assert is_gate_excluded(["gate:milestone"])
        assert not is_gate_excluded(["gate:milestone", "gate-due"])


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
        # 1 low + 3 mid + 0 high at the high slot: queue is 4 < floor -> skip
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 0})
        assert not d.launch
        # ...but with 5 high it launches, using the high tier's own model
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 5})
        assert d.launch and d.issue_filter == "high+mid+low"
        assert d.model == TIER_MODELS["high"]

    def test_model_is_highest_present_not_slot(self):
        # high slot, no high issues, bundle of starving low+mid -> mid's model.
        d = decide_slot("high", {"low": 5, "mid": 6, "high": 0})
        assert d.launch  # 11 >= floor
        assert d.issue_filter == "mid+low"
        assert d.model == TIER_MODELS["mid"]  # never the high tier's model without high issues

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
        assert set(rec) == {
            "launch", "tiers", "issue_filter", "model", "reason",
        }


class TestDecideTrigger:
    def _launches(self, counts, **kw):
        from nousergon_lib.groom_eligibility import decide_trigger
        return decide_trigger(counts, **kw)

    def test_brians_8_9_10_all_three_spin_up_same_trigger(self):
        decisions = [d for d in self._launches({"low": 8, "mid": 9, "high": 10}) if d.launch]
        assert [(d.issue_filter, d.model) for d in sorted(decisions, key=lambda x: x.issue_filter)] == [
            ("high-only", "claude-sonnet-5"),
            ("low-only", "claude-haiku-4-5"),
            ("mid-only", "claude-sonnet-5"),
        ]

    def test_thin_low_attaches_to_nearest_standalone_above(self):
        decisions = [d for d in self._launches({"low": 6, "mid": 9, "high": 10}) if d.launch]
        filters = {d.issue_filter for d in decisions}
        assert filters == {"mid+low", "high-only"}  # low rides mid, not high

    def test_leftover_thin_pool_launches_at_highest_model_when_over_floor(self):
        decisions = [d for d in self._launches({"low": 4, "mid": 5, "high": 2}) if d.launch]
        assert len(decisions) == 1
        assert decisions[0].issue_filter == "high+mid+low"
        assert decisions[0].model == TIER_MODELS["high"]  # high present in pool

    def test_thin_pool_under_floor_skips_with_reason(self):
        decisions = self._launches({"low": 1, "mid": 2, "high": 1})
        assert len(decisions) == 1 and not decisions[0].launch
        assert "deferred" in decisions[0].reason

    def test_thin_pool_p0_valve(self):
        decisions = [d for d in self._launches({"low": 1, "mid": 2, "high": 0}, p0_tiers=["mid"]) if d.launch]
        assert len(decisions) == 1 and decisions[0].model == "claude-sonnet-5"  # no high -> Sonnet

    def test_thin_pool_age_valve(self):
        decisions = [d for d in self._launches({"low": 3, "mid": 0, "high": 0},
                                        oldest_wait_hours={"low": 96.0}) if d.launch]
        assert len(decisions) == 1 and decisions[0].model == "claude-haiku-4-5"

    def test_high_never_rides_below_its_own_model(self):
        # standalone low, thin high: high must NOT attach downward, and any
        # launch containing high uses high's own model (Sonnet, same as mid,
        # as of config#2409 — the guardrail is about tier isolation, not a
        # distinct model anymore).
        decisions = [d for d in self._launches({"low": 9, "mid": 0, "high": 2}) if d.launch]
        by_filter = {d.issue_filter: d for d in decisions}
        assert "low-only" in by_filter
        assert all("high" not in f or d.model == TIER_MODELS["high"] for f, d in by_filter.items())

    def test_empty_backlog_no_launches(self):
        assert all(not d.launch for d in self._launches({"low": 0, "mid": 0, "high": 0}))

    def test_solo_launch_when_only_high_clears_floor(self):
        # Only high clears the floor -> a single launch.
        decisions = [d for d in self._launches({"low": 0, "mid": 0, "high": 10}) if d.launch]
        assert len(decisions) == 1
        assert decisions[0].issue_filter == "high-only"

    def test_skip_decisions_emitted_alongside_a_launch(self):
        # One standalone launch (low) plus a thin `high` with no standalone
        # tier above it (high is the top tier) that falls to the leftover
        # pool and doesn't clear the floor there either -> skip. Both a
        # launching and a skipped decision are returned.
        decisions = self._launches({"low": 10, "mid": 0, "high": 3})
        launching = [d for d in decisions if d.launch]
        skipped = [d for d in decisions if not d.launch]
        assert len(launching) == 1 and len(skipped) == 1
        assert launching[0].issue_filter == "low-only"

    def test_launch_emit_order_is_high_first(self):
        # decide_trigger's own pool ordering sorts high-first; assert the
        # emitted launch order doesn't silently reshuffle.
        decisions = [d for d in self._launches({"low": 8, "mid": 9, "high": 10}) if d.launch]
        assert [d.issue_filter for d in decisions] == ["high-only", "mid-only", "low-only"]


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


class TestCiExpectedRed:
    def test_expected_check_returns_label(self):
        result = expected_red_labels_for_checks(["iam-drift"])
        assert result == [CI_EXPECTED_RED_LABEL]

    def test_unknown_check_returns_empty(self):
        result = expected_red_labels_for_checks(["iam-drift", "ci.yml"])
        assert result == []

    def test_empty_input_returns_empty(self):
        result = expected_red_labels_for_checks([])
        assert result == []

    def test_label_outside_gate_namespace(self):
        assert not CI_EXPECTED_RED_LABEL.startswith("gate:")
