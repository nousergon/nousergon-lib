"""Tests for nousergon_lib.groom_eligibility (config#1933)."""

from __future__ import annotations

import pytest
from krepis.router import TIER_GROUPS, group_for_tier

from nousergon_lib import groom_eligibility
from nousergon_lib.groom_eligibility import (
    BUNDLED_FILTERS,
    CI_EXPECTED_RED_LABEL,
    GATE_SOFT_EXCLUDE_LABELS,
    RULING_PENDING_LABEL,
    TIERS,
    VALID_ISSUE_FILTERS,
    comment_only_strikes_exceeded,
    decide_slot,
    expected_red_labels_for_checks,
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


class TestTierModelTablesRemoved:
    """alpha-engine-config-I9297 (Brian ruling 2026-08-29): the module used
    to hardcode its own tier -> vendor model id tables (``TIER_MODELS``,
    ``FALLBACK_TIER_MODELS``) — a second routing plane. Both are gone; a
    decided run now carries a krepis registry GROUP, resolved via
    ``krepis.router.group_for_tier`` (the ONE tier->group mapping,
    fleet-wide), never a model id."""

    def test_tier_models_no_longer_exported(self):
        assert not hasattr(groom_eligibility, "TIER_MODELS")

    def test_fallback_tier_models_no_longer_exported(self):
        assert not hasattr(groom_eligibility, "FALLBACK_TIER_MODELS")

    def test_fallback_model_config_no_longer_exported(self):
        assert not hasattr(groom_eligibility, "FallbackModelConfig")

    def test_group_for_tier_covers_every_declared_tier(self):
        assert set(TIER_GROUPS) == set(TIERS)

    def test_group_for_tier_returns_a_registry_group_not_a_model_id(self):
        for tier in TIERS:
            group = group_for_tier(tier)
            assert group in {"low", "med", "high", "ultra"}
            assert "deepseek" not in group and "claude" not in group

    def test_group_for_tier_refuses_an_unknown_tier(self):
        # Fail closed (model-router-policy R20): no silent default-to-mid.
        with pytest.raises(ValueError):
            group_for_tier("nope")


class TestDecideSlot:
    def test_all_tiers_above_floor_each_slot_runs_own_tier(self):
        # Brian's 8/9/10 example: every slot launches, each works ONLY its
        # own tier (no lower tier is below floor, so nothing bundles).
        counts = {"low": 8, "mid": 9, "high": 10}
        for slot, expected_filter, expected_tier in [
            ("low", "low-only", "low"),
            ("mid", "mid-only", "mid"),
            ("high", "high-only", "high"),
        ]:
            d = decide_slot(slot, counts)
            assert d.launch and d.issue_filter == expected_filter
            assert d.model_group == group_for_tier(expected_tier)

    def test_starving_low_bundles_into_mid_slot(self):
        # Brian's example: low=6 (< floor) rides the mid slot.
        d = decide_slot("mid", {"low": 6, "mid": 9, "high": 0})
        assert d.launch
        assert d.issue_filter == "mid+low"
        assert d.model_group == group_for_tier("mid")

    def test_thin_everything_bundles_at_high_slot_on_cheapest_adequate_model(self):
        # 1 low + 3 mid + 0 high at the high slot: queue is 4 < floor -> skip
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 0})
        assert not d.launch
        # ...but with 5 high it launches, using the high tier's own model
        d = decide_slot("high", {"low": 1, "mid": 3, "high": 5})
        assert d.launch and d.issue_filter == "high+mid+low"
        assert d.model_group == group_for_tier("high")

    def test_model_is_highest_present_not_slot(self):
        # high slot, no high issues, bundle of starving low+mid -> mid's model.
        d = decide_slot("high", {"low": 5, "mid": 6, "high": 0})
        assert d.launch  # 11 >= floor
        assert d.issue_filter == "mid+low"
        assert d.model_group == group_for_tier("mid")  # never the high tier's group without high issues

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
        assert d.model_group == group_for_tier("low")

    def test_empty_queue(self):
        d = decide_slot("high", {"low": 0, "mid": 0, "high": 0})
        assert not d.launch and "empty" in d.reason

    def test_record_shape(self):
        rec = decide_slot("mid", {"low": 0, "mid": 9, "high": 0}).as_record()
        assert set(rec) == {
            "launch", "tiers", "issue_filter", "model_group", "reason",
        }


class TestDecideTrigger:
    def _launches(self, counts, **kw):
        from nousergon_lib.groom_eligibility import decide_trigger
        return decide_trigger(counts, **kw)

    def test_brians_8_9_10_all_three_spin_up_same_trigger(self):
        decisions = [d for d in self._launches({"low": 8, "mid": 9, "high": 10}) if d.launch]
        assert [(d.issue_filter, d.model_group)
                for d in sorted(decisions, key=lambda x: x.issue_filter)] == [
            ("high-only", group_for_tier("high")),
            ("low-only", group_for_tier("low")),
            ("mid-only", group_for_tier("mid")),
        ]

    def test_thin_tier_launches_independently_instead_of_attaching(self):
        decisions = [d for d in self._launches({"low": 6, "mid": 9, "high": 10}) if d.launch]
        filters = {d.issue_filter for d in decisions}
        assert filters == {"low-only", "mid-only", "high-only"}

    def test_all_tiers_launch_independently(self):
        decisions = [d for d in self._launches({"low": 4, "mid": 5, "high": 2}) if d.launch]
        assert len(decisions) == 3
        assert {d.issue_filter for d in decisions} == {"low-only", "mid-only", "high-only"}

    def test_no_skip_for_thin_tiers(self):
        decisions = self._launches({"low": 1, "mid": 2, "high": 1})
        assert all(d.launch for d in decisions)
        assert len(decisions) == 3
        assert {d.issue_filter for d in decisions} == {"low-only", "mid-only", "high-only"}

    def test_each_tier_gets_own_model_group(self):
        decisions = [d for d in self._launches({"low": 1, "mid": 2, "high": 0}) if d.launch]
        by_filter = {d.issue_filter: d for d in decisions}
        assert by_filter["low-only"].model_group == group_for_tier("low")
        assert by_filter["mid-only"].model_group == group_for_tier("mid")

    def test_high_launches_independently_its_own_model_group(self):
        decisions = [d for d in self._launches({"low": 9, "mid": 0, "high": 2}) if d.launch]
        by_filter = {d.issue_filter: d for d in decisions}
        assert "low-only" in by_filter
        assert "high-only" in by_filter
        assert by_filter["high-only"].model_group == group_for_tier("high")

    def test_empty_backlog_no_launches(self):
        assert all(not d.launch for d in self._launches({"low": 0, "mid": 0, "high": 0}))

    def test_solo_high_launches_when_count_positive(self):
        decisions = [d for d in self._launches({"low": 0, "mid": 0, "high": 10}) if d.launch]
        assert len(decisions) == 1
        assert decisions[0].issue_filter == "high-only"

    def test_skip_zero_count_tiers(self):
        # Only tiers with count > 0 launch; zero-count tiers are absent.
        decisions = self._launches({"low": 10, "mid": 0, "high": 3})
        assert len(decisions) == 2
        assert {d.issue_filter for d in decisions if d.launch} == {"low-only", "high-only"}

    def test_launch_emit_order_is_high_first(self):
        decisions = [d for d in self._launches({"low": 8, "mid": 9, "high": 10}) if d.launch]
        assert [d.issue_filter for d in decisions] == ["high-only", "mid-only", "low-only"]


class TestFreshSkipRetired:
    """config#2146 (Brian ruling 2026-07-10): the 72h fresh_skip_active
    time-window cooldown is retired — eligibility is disposition-structural
    now (BASE_EXCLUDE_LABELS / is_gate_excluded), never a wall-clock skip."""

    def test_fresh_skip_active_no_longer_exported(self):
        import nousergon_lib.groom_eligibility as ge
        assert not hasattr(ge, "fresh_skip_active")

    def test_fresh_skip_hours_no_longer_exported(self):
        import nousergon_lib.groom_eligibility as ge
        assert not hasattr(ge, "FRESH_SKIP_HOURS")


class TestCommentOnlyStrikes:
    """config#2146 deliverable 2: 2 CONSECUTIVE comment-only engagements with
    no intervening state change routes an issue to the human Decision Queue
    (groom:stalled + triage:session) instead of a time-based cooldown."""

    def test_no_history_is_zero_strikes(self):
        assert not comment_only_strikes_exceeded([])

    def test_single_comment_only_is_not_a_strike_out(self):
        assert not comment_only_strikes_exceeded(["commented"])

    def test_two_consecutive_comment_only_exceeds(self):
        assert comment_only_strikes_exceeded(["commented", "commented"])

    def test_state_change_resets_the_streak(self):
        # Most-recent-first: labeled (a real state change) intervened between
        # the two comment-only passes, so this is NOT 2 consecutive strikes.
        assert not comment_only_strikes_exceeded(["commented", "labeled", "commented"])

    def test_closed_or_pr_opened_also_resets(self):
        assert not comment_only_strikes_exceeded(["commented", "pr_opened"])
        assert not comment_only_strikes_exceeded(["commented", "closed"])

    def test_three_consecutive_still_exceeds(self):
        assert comment_only_strikes_exceeded(["commented", "commented", "commented"])

    def test_limit_is_configurable(self):
        assert not comment_only_strikes_exceeded(["commented", "commented"], limit=3)
        assert comment_only_strikes_exceeded(
            ["commented", "commented", "commented"], limit=3)


class TestFreshSkipConstantsContract:
    """config#2038: these constants are the SSoT both groom consumers
    (groom_driver.py on-box, contract-tested against this module; the
    scheduled-groom-dispatcher Lambda, imported directly) must use — pins the
    values so a future edit here can't silently re-drift one consumer from
    the other the way FRESH_SKIP_SLACK_SEC (900 vs the driver's 1800) and the
    3-vs-4-day lookback did. Retained post-config#2146: these now back the
    comment-only-strike scan instead of the retired fresh-skip cooldown."""

    def test_slack_matches_driver_value(self):
        from nousergon_lib.groom_eligibility import FRESH_SKIP_SLACK_SEC
        assert FRESH_SKIP_SLACK_SEC == 1800.0

    def test_engaged_dispositions_matches_driver_value(self):
        from nousergon_lib.groom_eligibility import ENGAGED_DISPOSITIONS
        assert ENGAGED_DISPOSITIONS == ("closed", "pr_opened", "commented", "labeled")

    def test_strike_lookback_covers_the_strike_limit_generously(self):
        from nousergon_lib.groom_eligibility import (
            COMMENT_ONLY_STRIKE_LOOKBACK_DAYS,
            ENGAGEMENT_LOOKBACK_DAYS,
        )
        # The strike scan's own window must be at least as wide as the
        # general engagement lookback (a P2/P3 issue isn't re-groomed daily,
        # so 2 strikes can legitimately span more calendar time).
        assert COMMENT_ONLY_STRIKE_LOOKBACK_DAYS >= ENGAGEMENT_LOOKBACK_DAYS


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


# -- incident records are never work items (nous-ergon-ops-I127 / config-I5222) --


def test_incident_label_is_excluded_from_every_tier():
    """An incident record is a record, not a work item."""
    assert groom_eligibility.tier_of({"incident"}) is None


def test_incident_beats_an_explicit_complexity_label():
    """The realistic breach: a record filed WITH a complexity label, or without
    one at all — unlabelled defaults to "mid", so silence is not protection."""
    for extra in ("complexity:low", "complexity:mid", "complexity:high"):
        assert groom_eligibility.tier_of({"incident", extra}) is None, extra


def test_incident_beats_a_priority_label():
    assert groom_eligibility.tier_of({"incident", "P0"}) is None
    assert groom_eligibility.tier_of({"incident", "P1", "sev1"}) is None


def test_a_severity_label_alone_does_not_exclude():
    """Only `incident` excludes. A sev label on an ordinary issue is not a
    reason to hide it from the groom — the record label is what carries meaning."""
    assert groom_eligibility.tier_of({"sev3"}) == "mid"


def test_incident_label_is_in_the_documented_exclusion_set():
    assert groom_eligibility.INCIDENT_LABEL in groom_eligibility.BASE_EXCLUDE_LABELS
