"""GuardStaging — observe-then-enforce (sf-pipeline-policy.md §7a).

The rule exists because of the 2026-08-17 EOD gap: a correct zero-variance
guard merged at 18:59 UTC, ran for the first time anywhere on the 20:00 UTC
production EOD, and cost the day its ArcticDB append and its reconcile. Not
because the predicate was wrong — because its blast radius had never been
observed.

These tests pin the three obligations of §7a, and in particular the two that
are easy to lose: the promotion criterion (a guard parked in observe mode
forever is the same defect one direction over) and the loudness (a verdict
nobody reads is a suppression, not an observation).
"""

from __future__ import annotations

import logging

import pytest

from nousergon_lib.guard_mode import GuardMode, GuardStaging


def _staging(mode: GuardMode = GuardMode.OBSERVE, **over) -> GuardStaging:
    kwargs = dict(
        name="feature_store_zero_variance",
        mode=mode,
        promotion_criterion="enforce after 10 clean daily runs",
        tracked_issue="alpha-engine-config-I7539",
    )
    kwargs.update(over)
    return GuardStaging(**kwargs)


class TestObserveDoesNotHalt:
    def test_a_failing_verdict_does_not_raise(self, caplog):
        with caplog.at_level(logging.ERROR):
            _staging().verdict(logging.getLogger("t"), "8 constant columns")
        assert "8 constant columns" in caplog.text

    def test_enforce_raises_the_same_verdict(self):
        with pytest.raises(RuntimeError, match="8 constant columns"):
            _staging(GuardMode.ENFORCE).verdict(logging.getLogger("t"), "8 constant columns")

    def test_the_exception_type_is_the_caller_s(self):
        class FeatureStoreError(RuntimeError):
            pass

        with pytest.raises(FeatureStoreError):
            _staging(GuardMode.ENFORCE).verdict(
                logging.getLogger("t"), "boom", raise_with=FeatureStoreError
            )

    def test_the_raised_message_carries_the_staging(self):
        """So a production traceback says which guard, in which mode, with what
        promotion criterion — without anyone going to look it up."""
        with pytest.raises(RuntimeError) as exc:
            _staging(GuardMode.ENFORCE).verdict(logging.getLogger("t"), "boom")
        assert "guard=feature_store_zero_variance" in str(exc.value)
        assert "alpha-engine-config-I7539" in str(exc.value)


class TestObservingIsLoud:
    """§7a obligation 3. An observe-mode guard that logged at INFO, or not at
    all, would be indistinguishable from a guard that never fired — and the
    observation period would produce no evidence for the promotion decision it
    exists to inform."""

    @pytest.mark.parametrize("mode", [GuardMode.OBSERVE, GuardMode.ENFORCE])
    def test_both_modes_log_at_error(self, caplog, mode):
        with caplog.at_level(logging.DEBUG):
            try:
                _staging(mode).verdict(logging.getLogger("t"), "constant column")
            except RuntimeError:
                pass
        levels = {r.levelno for r in caplog.records}
        assert levels == {logging.ERROR}

    def test_the_log_line_names_the_mode(self, caplog):
        with caplog.at_level(logging.ERROR):
            _staging().verdict(logging.getLogger("t"), "constant column")
        assert "OBSERVE" in caplog.text

    def test_the_log_line_is_greppable_by_guard_name(self, caplog):
        """Observe-mode verdicts have to be COUNTABLE before promotion —
        'ten clean runs' is not a criterion anyone can evaluate otherwise."""
        with caplog.at_level(logging.ERROR):
            _staging().verdict(logging.getLogger("t"), "constant column")
        assert "guard=feature_store_zero_variance" in caplog.text


class TestPromotionCriterionIsMandatory:
    """A guard parked in observe mode forever is the same defect one direction
    over. The criterion is part of the type, not advice in a docstring."""

    @pytest.mark.parametrize("field", ["promotion_criterion", "tracked_issue", "name"])
    def test_missing_is_rejected_at_construction(self, field):
        with pytest.raises(ValueError, match=field):
            _staging(**{field: ""})

    @pytest.mark.parametrize("field", ["promotion_criterion", "tracked_issue", "name"])
    def test_whitespace_only_is_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            _staging(**{field: "   "})

    def test_enforce_mode_still_requires_one(self):
        """In ENFORCE the criterion is the record of WHY the guard was
        promoted, which is what a later reader needs when deciding whether a
        regression means demoting it again."""
        with pytest.raises(ValueError):
            _staging(GuardMode.ENFORCE, promotion_criterion="")

    def test_it_fails_at_construction_not_at_first_verdict(self):
        """A malformed guard must not reach production and discover it on the
        one run where the predicate fires."""
        with pytest.raises(ValueError):
            GuardStaging(
                name="x", mode=GuardMode.OBSERVE, promotion_criterion="", tracked_issue="I1"
            )

    def test_a_non_mode_value_is_rejected(self):
        with pytest.raises(ValueError, match="GuardMode"):
            _staging(mode="observe")


class TestSurface:
    def test_describe_carries_all_four_facts(self):
        d = _staging().describe()
        for fragment in (
            "guard=feature_store_zero_variance",
            "mode=observe",
            "promotion=",
            "tracked=alpha-engine-config-I7539",
        ):
            assert fragment in d

    def test_enforcing_property(self):
        assert _staging(GuardMode.ENFORCE).enforcing is True
        assert _staging(GuardMode.OBSERVE).enforcing is False

    def test_staging_is_immutable(self):
        """Mode is a deployment decision, not a runtime toggle — a guard that
        can be flipped mid-run cannot be reasoned about from its declaration."""
        with pytest.raises(Exception):
            _staging().mode = GuardMode.ENFORCE  # type: ignore[misc]

    def test_mode_values_are_stable_strings(self):
        """They appear in logs and may be read back from config."""
        assert GuardMode.OBSERVE.value == "observe"
        assert GuardMode.ENFORCE.value == "enforce"
