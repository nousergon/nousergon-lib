"""Contract tests for the fleet's ``pipeline_role`` vocabulary.

alpha-engine-config#5592. The vocabulary's whole job is to be TOTAL: a role
minted by a producer that no consumer classifies fails silently (the walk
skips it and an older run renders as current — alpha-engine-config#5590).
These tests pin the properties consumers rely on.
"""

from __future__ import annotations

import itertools

import pytest

from nousergon_lib.pipeline_status.roles import (
    ADHOC_ROLES,
    ALL_ROLES,
    CADENCE_ROLES,
    EXERCISE_ROLES,
    RECOVERY_ROLES,
    cadence_filter,
    classify,
)

_BUCKETS = {
    "cadence": CADENCE_ROLES,
    "recovery": RECOVERY_ROLES,
    "exercise": EXERCISE_ROLES,
    "adhoc": ADHOC_ROLES,
}


class TestVocabularyIsTotal:
    def test_every_bucket_is_non_empty(self):
        for name, bucket in _BUCKETS.items():
            assert bucket, f"{name} bucket is empty"

    def test_buckets_are_pairwise_disjoint(self):
        for (a_name, a), (b_name, b) in itertools.combinations(
            _BUCKETS.items(), 2
        ):
            assert not (a & b), (
                f"{a_name} and {b_name} both claim {sorted(a & b)} — a role "
                f"must classify into exactly one bucket"
            )

    def test_all_roles_is_exactly_the_union(self):
        union = set().union(*_BUCKETS.values())
        assert set(ALL_ROLES) == union

    def test_every_role_classifies(self):
        for role in ALL_ROLES:
            assert classify(role) in _BUCKETS


class TestExerciseIsNotRecovery:
    """Binding invariant, not a preference.

    A recovery role may complete a cadence cycle. An exercise run is a
    debugging cycle on the same state machine — if it were classified as
    recovery, a Tuesday exercise SUCCESS would mark the week's cadence
    COMPLETE and hide a failed Saturday belief refresh behind a green dot.
    """

    def test_exercise_and_recovery_do_not_overlap(self):
        assert not (EXERCISE_ROLES & RECOVERY_ROLES)

    def test_cadence_filter_never_admits_an_exercise_role(self):
        for cadence_role in CADENCE_ROLES:
            assert not (cadence_filter(cadence_role) & EXERCISE_ROLES)

    def test_cadence_filter_never_admits_an_adhoc_role(self):
        for cadence_role in CADENCE_ROLES:
            assert not (cadence_filter(cadence_role) & ADHOC_ROLES)


class TestCadenceFilter:
    def test_unions_the_cadence_role_with_recovery_overlays(self):
        assert cadence_filter("weekly") == frozenset(
            {"weekly", "watch-rerun", "recovery"}
        )

    def test_excludes_the_other_cadence_roles(self):
        got = cadence_filter("weekly")
        assert "daily" not in got and "eod" not in got

    def test_rejects_a_non_cadence_role_loudly(self):
        # Passing "exercise" or "smoke" here is a caller bug; returning a
        # plausible-looking filter would hide it.
        for bad in ("exercise", "smoke", "watch-rerun", "nope"):
            with pytest.raises(ValueError):
                cadence_filter(bad)


class TestClassify:
    def test_untagged_is_not_a_violation(self):
        # Manual runs legitimately carry no role — absence must not be
        # reported as an unknown role.
        assert classify(None) is None
        assert classify("") is None

    def test_unknown_role_is_unclassified(self):
        assert classify("exercise-v2") is None

    @pytest.mark.parametrize(
        "role,bucket",
        [
            ("weekly", "cadence"),
            ("eod", "cadence"),
            ("watch-rerun", "recovery"),
            ("exercise", "exercise"),
            ("operator-replay", "adhoc"),
        ],
    )
    def test_known_roles_land_in_their_bucket(self, role, bucket):
        assert classify(role) == bucket


class TestKnownProducerLiterals:
    """The roles the fleet's producers mint today, named explicitly so a
    removal from the vocabulary breaks here rather than in a consumer."""

    @pytest.mark.parametrize(
        "role",
        [
            "weekly",  # EventBridge alpha-engine-saturday input
            "daily",  # EventBridge pre-open input
            "eod",  # crucible-executor daemon.py
            "exercise",  # nousergon-data step_function_eod.json
            "watch-rerun",  # nousergon-data scripts/weekly_sf_rerun.py
            "recovery",  # operator/agent convention
            "smoke",
            "shell-run",
            "backfill",
            "operator-replay",
        ],
    )
    def test_producer_literal_is_in_the_vocabulary(self, role):
        assert role in ALL_ROLES
