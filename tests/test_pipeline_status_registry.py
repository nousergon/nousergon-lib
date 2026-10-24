"""Unit tests for ``nousergon_lib.pipeline_status.registry``.

Pins the substantive-state filter + state→archive-page registry against
the SF JSON contract. The downstream-repo CI test (planned for the
alpha-engine-data PR in Phase 3) walks the live SF JSONs and asserts
every Task state name is in the registry; these tests guard the lib-side
invariants (every entry is a non-empty :class:`ArchivePageRef` or
:class:`ArtifactReason`, Wait-grouping is exhaustive for the parent
states named in the registry, pretty labels match `sf-telegram-notifier`
verbatim).
"""

from __future__ import annotations

import pytest

from nousergon_lib.pipeline_status import registry


def test_state_to_archive_page_entries_are_non_empty():
    """Every registry entry is either a valid ArchivePageRef (non-empty
    page + label) or a valid ArtifactReason (non-empty reason string)."""
    for name, entry in registry.STATE_TO_ARCHIVE_PAGE.items():
        if isinstance(entry, registry.ArchivePageRef):
            assert entry.page, f"ArchivePageRef for {name!r} has empty page slug"
            assert entry.artifact_label, (
                f"ArchivePageRef for {name!r} has empty artifact_label"
            )
        elif isinstance(entry, registry.ArtifactReason):
            assert entry.reason, f"ArtifactReason for {name!r} has empty reason"
            # Per feedback_no_silent_fails — explicit non-generic reason
            assert "no artifact" not in entry.reason.lower() or len(entry.reason) > 30, (
                f"ArtifactReason for {name!r} reads as a generic placeholder; "
                f"reasons must be specific per feedback_no_silent_fails"
            )
        else:
            pytest.fail(f"Registry entry for {name!r} is wrong type: {type(entry)}")


def test_pipeline_labels_match_sf_telegram_notifier():
    """Pretty labels are kept verbatim with sf-telegram-notifier's _SF_LABELS
    (alpha-engine-data/infrastructure/lambdas/sf-telegram-notifier/index.py L43-47).
    The two MUST agree so Telegram + email + page channels render the same
    label."""
    assert registry.PIPELINE_LABELS == {
        "ne-weekly-freshness-pipeline": "Weekly Freshness SF",
        "ne-preopen-trading-pipeline": "Pre-open Trading SF",
        "ne-postclose-trading-pipeline": "Post-close Trading SF",
        "ne-postclose-reconcile-pipeline": "Post-close Reconcile SF",
    }


def test_wait_grouping_parents_are_in_registry():
    """Every Wait-grouping target (the parent state name) MUST exist in the
    registry — otherwise the wait companion gets absorbed into a parent
    that doesn't render, dropping the row entirely."""
    missing = []
    for wait_name, parent in registry.WAIT_GROUPING.items():
        if parent not in registry.STATE_TO_ARCHIVE_PAGE:
            missing.append((wait_name, parent))
    assert not missing, (
        f"WAIT_GROUPING parents missing from STATE_TO_ARCHIVE_PAGE: {missing}"
    )


def test_wait_for_instance_ready_maps_to_start_executor_ec2():
    """Regression pin (v0.28.1) — `WaitForInstanceReady` is the post-boot
    settle delay after `StartExecutorEC2` in the Pre-open Trading SF. Caught by
    the alpha-engine-dashboard registry-drift CI test on first Phase-2
    run; v0.28.0 shipped without it because the original `jq` walk
    filtered to `Task`-type states and missed bare `Wait`-type companions.
    """
    assert registry.WAIT_GROUPING.get("WaitForInstanceReady") == "StartExecutorEC2"


def test_wait_grouping_keys_are_wait_prefix():
    """Every key in WAIT_GROUPING is a ``WaitFor*`` state name (sanity check
    that we haven't accidentally rolled up a non-wait state)."""
    for wait_name in registry.WAIT_GROUPING.keys():
        assert wait_name.startswith("WaitFor"), (
            f"WAIT_GROUPING key {wait_name!r} does not start with 'WaitFor' — "
            f"may be a substantive state mis-rolled-up"
        )


def test_substantive_resources_covers_expected_arn_set():
    """Pin the SUBSTANTIVE_RESOURCES set — these are the Resource ARN
    values the SF JSON walk must filter to. Drift between this set and the
    SF JSONs is caught at the dashboard PR's CI test in Phase 2."""
    assert registry.SUBSTANTIVE_RESOURCES == frozenset(
        {
            "arn:aws:states:::lambda:invoke",
            "arn:aws:states:::aws-sdk:ssm:sendCommand",
            "arn:aws:states:::sns:publish",
            "arn:aws:states:::aws-sdk:ec2:startInstances",
            "arn:aws:states:::aws-sdk:ec2:stopInstances",
        }
    )


def test_registry_covers_known_saturday_substantive_states():
    """Spot-check: the operator-meaningful Weekly Freshness SF states (per plan
    doc §2.1) ALL appear in the registry. Backstop against a future
    refactor accidentally dropping one."""
    required = {
        "MorningEnrich",
        "DataPhase1",
        "Scanner",
        "RAGIngestion",
        "RegimeSubstrate",
        "Research",
        "DataPhase2",
        "EvalJudgeSubmitFirstSaturday",
        "EvalJudgeSubmitWeekly",
        "EvalRollingMean",
        "PredictorTraining",
        "Backtester",
        "Parity",
        # alpha-engine-config-I3112 deliverable 3: the merged "Evaluator"
        # state no longer exists in the Saturday SF. Both halves are required
        # here, not one — a registry entry for only the first would let the
        # second render as an unknown state on the pipeline-status surface,
        # which is the config#1115/#1120/#2372/#2430 drift class this test
        # exists to catch.
        "EvaluatorDiagnostics",
        "EvaluatorOptimize",
        "DriftDetection",
        "SaturdayHealthCheck",
        "WeeklySubstrateHealthCheck",
        "NotifyComplete",
        "HandleFailure",
        "WeeklyRunDayGate",  # config#1824 run-day gate (2026-07-06)
    }
    missing = required - set(registry.STATE_TO_ARCHIVE_PAGE.keys())
    assert not missing, f"Saturday substantive states missing from registry: {missing}"


def test_registry_covers_known_weekday_substantive_states():
    required = {
        "DeployDriftCheck",
        "StartExecutorEC2",
        "MorningEnrich",
        "PredictorInference",
        "PredictorHealthCheck",
        "RunMorningPlanner",
        "RunDaemon",
        "HandleFailure",
        "CodeFreshnessGate",  # config#1811 Part A (2026-07-06)
        "ForceStopUnresponsiveInstance",  # config#1811 Part B watchdog branch
        "LaunchDailyDataSpot",  # config#1807 daily data spot decouple
        "LaunchMorningEnrichSpot",  # config#1687 per-stage data spots
        "LaunchMorningArcticAppendSpot",  # config#1687
        "PublishDataSpotFailureImmediate",  # config#1687 fail-open alert
    }
    missing = required - set(registry.STATE_TO_ARCHIVE_PAGE.keys())
    assert not missing, f"Weekday substantive states missing from registry: {missing}"


def test_registry_covers_known_eod_substantive_states():
    required = {
        "PostMarketData",
        "CaptureSnapshot",
        "EODReconcile",
        "DailySubstrateHealthCheck",
        "StopTradingInstance",
        "HandleFailure",
        "LaunchPostMarketDataSpot",  # config#1687 per-stage data spots
        "LaunchPostMarketArcticAppendSpot",  # config#1687
        "PublishDataSpotFailureImmediate",  # config#1687 fail-open alert
    }
    missing = required - set(registry.STATE_TO_ARCHIVE_PAGE.keys())
    assert not missing, f"EOD substantive states missing from registry: {missing}"


def test_lookup_registry_returns_none_for_unknown_state():
    """``lookup_registry`` MUST return None for unknown states — the
    dashboard treats None as a CI-time registry-drift signal, NOT as a
    renderable placeholder."""
    assert registry.lookup_registry("NoSuchState") is None


def test_lookup_registry_returns_entry_for_known_state():
    entry = registry.lookup_registry("Research")
    assert isinstance(entry, registry.ArchivePageRef)
    assert entry.page == "host_research_signals?tab=Briefing+Archive"


def test_archive_page_ref_is_frozen():
    """ArchivePageRef + ArtifactReason are frozen Pydantic models —
    mutation is forbidden so consumers can rely on registry stability
    across a process lifetime. Pydantic V2 raises ValidationError with
    ``type='frozen_instance'`` on attribute set; older stdlib-dataclass
    consumers would have seen TypeError / FrozenInstanceError, both
    accepted here so the test survives the dataclass → BaseModel
    transition without spurious churn."""
    from pydantic import ValidationError

    ref = registry.ArchivePageRef(page="x", artifact_label="y")
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        ref.page = "z"  # type: ignore[misc]


def test_artifact_reason_is_frozen():
    from pydantic import ValidationError

    reason = registry.ArtifactReason(reason="x")
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        reason.reason = "y"  # type: ignore[misc]


def test_postclose_split_moves_the_collector_stages_to_the_reconcile_machine():
    """nousergon-data-PR1996 split the post-close run in two
    (alpha-engine-config-I11269 follow-up). The three collector-dependent
    spine stages moved, unrenamed, to ``ne-postclose-reconcile-pipeline``.

    They must leave the post-close spine OUTRIGHT, not via
    RETIRING_DEFINITION_STAGES: nousergon-data's contract test is strict about
    a retiring marker whose state is gone, and a stage declared on both
    machines would make every post-close cycle ``partial_success`` for a stage
    that machine can no longer enter.
    """
    moved = ("WaitForCollectionManifests", "EODReconcile", "StopTradingInstance")
    postclose = registry.PIPELINE_STAGE_ORDER["ne-postclose-trading-pipeline"]
    reconcile = registry.PIPELINE_STAGE_ORDER["ne-postclose-reconcile-pipeline"]
    assert reconcile == moved
    assert not set(moved) & set(postclose)
    assert "CaptureSnapshot" in postclose
    for pipeline in ("ne-postclose-trading-pipeline", "ne-postclose-reconcile-pipeline"):
        assert not registry.RETIRING_DEFINITION_STAGES[pipeline]
        assert not registry.PENDING_DEFINITION_STAGES[pipeline]
    # Every moved stage keeps the registry entry it had before the split.
    for stage in moved:
        assert stage in registry.STATE_TO_ARCHIVE_PAGE


def test_director_poll_companion_rolls_up_into_the_director_row():
    """alpha-engine-config-I11936: the weekly Director runs over SSM on the
    launcher spot, polled by ``WaitForDirector``; the poll must roll up into
    the existing ``Director`` row rather than render as its own."""
    assert registry.WAIT_GROUPING.get("WaitForDirector") == "Director"
    assert "Director" in registry.STATE_TO_ARCHIVE_PAGE


# ── alpha-engine-config-I11081: the declared substantive-stage subset ─────────


def test_every_spine_pipeline_declares_a_vacuity_spine():
    assert set(registry.VACUITY_SPINE) == set(registry.PIPELINE_STAGE_ORDER)


@pytest.mark.parametrize("pipeline", sorted(registry.PIPELINE_STAGE_ORDER))
def test_vacuity_spine_is_a_non_empty_ordered_subset_of_the_spine(pipeline):
    spine = registry.PIPELINE_STAGE_ORDER[pipeline]
    vac = registry.VACUITY_SPINE[pipeline]
    assert vac, f"{pipeline}: an empty vacuity spine makes every rerun vacuous"
    assert set(vac) <= set(spine), f"{pipeline}: {sorted(set(vac) - set(spine))} not in the spine"
    assert list(vac) == [s for s in spine if s in vac], f"{pipeline}: order differs from the spine"


def test_infrastructure_stages_are_not_substantive():
    """The stages no skip flag gates. Measured 2026-10-23: including any of them
    made the weekday guard structurally unable to fire (alpha-engine-config-I11081)."""
    pre = registry.VACUITY_SPINE["ne-preopen-trading-pipeline"]
    for stage in ("StartExecutorEC2", "CodeFreshnessGate", "WaitForCollectionManifests"):
        assert stage not in pre
    rec = registry.VACUITY_SPINE["ne-postclose-reconcile-pipeline"]
    for stage in ("WaitForCollectionManifests", "StopTradingInstance"):
        assert stage not in rec


def test_weekly_vacuity_spine_is_the_full_spine():
    # The weekly rerun guard (I11075) reads the full spine; this release must not
    # change its verdicts.
    assert (registry.VACUITY_SPINE["ne-weekly-freshness-pipeline"]
            == registry.PIPELINE_STAGE_ORDER["ne-weekly-freshness-pipeline"])


def test_vacuity_spine_for_accepts_arn_or_name_and_is_empty_when_undeclared():
    arn = "arn:aws:states:us-east-1:000000000000:stateMachine:ne-postclose-trading-pipeline"
    assert registry.vacuity_spine_for(arn) == ("CaptureSnapshot",)
    assert registry.vacuity_spine_for("ne-postclose-trading-pipeline") == ("CaptureSnapshot",)
    assert registry.vacuity_spine_for("no-such-pipeline") == ()
    assert registry.vacuity_spine_for("") == ()


def test_vacuity_spine_is_exported_from_the_package():
    import nousergon_lib.pipeline_status as ps
    assert ps.vacuity_spine_for is registry.vacuity_spine_for
    assert "VACUITY_SPINE" in ps.__all__ and "vacuity_spine_for" in ps.__all__
