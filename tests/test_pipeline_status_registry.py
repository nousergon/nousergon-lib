"""Unit tests for ``alpha_engine_lib.pipeline_status.registry``.

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

from alpha_engine_lib.pipeline_status import registry


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
        "alpha-engine-saturday-pipeline": "Saturday SF",
        "alpha-engine-weekday-pipeline": "Weekday SF",
        "alpha-engine-eod-pipeline": "EOD SF",
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
    settle delay after `StartExecutorEC2` in the Weekday SF. Caught by
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
    """Spot-check: the operator-meaningful Saturday SF states (per plan
    doc §2.1) ALL appear in the registry. Backstop against a future
    refactor accidentally dropping one."""
    required = {
        "MorningEnrich",
        "DataPhase1",
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
        "Evaluator",
        "DriftDetection",
        "WeeklySubstrateHealthCheck",
        "NotifyComplete",
        "HandleFailure",
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
    assert entry.page == "17_Research_Briefing_Archive"


def test_archive_page_ref_is_frozen():
    """ArchivePageRef + ArtifactReason are frozen dataclasses — mutation
    is forbidden so consumers can rely on registry stability across
    a process lifetime."""
    ref = registry.ArchivePageRef(page="x", artifact_label="y")
    with pytest.raises((TypeError, AttributeError)):
        ref.page = "z"  # type: ignore[misc]


def test_artifact_reason_is_frozen():
    reason = registry.ArtifactReason(reason="x")
    with pytest.raises((TypeError, AttributeError)):
        reason.reason = "y"  # type: ignore[misc]
