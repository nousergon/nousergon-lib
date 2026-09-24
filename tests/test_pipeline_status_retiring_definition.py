"""A spine stage declared retiring tolerates EITHER merge order — the mirror
image of ``test_pipeline_status_pending_definition.py`` for the REMOVE
direction.

alpha-engine-config-I11267. ``MorningEnrich`` / ``DataPhase1`` and their
siblings across the three v1 SFs are being removed by the decoupled data
cutover (``alpha-engine-config-I11269``, ``PR11263`` §6.2c), but
``nousergon-data`` ``main`` still defines them until that PR merges (no
earlier than 2026-09-23). Dropping their registry entries the moment removal
is planned would redden ``crucible-dashboard``'s
``test_every_substantive_state_has_registry_entry`` for the whole pre-cutover
window. ``RETIRING_DEFINITION_STAGES`` keeps a retiring stage's registry
entries in place while excluding it from the "a declared stage must exist in
the definition" grading, so a definition that still has the state passes AND
a definition that has dropped it also passes.

These are mutation tests: each one simulates the mirror PR's view by patching
the registry and asserts the verdict that view must produce. A synthetic
stage name is used throughout (mirroring the pending suite's ``NEW``) rather
than a real retiring stage, so these tests exercise the mechanism
independently of which real stages happen to be retiring today.
"""

from __future__ import annotations

import pytest

from nousergon_lib.pipeline_status import (
    PIPELINE_STAGE_ORDER,
    RETIRING_DEFINITION_STAGES,
    STATE_TO_ARCHIVE_PAGE,
    ArtifactReason,
    RunStatus,
    WorkVerdict,
    classify_work,
    registry,
    retiring_definition_stages_for,
    stale_retiring_stages,
    undefined_spine_stages,
)
from nousergon_lib.pipeline_status.cycle_shape import CycleVerdict, build_cycle_shape

EOD = "ne-postclose-trading-pipeline"
OLD = "LaunchSomeOldDailySpot"


@pytest.fixture
def old_stage_declared(monkeypatch):
    """The definition-owner side: a spine stage that still HAS a registry
    entry (retiring keeps the entry — it never substitutes for one)."""
    order = dict(PIPELINE_STAGE_ORDER)
    order[EOD] = (*order[EOD], OLD)
    monkeypatch.setattr(registry, "PIPELINE_STAGE_ORDER", order)
    pages = dict(STATE_TO_ARCHIVE_PAGE)
    pages[OLD] = ArtifactReason(reason="test fixture — retiring stage's kept registry entry")
    monkeypatch.setattr(registry, "STATE_TO_ARCHIVE_PAGE", pages)
    return order


@pytest.fixture
def old_stage_retiring(old_stage_declared, monkeypatch):
    retiring = {k: dict(v) for k, v in RETIRING_DEFINITION_STAGES.items()}
    retiring[EOD][OLD] = "alpha-engine-config-I11267 (test fixture)"
    monkeypatch.setattr(registry, "RETIRING_DEFINITION_STAGES", retiring)
    return retiring


def _definition(*, with_old: bool) -> set[str]:
    states = set(PIPELINE_STAGE_ORDER[EOD]) | {"MarketHoursBlocked"}
    return states | {OLD} if with_old else states


def test_every_pipeline_has_a_retiring_map_and_every_entry_names_a_spine_stage():
    assert set(RETIRING_DEFINITION_STAGES) == set(PIPELINE_STAGE_ORDER)
    for pipeline, entries in RETIRING_DEFINITION_STAGES.items():
        for stage, ref in entries.items():
            assert stage in PIPELINE_STAGE_ORDER[pipeline], (
                f"{pipeline}: retiring marker {stage!r} names no spine stage — a "
                f"marker for nothing hides nothing and should be deleted"
            )
            assert ref.strip(), f"{pipeline}:{stage} retiring marker must name its tracking issue"


def test_every_retiring_stage_still_has_a_registry_entry():
    """A retiring stage with NO registry entry fails — the whole point of
    RETIRING_DEFINITION_STAGES is that the entry is KEPT, never dropped."""
    for pipeline, entries in RETIRING_DEFINITION_STAGES.items():
        for stage in entries:
            assert stage in STATE_TO_ARCHIVE_PAGE, (
                f"{pipeline}:{stage} is marked retiring but has no "
                f"STATE_TO_ARCHIVE_PAGE entry — retiring KEEPS the registry entry, "
                f"it does not substitute for one"
            )


def test_unmarked_old_stage_is_undefined_against_a_definition_without_it(old_stage_declared):
    """The RED proof: without the marker, a dropped stage fails the existence
    check exactly like any other undeclared state."""
    assert undefined_spine_stages(EOD, _definition(with_old=False)) == (OLD,)


def test_definition_still_has_it_is_not_red_while_retiring(old_stage_retiring):
    """Pre-cutover: the state is still live on nousergon-data main."""
    assert undefined_spine_stages(EOD, _definition(with_old=True)) == ()
    assert stale_retiring_stages(EOD, _definition(with_old=True)) == ()


def test_definition_dropped_it_is_not_red_while_retiring(old_stage_retiring):
    """Post-cutover: the cutover PR removed the state from the ASL."""
    assert undefined_spine_stages(EOD, _definition(with_old=False)) == ()


def test_the_definition_owner_sees_the_marker_as_stale_once_the_state_is_gone(old_stage_retiring):
    assert stale_retiring_stages(EOD, _definition(with_old=False)) == (OLD,)


def test_a_misspelled_stage_is_still_undefined_while_another_is_retiring(old_stage_retiring, monkeypatch):
    order = dict(registry.PIPELINE_STAGE_ORDER)
    order[EOD] = (*order[EOD], "CaptureSnapshto")
    monkeypatch.setattr(registry, "PIPELINE_STAGE_ORDER", order)
    assert undefined_spine_stages(EOD, _definition(with_old=True)) == ("CaptureSnapshto",)


def test_arn_and_bare_name_resolve_the_same_retiring_set(old_stage_retiring):
    arn = f"arn:aws:states:us-east-1:000000000000:stateMachine:{EOD}"
    # EOD already carries real retiring entries (alpha-engine-config-I11267);
    # the fixture adds OLD alongside them.
    assert retiring_definition_stages_for(arn) == retiring_definition_stages_for(EOD)
    assert OLD in retiring_definition_stages_for(EOD)


def _eod_run(entered):
    return classify_work(
        state_machine_name=EOD, status=RunStatus.SUCCEEDED, entered_states=entered
    )


def test_a_run_missing_the_retiring_stage_is_still_complete(old_stage_retiring):
    """Post-cutover execution: the retiring state can never be entered again,
    and that must not read as partial_success forever."""
    outcome = _eod_run(list(PIPELINE_STAGE_ORDER[EOD]))
    assert outcome.verdict is WorkVerdict.COMPLETED
    assert OLD not in outcome.stages_missing


def test_an_unmarked_stage_not_entered_is_still_partial(old_stage_declared):
    outcome = _eod_run(list(PIPELINE_STAGE_ORDER[EOD]))
    assert outcome.verdict is WorkVerdict.INCOMPLETE
    assert outcome.stages_missing == (OLD,)


def test_a_retiring_stage_that_was_entered_still_counts_as_entered(old_stage_retiring):
    """Pre-cutover: the state still exists and a normal run still enters it."""
    outcome = _eod_run([*PIPELINE_STAGE_ORDER[EOD], OLD])
    assert outcome.verdict is WorkVerdict.COMPLETED
    assert OLD in outcome.stages_entered


def test_a_cycle_missing_the_retiring_stage_is_still_complete(old_stage_retiring):
    entered = list(PIPELINE_STAGE_ORDER[EOD])
    shape = build_cycle_shape(
        pipeline=EOD,
        run_date="2026-09-21",
        outcomes=[(_eod_run(entered), "eod", entered)],
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert OLD not in shape.stages_missing


def test_a_cycle_missing_an_unmarked_stage_is_not_complete(old_stage_declared):
    entered = list(PIPELINE_STAGE_ORDER[EOD])
    shape = build_cycle_shape(
        pipeline=EOD,
        run_date="2026-09-21",
        outcomes=[(_eod_run(entered), "eod", entered)],
    )
    assert shape.verdict is not CycleVerdict.COMPLETED
    assert shape.stages_missing == (OLD,)


# ── The decoupled data cutover's retirement, completed (alpha-engine-config-I11269) ──

_RETIRED_BY_THE_CUTOVER = {
    "ne-weekly-freshness-pipeline": ("MorningEnrich", "DataPhase1"),
    "ne-preopen-trading-pipeline": ("LaunchMorningEnrichSpot", "LaunchMorningArcticAppendSpot"),
    "ne-postclose-trading-pipeline": (
        "LaunchPostMarketDataSpot",
        "LaunchPostMarketArcticAppendSpot",
        "LaunchEdgarPitFundamentalsDailySpot",
    ),
}


@pytest.mark.parametrize("pipeline", sorted(_RETIRED_BY_THE_CUTOVER))
def test_the_cutover_stages_left_the_spine_and_their_markers_with_them(pipeline):
    """The lockstep release for the cutover PR: the retired stages are gone
    from the spine AND from the retiring map (a marker that outlives its state
    fails nousergon-data's stale-retiring contract test), and the stage that
    replaced them is no longer pending (a pending marker on a landed state
    fails its stale-pending test)."""
    spine = PIPELINE_STAGE_ORDER[pipeline]
    retired = _RETIRED_BY_THE_CUTOVER[pipeline]
    assert not set(retired) & set(spine), f"{pipeline}: {set(retired) & set(spine)}"
    assert not set(retired) & retiring_definition_stages_for(pipeline)
    assert "WaitForCollectionManifests" in spine
    assert "WaitForCollectionManifests" not in registry.pending_definition_stages_for(pipeline)


@pytest.mark.parametrize("pipeline", sorted(_RETIRED_BY_THE_CUTOVER))
def test_the_cutover_stages_keep_their_registry_entries_for_history(pipeline):
    """Leaving the spine is not leaving the registry: executions from before the
    cutover still carry these states, and a state with no entry drops its row
    from every historical execution the dashboard renders."""
    for stage in _RETIRED_BY_THE_CUTOVER[pipeline]:
        assert stage in STATE_TO_ARCHIVE_PAGE, f"{pipeline}:{stage}"
