"""A spine stage declared ahead of its definition tolerates EITHER merge order.

alpha-engine-config-I10762. ``LaunchEdgarPitFundamentalsDailySpot`` entered
:data:`PIPELINE_STAGE_ORDER` (nousergon-lib-PR410) before any definition had
it, and two PRs deadlocked: crucible-dashboard-PR860 (library pin bump) read
``nousergon-data`` ``main`` and found no such state; nousergon-data-PR1702 (the
state) ran the dashboard's tests on the dashboard's OLD pin. Each was red on
the other's ``main``.

These are mutation tests: each one simulates the mirror PR's view by patching
the registry and asserts the verdict that view must produce.
"""

from __future__ import annotations

import pytest

from nousergon_lib.pipeline_status import (
    PENDING_DEFINITION_STAGES,
    PIPELINE_STAGE_ORDER,
    RunStatus,
    WorkVerdict,
    classify_work,
    pending_definition_stages_for,
    registry,
    stale_pending_stages,
    undefined_spine_stages,
)
from nousergon_lib.pipeline_status.cycle_shape import CycleVerdict, build_cycle_shape

EOD = "ne-postclose-trading-pipeline"
NEW = "LaunchSomeNewDailySpot"


@pytest.fixture
def new_stage_declared(monkeypatch):
    """The library side of the pair: a new stage in the spine, NOT yet pending."""
    order = dict(PIPELINE_STAGE_ORDER)
    order[EOD] = (*order[EOD], NEW)
    monkeypatch.setattr(registry, "PIPELINE_STAGE_ORDER", order)
    return order


@pytest.fixture
def new_stage_pending(new_stage_declared, monkeypatch):
    pending = {k: dict(v) for k, v in PENDING_DEFINITION_STAGES.items()}
    pending[EOD][NEW] = "alpha-engine-config-I10762 (test fixture)"
    monkeypatch.setattr(registry, "PENDING_DEFINITION_STAGES", pending)
    return pending


def _definition(*, with_new: bool) -> set[str]:
    # Subtract any REAL pending stages (alpha-engine-config-I11267 added
    # WaitForCollectionManifests to EOD's own pending map) — this helper
    # models "the live definition", which by construction does not yet
    # contain a stage the library declares ahead of its own definition.
    states = (set(PIPELINE_STAGE_ORDER[EOD]) | {"MarketHoursBlocked"}) - pending_definition_stages_for(EOD)
    return states | {NEW} if with_new else states


def test_every_pipeline_has_a_pending_map_and_every_entry_names_a_spine_stage():
    assert set(PENDING_DEFINITION_STAGES) == set(PIPELINE_STAGE_ORDER)
    for pipeline, entries in PENDING_DEFINITION_STAGES.items():
        for stage, ref in entries.items():
            assert stage in PIPELINE_STAGE_ORDER[pipeline], (
                f"{pipeline}: pending marker {stage!r} names no spine stage — a "
                f"marker for nothing hides nothing and should be deleted"
            )
            assert ref.strip(), f"{pipeline}:{stage} pending marker must name its tracking issue"


def test_unmarked_new_stage_is_undefined_against_a_definition_without_it(new_stage_declared):
    """The RED proof: without the marker the consumer's existence check fails."""
    assert undefined_spine_stages(EOD, _definition(with_new=False)) == (NEW,)


def test_consumer_first_order_is_not_red_when_the_stage_is_pending(new_stage_pending):
    """Mirror of crucible-dashboard-PR860: pin bumped, definition not yet on main."""
    assert undefined_spine_stages(EOD, _definition(with_new=False)) == ()
    assert stale_pending_stages(EOD, _definition(with_new=False)) == ()


def test_definition_first_order_is_not_red_for_a_consumer(new_stage_pending):
    """Mirror of nousergon-data-PR1702 seen from a consumer on the pending pin."""
    assert undefined_spine_stages(EOD, _definition(with_new=True)) == ()


def test_the_definition_owner_sees_the_marker_as_stale_once_the_state_exists(new_stage_pending):
    assert stale_pending_stages(EOD, _definition(with_new=True)) == (NEW,)


def test_a_misspelled_stage_is_still_undefined_while_another_is_pending(new_stage_pending, monkeypatch):
    order = dict(registry.PIPELINE_STAGE_ORDER)
    order[EOD] = (*order[EOD], "CaptureSnapshto")
    monkeypatch.setattr(registry, "PIPELINE_STAGE_ORDER", order)
    assert undefined_spine_stages(EOD, _definition(with_new=False)) == ("CaptureSnapshto",)


def test_arn_and_bare_name_resolve_the_same_pending_set(new_stage_pending):
    arn = f"arn:aws:states:us-east-1:000000000000:stateMachine:{EOD}"
    # >=, not ==: alpha-engine-config-I11267 declared a real pending entry
    # (WaitForCollectionManifests) for this pipeline too, alongside the
    # fixture's synthetic NEW.
    assert pending_definition_stages_for(arn) == pending_definition_stages_for(EOD)
    assert NEW in pending_definition_stages_for(EOD)


def test_undeclared_pipeline_raises_rather_than_passing():
    with pytest.raises(KeyError):
        undefined_spine_stages("ne-no-such-pipeline", set())


def _eod_run(entered):
    return classify_work(
        state_machine_name=EOD, status=RunStatus.SUCCEEDED, entered_states=entered
    )


def test_a_run_of_the_old_definition_is_complete_while_the_stage_is_pending(new_stage_pending):
    outcome = _eod_run(list(PIPELINE_STAGE_ORDER[EOD]))
    assert outcome.verdict is WorkVerdict.COMPLETED
    assert NEW not in outcome.stages_missing


def test_an_unmarked_stage_not_entered_is_still_partial(new_stage_declared):
    outcome = _eod_run(list(PIPELINE_STAGE_ORDER[EOD]))
    assert outcome.verdict is WorkVerdict.INCOMPLETE
    assert outcome.stages_missing == (NEW,)


def test_a_pending_stage_that_was_entered_counts_as_entered(new_stage_pending):
    outcome = _eod_run([*PIPELINE_STAGE_ORDER[EOD], NEW])
    assert outcome.verdict is WorkVerdict.COMPLETED
    assert NEW in outcome.stages_entered


def test_a_cycle_of_the_old_definition_is_complete_while_the_stage_is_pending(new_stage_pending):
    entered = list(PIPELINE_STAGE_ORDER[EOD])
    shape = build_cycle_shape(
        pipeline=EOD,
        run_date="2026-09-14",
        outcomes=[(_eod_run(entered), "eod", entered)],
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert NEW not in shape.stages_missing


def test_a_cycle_missing_an_unmarked_stage_is_not_complete(new_stage_declared):
    entered = list(PIPELINE_STAGE_ORDER[EOD])
    shape = build_cycle_shape(
        pipeline=EOD,
        run_date="2026-09-14",
        outcomes=[(_eod_run(entered), "eod", entered)],
    )
    assert shape.verdict is not CycleVerdict.COMPLETED
    assert shape.stages_missing == (NEW,)
