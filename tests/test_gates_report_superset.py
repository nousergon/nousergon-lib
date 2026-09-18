"""The protections `crucible/morning.py` carries that this package did not.

`shared-code-policy` §3.1: a fork is closed only onto a destination that is
already a SUPERSET of it. These are the guards enumerated on the fork before
the migration (`alpha-engine-config-I10953`), each asserted here so the
migration lands on a library that is at least as strong — and so a later
simplification of any of them fails a test that names the incident rather than
reading as tidying.

Each test names the fork behaviour it is importing, not just the API.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nousergon_lib.gates.report import (
    Move,
    UndeliveredError,
    deliver,
    moved_since,
    resolve_trigger,
    staleness,
)

NOW = dt.datetime(2026, 9, 18, 13, 0, tzinfo=dt.timezone.utc)


def _board(rows):
    return {"rows": rows}


# ---------------------------------------------------------------------------
# moved_since: a schedule phase is not a move (alpha-engine-config-I10872)


def _running_either_side(before, after):
    """The shape of crucible's `console.classify.not_yet_due`, in miniature."""
    transient = {"RUNNING", "ARMED"}
    states = [(row or {}).get("component_state") for row in (before, after)]
    return any(state in transient for state in states)


def test_a_row_transient_in_either_reading_is_deferred_not_counted_as_a_move():
    """I10872: 30 RUNNING rows were counted as having moved. Correct, and news
    about nothing."""
    previous = _board([{"id": "a", "state": "UNMET", "component_state": "RUNNING"}])
    current = _board([{"id": "a", "state": "MET", "component_state": "IDLE"}])
    result = moved_since(
        previous=previous,
        current=current,
        previous_reason=None,
        defer=_running_either_side,
    )
    assert result.count == 0
    assert result.lines == []
    assert result.deferred == (Move("a", "UNMET", "MET"),)
    assert result.deferred_count == 1


def test_a_deferred_row_is_still_rendered_so_it_is_set_aside_and_not_dropped():
    previous = _board([{"id": "a", "state": "UNMET", "component_state": "RUNNING"}])
    current = _board([{"id": "a", "state": "MET", "component_state": "RUNNING"}])
    result = moved_since(previous=previous, current=current, previous_reason=None, defer=_running_either_side)
    assert result.deferred_lines == ["- a: UNMET -> MET"]


def test_a_row_with_no_classifiable_field_is_always_a_move():
    """A board written before the field existed must not silently defer."""
    previous = _board([{"id": "a", "state": "UNMET"}])
    current = _board([{"id": "a", "state": "MET"}])
    result = moved_since(previous=previous, current=current, previous_reason=None, defer=_running_either_side)
    assert result.count == 1
    assert result.deferred == ()


def test_the_predicate_sees_whole_rows_so_it_can_classify_on_a_field_the_diff_ignores():
    seen: list[tuple[dict | None, dict | None]] = []

    def record(before, after):
        seen.append((before, after))
        return False

    moved_since(
        previous=_board([{"id": "a", "state": "UNMET", "component_state": "RUNNING"}]),
        current=_board([{"id": "a", "state": "MET", "component_state": "IDLE"}]),
        previous_reason=None,
        defer=record,
    )
    assert seen == [
        (
            {"id": "a", "state": "UNMET", "component_state": "RUNNING"},
            {"id": "a", "state": "MET", "component_state": "IDLE"},
        )
    ]


def test_a_vanished_row_reaches_the_predicate_with_None_on_the_missing_side():
    calls: list[tuple[dict | None, dict | None]] = []
    moved_since(
        previous=_board([{"id": "gone", "state": "MET"}]),
        current=_board([]),
        previous_reason=None,
        defer=lambda b, a: calls.append((b, a)) or False,
    )
    assert calls == [({"id": "gone", "state": "MET"}, None)]


def test_without_a_predicate_the_diff_is_unchanged_and_nothing_is_deferred():
    result = moved_since(
        previous=_board([{"id": "a", "state": "UNMET"}]),
        current=_board([{"id": "a", "state": "MET"}, {"id": "b", "state": "MET"}]),
        previous_reason=None,
    )
    assert result.lines == ["- a: UNMET -> MET", "- b: ABSENT -> MET"]
    assert result.count == 2
    assert result.deferred == ()


def test_a_deferred_count_over_an_unreadable_previous_is_None_never_zero():
    """Zero deferred over a comparison that did not happen is a claim."""
    result = moved_since(previous=None, current=_board([]), previous_reason="denied", defer=_running_either_side)
    assert result.count is None
    assert result.deferred_count is None


# ---------------------------------------------------------------------------
# staleness: the caller's own alarm word


@pytest.mark.parametrize(
    "generated_at",
    [None, "", "not-a-timestamp", "2026-09-01T00:00:00Z"],
)
def test_every_stale_branch_opens_with_the_callers_prefix(generated_at):
    """A live surface's first line is recognised by its first word; changing it
    as a side effect of consolidating two implementations is the un-shipped
    behaviour `shared-code-policy` §3.1 is about."""
    line = staleness(
        generated_at=generated_at,
        now=NOW,
        stale_after=dt.timedelta(days=1),
        label="board/current.json",
        prefix="STALE BOARD",
    )
    assert line is not None
    assert line.startswith("STALE BOARD: ")


def test_a_fresh_reading_has_no_headline_whatever_the_prefix():
    assert (
        staleness(
            generated_at="2026-09-18T12:00:00Z",
            now=NOW,
            stale_after=dt.timedelta(days=1),
            label="board/current.json",
            prefix="STALE BOARD",
        )
        is None
    )


# ---------------------------------------------------------------------------
# resolve_trigger: an explicit environment


def test_the_trigger_can_be_resolved_from_a_supplied_mapping_without_touching_the_process():
    assert resolve_trigger(environ={"GITHUB_EVENT_NAME": "schedule"}) == "schedule"


def test_an_empty_supplied_mapping_is_unknown_even_when_the_process_says_otherwise(monkeypatch):
    """The substitution is total: a stray variable in the real environment must
    not make an assertion about `unknown` pass for the wrong reason."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert resolve_trigger(environ={}) == "unknown"


def test_a_supplied_override_var_is_read_after_the_reserved_github_one():
    assert (
        resolve_trigger(
            override_var="CRUCIBLE_TRIGGER",
            environ={"GITHUB_EVENT_NAME": "schedule", "CRUCIBLE_TRIGGER": "manual"},
        )
        == "schedule"
    )
    assert resolve_trigger(override_var="CRUCIBLE_TRIGGER", environ={"CRUCIBLE_TRIGGER": "manual"}) == "manual"


# ---------------------------------------------------------------------------
# deliver: where it went, from the transport rather than from the request


class _Result:
    def __init__(self, **kwargs):
        self.any_ok = True
        self.dedup_skipped = False
        self.muted = False
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_deliver_returns_the_destination_the_transport_reports():
    sent: list[dict] = []

    def transport(message, **kwargs):
        sent.append(kwargs)
        return _Result(telegram_destination="operator_chat")

    where = deliver(
        "body",
        severity="info",
        source="crucible-v2/report.morning",
        console_artifact="artifact",
        destination="operator_chat",
        transport=transport,
    )
    assert where == "operator_chat"
    assert sent[0]["destination"] == "operator_chat"


def test_deliver_falls_back_to_the_transport_name_rather_than_to_what_it_asked_for():
    """Recording the REQUEST would keep reading `ok` on the day the resolution
    changed underneath it."""
    where = deliver(
        "body",
        severity="info",
        source="s",
        console_artifact="a",
        destination="operator_chat",
        transport=lambda message, **kwargs: _Result(),
    )
    assert where == "telegram"


def test_an_injected_transport_is_still_held_to_the_undelivered_check():
    with pytest.raises(UndeliveredError):
        deliver(
            "body",
            severity="info",
            source="s",
            console_artifact="a",
            destination="operator_chat",
            transport=lambda message, **kwargs: _Result(muted=True),
        )
