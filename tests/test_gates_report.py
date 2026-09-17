"""The daily-report core's invariants — the ones paid for in incidents.

Every test here grades a property a well-intentioned refactor would otherwise
be free to break: an unreadable timestamp is never assumed fresh, a denial is
never rendered as absence, an over-long body raises rather than truncating,
a withheld clause leaves a trace, and a corrupt history row neither blanks the
index nor passes for a missing field.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from nousergon_lib.gates.report import (
    TRIGGER_RE,
    TRIGGER_UNKNOWN,
    TRIGGER_VARS,
    MessageTooLongError,
    Read,
    UndeliveredError,
    assert_within_budget,
    deliver,
    escape,
    filter_withheld_clauses,
    history_row,
    moved_since,
    read_optional,
    read_required,
    render_history_index,
    resolve_trigger,
    staleness,
    transport_prefix,
    wire_budget,
    wire_length,
)

NOW = dt.datetime(2026, 9, 17, 13, 0, tzinfo=dt.timezone.utc)
DAY = dt.timedelta(days=1)


# --------------------------------------------------------------------------
# stores


class JsonStore:
    """A store spelling fetch the way the contract names it: ``get_json``."""

    def __init__(self, documents, raises=None):
        self.documents = documents
        self.raises = raises or {}

    def get_json(self, key):
        if key in self.raises:
            raise self.raises[key]
        if key not in self.documents:
            raise KeyError(key)
        return self.documents[key]


class BytesStore:
    """A store spelling fetch the way `gates.store.GateStore` names it."""

    def __init__(self, documents):
        self.documents = documents

    def get_bytes(self, key):
        if key not in self.documents:
            raise KeyError(key)
        return json.dumps(self.documents[key]).encode("utf-8")


class RawBytesStore(BytesStore):
    def get_bytes(self, key):
        return self.documents[key]


def client_error(code):
    """The botocore shape, built without importing the SDK."""
    exc = Exception(f"An error occurred ({code})")
    exc.response = {"Error": {"Code": code}}
    return exc


# --------------------------------------------------------------------------
# present / absent / DENIED


def test_a_present_document_reads_as_present():
    read = read_optional(JsonStore({"a/b.json": {"x": 1}}), "a/b.json")
    assert read.document == {"x": 1}
    assert not read.absent
    assert not read.denied
    assert read.reason is None


def test_a_missing_key_reads_as_absent_and_names_the_key():
    read = read_optional(JsonStore({}), "a/b.json")
    assert read.absent
    assert not read.denied
    assert read.denied_code is None
    assert "a/b.json" in read.reason


def test_a_denied_read_is_a_third_fact_never_folded_into_absent():
    # The live shape: GHA 33766008781, 2026-09-03T14:18Z, AccessDenied on an
    # OPTIONAL artifact. S3 answers 403 for a MISSING key when the caller also
    # lacks ListBucket, so absence is a listing, not a get.
    store = JsonStore({}, raises={"a/b.json": client_error("AccessDenied")})
    read = read_optional(store, "a/b.json")
    assert read.denied
    assert read.denied_code == "AccessDenied"
    assert not read.absent
    assert "AccessDenied" in read.reason


def test_a_not_found_coded_client_error_is_still_absent_not_denied():
    store = JsonStore({}, raises={"a/b.json": client_error("NoSuchKey")})
    read = read_optional(store, "a/b.json")
    assert read.absent
    assert not read.denied


def test_a_corrupt_document_is_unreadable_and_not_denied():
    store = RawBytesStore({"a/b.json": b"{not json"})
    read = read_optional(store, "a/b.json")
    assert read.document is None
    assert not read.denied
    assert "unreadable" in read.reason


def test_an_array_bodied_document_is_unreadable_not_a_document():
    store = RawBytesStore({"a/b.json": b"[1, 2]"})
    read = read_optional(store, "a/b.json")
    assert read.document is None
    assert not read.denied


def test_a_get_bytes_only_store_is_accepted():
    read = read_optional(BytesStore({"a/b.json": {"x": 1}}), "a/b.json")
    assert read.document == {"x": 1}


def test_a_non_botocore_exception_propagates_rather_than_being_classified():
    store = JsonStore({}, raises={"a/b.json": RuntimeError("a defect in this job")})
    with pytest.raises(RuntimeError, match="a defect in this job"):
        read_optional(store, "a/b.json")


def test_read_required_raises_so_the_manifest_reads_failed():
    with pytest.raises(KeyError):
        read_required(JsonStore({}), "a/b.json")


def test_read_required_returns_the_document_when_it_is_there():
    assert read_required(JsonStore({"a/b.json": {"x": 1}}), "a/b.json") == {"x": 1}


def test_the_read_record_is_frozen():
    with pytest.raises(AttributeError):
        Read(None, None, None).document = {}


# --------------------------------------------------------------------------
# staleness


def test_a_fresh_reading_has_no_headline():
    generated = (NOW - dt.timedelta(hours=2)).isoformat()
    assert staleness(generated_at=generated, now=NOW, stale_after=DAY, label="board") is None


def test_a_stale_reading_headlines_with_its_age():
    generated = (NOW - dt.timedelta(hours=50)).isoformat()
    line = staleness(generated_at=generated, now=NOW, stale_after=DAY, label="board")
    assert line.startswith("STALE")
    assert "50h ago" in line
    assert "board" in line


def test_an_unparseable_generated_at_is_stale_never_assumed_fresh():
    line = staleness(generated_at="yesterday-ish", now=NOW, stale_after=DAY, label="board")
    assert line is not None
    assert line.startswith("STALE")
    assert "is not a timestamp" in line


def test_a_missing_generated_at_is_stale_too():
    for value in (None, "", "   "):
        line = staleness(generated_at=value, now=NOW, stale_after=DAY, label="board")
        assert line is not None and line.startswith("STALE")


def test_a_naive_generated_at_is_read_as_utc_rather_than_crashing():
    generated = (NOW - dt.timedelta(hours=2)).replace(tzinfo=None).isoformat()
    assert staleness(generated_at=generated, now=NOW, stale_after=DAY, label="b") is None


def test_a_z_suffixed_timestamp_parses():
    generated = "2026-09-17T11:00:00Z"
    assert staleness(generated_at=generated, now=NOW, stale_after=DAY, label="b") is None


def test_a_naive_now_is_read_as_utc():
    generated = "2026-09-17T11:00:00Z"
    assert staleness(generated_at=generated, now=NOW.replace(tzinfo=None), stale_after=DAY, label="b") is None


# --------------------------------------------------------------------------
# moved_since


def _doc(**states):
    return {"rows": [{"id": k, "state": v} for k, v in states.items()]}


def test_a_changed_row_is_reported_old_to_new():
    result = moved_since(previous=_doc(a="UNMET"), current=_doc(a="MET"), previous_reason=None)
    assert result.lines == ["- a: UNMET -> MET"]
    assert result.count == 1
    assert result.cannot_say is None


def test_an_unchanged_board_reports_zero_movement_not_a_failure():
    result = moved_since(previous=_doc(a="MET"), current=_doc(a="MET"), previous_reason=None)
    assert result.lines == []
    assert result.count == 0
    assert result.cannot_say is None


def test_an_appearing_row_reads_absent_and_a_vanishing_row_reads_vanished():
    result = moved_since(previous=_doc(a="MET"), current=_doc(b="MET"), previous_reason=None)
    assert "- a: MET -> VANISHED" in result.lines
    assert "- b: ABSENT -> MET" in result.lines


def test_an_unreadable_previous_reading_is_never_nothing_moved():
    result = moved_since(previous=None, current=_doc(a="MET"), previous_reason="unreadable at x: AccessDenied")
    assert result.count is None
    assert result.cannot_say is not None
    assert "AccessDenied" in result.cannot_say
    assert "nothing moved" not in " ".join(result.lines)


def test_an_unreadable_previous_reading_with_no_reason_still_says_cannot_say():
    result = moved_since(previous=None, current=_doc(a="MET"), previous_reason=None)
    assert result.cannot_say is not None
    assert result.count is None


def test_custom_row_and_state_keys_are_honoured():
    previous = {"rows": [{"unit": "D33", "verdict": "RED"}]}
    current = {"rows": [{"unit": "D33", "verdict": "GREEN"}]}
    result = moved_since(
        previous=previous,
        current=current,
        previous_reason=None,
        row_id_key="unit",
        state_key="verdict",
    )
    assert result.lines == ["- D33: RED -> GREEN"]


# --------------------------------------------------------------------------
# withholding


TOKENS = frozenset({" percent ", " complete "})


def test_clean_text_passes_through_byte_for_byte():
    text = "three units read; two are green"
    assert filter_withheld_clauses(text, tokens=TOKENS) == text


def test_a_forbidden_clause_is_replaced_never_dropped():
    text = "two units read; 40 percent complete; one is green"
    out = filter_withheld_clauses(text, tokens=TOKENS, note="plan rule 3")
    assert "40 percent" not in out
    assert "[1 clause withheld — plan rule 3]" in out
    assert "two units read" in out
    assert "one is green" in out


def test_the_marker_pluralizes_and_counts():
    text = "40 percent complete; 60 percent complete; fine"
    out = filter_withheld_clauses(text, tokens=TOKENS)
    assert "[2 clauses withheld — policy]" in out


def test_a_wholly_withheld_detail_is_the_marker_alone_not_an_empty_string():
    out = filter_withheld_clauses("40 percent complete", tokens=TOKENS)
    assert out == "[1 clause withheld — policy]"


def test_a_custom_separator_is_honoured():
    out = filter_withheld_clauses("a | 40 percent complete", tokens=TOKENS, separator=" | ")
    assert out == "a | [1 clause withheld — policy]"


# --------------------------------------------------------------------------
# the wire budget


class TestTheWireBodyFitsTheCap:
    """The budget is pinned against krepis' REAL formatter, not a literal."""

    def test_the_prefix_matches_the_one_krepis_actually_prepends(self):
        from krepis.alerts import _format_message

        severity, source = "info", "data-collector/report.daily"
        theirs = _format_message("BODY", severity, source)
        ours = transport_prefix(severity=severity, source=source)
        assert theirs == f"{ours}BODY"

    def test_the_budget_is_the_cap_minus_that_prefix(self):
        from krepis.alerts import _format_message

        severity, source = "info", "data-collector/report.daily"
        prefix = len(_format_message("", severity, source))
        assert wire_budget(severity=severity, source=source, max_chars=4096) == 4096 - prefix

    def test_the_crucible_measurement_reproduces(self):
        # A 4117-character body POSTed at 4152 and was refused `400 message is
        # too long` — not an entity-parse error, so krepis' plain-text retry
        # never fired and the report was not delivered at all.
        #
        # The prefix is 35 characters, not the 36-or-37 the prose around that
        # incident has carried: 4117 + 35 = 4152, which is the number the wire
        # actually showed. Pinned here as the arithmetic rather than as a
        # remembered figure, which is the whole reason this budget is derived
        # from krepis' own formatter instead of restated as a literal.
        prefix = transport_prefix(severity="info", source="crucible-v2/report.morning")
        assert len(prefix) == 35
        assert 4117 + len(prefix) == 4152
        budget = wire_budget(severity="info", source="crucible-v2/report.morning", max_chars=4096)
        with pytest.raises(MessageTooLongError):
            assert_within_budget("x" * 4117, budget=budget)

    def test_an_over_budget_body_raises_and_is_never_truncated(self):
        with pytest.raises(MessageTooLongError) as excinfo:
            assert_within_budget("x" * 101, budget=100)
        assert "1 over" in str(excinfo.value)

    def test_a_body_exactly_at_budget_is_accepted(self):
        assert assert_within_budget("x" * 100, budget=100) is None


def test_wire_length_is_the_declared_unit_of_measure():
    assert wire_length("abc") == 3


def test_escape_delegates_to_the_one_escaper():
    from krepis.telegram import escape_html

    assert escape("<b>&</b>") == escape_html("<b>&</b>")


# --------------------------------------------------------------------------
# resolve_trigger


def test_github_event_name_is_read_first(monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("DATA_TRIGGER", "workflow_dispatch")
    assert resolve_trigger(override_var="DATA_TRIGGER") == "schedule"


def test_the_override_var_answers_when_github_says_nothing(monkeypatch):
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv("DATA_TRIGGER", "cron")
    assert resolve_trigger(override_var="DATA_TRIGGER") == "cron"


def test_an_unset_trigger_is_declared_unknown_not_guessed(monkeypatch):
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("DATA_TRIGGER", raising=False)
    assert resolve_trigger(override_var="DATA_TRIGGER") == TRIGGER_UNKNOWN
    assert TRIGGER_RE.match(TRIGGER_UNKNOWN) is not None


def test_a_trigger_that_is_not_key_safe_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule/../../etc")
    with pytest.raises(ValueError, match="not a usable trigger name"):
        resolve_trigger()


def test_github_event_name_leads_the_declared_order():
    assert TRIGGER_VARS[0] == "GITHUB_EVENT_NAME"


# --------------------------------------------------------------------------
# deliver


class Result:
    def __init__(self, any_ok=True, dedup_skipped=False, muted=False):
        self.any_ok = any_ok
        self.dedup_skipped = dedup_skipped
        self.muted = muted


def _capture(monkeypatch, result):
    calls = {}

    def fake(message, **kwargs):
        calls["message"] = message
        calls.update(kwargs)
        return result

    monkeypatch.setattr("nousergon_lib.gates.report._krepis_publish", fake)
    return calls


def test_delivery_is_never_silent_and_never_deduped(monkeypatch):
    # Brian ruling, alpha-engine-config-I9916: the first crucible delivery
    # went out silent, the manifest read ok, and he never saw it.
    calls = _capture(monkeypatch, Result())
    deliver("body", severity="info", source="src", console_artifact="runs/x.txt")
    assert calls["silent"] is False
    assert calls["dedup_key"] is None
    assert calls["sns"] is False
    assert calls["telegram"] is True
    assert calls["raise_on_total_failure"] is True
    assert calls["console_artifact"] == "runs/x.txt"
    assert calls["parse_mode"] == "HTML"


def test_the_destination_is_explicit_never_left_to_the_resolver(monkeypatch):
    from krepis.alerts import DESTINATION_OPERATOR_CHAT

    calls = _capture(monkeypatch, Result())
    deliver("body", severity="info", source="src", console_artifact="a")
    assert calls["destination"] == DESTINATION_OPERATOR_CHAT


def test_an_explicit_destination_wins(monkeypatch):
    calls = _capture(monkeypatch, Result())
    deliver("b", severity="info", source="s", console_artifact="a", destination="log_chat")
    assert calls["destination"] == "log_chat"


def test_the_override_var_redirects_delivery(monkeypatch):
    monkeypatch.setenv("DATA_DESTINATION", "console_only")
    calls = _capture(monkeypatch, Result())
    deliver(
        "b",
        severity="info",
        source="s",
        console_artifact="a",
        destination_override_var="DATA_DESTINATION",
    )
    assert calls["destination"] == "console_only"


def test_a_publish_that_reached_nobody_raises(monkeypatch):
    _capture(monkeypatch, Result(any_ok=False))
    with pytest.raises(UndeliveredError):
        deliver("b", severity="info", source="s", console_artifact="a")


def test_a_dedup_suppressed_publish_raises_although_krepis_calls_it_ok(monkeypatch):
    _capture(monkeypatch, Result(any_ok=True, dedup_skipped=True))
    with pytest.raises(UndeliveredError, match="dedup_skipped=True"):
        deliver("b", severity="info", source="s", console_artifact="a")


def test_a_muted_publish_raises_although_krepis_calls_it_ok(monkeypatch):
    _capture(monkeypatch, Result(any_ok=True, muted=True))
    with pytest.raises(UndeliveredError, match="muted=True"):
        deliver("b", severity="info", source="s", console_artifact="a")


def test_a_transport_result_that_cannot_say_is_not_recorded_as_delivered(monkeypatch):
    _capture(monkeypatch, object())
    with pytest.raises(TypeError, match="any_ok"):
        deliver("b", severity="info", source="s", console_artifact="a")


# --------------------------------------------------------------------------
# the history index


TITLES = ["units", "acceptance"]


def _row(day, url="https://x/1", **columns):
    return history_row(
        trading_day=day,
        delivered_local="2026-09-17 06:00 PDT",
        columns=columns or {"units": "3/3", "acceptance": "2/5"},
        update_url=url,
    )


def test_a_row_renders_its_cells_and_its_update_link():
    body = render_history_index(
        rows=[("runs/2026-09-17/history_row.json", _row("2026-09-17"), None)],
        column_titles=TITLES,
    )
    assert "| 2026-09-17 | 2026-09-17 06:00 PDT | 3/3 | 2/5 | [update](https://x/1) |" in body


def test_the_last_row_for_a_trading_day_wins():
    rows = [
        ("runs/2026-09-17/a/history_row.json", _row("2026-09-17", units="1/3"), None),
        ("runs/2026-09-17/b/history_row.json", _row("2026-09-17", units="3/3"), None),
    ]
    body = render_history_index(rows=rows, column_titles=TITLES)
    assert body.count("2026-09-17 |") == 1
    assert "3/3" in body
    assert "1/3" not in body


def test_days_render_newest_first():
    rows = [
        ("runs/2026-09-16/history_row.json", _row("2026-09-16"), None),
        ("runs/2026-09-17/history_row.json", _row("2026-09-17"), None),
    ]
    body = render_history_index(rows=rows, column_titles=TITLES)
    assert body.index("2026-09-17") < body.index("2026-09-16")


def test_a_corrupt_row_gets_its_own_unreadable_row_and_does_not_blank_the_rest():
    rows = [
        ("runs/2026-09-16/history_row.json", _row("2026-09-16"), None),
        ("runs/2026-09-17/history_row.json", None, "not valid JSON"),
    ]
    body = render_history_index(rows=rows, column_titles=TITLES)
    assert "| 2026-09-17 | unreadable: not valid JSON |" in body
    assert "2026-09-16" in body


def test_a_corrupt_row_replaces_that_days_earlier_good_row():
    rows = [
        ("runs/2026-09-17/a/history_row.json", _row("2026-09-17"), None),
        ("runs/2026-09-17/b/history_row.json", None, "vanished between listing and read"),
    ]
    body = render_history_index(rows=rows, column_titles=TITLES)
    assert "unreadable: vanished between listing and read" in body
    assert "[update]" not in body


def test_a_corrupt_row_with_no_stated_problem_still_says_unreadable():
    body = render_history_index(rows=[("runs/2026-09-17/history_row.json", None, None)], column_titles=TITLES)
    assert "unreadable:" in body


def test_a_corrupt_row_whose_key_carries_no_date_falls_back_to_the_key():
    body = render_history_index(rows=[("runs/latest.json", None, "bad")], column_titles=TITLES)
    assert "runs/latest.json" in body


def test_a_missing_cell_renders_a_dash_not_a_fabricated_value():
    row = _row("2026-09-17", units="3/3")
    body = render_history_index(rows=[("runs/2026-09-17/history_row.json", row, None)], column_titles=TITLES)
    assert "| 3/3 | — |" in body


def test_a_row_with_no_update_url_renders_a_dash_not_an_empty_link():
    row = _row("2026-09-17", url="")
    body = render_history_index(rows=[("runs/2026-09-17/history_row.json", row, None)], column_titles=TITLES)
    assert "[update]" not in body


def test_an_empty_history_says_so_rather_than_rendering_a_bare_header():
    body = render_history_index(rows=[], column_titles=TITLES)
    assert "no delivery has filed a history row yet" in body


def test_the_header_carries_every_declared_column():
    body = render_history_index(rows=[], column_titles=TITLES)
    assert "| trading day | delivered | units | acceptance | update |" in body


def test_history_row_stringifies_its_cells_so_the_index_never_re_derives_one():
    row = history_row(
        trading_day="2026-09-17",
        delivered_local="06:00",
        columns={"units": 3},
        update_url="https://x/1",
    )
    assert row["columns"] == {"units": "3"}
    assert row["trading_day"] == "2026-09-17"
