"""The gate engine's invariants — the ones paid for in incidents.

Every test here grades a property that a well-intentioned refactor would
otherwise be free to break: no-data is never green, an access failure is never
a finding about a producer, and a clause that raises never darkens the ladder.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from nousergon_lib.gates import (
    CLAUSE_MEMBER_RANK,
    LADDER_CONSOLE_STATE,
    LADDER_STATES,
    Clause,
    ClauseMisconfiguredError,
    GateResult,
    Phase,
    build_ladder,
    clause_member_status,
    contain_clause_exceptions,
    contained,
    fault_excused_run_ids,
    gate_key,
    gate_prefix,
    gate_state_for,
    ladder_payload,
    ladder_schema,
    last_read,
    list_store_keys,
    read_store_document,
    unmeasurable,
    validate_ladder_document,
)

DAY = dt.date(2026, 9, 14)


class FakeStore:
    """The two reads a gate takes, over a dict. Optionally denied."""

    def __init__(self, objects=None, *, deny=False):
        self.objects = dict(objects or {})
        self.deny = deny

    def get_bytes(self, key):
        if self.deny:
            raise PermissionError("AccessDenied")
        try:
            return self.objects[key]
        except KeyError:
            raise FileNotFoundError(key) from None

    def list_keys(self, prefix=""):
        if self.deny:
            raise PermissionError("AccessDenied")
        return [k for k in sorted(self.objects) if k.startswith(prefix)]


def _phase(gate=None):
    return Phase.on_alpha_engine_config(
        id="data-phase0", number=0, title="Instrument the board", issue=10748, gate=gate
    )


# --------------------------------------------------------------------------
# Clause state machine
# --------------------------------------------------------------------------


def test_unmeasurable_is_never_met():
    clause = unmeasurable("x", "x holds", "the bucket said AccessDenied")
    assert clause.met is False
    assert clause.unmeasurable is True
    assert clause_member_status(clause) == "UNMEASURABLE"


def test_unmeasurable_ranks_worse_than_unmet():
    # "we could not read this" must never render better than "it said no".
    assert CLAUSE_MEMBER_RANK["UNMEASURABLE"] > CLAUSE_MEMBER_RANK["UNMET"]
    assert CLAUSE_MEMBER_RANK["UNMET"] > CLAUSE_MEMBER_RANK["MET"]


def test_clause_to_dict_carries_the_console_four_fields():
    clause = Clause(
        "a", "a holds", True, "read", ("k1", "k0"), phase="data-phase0", source="descriptor", as_of="2026-09-14"
    )
    payload = clause.to_dict()
    assert payload["evidence"] == ["k0", "k1"]
    assert payload["phase"] == "data-phase0"
    assert payload["source"] == "descriptor"
    assert payload["as_of"] == "2026-09-14"


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------


def test_contained_turns_a_raising_clause_into_unmeasurable():
    @contained
    def _clause_boom():
        raise RuntimeError("NoRegionError")

    clause = _clause_boom()
    assert clause.unmeasurable is True
    assert clause.met is False
    assert "RuntimeError" in clause.detail
    assert clause.name == "boom"


def test_contained_reraises_a_misconfigured_clause():
    @contained
    def _clause_bad_args():
        raise ClauseMisconfiguredError("minimum > maximum")

    with pytest.raises(ClauseMisconfiguredError):
        _clause_bad_args()


def test_contain_clause_exceptions_refuses_to_wrap_nothing():
    with pytest.raises(ClauseMisconfiguredError):
        contain_clause_exceptions({"not_a_clause": lambda: None})


def test_contain_clause_exceptions_is_idempotent():
    namespace = {"_clause_a": lambda: Clause("a", "a", True, "")}
    assert contain_clause_exceptions(namespace) == 1
    with pytest.raises(ClauseMisconfiguredError):
        contain_clause_exceptions(namespace)


# --------------------------------------------------------------------------
# Store reads: absence, malformed, and access failure are three answers
# --------------------------------------------------------------------------


def test_absent_key_is_an_answer_not_a_problem():
    read = read_store_document(FakeStore(), "missing.json")
    assert read.absent is True
    assert read.problem is None


def test_denied_read_is_a_problem_not_an_absence():
    read = read_store_document(FakeStore(deny=True), "anything.json")
    assert read.absent is False
    assert read.problem is not None
    assert read.access_problem is True


def test_malformed_payload_is_a_problem_but_not_an_access_problem():
    read = read_store_document(FakeStore({"x.json": b"not json"}), "x.json")
    assert read.problem is not None
    assert read.access_problem is False


def test_failed_listing_is_not_an_empty_listing():
    listed = list_store_keys(FakeStore(deny=True), "gates/")
    assert listed.keys is None
    assert listed.problem is not None


# --------------------------------------------------------------------------
# GateResult
# --------------------------------------------------------------------------


def test_empty_gate_ratio_is_none_never_zero_or_one():
    reading = GateResult(gate="g", trading_day=DAY)
    assert reading.met_ratio is None
    assert reading.met is False


def test_one_unmeasurable_clause_makes_the_ratio_none():
    reading = GateResult(
        gate="g",
        trading_day=DAY,
        clauses=[Clause("a", "a", True, ""), unmeasurable("b", "b", "denied")],
    )
    assert reading.met_ratio is None
    assert gate_state_for(reading) == "UNMEASURABLE"


def test_gate_state_unmeasurable_outranks_unmet():
    reading = GateResult(
        gate="g",
        trading_day=DAY,
        clauses=[Clause("a", "a", False, ""), unmeasurable("b", "b", "denied")],
    )
    assert gate_state_for(reading) == "UNMEASURABLE"


def test_members_cannot_drift_from_clauses():
    reading = GateResult(
        gate="g",
        trading_day=DAY,
        clauses=[Clause("a", "a", True, ""), Clause("b", "b", False, "")],
    )
    assert [m["status"] for m in reading.members] == ["MET", "UNMET"]
    assert reading.met is False


def test_earliest_satisfiable_is_the_max_over_clauses():
    reading = GateResult(
        gate="g",
        trading_day=DAY,
        clauses=[
            Clause("a", "a", False, "", earliest_satisfiable=dt.date(2026, 9, 20)),
            Clause("b", "b", False, "", earliest_satisfiable=dt.date(2026, 10, 1)),
        ],
    )
    assert reading.earliest_satisfiable == dt.date(2026, 10, 1)
    assert reading.earliest_satisfiable_clause == "b"


# --------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------


def test_every_ladder_state_declares_a_console_rendering():
    assert set(LADDER_STATES) == set(LADDER_CONSOLE_STATE)
    # The one that matters: no reading renders green.
    assert LADDER_CONSOLE_STATE["UNMEASURED"] == "UNREPORTED"
    assert LADDER_CONSOLE_STATE["UNMEASURABLE"] == "FAILED"


def test_a_phase_with_no_gate_is_unmeasured_never_met():
    ladder = build_ladder(
        FakeStore(),
        phases=[_phase(gate=None)],
        trading_day=DAY,
        evaluate=lambda gate: pytest.fail("evaluate must not be called for an ungated phase"),
    )
    row = ladder.rows[0]
    assert row.state == "UNMEASURED"
    assert row.clauses_total is None
    assert row.met_ratio is None
    assert ladder.current_phase == "data-phase0"


def test_a_gate_with_no_clauses_is_unmeasured_never_vacuously_met():
    ladder = build_ladder(
        FakeStore(),
        phases=[_phase(gate="data-phase0")],
        trading_day=DAY,
        evaluate=lambda gate: GateResult(gate=gate, trading_day=DAY),
    )
    assert ladder.rows[0].state == "UNMEASURED"
    assert ladder.rows[0].met_ratio is None


def test_withholding_the_store_renders_unmeasurable_never_green():
    """The commissioning property: a denied store never produces a met ladder."""
    ladder = build_ladder(
        FakeStore(deny=True),
        phases=[_phase(gate="data-phase0")],
        trading_day=DAY,
        evaluate=lambda gate: GateResult(gate=gate, trading_day=DAY, clauses=[unmeasurable("a", "a", "AccessDenied")]),
    )
    row = ladder.rows[0]
    assert row.state == "UNMEASURABLE"
    assert LADDER_CONSOLE_STATE[row.state] == "FAILED"
    assert row.met_ratio is None


def test_last_read_distinguishes_never_read_from_unreadable():
    assert last_read(FakeStore(), "data-phase0") == (None, False)
    assert last_read(FakeStore(deny=True), "data-phase0") == (None, True)
    store = FakeStore({gate_key("data-phase0", "2026-09-11"): b"{}", gate_key("data-phase0", "2026-09-12"): b"{}"})
    assert last_read(store, "data-phase0") == ("2026-09-12", False)


def test_gate_prefix_matches_gate_key():
    assert gate_key("g", "2026-09-14").startswith(gate_prefix("g"))


def test_out_of_order_when_a_later_phase_is_graded_ahead():
    early = Phase.on_alpha_engine_config(id="p0", number=0, title="p0", issue=1, gate="g0")
    late = Phase.on_alpha_engine_config(id="p1", number=1, title="p1", issue=2, gate="g1")
    store = FakeStore({gate_key("g1", "2026-09-14"): b"{}"})
    readings = {
        "g0": GateResult(gate="g0", trading_day=DAY, clauses=[Clause("a", "a", False, "no")]),
        "g1": GateResult(gate="g1", trading_day=DAY, clauses=[Clause("b", "b", True, "yes")]),
    }
    ladder = build_ladder(
        store,
        phases=[early, late],
        trading_day=DAY,
        evaluate=lambda gate: readings[gate],
        readings=readings,
    )
    assert ladder.out_of_order == ["p1"]
    assert ladder.rows[1].blocked_by == "p0"
    assert LADDER_CONSOLE_STATE[ladder.rows[1].state] == "FAILED"


def test_ladder_payload_validates_against_the_shipped_schema():
    readings = {"g0": GateResult(gate="g0", trading_day=DAY, clauses=[Clause("a", "a", True, "ok")])}
    ladder = build_ladder(
        FakeStore(),
        phases=[Phase.on_alpha_engine_config(id="p0", number=0, title="p0", issue=1, gate="g0")],
        trading_day=DAY,
        evaluate=lambda gate: readings[gate],
        name="data",
    )
    document = json.loads(ladder_payload(ladder))
    assert document["schema_version"] == "phase_ladder.v1"
    assert document["ladder"] == "data"
    assert document["phases"][0]["console_state"] == "HEALTHY"


def test_a_malformed_ladder_is_refused_before_it_reaches_the_store():
    with pytest.raises(ValueError):
        validate_ladder_document({"schema_version": "phase_ladder.v1"})


def test_the_shipped_schema_forbids_unknown_fields():
    schema = ladder_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["PhaseLadderRow"]["additionalProperties"] is False


def test_a_crucible_shaped_row_still_validates_here():
    """The lift is a strict generalization: crucible's own document must pass."""
    document = {
        "schema_version": "phase_ladder.v1",
        "trading_day": "2026-09-14",
        "generated_utc": "2026-09-14T00:00:00Z",
        "current_phase": "phase3",
        "phases_total": 1,
        "phases_met": 0,
        "unmeasured": 0,
        "out_of_order": [],
        "phases": [
            {
                "decision_id": "alpha-engine-config-I10600",
                "phase": "phase3",
                "number": 3,
                "title": "Autonomy",
                "tracker": "alpha-engine-config-I10600",
                "tracker_url": "https://github.com/nousergon/alpha-engine-config/issues/10600",
                "gate": "phase3",
                "state": "UNMET",
                "console_state": "DEGRADED",
                "gate_state": "UNMET",
                "detail": "3/6 clauses met",
                "clauses_met": 3,
                "clauses_total": 6,
                "clauses_unmeasurable": 0,
                "met_ratio": 0.5,
                "read_on": "2026-09-13",
                "blocked_by": None,
                "generated_utc": "2026-09-14T00:00:00Z",
            }
        ],
    }
    validate_ladder_document(document)


# --------------------------------------------------------------------------
# Fault exclusion — the one mechanism that can turn a red clause green
# --------------------------------------------------------------------------


def _fault(run_id, outcome):
    return json.dumps({"run_id": run_id, "outcome": outcome}).encode("utf-8")


def test_only_an_induced_record_excuses_a_run():
    store = FakeStore(
        {
            "faults/2026-09-11/a.json": _fault("run-a", "induced"),
            "faults/2026-09-11/b.json": _fault("run-b", "absorbed"),
            "faults/2026-09-11/c.json": _fault("run-c", "unreachable"),
        }
    )
    excused, problem = fault_excused_run_ids(store, "faults/")
    assert problem is None
    assert excused == frozenset({"run-a"})


def test_a_malformed_fault_record_excuses_nothing():
    store = FakeStore({"faults/2026-09-11/a.json": b"{not json", "faults/2026-09-11/b.json": _fault("", "induced")})
    excused, problem = fault_excused_run_ids(store, "faults/")
    assert problem is None
    assert excused == frozenset()


def test_an_unreadable_fault_listing_is_none_not_an_empty_set():
    excused, problem = fault_excused_run_ids(FakeStore(deny=True), "faults/")
    assert excused is None
    assert problem is not None
