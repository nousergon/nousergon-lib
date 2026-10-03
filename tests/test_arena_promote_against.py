"""``ArenaConfig.promote_against`` — who a promotable challenger must beat.

Brian's ruling of 2026-10-03 (`alpha-engine-config#11849`), verbatim:

    "All arms should be compared each week, performance tracked, and if after
    minimum two weeks an arm outperforms the champion and all other
    challengers then it gets promoted. Otherwise we compare the common window
    of weeks for each arm in making our comparison."

``every_arm`` encodes it: a challenger takes the pointer only when it leads
the incumbent on a paired window of at least ``promote_min_weeks`` AND beats
every other eligible challenger head to head, each pair on its own longest
common window. ``incumbent`` — the default — is every slot's behaviour before
the field existed and must stay byte-identical.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from nousergon_lib.arena import (
    PROMOTE_AGAINST_EVERY_ARM,
    PROMOTE_AGAINST_INCUMBENT,
    ArenaConfig,
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    PointerDecision,
    ServingPrecondition,
    decide_pointer,
    run_cycle,
)
from nousergon_lib.arena.engine import ArenaConfigError
from nousergon_lib.arena.ranking import EVIDENCE_ANYTIME_VALID, EVIDENCE_POINT

AS_OF = "2026-10-03"


def _dates(n, end=AS_OF):
    d_end = date.fromisoformat(end)
    return [(d_end - timedelta(days=(n - 1 - i) * 7)).isoformat() for i in range(n)]


def _weekly(arm_id, values):
    """``values`` on a weekly cadence ending at ``AS_OF`` (newest last)."""
    return ArmSeries(arm_id=arm_id, scores=dict(zip(_dates(len(values)), values)))


def _config(**kw):
    base = {
        "slot": "m",
        "slot_kind": "model",
        "diff_clip": 0.05,
        "promote_evidence": EVIDENCE_POINT,
        "promote_min_weeks": 2,
        "promote_against": PROMOTE_AGAINST_EVERY_ARM,
    }
    base.update(kw)
    return ArenaConfig(**base)


def _cross_window_fixture():
    """The case the ruling exists for.

    ``b`` registered two weeks after ``a``. Against the incumbent, ``b``'s lead
    (+0.035 over its 2 weeks) is LARGER than ``a``'s (+0.030 over 4 weeks) —
    but those leads sit on different windows. On the two weeks the
    challengers actually share, ``a`` beats ``b`` (0.000 vs -0.005).
    """
    return {
        "inc": _weekly("inc", [0.00, 0.00, -0.04, -0.04]),
        "a": _weekly("a", [0.02, 0.02, 0.00, 0.00]),
        "b": _weekly("b", [-0.005, -0.005]),
    }


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_promote_against_defaults_to_the_incumbent():
    """A default of `every_arm` would change every slot that never declared it."""
    assert ArenaConfig(slot="r", slot_kind="model").promote_against == PROMOTE_AGAINST_INCUMBENT


def test_an_unknown_rule_is_refused():
    with pytest.raises(ArenaConfigError, match="promote_against"):
        _config(promote_against="everyone")


def test_every_arm_with_the_anytime_valid_sequence_is_refused():
    """The ruling is defined on the point estimate. The anytime-valid version
    across several simultaneous head-to-heads is a different false-promotion
    question no slot has been ruled onto, so it is refused, not guessed."""
    with pytest.raises(ArenaConfigError, match="every_arm"):
        _config(promote_evidence=EVIDENCE_ANYTIME_VALID)


def test_the_rule_in_force_is_emitted_and_reconstructed():
    config = _config()
    payload = config.to_dict()
    assert payload["promote_against"] == PROMOTE_AGAINST_EVERY_ARM
    restored = ArenaConfig.from_dict(payload, slot="m", slot_kind="model", benchmark="population")
    assert restored.promote_against == PROMOTE_AGAINST_EVERY_ARM


def test_a_config_recorded_before_the_field_existed_reads_as_incumbent():
    """§2.1: a cycle written before `promote_against` existed was decided
    against the incumbent alone, and must be reconstructed that way."""
    payload = _config().to_dict()
    del payload["promote_against"]
    restored = ArenaConfig.from_dict(payload, slot="m", slot_kind="model", benchmark="population")
    assert restored.promote_against == PROMOTE_AGAINST_INCUMBENT


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def test_incumbent_mode_picks_the_bigger_lead_on_a_different_window():
    """Pins TODAY's `point` behaviour, which the ruling changes: leads on two
    different windows are compared as if they were one, so ``b`` wins although
    ``a`` beats it on every week the two share."""
    decision = decide_pointer(
        _config(promote_against=PROMOTE_AGAINST_INCUMBENT), AS_OF, "inc", _cross_window_fixture()
    )
    assert decision.champion == "b"
    assert decision.rivals == ()


def test_every_arm_promotes_the_arm_that_beats_the_champion_and_every_other_challenger():
    decision = decide_pointer(_config(), AS_OF, "inc", _cross_window_fixture())
    assert decision.champion == "a"
    assert decision.moved
    assert decision.status == "decided"
    assert "promote_against=every_arm" in decision.reason
    assert "outperforms every other eligible challenger" in decision.reason
    by_arm = {c.challenger: c for c in decision.comparisons}
    assert "not ahead of a" in by_arm["b"].reason
    assert "outperforms every other eligible challenger" in by_arm["a"].reason


def test_the_head_to_head_is_on_the_pair_s_own_common_window():
    """"Otherwise we compare the common window of weeks for each arm": the
    a-vs-b verdict rests on the two weeks both produced, never on a's four."""
    decision = decide_pointer(_config(), AS_OF, "inc", _cross_window_fixture())
    (verdict,) = decision.rivals
    assert {verdict.arm_a, verdict.arm_b} == {"a", "b"}
    assert verdict.window.n_dates == 2
    assert verdict.window.weeks == 2
    assert verdict.window.start_date == _dates(2)[0]
    assert verdict.winner == "a"


def test_every_pair_of_challengers_is_compared_and_carries_its_bound():
    """"All arms should be compared each week, performance tracked." Every
    head-to-head is on the record with its confidence sequence, which is
    computed although it does not decide."""
    series = {
        "inc": _weekly("inc", [0.0] * 4),
        "a": _weekly("a", [0.01] * 4),
        "b": _weekly("b", [0.02] * 4),
        "c": _weekly("c", [0.03] * 4),
        "d": _weekly("d", [-0.01] * 4),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    assert len(decision.rivals) == 6  # 4 challengers, C(4, 2)
    assert all(v.bound is not None for v in decision.rivals)
    assert decision.champion == "c"


def test_rivals_are_recorded_on_a_hold_too():
    series = {
        "inc": _weekly("inc", [0.05] * 4),
        "a": _weekly("a", [0.01] * 4),
        "b": _weekly("b", [0.02] * 4),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    assert decision.status == "held"
    assert decision.champion == "inc"
    assert len(decision.rivals) == 1


def test_beating_the_champion_is_not_enough_when_a_rival_beats_you():
    """``a`` is the only age-eligible leader, but ``young`` — one week old —
    beats it on the week they share. Read literally, ``a`` has not
    outperformed all other challengers, so the incumbent holds, and the
    reason names the arm that held it."""
    series = {
        "inc": _weekly("inc", [0.00, 0.00, 0.00]),
        "a": _weekly("a", [0.01, 0.01, 0.01]),
        "young": _weekly("young", [0.09]),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    assert decision.champion == "inc"
    assert decision.status == "held"
    assert "a leads the incumbent but is not ahead of young" in decision.reason
    assert "Below promote_min_weeks=2: young: 1 paired week(s)" in decision.reason


def test_a_tie_with_a_rival_is_not_outperforming_it():
    series = {
        "inc": _weekly("inc", [0.00] * 3),
        "a": _weekly("a", [0.02] * 3),
        "b": _weekly("b", [0.02] * 3),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    assert decision.status == "held"
    assert decision.champion == "inc"
    (verdict,) = decision.rivals
    assert verdict.winner is None
    assert "tied" in verdict.reason


def test_a_condorcet_cycle_among_the_leaders_holds_the_incumbent():
    """No arm outperforms all the others, so none is promoted — no
    tie-break is invented. Each pair overlaps on a different slice."""
    d = _dates(3)
    series = {
        "inc": ArmSeries("inc", {d[0]: 0.0, d[1]: 0.0, d[2]: 0.0}),
        # a beats b on d0..d1; b beats c on d0, d2; c beats a on d0.
        "a": ArmSeries("a", {d[0]: 0.03, d[1]: 0.02}),
        "b": ArmSeries("b", {d[0]: 0.01, d[1]: 0.01, d[2]: 0.04}),
        "c": ArmSeries("c", {d[0]: 0.04, d[2]: 0.00}),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    winners = {(v.winner, v.loser) for v in decision.rivals}
    assert winners == {("a", "b"), ("b", "c"), ("c", "a")}, "fixture must be a cycle"
    assert decision.status == "held"
    assert decision.champion == "inc"


def test_a_rival_with_no_shared_window_does_not_block():
    """No comparison exists for the arm to have lost; an absent comparison is
    recorded, never scored as a loss."""
    d = _dates(4)
    series = {
        "inc": ArmSeries("inc", dict.fromkeys(d, 0.0)),
        "a": ArmSeries("a", {d[0]: 0.01, d[1]: 0.01}),
        "b": ArmSeries("b", {d[2]: 0.005, d[3]: 0.005}),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    (verdict,) = decision.rivals
    assert not verdict.measurable
    assert "common_window_too_short" in verdict.reason
    assert decision.champion == "a"


def test_the_age_bar_still_binds_the_candidate_against_the_incumbent():
    """"After minimum two weeks": the promoted arm's paired window with the
    incumbent must reach `promote_min_weeks`, however far ahead it is."""
    series = {
        "inc": _weekly("inc", [0.0, 0.0, 0.0]),
        "young": _weekly("young", [0.09]),
    }
    decision = decide_pointer(_config(), AS_OF, "inc", series)
    assert decision.champion == "inc"
    assert decision.status == "held"


# --------------------------------------------------------------------------
# Serving preconditions are untouched
# --------------------------------------------------------------------------


def test_a_vetoed_arm_is_neither_a_candidate_nor_a_rival():
    """An arm that may not serve cannot hold the pointer, so it cannot keep
    another arm off it — and it never serves, however far ahead."""
    series = {
        "inc": _weekly("inc", [0.0] * 4),
        "a": _weekly("a", [0.01] * 4),
        "vetoed": _weekly("vetoed", [0.09] * 4),
    }
    decision = decide_pointer(
        _config(),
        AS_OF,
        "inc",
        series,
        preconditions={"vetoed": [ServingPrecondition("behavioural_veto", False, "zero high-confidence names")]},
    )
    assert decision.champion == "a"
    assert "vetoed" in decision.ineligible
    assert decision.rivals == ()


def test_a_vetoed_incumbent_still_forces_the_pointer_to_move():
    series = {
        "inc": _weekly("inc", [0.09] * 4),
        "a": _weekly("a", [0.01] * 4),
        "b": _weekly("b", [0.02] * 4),
    }
    decision = decide_pointer(
        _config(),
        AS_OF,
        "inc",
        series,
        preconditions={"inc": [ServingPrecondition("behavioural_veto", False, "collapsed")]},
    )
    assert decision.moved
    assert decision.champion == "b"


def test_every_arm_vetoed_is_still_unservable():
    series = {"inc": _weekly("inc", [0.0] * 3), "a": _weekly("a", [0.01] * 3)}
    veto = [ServingPrecondition("behavioural_veto", False, "collapsed")]
    decision = decide_pointer(_config(), AS_OF, "inc", series, preconditions={"inc": veto, "a": veto})
    assert decision.status == "unservable"
    assert decision.champion is None


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_the_decision_round_trips_with_its_rivals():
    decision = decide_pointer(_config(), AS_OF, "inc", _cross_window_fixture())
    payload = decision.to_dict()
    assert len(payload["rivals"]) == 1
    assert PointerDecision.from_dict(payload).to_dict() == payload


def test_a_decision_recorded_before_rivals_existed_reads_as_none():
    payload = decide_pointer(
        _config(promote_against=PROMOTE_AGAINST_INCUMBENT), AS_OF, "inc", _cross_window_fixture()
    ).to_dict()
    del payload["rivals"]
    assert PointerDecision.from_dict(payload).rivals == ()


def _register(names):
    reg = ArmRegister()
    ids = {}
    for name in names:
        reg, record = reg.register(
            slot="m", name=name, spec={"recipe": name}, created_date="2026-08-01", filed_on="2026-08-01"
        )
        ids[name] = record.arm_id
    return reg, ids


def test_a_whole_cycle_under_every_arm_conforms_and_round_trips():
    pytest.importorskip("jsonschema")
    from nousergon_lib import contracts

    reg, ids = _register(["inc", "a", "b"])
    raw = _cross_window_fixture()
    series = {ids[k]: ArmSeries(ids[k], dict(v.scores)) for k, v in raw.items()}
    cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["inc"])
    assert cycle.decision.champion == ids["a"]
    payload = cycle.to_dict()
    assert payload["config"]["promote_against"] == PROMOTE_AGAINST_EVERY_ARM
    assert contracts.conformance_errors("arena_cycle", payload) == []
    assert ArenaCycle.from_dict(payload).to_dict() == payload


def test_incumbent_mode_records_no_rivals_and_does_not_name_the_rule():
    """Slots that never declared the rule — every `anytime_valid` slot among
    them — must decide and render exactly as before it existed."""
    series = {"inc": _weekly("inc", [0.0] * 60), "chal": _weekly("chal", [0.03] * 60)}
    decision = decide_pointer(
        ArenaConfig(slot="r", slot_kind="model", diff_clip=0.05), AS_OF, "inc", series
    )
    assert decision.champion == "chal"
    assert decision.rivals == ()
    assert "promote_against" not in decision.reason
    assert all("promote_against" not in c.reason for c in decision.comparisons)
