"""Tests for nousergon_lib.arena — the shared champion/challenger engine.

Normative source: nous-ergon-ops/policies/champion-challenger-policy.md.
Every test below either encodes one of Brian's 2026-08-29 rulings or pins a
defect measured in production; the docstring names which, so a future reader
knows what a failure means before reading the assertion.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from nousergon_lib.arena import (
    ArenaConfig,
    ArmRegister,
    ArmSeries,
    ImmutableArmError,
    ServingPrecondition,
    TrainingIntegrityError,
    TrainingStatus,
    build_ladder,
    confidence_sequence,
    decide_pointer,
    derive_arm_id,
    evaluate_retirements,
    pair_on_common_window,
    rank_pairwise,
    run_cycle,
)
from nousergon_lib.arena.engine import ArenaConfigError
from nousergon_lib.arena.ranking import (
    EVIDENCE_ANYTIME_VALID,
    ArmStanding,
    PairwiseRanking,
)
from nousergon_lib.arena.selection import rank_by_alpha, rank_to_score

AS_OF = "2026-08-29"


def _dates(n, step=7, end=AS_OF):
    """``n`` observation dates on a ``step``-day cadence, ENDING at ``end``.

    Anchored at the end rather than the start so a fixture's history always
    runs up to the evaluation date — a series that stops months before
    ``as_of`` is a legitimate but different scenario, and it is tested
    explicitly rather than arrived at by accident.
    """
    d_end = date.fromisoformat(end)
    return [(d_end - timedelta(days=(n - 1 - i) * step)).isoformat() for i in range(n)]


def _series(arm_id, values, step=7, misses=()):
    ds = _dates(len(values), step=step)
    return ArmSeries(arm_id=arm_id, scores=dict(zip(ds, values)), misses=frozenset(misses))


def _config(**kw):
    base = {"slot": "model", "slot_kind": "model", "diff_clip": 0.05}
    base.update(kw)
    return ArenaConfig(**base)


def _register(arms, created="2026-01-05"):
    """Register ``arms`` as {name: created_date} and return (register, ids)."""
    reg = ArmRegister()
    ids = {}
    for name, created_date in arms.items():
        reg, record = reg.register(
            slot="model", name=name, spec={"recipe": name}, created_date=created_date
        )
        ids[name] = record.arm_id
    return reg, ids


# --------------------------------------------------------------------------
# §4 — longest common window, paired per date
# --------------------------------------------------------------------------


def test_comparison_uses_only_the_intersection_of_both_arms_dates():
    """§4 regression, measured 2026-08-29.

    An incumbent scored over 2 dates successfully defended against a
    challenger rejected for having 4 — bases 3.6x apart. The engine must
    compare on the 2 dates they SHARE, and must report that count.
    """
    all_dates = _dates(4)
    incumbent = ArmSeries("inc", {all_dates[2]: 0.01, all_dates[3]: 0.01})
    challenger = ArmSeries("chal", dict.fromkeys(all_dates, 0.02))

    window = pair_on_common_window(challenger, incumbent)

    assert window.n_dates == 2
    assert window.dates == (all_dates[2], all_dates[3])
    assert window.mean_diff == pytest.approx(0.01)
    # The intersection is reported alongside the metric, not inferable.
    assert window.to_dict()["n_dates"] == 2
    assert window.to_dict()["start_date"] == all_dates[2]


def test_no_common_window_is_unmeasurable_with_a_reason_never_a_tie():
    """§7.2 — an unmeasurable result fails loud; it never renders as an empty pass."""
    a = ArmSeries("a", {"2026-01-05": 0.01})
    b = ArmSeries("b", {"2026-03-02": 0.01})

    window = pair_on_common_window(a, b)

    assert not window.measurable
    assert "common_window_too_short" in window.unmeasurable_reason
    with pytest.raises(ValueError, match="unmeasurable"):
        _ = window.mean_diff


def test_an_arm_cannot_be_paired_against_itself():
    """§4 — the 2026-08-28 promotion gate compared a model to itself and always promoted."""
    a = ArmSeries("a", {"2026-01-05": 0.01})
    with pytest.raises(ValueError, match="against itself"):
        pair_on_common_window(a, a)


def test_a_nan_score_is_refused_because_a_missing_score_is_a_miss():
    """§3 — silent absence and a genuine zero must never render identically."""
    with pytest.raises(ValueError, match="NaN"):
        ArmSeries("a", {"2026-01-05": float("nan")})


def test_a_date_cannot_be_both_scored_and_missed():
    with pytest.raises(ValueError, match="BOTH scored and missed"):
        ArmSeries("a", {"2026-01-05": 0.01}, misses=frozenset({"2026-01-05"}))


# --------------------------------------------------------------------------
# Anytime-valid confidence sequence
# --------------------------------------------------------------------------


def test_confidence_sequence_is_wide_at_one_observation_so_no_lead_is_supported():
    """The sequence subsumes minimum-evidence floors — that is why they are removed."""
    bound = confidence_sequence([0.04], clip=0.05)
    assert not bound.supported
    assert bound.lower < 0 < bound.upper


def test_confidence_sequence_narrows_and_eventually_supports_a_real_lead():
    bound_short = confidence_sequence([0.03] * 5, clip=0.05)
    bound_long = confidence_sequence([0.03] * 200, clip=0.05)
    assert bound_long.radius < bound_short.radius
    assert not bound_short.supported
    assert bound_long.supported


def test_confidence_sequence_covers_uniformly_over_repeated_weekly_looks():
    """Anytime validity, empirically.

    52 looks a year at a nominal 5% drifts toward ~40% false positives with a
    naive fixed-sample test. The whole point of the sequence is that peeking
    every cycle does not inflate that rate. Under a true null (mean 0), the
    fraction of runs in which ANY look excludes zero must stay well under
    alpha.
    """
    rng = random.Random(20260829)
    alpha = 0.05
    clip = 0.05
    breaches = 0
    runs = 300
    for _ in range(runs):
        diffs = []
        breached = False
        for _ in range(52):
            diffs.append(rng.uniform(-clip, clip))  # true mean 0
            bound = confidence_sequence(diffs, alpha=alpha, clip=clip)
            if bound.lower > 0 or bound.upper < 0:
                breached = True
                break
        breaches += int(breached)
    assert breaches / runs <= alpha, "time-uniform coverage violated"


def test_declared_variance_mode_refuses_to_run_without_a_declared_bound():
    """Validity must be checkable from configuration alone."""
    with pytest.raises(ValueError, match="requires an explicit `clip`"):
        confidence_sequence([0.01, 0.02], clip=None, variance_mode="declared")


def test_confidence_sequence_refuses_an_empty_window():
    with pytest.raises(ValueError, match="unmeasurable"):
        confidence_sequence([], clip=0.05)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_ladder_carries_every_horizon_and_the_longest_is_the_full_history():
    """Brian 2026-08-29: "each arm should have 1, 2, 3, 4, 52, 53, 54 etc week scores"."""
    series = _series("a", [0.01] * 10)  # 10 weekly observations
    ladder = build_ladder(series, AS_OF)

    weeks = [rung.weeks for rung in ladder.rungs]
    assert weeks == sorted(weeks)
    assert weeks[0] == 1
    assert ladder.longest.weeks == max(weeks) == ladder.total_weeks
    assert ladder.longest.n_dates == 10


def test_ladder_exposes_no_best_rung_selector():
    """Picking the best-looking horizon converts a pre-registered statistic into a
    search over 52 horizons. That IS mining, and the DSR/PBO battery does not
    deflate for it. The absence of such a selector is the guard."""
    import nousergon_lib.arena.ladder as ladder_module

    assert not hasattr(ladder_module, "best_rung")
    ladder = build_ladder(_series("a", [0.01] * 5), AS_OF)
    assert not any("best" in name for name in dir(ladder))


# --------------------------------------------------------------------------
# Immutability (Brian ruling 3) and the recipe/refit distinction
# --------------------------------------------------------------------------


def test_a_changed_spec_cannot_reuse_an_arm_id():
    """Ruling: "if we make a change the updated version becomes a challenger arm"."""
    first = derive_arm_id("model", "zoo", {"depth": 3})
    second = derive_arm_id("model", "zoo", {"depth": 4})
    assert first != second


def test_spec_hash_ignores_key_order_so_a_reformat_is_not_a_new_arm():
    assert derive_arm_id("model", "z", {"a": 1, "b": 2}) == derive_arm_id("model", "z", {"b": 2, "a": 1})


def test_registering_the_same_arm_twice_is_refused():
    reg, ids = _register({"a": "2026-01-05"})
    with pytest.raises(ImmutableArmError, match="registered twice"):
        reg.register(slot="model", name="a", spec={"recipe": "a"}, created_date="2026-01-05")


def test_re_registering_an_id_with_a_different_record_is_refused():
    """An id is permanently bound to one vintage, created_date included: the
    ladder must not be able to start from a different day than it did."""
    reg, ids = _register({"a": "2026-01-05"})
    with pytest.raises(ImmutableArmError, match="DIFFERENT record"):
        reg.register(slot="model", name="a", spec={"recipe": "a"}, created_date="2026-02-01")


def test_retiring_an_arm_twice_is_refused_because_it_would_move_the_trailing_window():
    reg, ids = _register({"a": "2026-01-05"})
    reg = reg.retire(ids["a"], "2026-06-01", reason="beaten")
    with pytest.raises(ImmutableArmError, match="retired twice"):
        reg.retire(ids["a"], "2026-07-01", reason="beaten again")


def test_retirement_requires_a_reason():
    reg, ids = _register({"a": "2026-01-05"})
    with pytest.raises(ValueError, match="requires a reason"):
        reg.retire(ids["a"], "2026-06-01", reason="")


def test_a_refit_changes_no_id_and_resets_no_series():
    """Brian 2026-08-29: an arm is a RECIPE including its retrain cadence, so a
    scheduled refit is the arm doing its job and its series stays continuous."""
    reg, ids = _register({"a": "2026-01-05"})
    before = reg.state(ids["a"])
    reg = reg.refit(ids["a"], "2026-03-01")
    reg = reg.refit(ids["a"], "2026-04-01")
    after = reg.state(ids["a"])

    assert after.record == before.record
    assert after.record.created_date == "2026-01-05"
    assert reg.refits(ids["a"]) == ("2026-03-01", "2026-04-01")
    assert reg.active_arms() == (ids["a"],)


def test_the_register_is_append_only_and_rebuildable_from_its_log():
    reg, ids = _register({"a": "2026-01-05", "b": "2026-02-02"})
    reg = reg.refit(ids["a"], "2026-03-01").retire(ids["b"], "2026-06-01", reason="beaten")
    rebuilt = ArmRegister.from_dicts(reg.to_dicts())
    assert rebuilt.active_arms() == reg.active_arms()
    assert rebuilt.state(ids["b"]).retired_date == "2026-06-01"


def test_a_retired_arm_is_still_scored_for_its_trailing_window():
    """§3 — "we retired the wrong one" must stay detectable."""
    reg, ids = _register({"a": "2026-01-05"})
    reg = reg.retire(ids["a"], "2026-08-01", reason="beaten")
    assert ids["a"] in reg.scored_arms("2026-08-29", trailing_cycles=8)
    assert ids["a"] not in reg.scored_arms("2026-12-01", trailing_cycles=8)


# --------------------------------------------------------------------------
# Config guards
# --------------------------------------------------------------------------


def test_a_selection_slot_may_not_be_graded_against_spy():
    """Measured 2026-08-17: SPY trailed the drawn-from population by 140bp at
    21d, which inverted wins and losses outright."""
    with pytest.raises(ArenaConfigError, match="selection-stage slot"):
        ArenaConfig(slot="cut", slot_kind="universe_cut", benchmark="spy")


def test_min_active_arms_below_two_is_refused():
    """A slot holding one arm produced ZERO comparisons on 2026-08-21 and
    2026-08-28 and wrote `no_promotable_challenger` both times."""
    with pytest.raises(ArenaConfigError, match="ZERO comparisons"):
        _config(min_active_arms=1)


def test_a_floor_above_the_cap_is_refused():
    with pytest.raises(ArenaConfigError, match="exceeds cap"):
        _config(cap=3, min_active_arms=4)


# --------------------------------------------------------------------------
# Pointer decision
# --------------------------------------------------------------------------


def test_pointer_moves_to_a_supported_lead():
    inc = _series("inc", [0.00] * 60)
    chal = _series("chal", [0.03] * 60)
    decision = decide_pointer(_config(), AS_OF, "inc", {"inc": inc, "chal": chal})
    assert decision.champion == "chal"
    assert decision.moved
    assert decision.status == "decided"


def test_pointer_holds_when_no_lead_is_supported():
    """The sequence, not a minimum-week floor, is what withholds the promotion."""
    inc = _series("inc", [0.00, 0.00, 0.00])
    chal = _series("chal", [0.03, 0.03, 0.03])
    decision = decide_pointer(_config(), AS_OF, "inc", {"inc": inc, "chal": chal})
    assert decision.champion == "inc"
    assert not decision.moved
    assert decision.status == "held"


def test_pointer_moves_back_when_the_former_arm_regains_the_edge():
    """Brian ruling 4 — free movement in BOTH directions, no cooldown, no hysteresis."""
    cfg = _config()
    early, late = 50, 120
    v1 = [0.03] * early + [0.00] * late
    v2 = [0.00] * early + [0.06] * late
    all_dates = _dates(early + late)

    early_inc = ArmSeries("v2", dict(zip(all_dates[:early], v2[:early])))
    early_chal = ArmSeries("v1", dict(zip(all_dates[:early], v1[:early])))
    first = decide_pointer(cfg, AS_OF, "v2", {"v2": early_inc, "v1": early_chal})
    assert first.champion == "v1", "v1 leads the early cumulative window"

    late_inc = ArmSeries("v1", dict(zip(all_dates, v1)))
    late_chal = ArmSeries("v2", dict(zip(all_dates, v2)))
    # v2 has regained the edge cumulatively; the pointer moves straight back,
    # with no cooldown and no margin to clear.
    second = decide_pointer(cfg, AS_OF, "v1", {"v1": late_inc, "v2": late_chal})
    assert second.champion == "v2"
    assert second.moved


def test_a_slot_with_only_the_champion_reports_unmeasurable_not_success():
    """The `no_promotable_challenger` shape: zero comparisons is a finding."""
    decision = decide_pointer(_config(), AS_OF, "inc", {"inc": _series("inc", [0.01] * 10)})
    assert decision.status == "unmeasurable"
    assert decision.champion == "inc"
    assert "zero comparisons" in decision.reason


def test_no_shared_window_reports_unmeasurable_with_the_reason():
    inc = ArmSeries("inc", {"2026-01-05": 0.01})
    chal = ArmSeries("chal", {"2026-05-04": 0.09})
    decision = decide_pointer(_config(), AS_OF, "inc", {"inc": inc, "chal": chal})
    assert decision.status == "unmeasurable"
    assert "common_window_too_short" in decision.reason


def test_an_arm_that_fails_a_serving_precondition_cannot_serve_however_far_it_leads():
    """The M-slot behavioural veto and input completeness are HARD preconditions,
    independent of ranking. An arm may lead the ladder and still not serve."""
    inc = _series("inc", [0.00] * 60)
    chal = _series("chal", [0.09] * 60)
    decision = decide_pointer(
        _config(),
        AS_OF,
        "inc",
        {"inc": inc, "chal": chal},
        preconditions={
            "chal": [ServingPrecondition("behavioral_veto", False, "zero high-confidence names")]
        },
    )
    assert decision.champion == "inc"
    assert "chal" in decision.ineligible


def test_an_ineligible_incumbent_forces_the_pointer_to_move():
    inc = _series("inc", [0.09] * 60)
    chal = _series("chal", [0.00] * 60)
    decision = decide_pointer(
        _config(),
        AS_OF,
        "inc",
        {"inc": inc, "chal": chal},
        preconditions={"inc": [ServingPrecondition("input_completeness", False, "7 features zeroed")]},
    )
    assert decision.champion == "chal"
    assert decision.moved


def test_no_eligible_arm_at_all_is_unservable_and_names_no_champion():
    inc = _series("inc", [0.01] * 10)
    chal = _series("chal", [0.01] * 10)
    decision = decide_pointer(
        _config(),
        AS_OF,
        "inc",
        {"inc": inc, "chal": chal},
        preconditions={
            "inc": [ServingPrecondition("veto", False, "collapsed")],
            "chal": [ServingPrecondition("veto", False, "collapsed")],
        },
    )
    assert decision.status == "unservable"
    assert decision.champion is None


# --------------------------------------------------------------------------
# Pairwise ranking and retirement
# --------------------------------------------------------------------------


def test_a_young_arm_is_only_judged_on_the_slice_it_overlaps():
    """Ranking a 4-week arm against a 54-week arm on their own numbers is the
    incomparable-window defect. Pairwise comparison uses only the overlap."""
    old = _series("old", [0.01] * 54)
    young = ArmSeries("young", dict(zip(_dates(54)[-4:], [0.05] * 4)))
    ranking = rank_pairwise(
        {"old": old, "young": young},
        created_dates={"old": "2026-01-05", "young": "2026-08-01"},
        as_of=AS_OF,
        clip=0.05,
    )
    verdict = ranking.verdicts[0]
    assert verdict.window.n_dates == 4
    assert ranking.standings["young"].wins == 1


def test_not_in_the_top_cap_means_cap_arms_beat_it_pairwise():
    series = {
        f"a{i}": _series(f"a{i}", [0.10 - 0.01 * i] * 30) for i in range(7)
    }
    created = dict.fromkeys(series, "2026-01-05")
    ranking = rank_pairwise(series, created_dates=created, as_of=AS_OF, clip=0.05)
    # Strict total order: exactly `cap` arms carry fewer than `cap` losses.
    under_cap = [a for a in series if ranking.standings[a].losses < 5]
    assert len(under_cap) == 5


def test_retirement_requires_both_age_and_being_out_of_the_top_cap():
    names = [f"a{i}" for i in range(7)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        created = "2026-01-05" if i < 6 else "2026-08-25"  # the last one is days old
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date=created)
        ids[name] = record.arm_id
    series = {
        ids[name]: _series(ids[name], [0.10 - 0.01 * i] * 30) for i, name in enumerate(names)
    }
    ranking = rank_pairwise(
        series,
        created_dates={ids[n]: reg.state(ids[n]).record.created_date for n in names},
        as_of=AS_OF,
        clip=0.05,
    )
    verdicts = {v.arm_id: v for v in evaluate_retirements(_config(), AS_OF, reg, ranking, champion=ids["a0"])}

    assert verdicts[ids["a5"]].retire, "6th-ranked arm, old enough, must retire"
    assert not verdicts[ids["a6"]].retire
    assert "grace" in verdicts[ids["a6"]].reason
    assert not verdicts[ids["a0"]].retire


def test_the_champion_is_never_retired():
    """Ruling 5 must not be able to retire the serving arm. Asserted explicitly
    here, and structurally impossible besides: an arm that `cap` arms beat
    pairwise cannot hold a lead the pointer decision supports."""
    names = [f"a{i}" for i in range(7)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    series = {ids[n]: _series(ids[n], [0.10 - 0.01 * i] * 30) for i, n in enumerate(names)}
    ranking = rank_pairwise(
        series, created_dates={ids[n]: "2026-01-05" for n in names}, as_of=AS_OF, clip=0.05
    )
    # Force the WORST arm to be champion — the pathological case.
    worst = ids["a6"]
    verdicts = {v.arm_id: v for v in evaluate_retirements(_config(), AS_OF, reg, ranking, champion=worst)}
    assert not verdicts[worst].retire
    assert verdicts[worst].is_champion
    assert "champion" in verdicts[worst].reason


def test_the_floor_stops_a_slot_being_stranded_below_a_live_comparison():
    """Aggressive retirement stranding a slot at one arm IS the defect being
    fixed — `PROMOTABLE_CUTS` held exactly one arm on 2026-08-21 and -28."""
    names = [f"a{i}" for i in range(5)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    series = {ids[n]: _series(ids[n], [0.10 - 0.01 * i] * 30) for i, n in enumerate(names)}
    ranking = rank_pairwise(
        series, created_dates={ids[n]: "2026-01-05" for n in names}, as_of=AS_OF, clip=0.05
    )
    # Force the pathological distribution the floor exists for: every
    # non-champion arm carrying enough pairwise losses to be retired at once.
    # A strict total order cannot produce this (with min_active_arms <= cap the
    # eligible count is always <= the allowance), but ties, Condorcet cycles
    # and unmeasurable pairs perturb the loss distribution, so the guard is
    # tested directly rather than assumed unreachable.
    standings = {
        ids[n]: ArmStanding(
            arm_id=ids[n],
            wins=0,
            losses=9,
            ties=0,
            unmeasurable=0,
            mean_margin=-0.01,
            created_date="2026-01-05",
        )
        for n in names
    }
    forced = PairwiseRanking(
        as_of=AS_OF,
        evidence_mode="point",
        verdicts=ranking.verdicts,
        standings=standings,
        ordering=ranking.ordering,
        cycles_present=False,
    )
    cfg = _config(cap=5, min_active_arms=3)
    verdicts = evaluate_retirements(cfg, AS_OF, reg, forced, champion=ids["a0"])
    retired = [v for v in verdicts if v.retire]
    assert len(retired) == 2, "5 active - 2 retirements = 3, the floor"
    assert any("floor" in v.reason for v in verdicts if not v.retire)
    assert any("floor" in v.reason for v in verdicts if not v.retire)


def test_every_arm_gets_a_retirement_verdict_including_the_survivors():
    """A retirement list containing only retirements cannot be audited."""
    names = [f"a{i}" for i in range(3)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    series = {ids[n]: _series(ids[n], [0.10 - 0.01 * i] * 30) for i, n in enumerate(names)}
    ranking = rank_pairwise(
        series, created_dates={ids[n]: "2026-01-05" for n in names}, as_of=AS_OF, clip=0.05
    )
    verdicts = evaluate_retirements(_config(), AS_OF, reg, ranking, champion=ids["a0"])
    assert {v.arm_id for v in verdicts} == set(ids.values())
    assert all(v.reason for v in verdicts)


def test_a_controls_win_does_not_move_a_real_arm_toward_the_cap():
    """alpha-engine-config-I9770: a control beating a real arm must not count
    toward that arm's retirement cap. With cap=5 and 5 real arms, the worst
    real arm can be beaten by at most 4 OTHER real arms — adding an
    uncounted control win must not push it to 5 and trigger retirement."""
    names = [f"a{i}" for i in range(5)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    reg, ctrl = reg.register(
        slot="model", name="benchmark", spec={"k": "control"}, created_date="2026-01-05", control=True
    )
    series = {ids[n]: _series(ids[n], [0.10 - 0.01 * i] * 30) for i, n in enumerate(names)}
    series[ctrl.arm_id] = _series(ctrl.arm_id, [0.99] * 30)  # beats every real arm
    created = {ids[n]: "2026-01-05" for n in names}
    created[ctrl.arm_id] = "2026-01-05"
    ranking = rank_pairwise(series, created_dates=created, as_of=AS_OF, clip=0.05)

    # Sanity: the control's win IS counted in the control-blind ranking.
    assert ranking.standings[ids["a4"]].losses == 5

    verdicts = {
        v.arm_id: v
        for v in evaluate_retirements(_config(), AS_OF, reg, ranking, champion=ids["a0"])
    }
    assert ctrl.arm_id not in verdicts, "a control never receives a retirement verdict"
    worst = verdicts[ids["a4"]]
    assert not worst.retire, "the control's win must not push the worst real arm to the cap"
    assert worst.pairwise_losses == 4
    assert "in top 5" in worst.reason


def test_a_control_is_never_retired():
    """alpha-engine-config-I9770: a control stays in the pairwise ranking
    (§6.2 is unchanged) but can never itself be retired, however badly it
    underperforms and however old it is."""
    names = [f"a{i}" for i in range(6)]
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(names):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    reg, ctrl = reg.register(
        slot="model", name="benchmark", spec={"k": "control"}, created_date="2026-01-05", control=True
    )
    series = {ids[n]: _series(ids[n], [0.10 - 0.01 * i] * 30) for i, n in enumerate(names)}
    series[ctrl.arm_id] = _series(ctrl.arm_id, [-0.99] * 30)  # loses to every real arm
    created = {ids[n]: "2026-01-05" for n in names}
    created[ctrl.arm_id] = "2026-01-05"
    ranking = rank_pairwise(series, created_dates=created, as_of=AS_OF, clip=0.05)

    # Sanity: the control-blind ranking would otherwise retire it outright.
    assert ranking.standings[ctrl.arm_id].losses == 6

    verdicts = evaluate_retirements(_config(cap=5), AS_OF, reg, ranking, champion=ids["a0"])
    assert ctrl.arm_id not in {v.arm_id for v in verdicts}
    # The real arms are unaffected: with a control present, only the top 5
    # of 6 real arms is the effective real-arm cap, and it is judged as such.
    real_verdicts = {v.arm_id: v for v in verdicts}
    assert real_verdicts[ids["a5"]].retire


def test_anytime_valid_retirement_evidence_is_available_and_stricter():
    series = {
        "a": _series("a", [0.02] * 6),
        "b": _series("b", [0.01] * 6),
    }
    created = {"a": "2026-01-05", "b": "2026-01-05"}
    point = rank_pairwise(series, created_dates=created, as_of=AS_OF, clip=0.05)
    strict = rank_pairwise(
        series, created_dates=created, as_of=AS_OF, clip=0.05, evidence_mode=EVIDENCE_ANYTIME_VALID
    )
    assert point.standings["b"].losses == 1
    assert strict.standings["b"].losses == 0  # not yet supported at 6 observations


# --------------------------------------------------------------------------
# Training integrity (Brian ruling, 2026-08-29)
# --------------------------------------------------------------------------


def _cycle_fixture():
    reg, ids = ArmRegister(), {}
    for i, name in enumerate(["a", "b", "c"]):
        reg, record = reg.register(slot="model", name=name, spec={"n": i}, created_date="2026-01-05")
        ids[name] = record.arm_id
    series = {ids[n]: _series(ids[n], [0.03 - 0.01 * i] * 60) for i, n in enumerate(["a", "b", "c"])}
    return reg, ids, series


def test_an_improperly_trained_arm_fails_the_whole_cycle():
    """Brian: "if any of the arms is not trained properly then the predictor
    module should fail the task." Not a miss, not a degraded run. The week of
    2026-08-29 published a complete leaderboard with seven features hard-zeroed."""
    reg, ids, series = _cycle_fixture()
    training = {
        ids["a"]: TrainingStatus(ids["a"], True),
        ids["b"]: TrainingStatus(ids["b"], False, "7 features hard-zeroed by the collector outage"),
        ids["c"]: TrainingStatus(ids["c"], True),
    }
    with pytest.raises(TrainingIntegrityError, match="not trained properly"):
        run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"], training=training)


def test_an_unasserted_fit_is_treated_as_a_failed_fit():
    reg, ids, series = _cycle_fixture()
    training = {ids["a"]: TrainingStatus(ids["a"], True)}
    with pytest.raises(TrainingIntegrityError, match="no training status reported"):
        run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"], training=training)


def test_a_miss_is_not_a_training_failure():
    """§3's miss keeps its narrow meaning: this arm legitimately had nothing to
    say — e.g. a cut that selected zero names."""
    reg, ids, series = _cycle_fixture()
    series[ids["c"]] = ArmSeries(
        ids["c"], series[ids["c"]].scores, misses=frozenset({"2027-01-01"})
    )
    training = {i: TrainingStatus(i, True) for i in ids.values()}
    cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"], training=training)
    assert cycle.decision.status in ("decided", "held")


# --------------------------------------------------------------------------
# Whole-cycle wiring and the emitted artifact
# --------------------------------------------------------------------------


def test_run_cycle_refuses_an_unregistered_arm():
    """§3 — writing shadow output without a register row is the
    `thinktank_coverage` defect: the data rots unnoticed."""
    reg, ids, series = _cycle_fixture()
    series["ghost"] = _series("ghost", [0.05] * 60)
    with pytest.raises(ValueError, match="unregistered arm"):
        run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])


def test_run_cycle_refuses_a_registered_arm_with_no_series():
    """§3 — every registered arm is scored every cycle, champion or not."""
    reg, ids, series = _cycle_fixture()
    series.pop(ids["c"])
    with pytest.raises(ValueError, match="no series supplied"):
        run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])


def test_run_cycle_emits_a_ladder_for_every_scored_arm():
    reg, ids, series = _cycle_fixture()
    cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
    assert {ladder.arm_id for ladder in cycle.ladders} == set(ids.values())
    assert all(ladder.rungs for ladder in cycle.ladders)


def test_the_emitted_cycle_conforms_to_the_arena_cycle_contract():
    pytest.importorskip("jsonschema")
    from nousergon_lib import contracts

    reg, ids, series = _cycle_fixture()
    cycle = run_cycle(
        _config(),
        AS_OF,
        reg,
        series,
        incumbent=ids["b"],
        preconditions={ids["c"]: [ServingPrecondition("veto", False, "collapsed")]},
    )
    assert contracts.conformance_errors("arena_cycle", cycle.to_dict()) == []


# ── The lifted filling-arm selection rule (alpha-engine-config-I9338) ────────


class TestSelectionRule:
    """`nousergon_lib.arena.selection` — the rule crucible-research and
    crucible-executor both held a copy of."""

    def test_rank_to_score_maps_the_ends_onto_the_band(self):
        assert rank_to_score(0.0, 60.0, 95.0) == 95.0
        assert rank_to_score(1.0, 60.0, 95.0) == 60.0

    def test_rank_to_score_is_monotone_non_increasing(self):
        band = [rank_to_score(i / 20, 60.0, 95.0) for i in range(21)]
        assert band == sorted(band, reverse=True)

    def test_rank_to_score_is_linear_in_the_rank_fraction(self):
        assert rank_to_score(0.5, 60.0, 95.0) == pytest.approx(77.5)

    def test_rank_to_score_clamps_out_of_range_fractions(self):
        assert rank_to_score(-3.0, 60.0, 95.0) == 95.0
        assert rank_to_score(4.0, 60.0, 95.0) == 60.0

    @pytest.mark.parametrize("floor,ceiling", [(95.0, 95.0), (95.0, 60.0)])
    def test_rank_to_score_rejects_an_empty_or_inverted_band(self, floor, ceiling):
        with pytest.raises(ValueError):
            rank_to_score(0.0, floor, ceiling)

    def test_rank_by_alpha_sorts_descending(self):
        assert rank_by_alpha([("A", 0.1), ("B", 0.9), ("C", 0.5)]) == [
            ("B", 0.9), ("C", 0.5), ("A", 0.1),
        ]

    def test_rank_by_alpha_breaks_ties_on_ticker_not_on_input_order(self):
        """§3.1 — an unstable order makes the recorded track record
        unverifiable, so the tie-break cannot be 'whatever order the rows
        arrived in'."""
        forward = rank_by_alpha([("ZZZ", 0.4), ("AAA", 0.4), ("MMM", 0.4)])
        reverse = rank_by_alpha([("MMM", 0.4), ("AAA", 0.4), ("ZZZ", 0.4)])
        assert forward == reverse == [("AAA", 0.4), ("MMM", 0.4), ("ZZZ", 0.4)]

    def test_rank_by_alpha_does_not_mutate_its_input(self):
        rows = [("A", 0.1), ("B", 0.9)]
        rank_by_alpha(rows)
        assert rows == [("A", 0.1), ("B", 0.9)]

    def test_rank_by_alpha_handles_the_empty_and_single_cases(self):
        assert rank_by_alpha([]) == []
        assert rank_by_alpha([("A", 0.1)]) == [("A", 0.1)]

    def test_the_rule_is_reachable_from_the_package_root(self):
        import nousergon_lib.arena as arena

        assert arena.rank_by_alpha is rank_by_alpha
        assert arena.rank_to_score is rank_to_score
        assert "rank_by_alpha" in arena.__all__
        assert "rank_to_score" in arena.__all__

    def test_the_module_is_pure(self):
        """The rest of nousergon_lib.arena holds itself to importable-from-a-
        Lambda; a rule that dragged pandas or boto3 in would break that."""
        import pathlib

        from nousergon_lib.arena import selection

        src = pathlib.Path(selection.__file__).read_text()
        for forbidden in ("import boto3", "import pandas", "import logging", "import json"):
            assert forbidden not in src


# ── Slot-supplied lineage on the verdict surface (alpha-engine-config-I9903) ─


class TestSeriesLineageReachesTheVerdictArtifact:
    """The engine carries a slot's provenance without learning what it means.

    `alpha-engine-config-I9801` removed `ModelRecipe.feature_version` from the
    M recipe's hashed spec and left the resolved version as PRODUCE-time
    lineage in the run manifest's `inputs[]`. No VERDICT-surface artifact
    carried it, so "did this champion's score series span one feature-layer
    version or several" was answerable only by walking every constituent
    day's manifest. `ArmSeries.lineage` is the channel; the engine never
    reads a key or a value, which is what keeps `run_cycle` scoring U, R, M
    and S off one implementation (principles.md §2.8).
    """

    def test_the_ladder_names_the_versions_the_arms_series_read(self):
        reg, ids, series = _cycle_fixture()
        series[ids["a"]] = ArmSeries(
            ids["a"], series[ids["a"]].scores, lineage={"feature_version": ("f00d",)}
        )
        cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
        by_arm = {ladder.arm_id: ladder for ladder in cycle.ladders}
        assert by_arm[ids["a"]].lineage == {"feature_version": ("f00d",)}
        assert by_arm[ids["b"]].lineage == {}

    def test_a_series_spanning_two_versions_is_readable_as_spanning_two(self):
        """The whole point: one value means one layer, two means it moved."""
        reg, ids, series = _cycle_fixture()
        series[ids["a"]] = ArmSeries(
            ids["a"], series[ids["a"]].scores, lineage={"feature_version": ("beef", "f00d")}
        )
        cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
        emitted = cycle.to_dict()["ladders"]
        spans = {
            entry["arm_id"]: len(entry["lineage"].get("feature_version", [])) for entry in emitted
        }
        assert spans[ids["a"]] == 2
        assert spans[ids["b"]] == 0

    def test_an_empty_ladder_still_carries_the_lineage(self):
        """An arm with no scored date is the case a pass-through most easily
        drops — `build_ladder` returns early, on a separate constructor."""
        ladder = build_ladder(
            ArmSeries("m:x:1", {}, lineage={"feature_version": ("f00d",)}), AS_OF
        )
        assert ladder.rungs == ()
        assert ladder.lineage == {"feature_version": ("f00d",)}

    def test_values_are_sorted_and_deduplicated_so_two_runs_agree(self):
        series = ArmSeries("m:x:1", {}, lineage={"feature_version": ["f00d", "beef", "f00d"]})
        assert series.lineage == {"feature_version": ("beef", "f00d")}

    def test_a_dimension_declared_with_no_values_is_refused(self):
        """§7.2 — a well-formed record of nothing. A slot with no provenance
        omits the dimension; it never declares it empty."""
        with pytest.raises(ValueError, match="at least one non-empty"):
            ArmSeries("m:x:1", {}, lineage={"feature_version": ()})

    def test_a_bare_string_is_refused_rather_than_split_into_characters(self):
        with pytest.raises(ValueError, match="not a bare string"):
            ArmSeries("m:x:1", {}, lineage={"feature_version": "f00d"})

    def test_the_engine_never_reads_the_lineage(self):
        """Substitutability: a dimension no slot in this repo has ever heard of
        must pass through untouched. If the engine ever branches on a key, this
        is the test that fails."""
        reg, ids, series = _cycle_fixture()
        series[ids["a"]] = ArmSeries(
            ids["a"],
            series[ids["a"]].scores,
            lineage={"a_dimension_the_engine_cannot_know": ("x",)},
        )
        cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
        by_arm = {ladder.arm_id: ladder for ladder in cycle.ladders}
        assert by_arm[ids["a"]].lineage == {"a_dimension_the_engine_cannot_know": ("x",)}

    def test_a_cycle_carrying_lineage_still_conforms_to_arena_cycle_v1(self):
        """The field is ADDITIVE and optional, so the contract stays at v1:
        a new document validates against the old schema's `required` list and
        an old document (no `lineage` key at all) validates against this one."""
        pytest.importorskip("jsonschema")
        from nousergon_lib import contracts

        reg, ids, series = _cycle_fixture()
        series[ids["a"]] = ArmSeries(
            ids["a"], series[ids["a"]].scores, lineage={"feature_version": ("beef", "f00d")}
        )
        cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
        payload = cycle.to_dict()
        assert payload["schema_version"] == 1
        assert contracts.conformance_errors("arena_cycle", payload) == []

        # An artifact written before the field existed. ABSENT is legal and
        # means "not recorded"; `{}` means "the slot declares none".
        for entry in payload["ladders"]:
            entry.pop("lineage")
        assert contracts.conformance_errors("arena_cycle", payload) == []

    def test_the_schema_refuses_a_dimension_with_an_empty_value_list(self):
        """§7.4 — the guard has to FAIL on the shape it exists to catch."""
        pytest.importorskip("jsonschema")
        from nousergon_lib import contracts

        reg, ids, series = _cycle_fixture()
        cycle = run_cycle(_config(), AS_OF, reg, series, incumbent=ids["a"])
        payload = cycle.to_dict()
        payload["ladders"][0]["lineage"] = {"feature_version": []}
        assert contracts.conformance_errors("arena_cycle", payload) != []
