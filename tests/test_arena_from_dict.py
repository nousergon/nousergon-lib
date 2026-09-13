"""`ArenaCycle.from_dict` and the sibling `from_dict`s — the inverse of `to_dict`.

Normative source: `alpha-engine-config-I10679`. `crucible.promote` is moving
from RECOMPUTING the arena cycle to READING the one `experiment.grade`
already wrote and validated (`crucible/arena_io.py::read_arena_cycle`). That
requires every dataclass `ArenaCycle.to_dict()` reaches to have a real
inverse — a crucible-side reconstruction would be the second copy of the
library's own shape that drifts (repo AGENTS.md, "the arena is called, never
re-implemented").

Every test below constructs a REAL cycle through `run_cycle` — never a
hand-built dict — round-trips it through `to_dict()` / `from_dict()` /
`to_dict()` again, and asserts the two dicts are equal. That is the honest
meaning of "round trip" for this artifact: `to_dict()` is deliberately lossy
for a few fields (`PairedWindow`'s per-date `dates`/`diffs`, `ArenaConfig
.max_ladder_weeks`), documented at each `from_dict`, and the contract this
package can actually promise is "read back what was written", not "recover
what was never serialized".
"""

from __future__ import annotations

import json

from nousergon_lib.arena import ArenaCycle, ArmRegister, ArmSeries, ServingPrecondition
from nousergon_lib.arena.confseq import ConfSeqBound, confidence_sequence
from nousergon_lib.arena.engine import ArenaConfig, Comparison, PointerDecision, RetirementVerdict, run_cycle
from nousergon_lib.arena.ladder import ScoreLadder, build_ladder
from nousergon_lib.arena.ranking import ArmStanding, PairVerdict, PairwiseRanking, rank_pairwise
from nousergon_lib.arena.window import PairedWindow, pair_on_common_window
from tests.test_arena import _dates, _series

AS_OF = "2026-08-29"


def _fixture_register_and_series(n_arms=3, promote_min_weeks=1):
    reg = ArmRegister()
    ids = []
    for name in ["a", "b", "c", "d"][:n_arms]:
        reg, record = reg.register(
            slot="u", name=name, spec={"recipe": name}, created_date="2026-01-05"
        )
        ids.append(record.arm_id)
    series = {
        arm_id: _series(arm_id, [0.01 * (i + 1) * k for k in range(20)])
        for i, arm_id in enumerate(ids)
    }
    config = ArenaConfig(
        slot="u",
        slot_kind="universe_cut",
        benchmark="population",
        diff_clip=0.05,
        promote_min_weeks=promote_min_weeks,
    )
    return config, reg, ids, series


# ---------------------------------------------------------------------------
# The end-to-end contract: a REAL cycle, round-tripped whole.
# ---------------------------------------------------------------------------


def test_a_decided_cycle_round_trips_exactly_through_to_dict_and_from_dict():
    config, reg, ids, series = _fixture_register_and_series()
    cycle = run_cycle(
        config=config, as_of=AS_OF, register=reg, series_by_arm=series, incumbent=ids[0]
    )
    payload = cycle.to_dict()

    restored = ArenaCycle.from_dict(payload)

    assert restored.to_dict() == payload


def test_a_bootstrap_cycle_with_no_incumbent_round_trips_exactly():
    """§9.1's cold-start path takes a different branch in `decide_pointer`
    (empty `comparisons`, `status="bootstrap"`) — a real, distinct shape from
    the decided path above, and it must round-trip too."""
    config, reg, ids, series = _fixture_register_and_series()
    cycle = run_cycle(
        config=config, as_of=AS_OF, register=reg, series_by_arm=series, incumbent=None
    )
    payload = cycle.to_dict()

    restored = ArenaCycle.from_dict(payload)

    assert restored.to_dict() == payload
    assert payload["decision"]["status"] == "bootstrap"


def test_an_unservable_cycle_with_a_failed_precondition_round_trips_exactly():
    """The `ineligible` map and empty `comparisons` are their own shape."""
    config, reg, ids, series = _fixture_register_and_series()
    preconditions = {
        ids[0]: (ServingPrecondition(name="not_a_control_arm", passed=False, reason="synthetic"),)
    }
    cycle = run_cycle(
        config=config,
        as_of=AS_OF,
        register=reg,
        series_by_arm=series,
        incumbent=ids[0],
        preconditions=preconditions,
    )
    payload = cycle.to_dict()

    restored = ArenaCycle.from_dict(payload)

    assert restored.to_dict() == payload
    assert payload["decision"]["ineligible"]


def test_a_cycle_serialised_to_real_json_and_back_still_round_trips():
    """The artifact this deliverable exists for is JSON on the wire
    (`crucible.arena_io.write_arena_cycle`/`read_arena_cycle`), not a Python
    dict handed in-process — a fixture over real `json.dumps`/`json.loads`
    is what `read_arena_cycle`'s caller actually gets."""
    config, reg, ids, series = _fixture_register_and_series()
    cycle = run_cycle(
        config=config, as_of=AS_OF, register=reg, series_by_arm=series, incumbent=ids[0]
    )
    payload = cycle.to_dict()
    wire = json.loads(json.dumps(payload, sort_keys=True))

    restored = ArenaCycle.from_dict(wire)

    assert restored.to_dict() == payload


def test_a_cycle_with_no_ranking_round_trips_the_none():
    """A single-arm slot's cycle has `ranking=None` (`run_cycle`: ranking is
    built only when `len(active_series) >= 2`) — `ArenaCycle.to_dict` emits
    `"ranking": null`, and `from_dict` must not choke reconstructing it."""
    config, reg, ids, series = _fixture_register_and_series(n_arms=1)
    cycle = run_cycle(
        config=config, as_of=AS_OF, register=reg, series_by_arm=series, incumbent=ids[0]
    )
    payload = cycle.to_dict()
    assert payload["ranking"] is None

    restored = ArenaCycle.from_dict(payload)

    assert restored.ranking is None
    assert restored.to_dict() == payload


# ---------------------------------------------------------------------------
# Per-class round trips — the inverse each `from_dict` documents.
# ---------------------------------------------------------------------------


def test_conf_seq_bound_round_trips_and_recomputes_supported_rather_than_trusting_it():
    bound = confidence_sequence([0.01, 0.02, -0.01, 0.03], clip=0.05)
    payload = bound.to_dict()

    restored = ConfSeqBound.from_dict(payload)

    assert restored.to_dict() == payload
    # `supported` is dropped from the constructor and recomputed from `lower`.
    tampered = dict(payload, supported=not payload["supported"])
    assert ConfSeqBound.from_dict(tampered).to_dict()["supported"] == payload["supported"]


def test_paired_window_measurable_round_trips_its_aggregates_exactly():
    a = ArmSeries("a", dict(zip(_dates(6), [0.01, 0.02, 0.03, 0.015, 0.025, 0.005])))
    b = ArmSeries("b", dict(zip(_dates(6), [0.0, 0.01, 0.02, 0.01, 0.02, 0.0])))
    window = pair_on_common_window(a, b)
    payload = window.to_dict()

    restored = PairedWindow.from_dict(payload)

    assert restored.to_dict() == payload
    # `mean_diff` is exact, not approximate — see `PairedWindow.from_dict`'s
    # single-element `diffs` reconstruction.
    assert restored.mean_diff == payload["mean_diff"]


def test_paired_window_unmeasurable_round_trips_with_no_synthetic_dates():
    a = ArmSeries("a", {"2026-01-05": 0.01})
    b = ArmSeries("b", {"2026-03-02": 0.01})
    window = pair_on_common_window(a, b)
    payload = window.to_dict()
    assert not window.measurable

    restored = PairedWindow.from_dict(payload)

    assert not restored.measurable
    assert restored.dates == ()
    assert restored.to_dict() == payload


def test_paired_window_single_date_round_trips():
    a = ArmSeries("a", {"2026-01-05": 0.02})
    b = ArmSeries("b", {"2026-01-05": 0.01})
    window = pair_on_common_window(a, b)
    payload = window.to_dict()
    assert payload["n_dates"] == 1

    restored = PairedWindow.from_dict(payload)

    assert restored.to_dict() == payload


def test_comparison_round_trips_a_measured_verdict():
    config, reg, ids, series = _fixture_register_and_series()
    cycle = run_cycle(
        config=config, as_of=AS_OF, register=reg, series_by_arm=series, incumbent=ids[0]
    )
    comparison = cycle.decision.comparisons[0]
    payload = comparison.to_dict()

    restored = Comparison.from_dict(payload)

    assert restored.to_dict() == payload


def test_pointer_decision_round_trips_including_ineligible_arms():
    config, reg, ids, series = _fixture_register_and_series()
    preconditions = {
        ids[1]: (ServingPrecondition(name="not_a_control_arm", passed=False, reason="x"),)
    }
    decision = run_cycle(
        config=config,
        as_of=AS_OF,
        register=reg,
        series_by_arm=series,
        incumbent=ids[0],
        preconditions=preconditions,
    ).decision
    payload = decision.to_dict()

    restored = PointerDecision.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.ineligible


def test_retirement_verdict_round_trips():
    verdict = RetirementVerdict(
        arm_id="u:a:1",
        retire=True,
        reason="3 arm(s) beat it pairwise (cap 3) and it is 6 week(s) old (grace 4)",
        age_weeks=6,
        pairwise_losses=3,
        is_champion=False,
    )
    payload = verdict.to_dict()

    restored = RetirementVerdict.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored == verdict


def test_arena_config_round_trips_every_decision_governing_field():
    config = ArenaConfig(
        slot="u",
        slot_kind="universe_cut",
        benchmark="population",
        diff_clip=0.05,
        promote_min_weeks=2,
        promote_evidence="point",
        cap=4,
        grace_weeks=3,
        min_active_arms=2,
        retired_trailing_cycles=6,
        retire_evidence="point",
    )
    payload = config.to_dict()

    restored = ArenaConfig.from_dict(
        payload, slot=config.slot, slot_kind=config.slot_kind, benchmark=config.benchmark
    )

    assert restored.to_dict() == payload
    assert restored.slot == config.slot
    assert restored.slot_kind == config.slot_kind
    assert restored.benchmark == config.benchmark
    # The one documented exception: never serialized, so never recovered.
    assert restored.max_ladder_weeks is None


def test_ladder_rung_and_score_ladder_round_trip():
    series = _series("u:a:1", [0.01 * k for k in range(30)])
    ladder = build_ladder(series, AS_OF)
    payload = ladder.to_dict()
    assert payload["rungs"]

    restored = ScoreLadder.from_dict(payload)

    assert restored.to_dict() == payload


def test_score_ladder_with_lineage_round_trips():
    series = ArmSeries(
        "u:a:1",
        dict(zip(_dates(10), [0.01 * k for k in range(10)])),
        lineage={"feature_version": ("v1", "v2")},
    )
    ladder = build_ladder(series, AS_OF)
    payload = ladder.to_dict()
    assert payload["lineage"] == {"feature_version": ["v1", "v2"]}

    restored = ScoreLadder.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.lineage == {"feature_version": ("v1", "v2")}


def test_arm_standing_round_trips_and_recomputes_copeland():
    standing = ArmStanding(
        arm_id="u:a:1", wins=3, losses=1, ties=0, unmeasurable=0, mean_margin=0.01,
        created_date="2026-01-05",
    )
    payload = standing.to_dict()

    restored = ArmStanding.from_dict(payload)

    assert restored.to_dict() == payload
    tampered = dict(payload, copeland=999)
    assert ArmStanding.from_dict(tampered).to_dict()["copeland"] == payload["copeland"]


def test_pair_verdict_round_trips():
    a = ArmSeries("a", dict(zip(_dates(6), [0.01, 0.02, 0.03, 0.015, 0.025, 0.005])))
    b = ArmSeries("b", dict(zip(_dates(6), [0.0, 0.01, 0.02, 0.01, 0.02, 0.0])))
    ranking = rank_pairwise(
        {"a": a, "b": b},
        created_dates={"a": "2026-01-05", "b": "2026-01-05"},
        as_of=AS_OF,
        clip=0.05,
    )
    verdict = ranking.verdicts[0]
    payload = verdict.to_dict()

    restored = PairVerdict.from_dict(payload)

    assert restored.to_dict() == payload


def test_pairwise_ranking_round_trips_whole():
    config, reg, ids, series = _fixture_register_and_series()
    ranking = rank_pairwise(
        series, created_dates=dict.fromkeys(ids, "2026-01-05"), as_of=AS_OF, clip=0.05
    )
    payload = ranking.to_dict()

    restored = PairwiseRanking.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.ordering == ranking.ordering


def test_pairwise_ranking_with_a_condorcet_cycle_round_trips():
    """`cycles_present` is a real boolean field the reconstruction must not
    drop or invert."""
    # Rock-paper-scissors margins over three arms on a shared window.
    dates = _dates(6)
    a = ArmSeries("a", dict(zip(dates, [0.03, -0.01, 0.02, 0.03, -0.01, 0.02])))
    b = ArmSeries("b", dict(zip(dates, [-0.01, 0.03, -0.01, 0.02, 0.03, -0.01])))
    c = ArmSeries("c", dict(zip(dates, [0.02, -0.02, 0.03, -0.02, 0.02, 0.03])))
    ranking = rank_pairwise(
        {"a": a, "b": b, "c": c},
        created_dates={"a": "2026-01-05", "b": "2026-01-05", "c": "2026-01-05"},
        as_of=AS_OF,
        clip=0.05,
    )
    payload = ranking.to_dict()

    restored = PairwiseRanking.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.cycles_present == ranking.cycles_present


def test_serving_precondition_round_trips():
    precondition = ServingPrecondition(name="not_a_control_arm", passed=False, reason="x")
    payload = precondition.to_dict()

    restored = ServingPrecondition.from_dict(payload)

    assert restored == precondition
    assert restored.to_dict() == payload
