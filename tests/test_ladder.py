"""Direct tests for ``nousergon_lib.arena.ladder`` with literal expected values.

Calendar used throughout: as_of 2026-03-15; rung ``h`` covers the trailing
``7h`` days ending there (start = as_of - (7h - 1) days).
"""

from __future__ import annotations

from typing import cast

import pytest

from nousergon_lib.arena.ladder import LadderRung, ScoreLadder, build_ladder
from nousergon_lib.arena.window import ArmSeries

AS_OF = "2026-03-15"


def _series() -> ArmSeries:
    return ArmSeries(
        arm_id="arm-a",
        scores={
            "2026-02-20": 5.0,
            "2026-03-05": 3.0,
            "2026-03-15": 1.0,
            "2026-03-20": 100.0,  # after as_of: must be ignored
        },
        misses=frozenset({"2026-03-10", "2026-01-01", "2026-04-01"}),
    )


def test_every_rung_is_emitted_with_trailing_window_means() -> None:
    ladder = build_ladder(_series(), AS_OF)
    assert ladder.arm_id == "arm-a"
    assert ladder.as_of == AS_OF
    assert [r.weeks for r in ladder.rungs] == [1, 2, 3, 4]
    assert [r.mean_score for r in ladder.rungs] == [1.0, 2.0, 2.0, 3.0]
    assert [r.n_dates for r in ladder.rungs] == [1, 2, 2, 3]
    assert [r.n_misses for r in ladder.rungs] == [1, 1, 1, 1]
    assert ladder.rungs[1].start_date == "2026-03-05"
    assert ladder.rungs[1].end_date == "2026-03-15"
    assert ladder.rungs[3].start_date == "2026-02-20"


def test_totals_exclude_dates_after_as_of() -> None:
    ladder = build_ladder(_series(), AS_OF)
    assert ladder.total_weeks == 4
    assert ladder.total_dates == 3
    # 2026-04-01 miss is after as_of; 2026-01-01 and 2026-03-10 count.
    assert ladder.total_misses == 2


def test_longest_is_the_full_history_rung() -> None:
    longest = build_ladder(_series(), AS_OF).longest
    assert longest is not None
    assert longest.weeks == 4
    assert longest.mean_score == 3.0


def test_max_weeks_caps_rungs_but_not_totals() -> None:
    ladder = build_ladder(_series(), AS_OF, max_weeks=2)
    assert [r.weeks for r in ladder.rungs] == [1, 2]
    assert ladder.total_weeks == 4
    longest = ladder.longest
    assert longest is not None
    assert longest.weeks == 2


def test_max_weeks_above_history_does_not_extend_the_ladder() -> None:
    ladder = build_ladder(_series(), AS_OF, max_weeks=50)
    assert [r.weeks for r in ladder.rungs] == [1, 2, 3, 4]


def test_single_scored_date_is_one_week() -> None:
    ladder = build_ladder(ArmSeries("a", {"2026-03-15": 0.5}), AS_OF)
    assert [r.weeks for r in ladder.rungs] == [1]
    assert ladder.total_weeks == 1
    assert ladder.total_dates == 1


def test_no_history_yields_empty_ladder_not_zeros() -> None:
    series = ArmSeries(
        "a", {"2026-04-01": 1.0}, misses=frozenset({"2026-03-01", "2026-04-02"})
    )
    ladder = build_ladder(series, AS_OF)
    assert ladder.rungs == ()
    assert ladder.longest is None
    assert ladder.total_weeks == 0
    assert ladder.total_dates == 0
    assert ladder.total_misses == 1


def test_window_with_no_observations_is_omitted_not_zeroed() -> None:
    # Only score is 2026-02-01, six weeks before as_of; rungs 1-6 are empty.
    ladder = build_ladder(ArmSeries("a", {"2026-02-01": 4.0}), AS_OF)
    assert ladder.total_weeks == 7
    assert [r.weeks for r in ladder.rungs] == [7]
    assert ladder.rungs[0].mean_score == 4.0


def test_lineage_is_carried_through_unread() -> None:
    series = ArmSeries("a", {"2026-03-15": 1.0}, lineage={"feature_version": ("v1",)})
    ladder = build_ladder(series, AS_OF)
    assert dict(ladder.lineage) == {"feature_version": ("v1",)}
    assert ladder.to_dict()["lineage"] == {"feature_version": ["v1"]}
    empty = build_ladder(ArmSeries("a", {"2026-04-01": 1.0}, lineage={"k": ("v",)}), AS_OF)
    assert dict(empty.lineage) == {"k": ("v",)}


def test_malformed_as_of_raises() -> None:
    with pytest.raises(ValueError):
        build_ladder(_series(), "not-a-date")


def test_rung_to_dict_and_from_dict_are_exact_inverses() -> None:
    rung = LadderRung(3, 2, 1, "2026-03-05", "2026-03-15", 2.0)
    assert rung.to_dict() == {
        "weeks": 3,
        "n_dates": 2,
        "n_misses": 1,
        "start_date": "2026-03-05",
        "end_date": "2026-03-15",
        "mean_score": 2.0,
    }
    assert LadderRung.from_dict(rung.to_dict()) == rung


def test_rung_from_dict_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        LadderRung.from_dict({"weeks": 1})


def test_ladder_to_dict_shape_and_roundtrip() -> None:
    ladder = build_ladder(_series(), AS_OF)
    payload = ladder.to_dict()
    assert payload["arm_id"] == "arm-a"
    assert payload["total_weeks"] == 4
    rungs = cast(list[object], payload["rungs"])
    assert len(rungs) == 4
    assert ScoreLadder.from_dict(payload) == ladder


def test_ladder_from_dict_tolerates_absent_rungs_and_lineage() -> None:
    restored = ScoreLadder.from_dict(
        {
            "arm_id": "a",
            "as_of": AS_OF,
            "total_weeks": 0,
            "total_dates": 0,
            "total_misses": 0,
        }
    )
    assert restored.rungs == ()
    assert dict(restored.lineage) == {}
    assert restored.longest is None


def test_ladder_from_dict_missing_required_field_raises() -> None:
    with pytest.raises(KeyError):
        ScoreLadder.from_dict({"arm_id": "a", "as_of": AS_OF})
