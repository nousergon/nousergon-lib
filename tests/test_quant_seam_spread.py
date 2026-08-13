"""Tests for nousergon_lib.quant.seam_spread (alpha-engine-config#7214).

The module's whole value is that its *conventions* are enforced rather than
documented, so most of these tests are about the guards: the survivorship
trap, the two-sides-disagree trap, the renormalization trap, and the
"measured zero vs could not measure" trap. The arithmetic itself is one test.
"""

import json

import pytest

from nousergon_lib.quant.seam_spread import (
    OUTCOME_TOLERANCE,
    Observation,
    date_clustered_mean,
    recompute_spread,
    seam_spread,
)

D1 = "2026-07-04"
D2 = "2026-07-11"


def _obs(date, pairs):
    return [Observation(date, k, v) for k, v in pairs]


def _spread(**kw):
    kw.setdefault("seam", "scanner")
    kw.setdefault("outcome_unit", "log_alpha_21d")
    kw.setdefault("window_start", D1)
    kw.setdefault("window_end", D2)
    return seam_spread(**kw)


class TestObservation:
    def test_rejects_empty_cohort_date(self):
        with pytest.raises(ValueError, match="cohort_date"):
            Observation("", "AAPL", 0.01)

    def test_rejects_empty_key(self):
        with pytest.raises(ValueError, match="key"):
            Observation(D1, "", 0.01)

    def test_rejects_nan_outcome_because_nan_is_an_uncounted_drop(self):
        with pytest.raises(ValueError, match="finite or None"):
            Observation(D1, "AAPL", float("nan"))

    def test_rejects_infinite_outcome(self):
        with pytest.raises(ValueError, match="finite or None"):
            Observation(D1, "AAPL", float("inf"))

    def test_none_outcome_is_a_legitimate_unresolved_state(self):
        assert Observation(D1, "AAPL", None).outcome is None


class TestDateClusteredMean:
    def test_equal_weight_within_then_across_dates(self):
        obs = _obs(D1, [("A", 0.0), ("B", 0.2)]) + _obs(D2, [("C", 1.0)])
        result = date_clustered_mean(obs)
        # (0.0+0.2)/2 = 0.1 for D1; 1.0 for D2; (0.1+1.0)/2 = 0.55.
        assert result.mean == pytest.approx(0.55)
        assert result.n_dates == 2

    def test_pooled_mean_differs_and_is_reported_separately(self):
        obs = _obs(D1, [("A", 0.0), ("B", 0.2)]) + _obs(D2, [("C", 1.0)])
        result = date_clustered_mean(obs)
        # Pooled weights the 2-name date twice as heavily: (0+0.2+1.0)/3.
        assert result.pooled_mean == pytest.approx(0.4)
        assert result.pooled_mean != result.mean

    def test_unresolved_names_are_dropped_from_the_mean_and_counted(self):
        obs = _obs(D1, [("A", 0.1), ("B", None), ("C", None)])
        result = date_clustered_mean(obs)
        assert result.mean == pytest.approx(0.1)
        assert result.n_resolved == 1
        assert result.n_unresolved == 2
        assert result.unresolved_rate == pytest.approx(2 / 3)

    def test_empty_population_reports_none_not_zero(self):
        result = date_clustered_mean([])
        assert result.mean is None
        assert result.pooled_mean is None
        assert result.unresolved_rate is None, "could not measure must not render as measured zero"

    def test_thin_date_is_excluded_from_the_mean_but_still_counted(self):
        obs = _obs(D1, [("A", 0.5)]) + _obs(D2, [("B", 1.0), ("C", 2.0)])
        result = date_clustered_mean(obs, min_names_per_date=2)
        assert result.n_dates == 1
        assert result.mean == pytest.approx(1.5)
        assert result.n_resolved == 3, "the excluded date's names are still SEEN"

    def test_duplicate_name_in_one_cross_section_raises(self):
        obs = _obs(D1, [("A", 0.1)]) + _obs(D1, [("A", 0.2)])
        with pytest.raises(ValueError, match="duplicate observation"):
            date_clustered_mean(obs)

    def test_min_names_per_date_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="min_names_per_date"):
            date_clustered_mean([], min_names_per_date=0)


class TestSeamSpreadArithmetic:
    def test_headline_is_the_mean_of_per_date_spreads(self):
        inputs = _obs(D1, [("A", 0.0), ("B", 0.1), ("C", 0.2)]) + _obs(
            D2, [("A", 0.0), ("B", 0.4)]
        )
        outputs = _obs(D1, [("C", 0.2)]) + _obs(D2, [("B", 0.4)])
        result = _spread(input_population=inputs, output_population=outputs)
        # D1: 0.2 - 0.1 = 0.1 ; D2: 0.4 - 0.2 = 0.2 ; mean = 0.15.
        assert result.status == "ok"
        assert result.spread == pytest.approx(0.15)
        assert result.n_dates == 2

    def test_a_seam_that_selects_the_worst_names_grades_negative(self):
        inputs = _obs(D1, [("A", -0.3), ("B", 0.1), ("C", 0.2)])
        outputs = _obs(D1, [("A", -0.3)])
        result = _spread(input_population=inputs, output_population=outputs)
        assert result.spread < 0

    def test_selecting_the_whole_input_population_grades_exactly_zero(self):
        pairs = [("A", -0.3), ("B", 0.1), ("C", 0.2)]
        result = _spread(input_population=_obs(D1, pairs), output_population=_obs(D1, pairs))
        assert result.spread == pytest.approx(0.0)
        assert result.status == "ok", "a measured zero is a real grade, not an absence"

    def test_unit_is_carried_onto_the_record(self):
        result = _spread(
            outcome_unit="bps",
            input_population=_obs(D1, [("A", 1.0), ("B", 3.0)]),
            output_population=_obs(D1, [("B", 3.0)]),
        )
        assert result.to_dict()["outcome_unit"] == "bps"


class TestConventionGuards:
    def test_output_name_absent_from_the_input_population_raises(self):
        inputs = _obs(D1, [("A", 0.1)])
        outputs = _obs(D1, [("ZZZZ", 0.5)])
        with pytest.raises(ValueError, match="selected FROM"):
            _spread(input_population=inputs, output_population=outputs)

    def test_subset_guard_can_be_disabled_explicitly(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.3)])
        outputs = _obs(D1, [("ZZZZ", 0.5)])
        result = _spread(
            input_population=inputs, output_population=outputs, require_subset=False
        )
        assert result.spread == pytest.approx(0.5 - 0.2)

    def test_output_date_the_input_never_published_raises(self):
        inputs = _obs(D1, [("A", 0.1)])
        outputs = _obs(D1, [("A", 0.1)]) + _obs(D2, [("A", 0.9)])
        with pytest.raises(ValueError, match="never published"):
            _spread(input_population=inputs, output_population=outputs)

    def test_same_name_with_two_different_outcomes_raises(self):
        inputs = _obs(D1, [("A", 0.10), ("B", 0.20)])
        outputs = _obs(D1, [("A", 0.11)])
        with pytest.raises(ValueError, match="SAME return convention"):
            _spread(input_population=inputs, output_population=outputs)

    def test_float_round_trip_noise_is_tolerated(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.2)])
        outputs = _obs(D1, [("A", 0.1 + OUTCOME_TOLERANCE / 2)])
        result = _spread(input_population=inputs, output_population=outputs)
        assert result.status == "ok"

    def test_resolved_on_one_side_unresolved_on_the_other_raises(self):
        inputs = _obs(D1, [("A", None), ("B", 0.2)])
        outputs = _obs(D1, [("A", 0.5)])
        with pytest.raises(ValueError, match="resolved on one side"):
            _spread(input_population=inputs, output_population=outputs)

    def test_outcome_unit_is_mandatory(self):
        with pytest.raises(ValueError, match="outcome_unit"):
            seam_spread(
                seam="scanner",
                outcome_unit="",
                window_start=D1,
                window_end=D2,
                input_population=[],
                output_population=[],
            )

    def test_seam_name_is_mandatory(self):
        with pytest.raises(ValueError, match="seam must be"):
            _spread(seam="", input_population=[], output_population=[])

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_max_unresolved_rate_must_be_a_fraction(self, bad):
        with pytest.raises(ValueError, match="max_unresolved_rate"):
            _spread(input_population=[], output_population=[], max_unresolved_rate=bad)

    def test_min_dates_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="min_dates"):
            _spread(input_population=[], output_population=[], min_dates=0)


class TestSurvivorship:
    def test_input_side_losing_its_delistings_raises_at_the_ceiling(self):
        # The trap: the ~900-name input population quietly loses every name
        # that delisted inside the window, while the 1-name output survives.
        inputs = _obs(D1, [("A", 0.1)] + [(f"D{i}", None) for i in range(9)])
        outputs = _obs(D1, [("A", 0.1)])
        with pytest.raises(ValueError, match="survivorship"):
            _spread(
                input_population=inputs,
                output_population=outputs,
                max_unresolved_rate=0.5,
            )

    def test_the_same_case_is_reported_rather_than_raised_when_no_ceiling_is_set(self):
        inputs = _obs(D1, [("A", 0.1)] + [(f"D{i}", None) for i in range(9)])
        outputs = _obs(D1, [("A", 0.1)])
        result = _spread(input_population=inputs, output_population=outputs)
        assert result.input_population.unresolved_rate == pytest.approx(0.9)
        assert result.output_population.unresolved_rate == pytest.approx(0.0)
        assert result.unresolved_rate_gap == pytest.approx(-0.9)

    def test_output_side_ceiling_is_enforced_too(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.2), ("C", None), ("D", None)])
        outputs = _obs(D1, [("C", None), ("D", None), ("A", 0.1)])
        with pytest.raises(ValueError, match="output population"):
            _spread(
                input_population=inputs,
                output_population=outputs,
                max_unresolved_rate=0.6,
            )

    def test_unresolved_rate_gap_is_none_when_a_side_is_empty(self):
        result = _spread(input_population=[], output_population=[])
        assert result.unresolved_rate_gap is None


class TestRenormalization:
    def test_a_date_present_on_only_one_side_is_dropped_and_NAMED(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.3)]) + _obs(D2, [("A", 9.0), ("B", 9.0)])
        outputs = _obs(D1, [("B", 0.3)])
        result = _spread(input_population=inputs, output_population=outputs, require_subset=False)
        assert result.dates_dropped == (D2,)
        assert result.n_dates == 1
        assert result.spread == pytest.approx(0.1)

    def test_the_dropped_date_does_not_leak_into_either_population_mean(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.3)]) + _obs(D2, [("A", 9.0), ("B", 9.0)])
        outputs = _obs(D1, [("B", 0.3)])
        result = _spread(input_population=inputs, output_population=outputs, require_subset=False)
        assert result.input_population.mean == pytest.approx(0.2)
        assert result.input_population.n_dates == 1


class TestInsufficientData:
    def test_no_usable_dates_reports_insufficient_data_not_zero(self):
        result = _spread(input_population=[], output_population=[])
        assert result.status == "insufficient_data"
        assert result.spread is None
        assert result.status_reason

    def test_min_dates_floor_blocks_a_one_week_grade(self):
        inputs = _obs(D1, [("A", 0.1), ("B", 0.3)])
        outputs = _obs(D1, [("B", 0.3)])
        result = _spread(input_population=inputs, output_population=outputs, min_dates=4)
        assert result.status == "insufficient_data"
        assert result.spread is None
        assert "need 4" in result.status_reason

    def test_status_ok_never_carries_a_none_spread(self):
        result = _spread(
            input_population=_obs(D1, [("A", 0.1), ("B", 0.3)]),
            output_population=_obs(D1, [("B", 0.3)]),
        )
        assert (result.status == "ok") is (result.spread is not None)


class TestPublishedRecordIsRecomputable:
    def _published(self):
        inputs = _obs(D1, [("A", 0.0), ("B", 0.1), ("C", 0.2)]) + _obs(
            D2, [("A", 0.0), ("B", 0.4)]
        )
        outputs = _obs(D1, [("C", 0.2)]) + _obs(D2, [("B", 0.4)])
        return _spread(input_population=inputs, output_population=outputs)

    def test_reader_reproduces_the_headline_from_the_record_alone(self):
        result = self._published()
        record = json.loads(json.dumps(result.to_dict()))
        assert recompute_spread(record) == pytest.approx(result.spread, abs=1e-9)

    def test_recompute_needs_only_the_per_date_rows(self):
        record = {"per_date": [{"input_mean": 0.1, "output_mean": 0.2}]}
        assert recompute_spread(record) == pytest.approx(0.1)

    def test_a_tampered_headline_is_caught_by_the_recompute(self):
        result = self._published()
        record = result.to_dict()
        record["spread"] = 0.99
        assert recompute_spread(record) != pytest.approx(record["spread"])

    def test_empty_per_date_recomputes_to_none(self):
        assert recompute_spread({"per_date": []}) is None

    def test_record_without_per_date_raises(self):
        with pytest.raises(ValueError, match="no 'per_date'"):
            recompute_spread({"spread": 0.15})

    def test_record_with_non_list_per_date_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            recompute_spread({"per_date": {"cohort_date": D1}})

    def test_row_that_is_not_an_object_raises(self):
        with pytest.raises(ValueError, match="not an object"):
            recompute_spread({"per_date": [[0.1, 0.2]]})

    @pytest.mark.parametrize("missing", ["input_mean", "output_mean"])
    def test_row_missing_a_mean_raises_rather_than_scoring(self, missing):
        row = {"input_mean": 0.1, "output_mean": 0.2}
        del row[missing]
        with pytest.raises(ValueError, match="missing"):
            recompute_spread({"per_date": [row]})

    def test_record_carries_every_row_it_averaged(self):
        result = self._published()
        record = result.to_dict()
        assert len(record["per_date"]) == result.n_dates
        assert record["estimator"] == "date_clustered_mean_spread"
        assert record["schema_version"] == 1

    def test_record_publishes_both_population_coverage_blocks(self):
        record = self._published().to_dict()
        for side in ("input_population", "output_population"):
            block = record[side]
            assert set(block) >= {
                "mean",
                "pooled_mean",
                "n_dates",
                "n_resolved",
                "n_unresolved",
                "unresolved_rate",
                "per_date",
            }

    def test_insufficient_data_record_still_round_trips(self):
        record = json.loads(json.dumps(_spread(input_population=[], output_population=[]).to_dict()))
        assert record["status"] == "insufficient_data"
        assert record["spread"] is None
        assert recompute_spread(record) is None


class TestSeamShapesFromTheDirective:
    """The four seams of alpha-engine-config#7214, each as a shape test."""

    def test_scanner_seam_top20_vs_universe(self):
        universe = _obs(D1, [(f"T{i:03d}", i / 1000.0) for i in range(900)])
        top20 = _obs(D1, [(f"T{i:03d}", i / 1000.0) for i in range(880, 900)])
        result = _spread(
            seam="scanner",
            input_population=universe,
            output_population=top20,
        )
        # universe mean = 0.4495 ; top-20 mean = mean(0.880..0.899) = 0.8895.
        assert result.spread == pytest.approx(0.8895 - 0.4495)

    def test_predictor_seam_portfolio_vs_top20(self):
        top20 = _obs(D1, [(f"T{i:02d}", i / 100.0) for i in range(20)])
        portfolio = _obs(D1, [(f"T{i:02d}", i / 100.0) for i in (0, 1, 2)])
        result = _spread(
            seam="predictor",
            input_population=top20,
            output_population=portfolio,
        )
        assert result.spread < 0, "a portfolio picking the bottom of its input grades negative"

    def test_executor_seam_is_a_measurement_seam_in_bps(self):
        planned = _obs(D1, [("ORD1", 0.0), ("ORD2", 0.0)])
        filled = _obs(D1, [("ORD1", -4.0), ("ORD2", -6.0)])
        result = _spread(
            seam="executor",
            outcome_unit="bps",
            input_population=planned,
            output_population=filled,
            require_outcome_agreement=False,
        )
        assert result.spread == pytest.approx(-5.0)
        assert result.to_dict()["seam_kind"] == "measurement"

    def test_backtester_seam_is_simulation_minus_realized(self):
        realized = _obs(D1, [("TRD1", 0.010), ("TRD2", -0.004)])
        simulated = _obs(D1, [("TRD1", 0.012), ("TRD2", -0.004)])
        result = _spread(
            seam="backtester",
            outcome_unit="simple_return",
            input_population=realized,
            output_population=simulated,
            require_subset=True,
            require_outcome_agreement=False,
        )
        # The simulation is optimistic by 10bps on average over the window.
        assert result.spread == pytest.approx(0.001)

    def test_a_simulation_that_invents_a_trade_still_raises(self):
        realized = _obs(D1, [("TRD1", 0.010)])
        simulated = _obs(D1, [("TRD1", 0.012), ("NEVER_HAPPENED", 0.5)])
        with pytest.raises(ValueError, match="selected FROM"):
            _spread(
                seam="backtester",
                input_population=realized,
                output_population=simulated,
                require_outcome_agreement=False,
            )

    def test_selection_seams_are_labelled_selection(self):
        result = _spread(
            input_population=_obs(D1, [("A", 0.1), ("B", 0.3)]),
            output_population=_obs(D1, [("B", 0.3)]),
        )
        assert result.to_dict()["seam_kind"] == "selection"
