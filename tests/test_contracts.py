"""Tests for nousergon_lib.contracts — slot boundary schemas + conformance kit (M0).

Fixtures are SYNTHETIC minimal-conforming payloads (never live artifacts — the
lib repo is public; thesis text / live scores stay out of it).
"""

import copy
import json

import pytest

jsonschema = pytest.importorskip("jsonschema")

from nousergon_lib import contracts
from nousergon_lib.contracts import ContractViolation


def _signal_entry(**overrides):
    entry = {
        "ticker": "TEST",
        "signal": "ENTER",
        "score": 72.5,
        "rating": "BUY",
        "conviction": "rising",
        "sector": "Information Technology",
        "sector_rating": "overweight",
        "price_target_upside": 0.18,
    }
    entry.update(overrides)
    return entry


def _signals_payload(**overrides):
    payload = {
        "date": "2026-06-11",
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "overweight", "modifier": 1.1}},
        "sector_modifiers": {"Technology": 1.1},
        "universe": [_signal_entry()],
        "buy_candidates": [_signal_entry(ticker="TST2", signal="HOLD", rating="HOLD")],
        "population": ["TEST"],
    }
    payload.update(overrides)
    return payload


def _research_intel_payload(**overrides):
    payload = {
        "schema_version": 1,
        "date": "2026-06-11",
        "generated_at": "2026-06-13T09:00:00Z",
        "market_regime": "neutral",
        "regime_narrative": "Macro backdrop is range-bound.",
        "sector_ratings": {
            "Technology": {"rating": "overweight", "rationale": "AI capex tailwind"},
        },
        "sector_modifiers": {"Technology": 1.1},
        "market_breadth": {
            "pct_above_50d_ma": 58.0,
            "pct_above_200d_ma": 62.0,
            "advance_decline_ratio": 1.4,
        },
        "attractiveness": {
            "AAA": {
                "ticker": "AAA",
                "score": 82.0,
                "sector": "Technology",
                "breakdown": {
                    "quant_score": 80.0,
                    "qual_score": 84.0,
                    "factor_subscore": None,
                    "weighted_base": 80.0,
                    "macro_shift": 1.0,
                },
                "thesis": {"bull_case": "Durable moat", "sector": "Technology"},
            },
        },
    }
    payload.update(overrides)
    return payload


def _prediction_entry(**overrides):
    entry = {
        "ticker": "TEST",
        "predicted_direction": "UP",
        "prediction_confidence": 0.62,
        "predicted_alpha": 0.031,
        "combined_rank": 1,
        "gbm_veto": False,
        "momentum_veto": False,
    }
    entry.update(overrides)
    return entry


def _predictions_payload(**overrides):
    payload = {
        "date": "2026-06-11",
        "model_version": "v3.0-test",
        "n_predictions": 1,
        "predictions": [_prediction_entry()],
    }
    payload.update(overrides)
    return payload


class TestSchemasAreWellFormed:
    @pytest.mark.parametrize("name", sorted(contracts.CONTRACT_SCHEMAS))
    def test_schema_passes_metaschema(self, name):
        jsonschema.Draft202012Validator.check_schema(contracts.load_schema(name))

    @pytest.mark.parametrize("name", sorted(contracts.CONTRACT_SCHEMAS))
    def test_schema_is_versioned(self, name):
        schema = contracts.load_schema(name)
        assert "$id" in schema, "every contract schema carries a stable $id"
        assert f"v{contracts.SCHEMA_VERSIONS[name]}" in schema["$id"]
        # additive evolution: the contract must NOT lock out new fields
        assert schema.get("additionalProperties", True) is not False

    def test_slot_schemas_are_a_subset_of_contract_schemas(self):
        # SLOT_SCHEMAS (R/M/S product boundaries) is a labelled subset of the
        # full contract registry; outcome_record is a contract but not a slot.
        assert set(contracts.SLOT_SCHEMAS) <= set(contracts.CONTRACT_SCHEMAS)
        assert "outcome_record" in contracts.CONTRACT_SCHEMAS
        assert "outcome_record" not in contracts.SLOT_SCHEMAS

    def test_unknown_contract_raises(self):
        with pytest.raises(KeyError):
            contracts.load_schema("nope")


class TestSignalsContract:
    def test_minimal_conforming_payload(self):
        contracts.validate("signals", _signals_payload())

    def test_schema_version_stamp_accepted(self):
        contracts.validate("signals", _signals_payload(schema_version=1))

    @pytest.mark.parametrize(
        "missing", ["date", "market_regime", "universe", "buy_candidates", "sector_modifiers"]
    )
    def test_missing_required_top_level_fails(self, missing):
        payload = _signals_payload()
        del payload[missing]
        errors = contracts.conformance_errors("signals", payload)
        assert errors and missing in " ".join(errors)

    @pytest.mark.parametrize(
        "field", ["ticker", "signal", "score", "conviction", "sector_rating", "price_target_upside"]
    )
    def test_missing_required_entry_field_fails(self, field):
        entry = _signal_entry()
        del entry[field]
        payload = _signals_payload(universe=[entry])
        assert contracts.conformance_errors("signals", payload)

    def test_null_score_is_tolerated(self):
        contracts.validate(
            "signals", _signals_payload(universe=[_signal_entry(score=None, signal="HOLD")])
        )

    def test_legacy_caution_regime_rejected_on_write_contract(self):
        assert contracts.conformance_errors("signals", _signals_payload(market_regime="caution"))

    def test_additive_fields_pass(self):
        entry = _signal_entry(brand_new_optional_field={"anything": 1})
        contracts.validate("signals", _signals_payload(universe=[entry], some_new_top_level=True))


class TestResearchIntelContract:
    def test_minimal_conforming_payload(self):
        contracts.validate("research_intel", _research_intel_payload())

    @pytest.mark.parametrize(
        "missing",
        [
            "schema_version",
            "date",
            "generated_at",
            "market_regime",
            "sector_ratings",
            "sector_modifiers",
            "market_breadth",
            "attractiveness",
        ],
    )
    def test_missing_required_top_level_fails(self, missing):
        payload = _research_intel_payload()
        del payload[missing]
        errors = contracts.conformance_errors("research_intel", payload)
        assert errors and missing in " ".join(errors)

    def test_null_regime_narrative_tolerated(self):
        contracts.validate("research_intel", _research_intel_payload(regime_narrative=None))

    def test_null_attractiveness_score_tolerated(self):
        payload = _research_intel_payload()
        payload["attractiveness"]["AAA"]["score"] = None
        contracts.validate("research_intel", payload)

    def test_legacy_caution_regime_rejected_on_write_contract(self):
        assert contracts.conformance_errors(
            "research_intel", _research_intel_payload(market_regime="caution")
        )

    def test_sector_modifier_out_of_range_fails(self):
        assert contracts.conformance_errors(
            "research_intel",
            _research_intel_payload(sector_modifiers={"Technology": 1.9}),
        )

    def test_invalid_sector_rating_fails(self):
        payload = _research_intel_payload(
            sector_ratings={"Technology": {"rating": "strong_buy"}}
        )
        assert contracts.conformance_errors("research_intel", payload)

    def test_attractiveness_entry_requires_ticker_and_score(self):
        payload = _research_intel_payload()
        del payload["attractiveness"]["AAA"]["ticker"]
        assert contracts.conformance_errors("research_intel", payload)

    def test_additive_fields_pass(self):
        payload = _research_intel_payload(some_new_top_level=True)
        payload["attractiveness"]["AAA"]["brand_new_field"] = {"x": 1}
        contracts.validate("research_intel", payload)


class TestPredictionsContract:
    def test_minimal_conforming_payload(self):
        contracts.validate("predictions", _predictions_payload())

    @pytest.mark.parametrize("missing", ["date", "model_version", "n_predictions", "predictions"])
    def test_missing_required_top_level_fails(self, missing):
        payload = _predictions_payload()
        del payload[missing]
        assert contracts.conformance_errors("predictions", payload)

    @pytest.mark.parametrize(
        "field",
        [
            "ticker",
            "predicted_direction",
            "prediction_confidence",
            "predicted_alpha",
            "combined_rank",
            "gbm_veto",
            "momentum_veto",
        ],
    )
    def test_missing_required_entry_field_fails(self, field):
        entry = _prediction_entry()
        del entry[field]
        payload = _predictions_payload(predictions=[entry])
        assert contracts.conformance_errors("predictions", payload)

    def test_flat_direction_rejected_on_write_contract(self):
        entry = _prediction_entry(predicted_direction="FLAT")
        assert contracts.conformance_errors("predictions", _predictions_payload(predictions=[entry]))

    def test_confidence_out_of_range_fails(self):
        entry = _prediction_entry(prediction_confidence=1.4)
        assert contracts.conformance_errors("predictions", _predictions_payload(predictions=[entry]))

    def test_null_combined_rank_tolerated(self):
        contracts.validate(
            "predictions",
            _predictions_payload(predictions=[_prediction_entry(combined_rank=None)]),
        )

    def test_null_observe_blocks_tolerated(self):
        # write_output emits these keys unconditionally, null when the stage
        # did not run (caught live by predictor conformance, 2026-06-11)
        contracts.validate(
            "predictions",
            _predictions_payload(output_distribution_gate=None, level_neutralization=None),
        )

    def test_optional_modern_fields_pass(self):
        entry = _prediction_entry(
            predicted_alpha_std=0.012,
            barrier_win_prob=0.55,
            stance="momentum",
            stance_loadings={"momentum": 0.7, "value": 0.1, "quality": 0.1, "catalyst": 0.1},
            catalyst_date=None,
            meta_alpha_tb=None,
        )
        contracts.validate("predictions", _predictions_payload(predictions=[entry]))


def _outcome_record(**overrides):
    record = {
        "schema_version": 1,
        "signal_id": "AAPL:2026-06-11",
        "score_date": "2026-06-11",
        "horizon_days": 21,
        "beat_spy": True,
        "stock_return": 0.043,
        "spy_return": 0.021,
        "log_alpha": 0.0215,
        "resolved_at": "2026-07-10T13:07:29Z",
        "is_primary": True,
    }
    record.update(overrides)
    return record


class TestOutcomeRecordContract:
    def test_minimal_primary_record(self):
        contracts.validate("outcome_record", _outcome_record())

    def test_diagnostic_record_may_have_null_log_alpha(self):
        # A non-primary (diagnostic) horizon carries no canonical alpha — null
        # log_alpha is valid when is_primary is False.
        contracts.validate(
            "outcome_record",
            _outcome_record(horizon_days=5, is_primary=False, log_alpha=None),
        )

    def test_primary_record_null_log_alpha_fails(self):
        # The canonical label must be present on the primary horizon (fail-loud
        # guardrail via the if/then conditional).
        errors = contracts.conformance_errors(
            "outcome_record", _outcome_record(log_alpha=None)
        )
        assert errors and "log_alpha" in " ".join(errors)

    @pytest.mark.parametrize(
        "missing",
        [
            "schema_version",
            "signal_id",
            "score_date",
            "horizon_days",
            "beat_spy",
            "stock_return",
            "spy_return",
            "log_alpha",
            "resolved_at",
            "is_primary",
        ],
    )
    def test_missing_required_field_fails(self, missing):
        record = _outcome_record()
        del record[missing]
        assert contracts.conformance_errors("outcome_record", record)

    def test_horizon_days_must_be_positive_integer(self):
        assert contracts.conformance_errors("outcome_record", _outcome_record(horizon_days=0))
        assert contracts.conformance_errors("outcome_record", _outcome_record(horizon_days=1.5))

    def test_bad_score_date_pattern_fails(self):
        assert contracts.conformance_errors(
            "outcome_record", _outcome_record(score_date="2026/06/11")
        )

    def test_additive_fields_pass(self):
        # Additive-only evolution — an unknown optional field must NOT break the
        # contract (deliberate, per S3 Contract Safety; the schema keeps
        # additionalProperties open rather than the sketch's false).
        contracts.validate("outcome_record", _outcome_record(some_future_field=1.23))


def _ic_block(**overrides):
    block = {
        "date_ic_mean": 0.06,
        "date_ic_t": 2.1,
        "date_ic_p": 0.04,
        "n_eval_dates": 9,
        "pooled_ic": 0.05,
        "pooled_ic_p": 0.01,
        "n": 430,
    }
    block.update(overrides)
    return block


def _trajectory_ic_block(**overrides):
    block = _ic_block(status="ok")
    block.update(overrides)
    return block


def _attractiveness_eval(**overrides):
    payload = {
        "schema_version": 2,
        "status": "ok",
        "as_of": "2026-07-06",
        "horizon_days": 21,
        "composite_ic": _ic_block(),
        "pillar_ic": {
            "quality": {"date_ic_mean": 0.04, "date_ic_p": 0.06, "n_eval_dates": 9},
        },
        "suggested_pillar_weights": {"quality": 0.55, "value": 0.45},
        "shrinkage": {"method": "demiguel_1overN", "lambda": 0.9, "n_eval_dates": 9},
        "trajectory_ic": {
            "pre_repricing_score": _trajectory_ic_block(),
            "attr_slope_z": _trajectory_ic_block(status="accruing"),
        },
        "counterfactual": {
            "top_n": [
                {"n": 10, "sector_balanced": False, "capture_rate": 0.34, "mean_alpha": 0.0182},
                {"n": 25, "sector_balanced": True, "capture_rate": 0.49, "mean_alpha": 0.0114},
            ],
            "live_gate": {"capture_rate": 0.41, "mean_alpha": 0.0063, "n_survivors": 27},
        },
    }
    payload.update(overrides)
    return payload


class TestAttractivenessEvalContract:
    def test_minimal_conforming_payload(self):
        contracts.validate("attractiveness_eval", _attractiveness_eval())

    def test_is_registered_at_v2(self):
        assert contracts.SCHEMA_VERSIONS["attractiveness_eval"] == 2
        schema = contracts.load_schema("attractiveness_eval")
        assert schema["properties"]["schema_version"]["const"] == 2
        assert "v2" in schema["$id"]

    def test_not_a_slot_boundary(self):
        # eval-storage contract, not an R/M/S product slot
        assert "attractiveness_eval" not in contracts.SLOT_SCHEMAS

    @pytest.mark.parametrize(
        "missing",
        [
            "schema_version",
            "status",
            "as_of",
            "horizon_days",
            "composite_ic",
            "counterfactual",
            "shrinkage",
            "trajectory_ic",
        ],
    )
    def test_missing_required_top_level_fails(self, missing):
        payload = _attractiveness_eval()
        del payload[missing]
        errors = contracts.conformance_errors("attractiveness_eval", payload)
        assert errors and missing in " ".join(errors)

    def test_v2_uses_mean_alpha_not_horizon_suffixed_name(self):
        # The v2 rename: mean_alpha is REQUIRED; the legacy horizon-suffixed
        # field name is no longer part of the contract (config#1861).
        payload = _attractiveness_eval()
        entry = payload["counterfactual"]["top_n"][0]
        entry["mean_alpha_21d"] = entry.pop("mean_alpha")
        errors = contracts.conformance_errors("attractiveness_eval", payload)
        assert errors and "mean_alpha" in " ".join(errors)

    def test_legacy_v1_stamp_rejected(self):
        # A v1 (schema_version=1) artifact must not silently validate as v2.
        assert contracts.conformance_errors(
            "attractiveness_eval", _attractiveness_eval(schema_version=1)
        )

    def test_null_alpha_and_capture_tolerated(self):
        payload = _attractiveness_eval()
        payload["counterfactual"]["top_n"][0]["mean_alpha"] = None
        payload["counterfactual"]["top_n"][0]["capture_rate"] = None
        payload["counterfactual"]["live_gate"]["mean_alpha"] = None
        contracts.validate("attractiveness_eval", payload)

    def test_insufficient_data_shape_is_additive(self):
        contracts.validate("attractiveness_eval", _attractiveness_eval(reason="warming up"))

    def test_additive_fields_pass(self):
        payload = _attractiveness_eval(brand_new_top_level={"x": 1})
        payload["counterfactual"]["top_n"][0]["n_cycles"] = 4
        contracts.validate("attractiveness_eval", payload)


def _loop_record(**overrides):
    record = {
        "outcome": "promoted",
        "blocked_by": None,
        "consecutive_blocked_weeks": 0,
        "detail": "live config written this run",
    }
    record.update(overrides)
    return record


def _apply_audit(**overrides):
    payload = {
        "schema_version": 1,
        "as_of": "2026-07-06",
        "loops": {
            "scoring_weights": _loop_record(),
            "executor_params": _loop_record(
                outcome="blocked",
                blocked_by=["min_trades_to_promote"],
                consecutive_blocked_weeks=3,
                detail="too few trades to promote",
            ),
            "predictor_params": _loop_record(outcome="insufficient_data", detail="thin inputs"),
            "research_params": _loop_record(outcome="disabled", detail="enforce flag off"),
        },
    }
    payload.update(overrides)
    return payload


class TestApplyAuditContract:
    def test_minimal_conforming_payload(self):
        contracts.validate("apply_audit", _apply_audit())

    def test_is_registered_at_v1(self):
        assert contracts.SCHEMA_VERSIONS["apply_audit"] == 1
        schema = contracts.load_schema("apply_audit")
        assert "v1" in schema["$id"]
        assert "apply_audit" not in contracts.SLOT_SCHEMAS

    @pytest.mark.parametrize("missing", ["schema_version", "as_of", "loops"])
    def test_missing_required_top_level_fails(self, missing):
        payload = _apply_audit()
        del payload[missing]
        errors = contracts.conformance_errors("apply_audit", payload)
        assert errors and missing in " ".join(errors)

    @pytest.mark.parametrize(
        "loop", ["scoring_weights", "executor_params", "predictor_params", "research_params"]
    )
    def test_all_four_loops_required(self, loop):
        payload = _apply_audit()
        del payload["loops"][loop]
        assert contracts.conformance_errors("apply_audit", payload)

    @pytest.mark.parametrize("field", ["outcome", "blocked_by", "consecutive_blocked_weeks", "detail"])
    def test_missing_required_loop_field_fails(self, field):
        payload = _apply_audit()
        del payload["loops"]["scoring_weights"][field]
        assert contracts.conformance_errors("apply_audit", payload)

    def test_unknown_outcome_rejected(self):
        payload = _apply_audit()
        payload["loops"]["scoring_weights"]["outcome"] = "vetoed"
        assert contracts.conformance_errors("apply_audit", payload)

    def test_unknown_blocked_by_slug_rejected(self):
        payload = _apply_audit()
        payload["loops"]["executor_params"]["blocked_by"] = ["mystery_gate"]
        assert contracts.conformance_errors("apply_audit", payload)

    def test_empty_blocked_by_array_rejected(self):
        # blocked_by is either null or a non-empty array (minItems 1)
        payload = _apply_audit()
        payload["loops"]["executor_params"]["blocked_by"] = []
        assert contracts.conformance_errors("apply_audit", payload)

    def test_extra_loop_via_additional_properties(self):
        # additionalProperties on loops is itself a loop_record — a new named
        # loop is additive, not a break.
        payload = _apply_audit()
        payload["loops"]["some_new_loop"] = _loop_record()
        contracts.validate("apply_audit", payload)


class TestConformanceKitApi:
    def test_validate_raises_with_full_error_list(self):
        payload = _predictions_payload()
        del payload["predictions"][0]["ticker"]
        del payload["date"]
        with pytest.raises(ContractViolation) as exc:
            contracts.validate("predictions", payload)
        assert exc.value.name == "predictions"
        assert len(exc.value.errors) == 2

    def test_conformance_errors_empty_on_valid(self):
        assert contracts.conformance_errors("signals", _signals_payload()) == []

    def test_payload_not_mutated(self):
        payload = _signals_payload()
        snapshot = copy.deepcopy(payload)
        contracts.validate("signals", payload)
        assert payload == snapshot


class TestCli:
    def test_cli_ok_and_fail(self, tmp_path, capsys):
        from nousergon_lib.contracts.__main__ import main

        good = tmp_path / "good.json"
        good.write_text(json.dumps(_predictions_payload()))
        assert main(["validate", "predictions", str(good)]) == 0
        assert "OK" in capsys.readouterr().out

        bad_payload = _predictions_payload()
        del bad_payload["predictions"][0]["gbm_veto"]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(bad_payload))
        assert main(["validate", "predictions", str(bad)]) == 1
        assert "gbm_veto" in capsys.readouterr().err

    def test_cli_validates_outcome_record(self, tmp_path, capsys):
        from nousergon_lib.contracts.__main__ import main

        good = tmp_path / "outcome.json"
        good.write_text(json.dumps(_outcome_record()))
        assert main(["validate", "outcome_record", str(good)]) == 0
        assert "OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# experiment.v1 (Crucible manifest — Phase A, config#1966)
# ---------------------------------------------------------------------------

def _manifest(**overrides):
    payload = {
        "schema_version": 1,
        "experiment": {
            "id": "single-agent-sonnet-baseline",
            "window": {"start": "2026-01-05", "end": "2026-06-26"},
            "universe": "sp500_400",
        },
        "slots": {
            "research": {"impl": "command", "run": "python my_agent.py --out ./out"},
            "model": {"impl": "stock"},
            "strategy": {"impl": "entry_point", "ref": "mypkg.rules:MyTrailingStop"},
        },
        "evaluation": {"horizons": ["21d", "5d"], "rubrics": ["default"], "gates": ["default"]},
        "data": {"snapshot": "latest"},
    }
    payload.update(overrides)
    return payload


class TestExperimentContract:
    def test_full_conforming_manifest(self):
        assert contracts.conformance_errors("experiment", _manifest()) == []

    def test_all_stock_minimal_manifest(self):
        # Run = config: the all-stock manifest needs zero code and zero
        # optional sections.
        payload = {
            "schema_version": 1,
            "experiment": {"id": "reference-rate",
                           "window": {"start": "2026-03-09", "end": "2026-07-07"}},
            "slots": {},
        }
        assert contracts.conformance_errors("experiment", payload) == []

    @pytest.mark.parametrize("missing", ["schema_version", "experiment", "slots"])
    def test_missing_required_top_level_fails(self, missing):
        payload = _manifest()
        payload.pop(missing)
        assert contracts.conformance_errors("experiment", payload)

    def test_artifact_impl_requires_path(self):
        payload = _manifest()
        payload["slots"]["research"] = {"impl": "artifact"}  # no path
        assert contracts.conformance_errors("experiment", payload)

    def test_command_impl_requires_run(self):
        payload = _manifest()
        payload["slots"]["model"] = {"impl": "command"}  # no run
        assert contracts.conformance_errors("experiment", payload)

    def test_entry_point_ref_shape_enforced(self):
        payload = _manifest()
        payload["slots"]["strategy"] = {"impl": "entry_point", "ref": "not a ref"}
        assert contracts.conformance_errors("experiment", payload)

    def test_strategy_rejects_artifact_impl(self):
        # Slot S is a code contract — artifact/command bindings are R/M-only.
        payload = _manifest()
        payload["slots"]["strategy"] = {"impl": "artifact", "path": "./x"}
        assert contracts.conformance_errors("experiment", payload)

    def test_unknown_slot_key_rejected(self):
        # slots is the ONE closed map in the manifest: a typo'd slot name must
        # fail loudly, not silently no-op the user's binding.
        payload = _manifest()
        payload["slots"]["resarch"] = {"impl": "stock"}
        assert contracts.conformance_errors("experiment", payload)

    def test_bad_horizon_shape_fails(self):
        payload = _manifest()
        payload["evaluation"]["horizons"] = ["21days"]
        assert contracts.conformance_errors("experiment", payload)


# ---------------------------------------------------------------------------
# experiment_record.v1 (Crucible run index — Phase A, config#1966)
# ---------------------------------------------------------------------------

def _run_record(**overrides):
    payload = {
        "schema_version": 1,
        "experiment_id": "reference-rate",
        "run_date": "2026-07-04",
        "generated_at": "2026-07-04T12:00:00+00:00",
        "status": "partial",
        "manifest": {"hash": "a41f9c2e", "inline": _manifest()},
        "slots": [
            {"slot": "research", "impl": "stock", "fingerprint": "stock@abc1234"},
            {"slot": "model", "impl": "stock", "fingerprint": "stock@def5678"},
        ],
        "data_snapshot": "2026-07-04T09:00Z",
        "git": {"crucible-research": "abc1234"},
        "artifacts": [
            {"name": "report_card", "status": "emitted",
             "key": "evaluator/2026-07-04/report_card.json"},
            {"name": "pit_parity", "status": "absent",
             "reason": "parity pass skipped: OOM guard on the spot instance"},
        ],
    }
    payload.update(overrides)
    return payload


class TestExperimentRecordContract:
    def test_conforming_record(self):
        assert contracts.conformance_errors("experiment_record", _run_record()) == []

    @pytest.mark.parametrize(
        "missing",
        ["schema_version", "experiment_id", "run_date", "status", "manifest", "slots", "artifacts"],
    )
    def test_missing_required_fails(self, missing):
        payload = _run_record()
        payload.pop(missing)
        assert contracts.conformance_errors("experiment_record", payload)

    def test_emitted_artifact_requires_key(self):
        payload = _run_record()
        payload["artifacts"] = [{"name": "report_card", "status": "emitted"}]
        assert contracts.conformance_errors("experiment_record", payload)

    def test_absent_artifact_requires_reason(self):
        # Honest absence is the contract: an absent artifact without a reason
        # is exactly the silent omission the record exists to prevent.
        payload = _run_record()
        payload["artifacts"] = [{"name": "pit_parity", "status": "absent"}]
        assert contracts.conformance_errors("experiment_record", payload)

    def test_bad_status_enum_fails(self):
        assert contracts.conformance_errors("experiment_record", _run_record(status="ok"))

    def test_envelope_contracts_are_not_slots(self):
        assert "experiment" in contracts.CONTRACT_SCHEMAS
        assert "experiment_record" in contracts.CONTRACT_SCHEMAS
        assert "experiment" not in contracts.SLOT_SCHEMAS
        assert "experiment_record" not in contracts.SLOT_SCHEMAS
