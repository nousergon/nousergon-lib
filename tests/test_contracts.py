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
    @pytest.mark.parametrize("name", sorted(contracts.SLOT_SCHEMAS))
    def test_schema_passes_metaschema(self, name):
        jsonschema.Draft202012Validator.check_schema(contracts.load_schema(name))

    @pytest.mark.parametrize("name", sorted(contracts.SLOT_SCHEMAS))
    def test_schema_is_versioned(self, name):
        schema = contracts.load_schema(name)
        assert "$id" in schema, "every contract schema carries a stable $id"
        assert f"v{contracts.SCHEMA_VERSIONS[name]}" in schema["$id"]
        # additive evolution: the contract must NOT lock out new fields
        assert schema.get("additionalProperties", True) is not False

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
