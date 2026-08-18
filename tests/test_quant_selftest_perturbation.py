"""Tests for nousergon_lib/quant/selftest_perturbation.py (alpha-engine-config-I7262)."""

from __future__ import annotations

import types

import pytest

from nousergon_lib.quant.selftest import Case, run_self_test
from nousergon_lib.quant.selftest_perturbation import assert_perturbation_caught

# A tiny stand-in "production module" with one tunable constant, mirroring the
# shape of regime.drawdown.INTENSITY_Z_TANH_SCALE etc.
_prod = types.SimpleNamespace(SCALE=2.0)


def _compute() -> float:
    return 10.0 * _prod.SCALE


def _cases() -> list[Case]:
    return [Case(name="scaled", description="d", inputs={"units": "n/a"},
                 expected=20.0, compute=_compute, tolerance=1e-9)]


def _run() -> dict:
    return run_self_test(
        "2026-08-18", case_provider=_cases, component="c", schema="s",
        resolved_libraries={}, code_sha_value="x",
    )


class TestAssertPerturbationCaught:
    def test_a_real_perturbation_is_caught(self, monkeypatch):
        out = assert_perturbation_caught(
            monkeypatch,
            module_path="tests.test_quant_selftest_perturbation",
            attr="_prod",
            perturbed=types.SimpleNamespace(SCALE=2.0001),
            run=_run,
        )
        assert out["verdict"] == "FAIL"

    def test_case_name_pins_the_specific_case(self, monkeypatch):
        assert_perturbation_caught(
            monkeypatch,
            module_path="tests.test_quant_selftest_perturbation",
            attr="_prod",
            perturbed=types.SimpleNamespace(SCALE=2.0001),
            run=_run,
            case_name="scaled",
        )

    def test_wrong_case_name_raises(self, monkeypatch):
        with pytest.raises(AssertionError, match="no case named"):
            assert_perturbation_caught(
                monkeypatch,
                module_path="tests.test_quant_selftest_perturbation",
                attr="_prod",
                perturbed=types.SimpleNamespace(SCALE=2.0001),
                run=_run,
                case_name="not_a_real_case",
            )

    def test_noop_perturbation_raises(self, monkeypatch):
        with pytest.raises(AssertionError, match="no-op"):
            assert_perturbation_caught(
                monkeypatch,
                module_path="tests.test_quant_selftest_perturbation",
                attr="_prod",
                perturbed=_prod,
                run=_run,
            )

    def test_a_perturbation_that_does_not_move_the_verdict_raises(self, monkeypatch):
        def _run_insensitive() -> dict:
            def cases() -> list[Case]:
                return [Case(name="const", description="d", inputs={"units": "n/a"},
                             expected=1.0, compute=lambda: 1.0, tolerance=1e-9)]
            return run_self_test(
                "2026-08-18", case_provider=cases, component="c", schema="s",
                resolved_libraries={}, code_sha_value="x",
            )

        with pytest.raises(AssertionError, match="did not move the battery"):
            assert_perturbation_caught(
                monkeypatch,
                module_path="tests.test_quant_selftest_perturbation",
                attr="_prod",
                perturbed=types.SimpleNamespace(SCALE=999.0),
                run=_run_insensitive,
            )
