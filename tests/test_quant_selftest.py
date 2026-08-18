"""Tests for nousergon_lib/quant/selftest.py — the lifted self-test runner
(alpha-engine-config-I7238).

These exercise the outcome taxonomy directly, since the four consumers
(backtester/evaluator/predictor/research) each wrap this module with their own
``build_cases()`` and thin identity/provenance bindings — that consumer-specific
wiring is covered by each repo's own ``tests/test_self_test.py``.
"""

from __future__ import annotations

import pytest

from nousergon_lib.quant.selftest import (
    CASE_TIMEOUT_SECONDS,
    FAIL,
    PASS,
    UNKNOWN,
    Case,
    SelfTestTimeout,
    _call_with_timeout,
    code_sha,
    resolved_library_versions,
    run_self_test,
    verdict_is_pass,
)


def _case(name="c", expected=1.0, compute=lambda: 1.0, tolerance=1e-9, **kw) -> Case:
    return Case(
        name=name, description="d", inputs={"units": "n/a"},
        expected=expected, compute=compute, tolerance=tolerance, **kw,
    )


class TestResolvedLibraryVersions:
    def test_missing_distribution_recorded_explicitly(self):
        resolved = resolved_library_versions(("definitely-not-a-package",))
        assert resolved["definitely-not-a-package"] == "<not installed>"

    def test_installed_distribution_resolves(self):
        resolved = resolved_library_versions(("pytest",))
        assert resolved["pytest"] != "<not installed>"


class TestCodeSha:
    def test_none_root_is_unknown_absent_env(self, monkeypatch):
        for key in ("GIT_SHA", "CODE_SHA", "GITHUB_SHA"):
            monkeypatch.delenv(key, raising=False)
        assert code_sha(None) == "unknown"

    def test_env_stamp_wins(self, monkeypatch):
        monkeypatch.setenv("GIT_SHA", "deadbeef")
        assert code_sha(None) == "deadbeef"


class TestCallWithTimeout:
    def test_raises_on_overrun(self):
        def _busy():
            import time
            time.sleep(0.3)
            return 1.0

        with pytest.raises(SelfTestTimeout):
            _call_with_timeout(_busy, 0.05)

    def test_returns_value_under_budget(self):
        assert _call_with_timeout(lambda: 42.0, 5.0) == 42.0


class TestRunSelfTest:
    def _run(self, cases, **kw):
        return run_self_test(
            "2026-08-18",
            case_provider=lambda: cases,
            component="test-component",
            schema="test-schema",
            resolved_libraries={"pytest": "0"},
            code_sha_value="abc123",
            **kw,
        )

    def test_agreeing_case_is_pass(self):
        out = self._run([_case(expected=1.0, compute=lambda: 1.0)])
        assert out["verdict"] == PASS
        assert out["cases"][0]["verdict"] == PASS
        assert out["n_failed"] == 0 and out["n_errored"] == 0

    def test_disagreeing_case_is_fail(self):
        out = self._run([_case(expected=1.0, compute=lambda: 2.0)])
        assert out["verdict"] == FAIL
        assert out["cases"][0]["verdict"] == FAIL

    def test_erroring_case_is_unknown_not_fail(self):
        def _boom():
            raise RuntimeError("no data")

        out = self._run([_case(compute=_boom)])
        assert out["cases"][0]["verdict"] == UNKNOWN
        assert out["verdict"] == UNKNOWN

    def test_timeout_is_fail_not_unknown(self):
        def _too_slow():
            raise SelfTestTimeout("case exceeded its budget")

        out = self._run([_case(compute=_too_slow)])
        assert out["cases"][0]["verdict"] == FAIL
        assert out["cases"][0]["timed_out"] is True
        assert out["verdict"] == FAIL

    def test_battery_that_cannot_be_built_is_unknown_never_raises(self):
        def _boom():
            raise ImportError("no lib")

        out = run_self_test(
            "2026-08-18", case_provider=_boom, component="c", schema="s",
            resolved_libraries={}, code_sha_value="x",
        )
        assert out["verdict"] == UNKNOWN
        assert out["status"] == "error"

    def test_empty_battery_is_unknown(self):
        out = self._run([])
        assert out["verdict"] == UNKNOWN

    def test_malformed_case_in_list_is_unknown_row_not_raise(self):
        out = self._run([1, 2])
        assert out["cases"][0]["verdict"] == UNKNOWN
        assert out["cases"][0]["errored"] is True

    def test_known_gap_case_annotates_body(self):
        out = self._run([
            _case(expected=1.0, compute=lambda: 1.0, known_gap=True,
                  gap_issue="alpha-engine-config-I9999"),
        ])
        assert out["cases"][0]["known_gap"] is True
        assert out["cases"][0]["gap_issue"] == "alpha-engine-config-I9999"
        assert out["n_known_gaps"] == 1

    def test_default_battery_has_zero_known_gaps(self):
        out = self._run([_case()])
        assert out["n_known_gaps"] == 0

    def test_extra_case_providers_composes_scopes(self):
        base = [_case(name="base")]
        extra = [_case(name="extra")]
        out = self._run(base, extra_case_providers={"training": lambda: extra},
                         primary_scope="inference")
        names = {c["case"] for c in out["cases"]}
        assert names == {"base", "extra"}
        assert out["scope"] == ["inference", "training"]

    def test_extra_case_providers_without_primary_scope_still_composes(self):
        base = [_case(name="base")]
        extra = [_case(name="extra")]
        out = self._run(base, extra_case_providers={"training": lambda: extra})
        names = {c["case"] for c in out["cases"]}
        assert names == {"base", "extra"}
        assert "scope" not in out

    def test_no_extra_case_providers_omits_scope_field_by_default(self):
        out = self._run([_case()])
        assert "scope" not in out

    def test_primary_scope_alone_publishes_single_element_scope(self):
        out = self._run([_case()], primary_scope="inference")
        assert out["scope"] == ["inference"]

    def test_extra_header_merges(self):
        out = self._run([_case()], extra_header={"scope_note": "n/a"})
        assert out["scope_note"] == "n/a"

    def test_case_timeout_seconds_default_is_thirty(self):
        assert CASE_TIMEOUT_SECONDS == 30.0


class TestVerdictIsPass:
    def test_only_explicit_pass_is_true(self):
        assert verdict_is_pass(PASS) is True
        assert verdict_is_pass(FAIL) is False
        assert verdict_is_pass(UNKNOWN) is False
        assert verdict_is_pass(None) is False
        assert verdict_is_pass("ok") is False
