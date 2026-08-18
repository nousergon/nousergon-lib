"""Perturbation-verification helper for self-test batteries (alpha-engine-config-I7262).

WHY THIS EXISTS
----------------
`alpha-engine-config-I7262`'s ``Closes-when`` requires each numeric self-test
battery to be **verified to fail against a deliberately perturbed
implementation** — a self-test that has never been shown to fail is not
evidence, and this fleet has shipped several detectors that could not fail
(``bugclass_a_guard_that_only_works_when_the_model_cooperates``,
``bugclass_a_check_written_over_already_clamped_values``, among others in the
2026-08-17 detector-integrity arc). This module is the standing guard against
adding another one to the self-test batteries specifically.

``crucible-predictor/tests/test_self_test.py`` and
``crucible-research/tests/test_self_test.py`` each independently grew the same
four-line pattern — monkeypatch one production attribute, rerun the battery,
assert it goes ``FAIL`` — which is exactly the shape ``shared-code-policy``'s
second-adoption trigger covers. This module is that lift, sitting alongside
``nousergon_lib.quant.selftest`` (the runner itself) so every current and
future battery inherits ONE proven-correct perturbation-assertion helper
rather than re-deriving the pattern per repo.

WHY THIS IS A SEPARATE MODULE FROM ``selftest.py``
----------------------------------------------------
``assert_perturbation_caught`` takes a ``pytest`` ``monkeypatch`` fixture as an
argument and is meaningful only inside a test — it has no place in
``run_self_test``'s own runtime import graph, which ships into a production
Lambda/spot-box image with no pytest present. Splitting it out mirrors the
existing ``nousergon_lib.pytest_time_shift`` precedent: a test-support module
that lives in the main package (no separate "dev" distribution needed) but is
imported ONLY from test code.

USAGE
-----
::

    from nousergon_lib.quant.selftest_perturbation import assert_perturbation_caught

    def test_a_perturbed_tanh_scale_is_caught(monkeypatch):
        assert_perturbation_caught(
            monkeypatch,
            module_path="regime.drawdown",
            attr="INTENSITY_Z_TANH_SCALE",
            perturbed=2.0001,
            run=lambda: run_self_test(run_date="2026-08-15"),
            case_name="composite_intensity_closed_form",  # optional
        )
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def assert_perturbation_caught(
    monkeypatch: MonkeyPatch,
    *,
    module_path: str,
    attr: str,
    perturbed: Any,
    run: Callable[[], dict],
    case_name: str | None = None,
    expect_verdict: str = "FAIL",
) -> dict:
    """Monkeypatch one production attribute, rerun the battery, and assert it
    goes ``FAIL`` (or ``expect_verdict``) — proving the battery is SENSITIVE to
    that value rather than merely asserted correct.

    ``module_path``/``attr`` name the production symbol to perturb (e.g.
    ``"regime.drawdown"`` / ``"INTENSITY_Z_TANH_SCALE"``, or a function such as
    ``"regime.composite"`` / ``"_zscore"``). ``perturbed`` is the replacement
    value or callable. A no-op perturbation (``perturbed == original``) is
    itself an assertion failure, since it would make the rest of this check
    vacuous.

    ``run`` is the caller's own zero-argument closure over
    ``run_self_test(...)`` — this module never imports a specific battery's
    runner, since ``run_date``, ``extra_case_providers`` and similar arguments
    are all caller-specific.

    ``case_name``, if given, additionally asserts that the SPECIFIC named case
    is what caught the perturbation — not merely that *some* case in the
    battery went red. A battery whose overall verdict flips while the intended
    case stays PASS is failing for the wrong reason, which this argument
    catches.

    Returns the perturbed run's body, for any further caller-specific
    assertions (e.g. checking a sibling case did NOT fire, as a metamorphic
    case should not fire under a convention-only perturbation).
    """
    module = importlib.import_module(module_path)
    original = getattr(module, attr)
    assert original != perturbed, (
        f"perturbation is a no-op: {module_path}.{attr} is already {perturbed!r}"
    )
    monkeypatch.setattr(module, attr, perturbed)

    out = run()
    assert out["verdict"] == expect_verdict, (
        f"perturbing {module_path}.{attr} to {perturbed!r} did not move the "
        f"battery to {expect_verdict} (got {out['verdict']!r}) — the self-test "
        f"cannot detect the thing it exists to detect"
    )
    if case_name is not None:
        matching = {c["verdict"] for c in out["cases"] if c["case"] == case_name}
        assert matching, f"no case named {case_name!r} in this battery's cases"
        assert expect_verdict in matching, (
            f"perturbing {module_path}.{attr} moved the battery to "
            f"{expect_verdict} but NOT case {case_name!r} specifically "
            f"(that case's verdict was {matching!r}) — verify the RIGHT case "
            f"caught it, not an unrelated one"
        )
    return out
