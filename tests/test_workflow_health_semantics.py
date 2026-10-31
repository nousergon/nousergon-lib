"""The CI guard runs with yaml and NOTHING else (alpha-engine-config-I9169).

Lifted from `alpha-engine-config/scripts/` (alpha-engine-config-I9707).
`alpha-engine-config`'s `test_check_scheduled_workflow_health.py` exercises the
same parser through the reconciler. These tests pin the two properties that
only matter about the STANDALONE module: that it imports nothing from this
package or its heavier dependencies (the first version of this guard died in
CI on `ModuleNotFoundError: nousergon_lib`, because the PR-validator job had no
reason to install it), and that its CLI exits non-zero on a broken
declaration.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nousergon_lib import workflow_health_semantics as whs

#: Every runtime dependency of this package other than pyyaml. Blocking them
#: proves the module's import cost really is yaml + stdlib.
_BLOCKED = ("nousergon_lib", "krepis", "boto3", "botocore", "pydantic", "requests")

_STEP = "Compare live alarms against the codified tree"


def _wf(tmp_path: Path, *, declared_step: str = _STEP,
        real_step: str = _STEP) -> Path:
    (tmp_path / "alarm-drift.yml").write_text(
        "name: alarm drift\n"
        "on:\n"
        "  schedule:\n"
        '    - cron: "0 6 * * 1"\n'
        "env:\n"
        "  WORKFLOW_HEALTH_SEMANTICS: finding-surface\n"
        "  WORKFLOW_HEALTH_FINDING_STEPS: |\n"
        f"    {declared_step}\n"
        "  WORKFLOW_HEALTH_FINDING_ISSUE: alpha-engine-config-I8064\n"
        "  WORKFLOW_HEALTH_FINDING_REASON: drift found\n"
        "jobs:\n"
        "  main:\n"
        "    steps:\n"
        f'      - name: "{real_step}"\n'
        '        run: "true"\n'
    )
    return tmp_path


def test_the_module_imports_nothing_beyond_the_standard_library_and_yaml(tmp_path):
    """Run in a subprocess with this package and its other runtime
    dependencies made unimportable. A guard job may install pyyaml and no
    more; an import creeping back in here would fail in CI, not here, and only
    on the repo that added it.

    The file is copied out of the package and imported BY PATH, the way a
    pyyaml-only guard job would run it. (Putting the package directory itself
    on `sys.path` would let its `logging.py` shadow the standard library's.)"""
    shutil.copy(whs.__file__, tmp_path / "workflow_health_semantics.py")
    code = (
        "import sys;"
        "sys.meta_path.insert(0, type('B', (), {"
        "'find_module': lambda s, n, p=None: None,"
        "'find_spec': lambda s, n, p=None, t=None: "
        f"(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in {_BLOCKED!r} else None"
        "})());"
        f"sys.path.insert(0, {str(tmp_path)!r});"
        "import workflow_health_semantics;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_cli_passes_on_a_valid_declaration(tmp_path, capsys):
    assert whs.main(["--workflow-dir", str(_wf(tmp_path))]) == 0
    assert "1 declared finding surface(s)" in capsys.readouterr().out


def test_cli_exits_non_zero_when_a_declared_step_does_not_exist(tmp_path):
    _wf(tmp_path, declared_step=_STEP, real_step="Renamed step")
    with pytest.raises(SystemExit) as exc:
        whs.main(["--workflow-dir", str(tmp_path)])
    assert "not a step in this workflow" in str(exc.value)


def test_cli_refuses_a_workflow_dir_that_is_not_a_directory(tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        whs.main(["--workflow-dir", str(tmp_path / "nope")])


# -- The reverse direction (alpha-engine-config-I9702) ------------------------


def _surface(steps, *, declared, exempt=None):
    """A minimal finding-surface workflow document.

    ``steps`` is ``[(name, continue_on_error), ...]``.
    """
    declaration = {
        whs.SEMANTICS_KEY: whs.SEMANTICS_FINDING_SURFACE,
        whs.FINDING_STEPS_KEY: "\n".join(declared),
        whs.FINDING_ISSUE_KEY: "alpha-engine-config-I1",
        whs.FINDING_REASON_KEY: "because",
    }
    if exempt is not None:
        declaration[whs.FINDING_EXEMPT_KEY] = "\n".join(exempt)
    return {
        "env": declaration,
        "jobs": {
            "j": {
                "steps": [
                    {"name": n, **({"continue-on-error": True} if coe else {})}
                    for n, coe in steps
                ]
            }
        },
    }


def test_a_tolerated_step_that_is_not_declared_is_an_error():
    """The defect this check exists for: ADDING a checker, not renaming one.

    `Lambda timeout drift (codified table vs live)` was added to
    nousergon-data's sf-arn-drift-check.yml on 2026-09-01 with an id, wired
    into the terminal gate, and left undeclared. CI was green, and its red
    would have been graded as a broken check rather than as the finding it is.
    """
    doc = _surface([("checker A", True), ("checker B", True)], declared=["checker A"])
    decl, errors = whs._parse_declaration("wf.yml", doc)
    assert decl is None
    assert any("checker B" in e for e in errors), errors


def test_a_tolerated_step_named_as_not_a_finding_is_accepted():
    """Remediation, credential setup and publish steps are legitimately not
    findings - but the omission has to be written down, not inferred."""
    doc = _surface(
        [("checker A", True), ("remediation", True)],
        declared=["checker A"],
        exempt=["remediation"],
    )
    decl, errors = whs._parse_declaration("wf.yml", doc)
    assert errors == []
    assert decl is not None and decl.steps == ("checker A",)


def test_an_exemption_naming_no_real_step_is_an_error():
    """An exemption that matches nothing exempts nothing while reading as a
    decision - the same shape as a stale suppression entry."""
    doc = _surface([("checker A", True)], declared=["checker A"], exempt=["ghost"])
    _, errors = whs._parse_declaration("wf.yml", doc)
    assert any("ghost" in e for e in errors), errors


def test_a_step_without_continue_on_error_needs_no_declaration():
    """A step whose failure already fails the job is not a finding surface -
    the check must not demand a declaration for every step in the file."""
    doc = _surface([("checker A", True), ("hard gate", False)], declared=["checker A"])
    decl, errors = whs._parse_declaration("wf.yml", doc)
    assert errors == []
    assert decl is not None


# -- operator-gated semantics (alpha-engine-config-I9874) --------------------

_GATE_STEP = "Live template matches main"


def _gated_wf(tmp_path: Path, *, declared_step: str = _GATE_STEP,
              real_step: str = _GATE_STEP, issue="alpha-engine-config-I9874",
              reason="operator apply gate") -> Path:
    lines = [
        "name: crucible v2 stack",
        "on:",
        "  schedule:",
        '    - cron: "0 22 * * 6"',
        "env:",
        "  WORKFLOW_HEALTH_SEMANTICS: operator-gated",
    ]
    if declared_step is not None:
        lines += ["  WORKFLOW_HEALTH_GATE_STEPS: |", f"    {declared_step}"]
    if issue is not None:
        lines.append(f"  WORKFLOW_HEALTH_GATE_ISSUE: {issue}")
    if reason is not None:
        lines.append(f"  WORKFLOW_HEALTH_GATE_REASON: {reason}")
    lines += ["jobs:", "  stack-live:", "    steps:",
              f'      - name: "{real_step}"', '        run: "true"']
    (tmp_path / "gated.yml").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_an_operator_gated_workflow_is_discovered_from_its_own_file(tmp_path):
    (decl,) = whs.declared_operator_gated(_gated_wf(tmp_path)).values()
    assert decl.steps == (_GATE_STEP,)
    assert decl.issue == "alpha-engine-config-I9874"
    assert decl.reason == "operator apply gate"


def test_an_operator_gated_workflow_is_not_a_finding_surface(tmp_path):
    """The two semantics are mutually exclusive per workflow, and each
    discovery function must ignore declarations of the other kind."""
    d = _gated_wf(tmp_path)
    assert whs.declared_finding_surfaces(d) == {}
    assert len(whs.declared_operator_gated(d)) == 1


def test_the_guard_fails_on_a_gated_step_the_workflow_does_not_have(tmp_path):
    _gated_wf(tmp_path, declared_step=_GATE_STEP, real_step="Renamed step")
    with pytest.raises(ValueError, match="not a step in this workflow"):
        whs.declared_operator_gated(tmp_path, strict=True)


def test_the_guard_fails_on_a_gated_declaration_with_no_steps(tmp_path):
    _gated_wf(tmp_path, declared_step=None)
    with pytest.raises(ValueError, match="no WORKFLOW_HEALTH_GATE_STEPS"):
        whs.declared_operator_gated(tmp_path, strict=True)


def test_the_guard_fails_on_a_gated_declaration_without_an_issue(tmp_path):
    _gated_wf(tmp_path, issue=None)
    with pytest.raises(ValueError, match="WORKFLOW_HEALTH_GATE_ISSUE"):
        whs.declared_operator_gated(tmp_path, strict=True)


def test_the_guard_fails_on_a_gated_declaration_without_a_reason(tmp_path):
    _gated_wf(tmp_path, reason=None)
    with pytest.raises(ValueError, match="WORKFLOW_HEALTH_GATE_REASON"):
        whs.declared_operator_gated(tmp_path, strict=True)


def test_an_invalid_semantics_value_is_reported_once_not_twice(tmp_path):
    """A typo must fail loudly exactly once — from the finding-surface
    validator, which owns the shared vocabulary check — not silently drop
    through the operator-gated parser unreported, and not duplicate the
    same error under two different messages."""
    (tmp_path / "wf.yml").write_text(
        "name: x\non:\n  schedule:\n    - cron: '0 6 * * 1'\n"
        "env:\n  WORKFLOW_HEALTH_SEMANTICS: findings-surface\njobs: {}\n"
    )
    with pytest.raises(ValueError, match="is not one of"):
        whs.declared_finding_surfaces(tmp_path, strict=True)
    # The operator-gated parser silently ignores it (not its vocabulary to
    # validate) rather than raising a second, differently-worded error.
    assert whs.declared_operator_gated(tmp_path, strict=True) == {}


def test_cli_reports_both_finding_surfaces_and_gated_detectors(tmp_path, capsys):
    _gated_wf(tmp_path)
    assert whs.main(["--workflow-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "0 declared finding surface(s), 1 declared operator-gated " \
           "detector(s)" in out
