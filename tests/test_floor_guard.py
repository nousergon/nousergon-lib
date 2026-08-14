"""The dependency-floor guard, and proof it fails against a below-floor
environment (alpha-engine-config#7217).

Root cause being guarded: `pip install <pkg>` is a no-op once any version of
that package is present (alpha-engine-config#I5854), so a local venv drifts
silently below its own repo's declared `>=X.Y.Z` floor -- no install failed,
nothing printed an error. CI always resolves the floor fresh and never
reproduces the resulting failures, so they get reported as "pre-existing"
local flakiness against code that is, in fact, fine. Measured 2026-08-13:
crucible-evaluator's local venv carried krepis 0.31.1 against its own
declared `krepis[flow_doctor,openai]>=0.40.0` -- 3 failed / 908 passed,
0 failed after upgrading to the floor.

Most cases call the module functions directly so their own lines are
measured by coverage; the pytest_sessionstart case runs a real pytest
subprocess against a synthetic repo layout, because the observable
contract this guard exists to provide is "a real pytest invocation refuses
to proceed" -- an in-process call to the function alone would not prove the
hook actually wires up and halts collection.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from nousergon_lib import floor_guard


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def test_parse_declared_floors_reads_requirements_txt(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        """
        krepis[flow_doctor,openai]>=0.40.0
        boto3>=1.34
        pinned-pkg==1.2.3
        """,
    )
    floors = floor_guard.parse_declared_floors(tmp_path)
    assert floors["krepis"] == ("0.40.0", "requirements.txt")
    assert floors["boto3"] == ("1.34", "requirements.txt")
    # == pins are a deliberate exact-version policy, not a floor.
    assert "pinned-pkg" not in floors


def test_parse_declared_floors_reads_pyproject_dependencies(tmp_path):
    _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        dependencies = [
            "pydantic>=2.0",
            "pyyaml>=6.0",
        ]
        [project.optional-dependencies]
        quant = ["numpy>=1.24", "pandas>=2.0"]
        """,
    )
    floors = floor_guard.parse_declared_floors(tmp_path)
    assert floors["pydantic"] == ("2.0", "pyproject.toml")
    assert floors["numpy"] == ("1.24", "pyproject.toml")


def test_parse_declared_floors_keeps_the_highest_floor_across_files(tmp_path):
    _write(tmp_path / "requirements.txt", "krepis>=0.40.0\n")
    _write(tmp_path / "requirements-lambda.txt", "krepis>=0.55.0\n")
    floors = floor_guard.parse_declared_floors(tmp_path)
    assert floors["krepis"][0] == "0.55.0"


def test_check_floor_conformance_flags_a_below_floor_package(tmp_path, monkeypatch):
    _write(tmp_path / "requirements.txt", "somepkg>=9.9.9\n")

    def fake_version(name):
        assert name == "somepkg"
        return "1.0.0"

    monkeypatch.setattr(floor_guard.importlib.metadata, "version", fake_version)
    violations = floor_guard.check_floor_conformance(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.package == "somepkg"
    assert v.installed == "1.0.0"
    assert v.floor == "9.9.9"
    assert v.fix_command == 'pip install -U "somepkg>=9.9.9"'


def test_check_floor_conformance_is_clean_when_installed_meets_floor(
    tmp_path, monkeypatch
):
    _write(tmp_path / "requirements.txt", "somepkg>=1.0.0\n")
    monkeypatch.setattr(
        floor_guard.importlib.metadata, "version", lambda name: "1.0.0"
    )
    assert floor_guard.check_floor_conformance(tmp_path) == []


def test_check_floor_conformance_skips_a_package_that_is_not_installed(
    tmp_path, monkeypatch
):
    _write(tmp_path / "requirements.txt", "somepkg>=1.0.0\n")

    def raise_not_found(name):
        raise floor_guard.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        floor_guard.importlib.metadata, "version", raise_not_found
    )
    # Missing is a different failure class (import error on collection);
    # the floor guard only speaks to "installed but stale".
    assert floor_guard.check_floor_conformance(tmp_path) == []


def test_format_violation_report_names_package_installed_floor_and_fix():
    violations = [
        floor_guard.FloorViolation("krepis", "0.31.1", "0.40.0", "requirements.txt")
    ]
    report = floor_guard.format_violation_report(violations)
    assert "krepis" in report
    assert "0.31.1" in report
    assert "0.40.0" in report
    assert 'pip install -U "krepis>=0.40.0"' in report
    assert "NOUSERGON_SKIP_FLOOR_GUARD" in report


def test_find_repo_root_walks_up_to_the_nearest_dot_git(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert floor_guard.find_repo_root(nested) == tmp_path


def test_find_repo_root_falls_back_to_start_when_no_dot_git_found(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert floor_guard.find_repo_root(nested) == nested


# --- proof it actually FAILS a real pytest run (I7217 closes-when #1) ------


def _run_pytest_subprocess(repo_root: Path, extra_env: dict | None = None):
    """Run a real pytest against a synthetic below-floor repo layout and
    return the completed process. Uses the current interpreter (which has
    nousergon_lib importable) so the plugin under test loads for real."""
    env = {**__import__("os").environ, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(repo_root), "-q"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_pytest_sessionstart_fails_a_real_run_below_its_own_repo_floor(tmp_path):
    (tmp_path / ".git").mkdir()
    # nousergon_lib itself is what's "installed" in this interpreter --
    # declare a floor far above the real installed version so the guard
    # is guaranteed to fire regardless of which nousergon-lib is on PATH.
    _write(tmp_path / "requirements.txt", "nousergon-lib>=999.0.0\n")
    _write(
        tmp_path / "tests" / "conftest.py",
        """
        pytest_plugins = ["nousergon_lib.floor_guard"]
        """,
    )
    _write(
        tmp_path / "tests" / "test_dummy.py",
        """
        def test_never_reached():
            assert True
        """,
    )
    result = _run_pytest_subprocess(tmp_path)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "FLOOR GUARD" in combined
    assert "nousergon-lib" in combined
    assert "999.0.0" in combined
    assert "test_never_reached" not in combined or "passed" not in combined


def test_pytest_sessionstart_passes_when_no_floor_is_violated(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path / "requirements.txt", "nousergon-lib>=0.0.1\n")
    _write(
        tmp_path / "tests" / "conftest.py",
        """
        pytest_plugins = ["nousergon_lib.floor_guard"]
        """,
    )
    _write(
        tmp_path / "tests" / "test_dummy.py",
        """
        def test_trivially_true():
            assert True
        """,
    )
    result = _run_pytest_subprocess(tmp_path)
    assert result.returncode == 0
    assert "1 passed" in result.stdout


def test_pytest_sessionstart_honors_the_skip_env_var(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path / "requirements.txt", "nousergon-lib>=999.0.0\n")
    _write(
        tmp_path / "tests" / "conftest.py",
        """
        pytest_plugins = ["nousergon_lib.floor_guard"]
        """,
    )
    _write(
        tmp_path / "tests" / "test_dummy.py",
        """
        def test_trivially_true():
            assert True
        """,
    )
    result = _run_pytest_subprocess(
        tmp_path, extra_env={"NOUSERGON_SKIP_FLOOR_GUARD": "1"}
    )
    assert result.returncode == 0
    assert "SKIPPED" in (result.stdout + result.stderr)
