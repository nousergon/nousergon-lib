"""Tests for ``nousergon_lib.testing.secret_scan`` (alpha-engine-config-I7925).

The scanner exists specifically to catch a pinned-secret ``os.environ.get``
read inside an INSTALLED first-party package (the surface a repo-tree-only
scan cannot see — see the module docstring for the incident this fixes).
The critical test below therefore does not just assert the scanner runs:
it builds a real importable package on disk, deliberately reintroduces
``os.environ.get("GITHUB_TOKEN")`` inside it, and proves
``scan_installed_packages`` — using the real ``importlib.util.find_spec``
resolution path, not a mock — flags it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from nousergon_lib.testing.secret_scan import (
    Violation,
    resolve_installed_package_root,
    scan_installed_packages,
    scan_tree,
)

_PINNED = frozenset({"GITHUB_TOKEN", "POLYGON_API_KEY"})


def _write_fake_package(tmp_path, package_name: str, module_source: dict[str, str]):
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    for filename, source in module_source.items():
        (pkg_dir / filename).write_text(source)
    return pkg_dir


def test_scan_tree_finds_pinned_secret_read(tmp_path):
    (tmp_path / "mod.py").write_text('import os\nos.environ.get("GITHUB_TOKEN")\n')
    violations = scan_tree(tmp_path, _PINNED)
    assert len(violations) == 1
    assert violations[0].name == "GITHUB_TOKEN"
    assert violations[0].lineno == 2


def test_scan_tree_ignores_non_pinned_env_reads(tmp_path):
    (tmp_path / "mod.py").write_text('import os\nos.environ.get("LANGCHAIN_PROJECT")\n')
    assert scan_tree(tmp_path, _PINNED) == []


def test_scan_tree_skips_default_skip_dirs(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "mod.py").write_text('import os\nos.environ.get("GITHUB_TOKEN")\n')
    assert scan_tree(tmp_path, _PINNED) == []


def test_resolve_installed_package_root_finds_real_package():
    # pytest is a real installed dependency of this repo's own test env —
    # exercises the real importlib.util.find_spec path, not a mock.
    root = resolve_installed_package_root("pytest")
    assert root is not None
    assert root.is_dir()


def test_resolve_installed_package_root_returns_none_for_missing_package():
    assert resolve_installed_package_root("definitely_not_a_real_package_xyz123") is None


def test_scan_installed_packages_catches_reintroduced_secret_read(tmp_path, monkeypatch):
    """The load-bearing regression test.

    Reproduces the exact failure class from alpha-engine-config-I7924/I7925:
    a first-party package installed in the environment (not the repo tree)
    reads a pinned secret via ``os.environ.get``. Builds a real importable
    package under a temp path, inserts it onto ``sys.path`` so
    ``importlib.util.find_spec`` resolves it exactly as it would resolve a
    real installed dependency, then asserts the scanner flags it.
    """
    package_name = "fake_nousergon_lib_i7925"
    _write_fake_package(
        tmp_path,
        package_name,
        {
            "preflight.py": (
                "import os\n\n"
                "def _github_auth_headers():\n"
                '    token = os.environ.get("GITHUB_TOKEN")\n'
                '    return {"Authorization": f"token {token}"}\n'
            )
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        violations, missing = scan_installed_packages([package_name], _PINNED)
    finally:
        sys.modules.pop(package_name, None)
        sys.modules.pop(f"{package_name}.preflight", None)

    assert missing == []
    assert len(violations) == 1
    violation = violations[0]
    assert isinstance(violation, Violation)
    assert violation.name == "GITHUB_TOKEN"
    assert violation.path.name == "preflight.py"


def test_scan_installed_packages_reports_missing_visibly_not_silently():
    violations, missing = scan_installed_packages(
        ["definitely_not_a_real_package_xyz123"], _PINNED
    )
    assert violations == []
    assert missing == ["definitely_not_a_real_package_xyz123"]


def test_scan_installed_packages_flags_violations_even_when_other_package_missing(
    tmp_path, monkeypatch
):
    package_name = "fake_krepis_i7925"
    _write_fake_package(
        tmp_path,
        package_name,
        {"secrets.py": 'import os\nos.environ.get("POLYGON_API_KEY")\n'},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        violations, missing = scan_installed_packages(
            [package_name, "definitely_not_a_real_package_xyz123"], _PINNED
        )
    finally:
        sys.modules.pop(package_name, None)
        sys.modules.pop(f"{package_name}.secrets", None)

    assert missing == ["definitely_not_a_real_package_xyz123"]
    assert len(violations) == 1
    assert violations[0].name == "POLYGON_API_KEY"


def test_nousergon_lib_itself_has_no_pinned_secret_reads():
    """Self-scan: nousergon-lib's own installed tree, using the real spec
    resolution path (no monkeypatch) — the invariant applied to itself,
    per the issue's "or lift the check into a shared nousergon-lib test
    helper that ... the library itself runs" requirement."""
    violations, missing = scan_installed_packages(["nousergon_lib"], _PINNED)
    assert missing == []
    assert violations == [], "\n".join(str(v) for v in violations)


def test_krepis_has_no_pinned_secret_reads_if_installed():
    violations, missing = scan_installed_packages(["krepis"], _PINNED)
    if missing:
        pytest.skip(f"krepis not installed in this environment: {missing}")
    assert violations == [], "\n".join(str(v) for v in violations)
