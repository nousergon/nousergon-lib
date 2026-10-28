"""Tests for scripts/autobump.py — the merge-time version writer (config-I2716).

The bump script is the ONLY routine writer of the version lines; a defect
here mis-versions a PyPI release (unrecoverable — PyPI forbids re-uploading
a version number). Covers: happy-path patch bump in both files, lockstep
violation fail-loud, malformed/duplicate version-line fail-loud, and that
the regexes match the REAL repo files.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "autobump", _REPO_ROOT / "scripts" / "autobump.py"
)
autobump = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(autobump)


def _write_pair(tmp_path: Path, py_ver: str, init_ver: str) -> tuple[Path, Path]:
    pyproject = tmp_path / "pyproject.toml"
    init = tmp_path / "__init__.py"
    pyproject.write_text(
        f'[project]\nname = "nousergon-lib"\nversion = "{py_ver}"\n'
    )
    init.write_text(f'"""Pkg."""\n\n__version__ = "{init_ver}"\n')
    return pyproject, init


def test_bump_increments_patch_in_both_files(tmp_path: Path) -> None:
    pyproject, init = _write_pair(tmp_path, "0.122.0", "0.122.0")
    new = autobump.bump(pyproject, init)
    assert new == "0.122.1"
    assert 'version = "0.122.1"' in pyproject.read_text()
    assert '__version__ = "0.122.1"' in init.read_text()


def test_bump_fails_loud_on_lockstep_violation(tmp_path: Path) -> None:
    pyproject, init = _write_pair(tmp_path, "0.122.0", "0.121.0")
    with pytest.raises(SystemExit, match="lockstep violated"):
        autobump.bump(pyproject, init)


def test_bump_fails_loud_on_missing_version_line(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    init = tmp_path / "__init__.py"
    pyproject.write_text('[project]\nname = "nousergon-lib"\n')
    init.write_text('__version__ = "0.122.0"\n')
    with pytest.raises(SystemExit, match="expected exactly one version line"):
        autobump.bump(pyproject, init)


def test_bump_fails_loud_on_duplicate_version_lines(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    init = tmp_path / "__init__.py"
    pyproject.write_text('version = "0.122.0"\nversion = "0.122.0"\n')
    init.write_text('__version__ = "0.122.0"\n')
    with pytest.raises(SystemExit, match="expected exactly one version line"):
        autobump.bump(pyproject, init)


def test_regexes_match_real_repo_files() -> None:
    py_text = (_REPO_ROOT / "pyproject.toml").read_text()
    init_text = (_REPO_ROOT / "src" / "nousergon_lib" / "__init__.py").read_text()
    assert len(autobump._PYPROJECT_RE.findall(py_text)) == 1
    assert len(autobump._INIT_RE.findall(init_text)) == 1
    # Lockstep on the real files, mirroring tests/test_version_pin.py.
    assert autobump._PYPROJECT_RE.search(py_text).group(0).split('"')[1] == re.search(
        r'__version__ = "([^"]+)"', init_text
    ).group(1)


# --- uv.lock (alpha-engine-config-I12753) -----------------------------------

_LOCK_TMPL = """version = 1
revision = 5
requires-python = ">=3.10"

[[package]]
name = "boto3"
version = "1.40.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "nousergon-lib"
version = "{ver}"
source = {{ editable = "." }}
dependencies = [
    {{ name = "boto3" }},
]
"""


def test_bump_sets_lock_project_version(tmp_path: Path) -> None:
    pyproject, init = _write_pair(tmp_path, "0.122.0", "0.122.0")
    lock = tmp_path / "uv.lock"
    lock.write_text(_LOCK_TMPL.format(ver="0.122.0"))
    assert autobump.bump(pyproject, init, lock) == "0.122.1"
    assert lock.read_text() == _LOCK_TMPL.format(ver="0.122.1")


def test_bump_heals_a_lagging_lock(tmp_path: Path) -> None:
    # The lock had already fallen behind (0.124.189 vs 0.124.192 on main,
    # 2026-10-28): it is SET to the new version, not bumped from its own.
    pyproject, init = _write_pair(tmp_path, "0.124.192", "0.124.192")
    lock = tmp_path / "uv.lock"
    lock.write_text(_LOCK_TMPL.format(ver="0.124.189"))
    assert autobump.bump(pyproject, init, lock) == "0.124.193"
    assert lock.read_text() == _LOCK_TMPL.format(ver="0.124.193")


def test_bump_fails_loud_without_lock_entry_and_writes_nothing(tmp_path: Path) -> None:
    pyproject, init = _write_pair(tmp_path, "0.122.0", "0.122.0")
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n\n[[package]]\nname = "boto3"\nversion = "1.40.0"\n')
    before = (pyproject.read_text(), init.read_text(), lock.read_text())
    with pytest.raises(SystemExit, match="exactly one editable nousergon-lib package"):
        autobump.bump(pyproject, init, lock)
    assert (pyproject.read_text(), init.read_text(), lock.read_text()) == before


def test_lock_regex_matches_real_repo_lock_in_lockstep() -> None:
    # The repo's own lock carries exactly one editable project entry, at the
    # pyproject version — the property `uv lock --check` enforces in CI.
    lock_text = (_REPO_ROOT / "uv.lock").read_text()
    matches = autobump._LOCK_RE.findall(lock_text)
    assert len(matches) == 1
    lock_ver = autobump._LOCK_RE.search(lock_text).group(0).split('"')[3]
    py_text = (_REPO_ROOT / "pyproject.toml").read_text()
    assert lock_ver == autobump._PYPROJECT_RE.search(py_text).group(0).split('"')[1]


def test_main_entry_point_passes_the_lock() -> None:
    src = (_REPO_ROOT / "scripts" / "autobump.py").read_text()
    assert "bump(PYPROJECT, INIT, LOCK)" in src
