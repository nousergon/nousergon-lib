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
