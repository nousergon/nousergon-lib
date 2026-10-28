#!/usr/bin/env python3
"""Merge-time patch-version bump (config-I2716, 2026-07-16).

PRs no longer touch the version line — version-bump-check.yml forbids it
(except under the release:manual label). This script is the single writer:
auto-version-bump.yml runs it on push to main and pushes the bump commit
directly via the AUTOBUMP_DEPLOY_KEY ruleset bypass.

Reads pyproject.toml [project].version and src/nousergon_lib/__init__.py
__version__, asserts they are in lockstep (same invariant as
tests/test_version_pin.py), bumps the patch component in BOTH files, sets
the editable project's own entry in uv.lock to the same version, and prints
the new version to stdout.

uv.lock (alpha-engine-config-I12753): the bump commit goes straight to main,
so a lock it leaves behind makes `uv lock --check` fail on main after every
release. The lock line is SET to the new version rather than bumped from the
old one, so a lock that already lagged heals on the next release. This is an
offline text edit of one block, deliberately not `uv lock`: re-resolving
dependencies inside an unreviewed push to main could move other pins and,
depending on the runner's uv, rewrite the lock's `revision`. The `__main__`
entry point always passes the lock; `bump()` itself takes it as an explicit
argument so a caller (a test) never edits the repo's lock by default.

Fails loud on any mismatch or parse failure — a wrong version string
published to PyPI is unrecoverable (PyPI forbids re-upload of a yanked
version number).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
INIT = Path("src/nousergon_lib/__init__.py")
LOCK = Path("uv.lock")

_PYPROJECT_RE = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
_INIT_RE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
# The editable project's own [[package]] block in uv.lock: name, then version,
# then the `source = { editable = "." }` line that distinguishes it from any
# registry package of the same name.
_LOCK_RE = re.compile(
    r'^(\[\[package\]\]\nname = "nousergon-lib"\nversion = ")[^"\n]+("\nsource = \{ editable = "\." \})$',
    re.MULTILINE,
)


def read_version(text: str, pattern: re.Pattern[str], path: Path) -> tuple[int, int, int]:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(
            f"autobump: expected exactly one version line in {path}, found {len(matches)}"
        )
    return tuple(int(p) for p in matches[0])  # type: ignore[return-value]


def set_lock_version(lock_text: str, new: str, path: Path) -> str:
    matches = _LOCK_RE.findall(lock_text)
    if len(matches) != 1:
        raise SystemExit(
            f"autobump: expected exactly one editable nousergon-lib package in {path}, "
            f"found {len(matches)}"
        )
    return _LOCK_RE.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", lock_text)


def bump(
    pyproject_path: Path = PYPROJECT,
    init_path: Path = INIT,
    lock_path: Path | None = None,
) -> str:
    py_text = pyproject_path.read_text()
    init_text = init_path.read_text()
    py_ver = read_version(py_text, _PYPROJECT_RE, pyproject_path)
    init_ver = read_version(init_text, _INIT_RE, init_path)
    if py_ver != init_ver:
        raise SystemExit(
            f"autobump: version lockstep violated — {pyproject_path} has "
            f"{'.'.join(map(str, py_ver))}, {init_path} has {'.'.join(map(str, init_ver))}"
        )
    major, minor, patch = py_ver
    new = f"{major}.{minor}.{patch + 1}"
    # Validate the lock BEFORE writing anything, so a malformed lock fails the
    # release with every file untouched.
    new_lock = (
        set_lock_version(lock_path.read_text(), new, lock_path) if lock_path is not None else None
    )
    pyproject_path.write_text(_PYPROJECT_RE.sub(f'version = "{new}"', py_text))
    init_path.write_text(_INIT_RE.sub(f'__version__ = "{new}"', init_text))
    if lock_path is not None and new_lock is not None:
        lock_path.write_text(new_lock)
    return new


if __name__ == "__main__":
    sys.stdout.write(bump(PYPROJECT, INIT, LOCK) + "\n")
