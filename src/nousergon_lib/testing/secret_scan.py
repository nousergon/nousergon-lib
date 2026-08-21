"""Shared ``os.environ.get(SECRET)`` regression scanner.

Lifted out of ``crucible-predictor/tests/test_no_secret_environ_reads.py``
and its ``nousergon-data`` twin (alpha-engine-config-I7925 deliverable 2).

Both repos' original tests only ``rglob``'d their own repo tree. That left
a first-party *dependency* — ``nousergon_lib.preflight._github_auth_headers()``
reading ``GITHUB_TOKEN`` from ``site-packages`` — invisible to the invariant
the tests exist to enforce (alpha-engine-config-I7924/I7925: the read used a
credential dead since 2026-06-03 and halted preopen trading). A repo-scoped
scan cannot see a surface the failure actually used, so this module scans
BOTH a repo's own tree and its installed first-party packages, resolved via
``importlib.util.find_spec`` rather than a hardcoded site-packages path (so
it works under any venv layout: editable installs, wheels, CI runners).

Consumers (a repo's own ``test_no_secret_environ_reads.py``, and this
library's own suite) import :func:`scan_tree` for the existing repo-tree
check and :func:`scan_installed_packages` for the new dependency check —
see ``crucible-predictor``/``nousergon-data`` for the call-site shape.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Matches os.environ.get("NAME") / os.getenv("NAME") — single- or
# double-quoted literal names only (matches the pattern the repo-level
# tests have always guarded against).
ENV_READ_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
)

# Directories to never descend into: virtualenvs, build artifacts, the
# test tree itself (tests are allowed to reference secret names as
# strings for fixtures/parametrization), VCS and cache dirs.
DEFAULT_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "build",
        "tests",
        "node_modules",
        "package",
        "__pycache__",
        ".git",
        "dist",
        ".eggs",
    }
)


@dataclass(frozen=True)
class Violation:
    """One ``os.environ.get(<pinned secret>)`` read found by the scanner."""

    path: Path
    lineno: int
    name: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.path}:{self.lineno}  {self.name}"


def iter_python_files(
    root: Path,
    *,
    skip_dir_names: frozenset[str] = DEFAULT_SKIP_DIR_NAMES,
    allowed_files: frozenset[str] = frozenset(),
) -> Iterable[Path]:
    """Yield every ``*.py`` file under ``root``, honoring the skip/allow sets."""
    for path in root.rglob("*.py"):
        if set(path.parts) & skip_dir_names:
            continue
        if path.name in allowed_files:
            continue
        yield path


def scan_tree(
    root: Path,
    pinned_secrets: frozenset[str],
    *,
    skip_dir_names: frozenset[str] = DEFAULT_SKIP_DIR_NAMES,
    allowed_files: frozenset[str] = frozenset(),
) -> list[Violation]:
    """Scan every ``*.py`` file under ``root`` for pinned-secret env reads."""
    violations: list[Violation] = []
    for path in iter_python_files(
        root, skip_dir_names=skip_dir_names, allowed_files=allowed_files
    ):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in ENV_READ_RE.finditer(line):
                name = match.group(1)
                if name in pinned_secrets:
                    violations.append(Violation(path, lineno, name))
    return violations


def resolve_installed_package_root(package_name: str) -> Path | None:
    """Resolve the on-disk root of an INSTALLED first-party package.

    Uses ``importlib.util.find_spec`` — never a hardcoded ``site-packages``
    path — so this resolves correctly for an editable install, a built
    wheel, or any CI runner's venv layout. Returns ``None`` when the
    package is not importable in the current environment; callers MUST
    surface that visibly (e.g. ``pytest.skip``) rather than treat it as
    a pass — the fail-loud rule (`~/Development/CLAUDE.md`) forbids a
    swallow that renders "not installed" indistinguishable from "clean".
    """
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def scan_installed_packages(
    package_names: Sequence[str],
    pinned_secrets: frozenset[str],
    *,
    skip_dir_names: frozenset[str] = DEFAULT_SKIP_DIR_NAMES,
) -> tuple[list[Violation], list[str]]:
    """Scan installed first-party packages for pinned-secret env reads.

    Returns ``(violations, missing)`` — ``missing`` lists any
    ``package_names`` entry that could not be resolved in the current
    environment (not installed). A non-empty ``missing`` list is not an
    error on its own (a given repo's venv may legitimately lack one of
    the first-party packages), but callers must render it visibly rather
    than silently treating an unverified package as clean.
    """
    violations: list[Violation] = []
    missing: list[str] = []
    for name in package_names:
        root = resolve_installed_package_root(name)
        if root is None:
            missing.append(name)
            continue
        violations.extend(scan_tree(root, pinned_secrets, skip_dir_names=skip_dir_names))
    return violations, missing
