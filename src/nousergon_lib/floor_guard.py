"""Fail fast when the active interpreter sits below its own repo's declared
dependency floors (alpha-engine-config#7217).

The bug class this exists to find: ``pip install <pkg>`` is a no-op once any
version of that package is already present (alpha-engine-config#I5854), so a
local venv drifts silently below a repo's stated ``>=X.Y.Z`` floor with no
action from anyone -- no install failed, no error was ever printed. CI always
resolves the floor fresh from a clean environment; a stale local venv does
not, and the resulting test failures get read as "pre-existing" or feed a
false "main is green" report passed on to other agents. Measured
2026-08-13: ``crucible-evaluator``'s local venv carried krepis 0.31.1
against its own declared ``krepis[flow_doctor,openai]>=0.40.0`` -- 23 minor
versions behind -- producing 3 failed / 908 passed that became 911 passed /
0 failed after ``pip install -U "krepis[flow_doctor,openai]>=0.59.0"``. The
tests were correct throughout; the environment was not.

Wire a repo in with one line in its ``tests/conftest.py`` (or any conftest
pytest loads first)::

    pytest_plugins = ["nousergon_lib.floor_guard"]

That fires ``pytest_sessionstart`` on every local ``pytest`` invocation,
before collection -- a one-time addition that cannot be silently skipped the
way a convention that must be *remembered* on every run can be. It reads
every ``>=`` floor declared in the repo's own ``requirements*.txt`` and
``pyproject.toml`` (``[project.dependencies]`` and
``[project.optional-dependencies]``), and compares each against
``importlib.metadata.version()`` for what is actually installed in *this*
interpreter. Exact pins (``==``, ``~=``) are out of scope -- those are a
deliberate exact-version policy in the repos that use it (see the
"pinned ... never floored" convention in e.g. ``crucible-predictor``) and
cannot silently drift below themselves the way a floor can.

CI is unaffected: it always builds a fresh environment that resolves the
floor, so the guard never fires there -- this exists for the local/dev loop
specifically, which is the loop that goes stale.

Escape hatch, for the rare deliberate below-floor run (documented, never
silent): ``NOUSERGON_SKIP_FLOOR_GUARD=1``, which prints a yellow warning
naming that the guard was skipped and why.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from dataclasses import dataclass
from pathlib import Path

_FLOOR_RE = re.compile(
    r'([A-Za-z][A-Za-z0-9_.\-]*)\s*(\[[^\]]+\])?\s*>=\s*([0-9][0-9A-Za-z.\-]*)'
)
_SKIP_ENV = "NOUSERGON_SKIP_FLOOR_GUARD"


@dataclass(frozen=True)
class FloorViolation:
    package: str
    installed: str
    floor: str
    source: str  # which file declared the floor

    @property
    def fix_command(self) -> str:
        return f'pip install -U "{self.package}>={self.floor}"'


def _version_tuple(v: str) -> tuple:
    parts = re.split(r"[.\-]", v)
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            # Non-numeric segment (e.g. a pre-release tag like "rc1") sorts
            # below a plain numeric one of the same prefix -- good enough
            # for a floor check, which only needs "is it at least this".
            out.append(-1)
    return tuple(out)


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the nearest directory containing ``.git``.

    Falls back to ``start`` itself if none is found (e.g. a shallow checkout
    or an extracted tarball with no ``.git``), rather than raising -- a
    guard that cannot find its own repo root should degrade to "no floors
    found" (a no-op), not crash every test run in an environment it wasn't
    anticipating.
    """
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def parse_declared_floors(repo_root: Path) -> dict[str, tuple[str, str]]:
    """Return ``{package_name_lowercased: (floor_version, source_file)}``.

    Scans every ``requirements*.txt`` in the repo root and the
    ``[project.dependencies]`` / ``[project.optional-dependencies]`` array
    literals in ``pyproject.toml``. Only ``>=`` specs are floors; ``==`` and
    ``~=`` pins are a different, deliberate policy and are skipped. When the
    same package appears more than once (e.g. base + lambda requirements,
    or repeated across optional-dependency extras) the *highest* declared
    floor wins -- that is the one a conforming environment must satisfy.
    """
    floors: dict[str, tuple[str, str]] = {}
    candidates = sorted(repo_root.glob("requirements*.txt"))
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        candidates.append(pyproject)

    for f in candidates:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for raw_line in text.splitlines():
            # Strip a trailing inline comment before matching so a comment
            # like "Bump to >=0.6.0 when it ships" can't be read as a
            # package floor.
            line_no_comment = raw_line.split("#", 1)[0]
            # pyproject array literals put several "pkg>=x" entries on one
            # line -- finditer (not a single match/search) so each is
            # matched independently rather than only the first. A naive
            # comma-split would break "krepis[flow_doctor,openai]>=0.40.0"
            # apart at the comma INSIDE the extras bracket, so don't split.
            for m in _FLOOR_RE.finditer(line_no_comment):
                pkg = m.group(1).lower()
                floor = m.group(3)
                prev = floors.get(pkg)
                if prev is None or _version_tuple(floor) > _version_tuple(prev[0]):
                    floors[pkg] = (floor, f.name)
    return floors


def check_floor_conformance(repo_root: Path) -> list[FloorViolation]:
    """Return every installed-below-floor package, empty if all conform."""
    violations: list[FloorViolation] = []
    for pkg, (floor, source) in parse_declared_floors(repo_root).items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Not installed at all is a different failure (pytest's own
            # import errors will surface it loudly on collection) -- this
            # guard is scoped to "installed but stale", not "missing".
            continue
        if _version_tuple(installed) < _version_tuple(floor):
            violations.append(FloorViolation(pkg, installed, floor, source))
    return violations


def format_violation_report(violations: list[FloorViolation]) -> str:
    lines = [
        "",
        "=" * 78,
        "FLOOR GUARD: this environment is BELOW its own repo's declared "
        "dependency floors.",
        "A local pytest run here can fail on packages the code does not "
        "actually have a problem with (alpha-engine-config#7217) -- CI "
        "resolves the floor fresh and will not reproduce this.",
        "",
    ]
    for v in violations:
        lines.append(
            f"  {v.package}: installed {v.installed}, repo floor >={v.floor} "
            f"(declared in {v.source})"
        )
        lines.append(f"    fix: {v.fix_command}")
    lines.append("")
    lines.append(
        f"Skip once, deliberately: {_SKIP_ENV}=1 pytest ..."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def pytest_sessionstart(session):  # pragma: no cover - pytest hook wiring
    if os.environ.get(_SKIP_ENV):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"[floor-guard] SKIPPED via {_SKIP_ENV}=1 -- a local run "
                f"may now fail on packages the code does not actually have "
                f"a problem with.",
                yellow=True,
            )
        return

    repo_root = find_repo_root(Path(str(session.config.rootpath)))
    violations = check_floor_conformance(repo_root)
    if not violations:
        return

    import pytest

    pytest.exit(format_violation_report(violations), returncode=1)
