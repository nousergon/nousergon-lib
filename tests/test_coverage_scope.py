"""The coverage gate's *scope* is asserted here, not only its number.

repository-baseline-policy.md §4.2 C5: the way a coverage gate stops being
honest is by narrowing what it measures rather than by lowering the number —
which reads as an improvement in every report. Measured on symposion, removing
one flag moved the reported figure from 34.76% to 92.36% with no new test
code.

So these tests assert what a passing suite cannot otherwise notice:

* the measured source is the WHOLE package tree under ``src`` (C1), never a
  path or submodule narrower than that;
* the floor is enforced by a non-zero exit (C2) and is a ratchet that may be
  raised and never lowered (C3);
* the ``omit`` list is pinned to its current, individually-justified members
  (``src/nousergon_lib/egress/proxy.py`` — a process-entry HTTP server, not
  unit-tested, see the comment above ``[tool.coverage.run]`` in
  pyproject.toml) — a future PR that widens it silently shrinks the
  denominator without this test noticing the change in words, so this test
  forces the diff to be reviewed instead of waved through as "coverage went
  up";
* every ``*.py`` module under ``src`` is inside the measured scope or on that
  pinned omit list — nothing is invisible to the gate by omission of a
  different kind;
* no CI-passed ``--cov-fail-under`` flag can silently shadow the
  pyproject.toml ratchet this file protects.
"""

from __future__ import annotations

import re
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.9 / 3.10 are still in test.yml's matrix — this
    import tomli as tomllib  # guard must run on EVERY leg. A scope check that
    # skips on one interpreter is indistinguishable from one that passed.

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_ROOT = REPO_ROOT / "src"

#: The floor may be RAISED here as coverage improves. Lowering it is a policy
#: amendment (repository-baseline-policy.md §4.2 C3), not a code change.
MINIMUM_FLOOR = 92

#: Individually justified in pyproject.toml's [tool.coverage.run] comment.
#: Widening this set is a scope decision, not a drive-by coverage bump —
#: update the comment there and this list together.
EXPECTED_OMIT = {"src/nousergon_lib/egress/proxy.py"}


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_coverage_source_is_the_whole_package() -> None:
    """C1 — ``source`` names the whole ``src`` tree, so unimported modules
    still count."""
    sources = _pyproject()["tool"]["coverage"]["run"]["source"]
    assert sources == ["src"], (
        f"coverage source must be exactly ['src'], got {sources!r}. Narrowing "
        "it to a submodule or a path measures the tested subset and reports "
        "it as the repository."
    )


def test_coverage_floor_is_enforced_and_never_lowered() -> None:
    """C2 + C3 — the gate exits non-zero below a floor that only ratchets up."""
    fail_under = _pyproject()["tool"]["coverage"]["report"]["fail_under"]
    assert isinstance(fail_under, int), (
        f"fail_under must be a single integer floor, got {fail_under!r}"
    )
    assert fail_under >= MINIMUM_FLOOR, (
        f"coverage floor {fail_under} is below the ratchet {MINIMUM_FLOOR}. "
        "A floor is raised as coverage improves and never lowered to make a "
        "change pass (repository-baseline-policy.md §4.2 C3)."
    )


def test_coverage_omit_matches_the_pinned_justified_set() -> None:
    """A shrunk denominator is a narrowing this test forces into review."""
    omit = set(_pyproject()["tool"]["coverage"]["run"].get("omit", []))
    added = omit - EXPECTED_OMIT
    removed = EXPECTED_OMIT - omit
    assert not added, (
        f"coverage omit gained unreviewed entries: {sorted(added)}. Each "
        "omitted path removes files from the denominator, raising the "
        "reported figure without adding a test — update EXPECTED_OMIT here "
        "alongside its justification in pyproject.toml if this is deliberate."
    )
    assert not removed, (
        f"coverage omit lost tracked entries: {sorted(removed)}. If these "
        "modules are now measured, good — also narrow EXPECTED_OMIT here."
    )


def test_every_source_module_is_inside_the_measured_package_or_pinned_omit() -> None:
    """No source file is invisible to the gate by living outside the measured
    scope."""
    modules = sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in SRC_ROOT.rglob("*.py")
    )
    assert modules, "no source modules found — the scope check would pass vacuously"
    stray = [m for m in modules if m not in EXPECTED_OMIT]
    assert stray, "every source module resolved to the pinned omit set — suspicious"
    # source == ["src"], so every module under src/ is by definition inside the
    # measured tree; this asserts the invariant explicitly rather than by
    # construction, so a future change to `source` trips a test here too.
    for module in modules:
        assert (REPO_ROOT / module).is_relative_to(SRC_ROOT)


def test_no_cov_fail_under_flag_shadows_the_pyproject_gate() -> None:
    """A CI-passed --cov-fail-under could silently override pyproject.toml's."""
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        for match in re.findall(r"--cov-fail-under=(\d+)", text):
            assert int(match) >= MINIMUM_FLOOR, (
                f"{workflow.name} passes --cov-fail-under={match} directly, "
                "bypassing the pyproject.toml ratchet this test protects."
            )
