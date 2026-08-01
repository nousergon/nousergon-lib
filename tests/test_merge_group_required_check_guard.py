"""The merge-group required-check guard must see job-ID-named jobs.

GitHub's status-check context is the job's `name:` **if present, else the job
ID**. The guard collected only `name:`, so a job declared as

    jobs:
      pytest:
        runs-on: ubuntu-latest

was invisible to it — and that is the single most common workflow shape in the
fleet.

Measured 2026-07-29: the guard reported `Produced by: UNKNOWN` for all 14
required checks across 7 repos, including `alpha-engine-config`'s `pytest`
(produced by `scripts-tests.yml`'s `jobs.pytest`, which has no `name:`).

Two failures, not one:
  1. it could not name the file to edit, so its own output was not actionable;
  2. it could not see a `merge_group` trigger that DID cover such a job, so it
     would keep reporting a gap after the gap was closed.

(2) is the blocking one. `scm-platform-policy.md` §3 requires this guard be
promoted from advisory to blocking before merge-queue re-adoption, and a guard
that cannot confirm its own remediation would then block every merge on all
seven repos carrying required checks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_merge_group_required_checks.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("mg_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mg_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "wf.yml"
    p.write_text(body)
    return p


def test_job_id_is_used_as_the_context_when_the_job_has_no_name(guard, tmp_path):
    """The real alpha-engine-config shape."""
    wf = _write(tmp_path, """
on:
  pull_request:
  merge_group:
    types: [checks_requested]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    has_mg, job_names = guard.parse_workflow_merge_group(wf)
    assert has_mg is not None, "merge_group trigger must be detected"
    assert "pytest" in job_names, (
        "a job with no name: must contribute its job ID — GitHub uses the ID as "
        "the check context")


def test_explicit_name_still_wins_over_the_job_id(guard, tmp_path):
    """The fallback must not shadow an explicit name — GitHub prefers `name:`."""
    wf = _write(tmp_path, """
on:
  merge_group:
    types: [checks_requested]
jobs:
  build:
    name: iam-policy-change-guard
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    _, job_names = guard.parse_workflow_merge_group(wf)
    assert "iam-policy-change-guard" in job_names
    assert "build" not in job_names, (
        "when name: is present it IS the context; adding the ID too would let a "
        "guard match a context GitHub never emits")


def test_a_covered_job_id_no_longer_reports_as_a_gap(guard, tmp_path):
    """The blocking half: the guard must be able to confirm its own fix."""
    wf = _write(tmp_path, """
on:
  merge_group:
    types: [checks_requested]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    has_mg, job_names = guard.parse_workflow_merge_group(wf)
    covered = any(guard.job_name_matches(j, "pytest") for j in job_names)
    assert has_mg is not None and covered, (
        "after adding merge_group to a job-ID-named job, the guard must see it "
        "as covered — otherwise it reports a gap that has already been closed")


def test_missing_merge_group_is_still_detected_for_a_job_id_named_job(guard, tmp_path):
    """The fallback must not turn into a blanket pass: a job-ID-named job in a
    workflow with NO merge_group trigger is still a genuine gap."""
    wf = _write(tmp_path, """
on:
  pull_request:
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    has_mg, job_names = guard.parse_workflow_merge_group(wf)
    assert has_mg is None, "no merge_group trigger must still read as absent"
    assert "pytest" in job_names, (
        "the producer must still be nameable so the report says which file to edit")


def test_matrix_expanded_names_are_still_collected(guard, tmp_path):
    """krepis' required checks are `pytest (py3.9)`..`(py3.13)` from a matrix."""
    wf = _write(tmp_path, """
on:
  merge_group:
    types: [checks_requested]
jobs:
  test:
    name: pytest (${{ matrix.py }})
    strategy:
      matrix:
        py: ["3.9", "3.13"]
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    _, job_names = guard.parse_workflow_merge_group(wf)
    assert any("${{" in j for j in job_names), (
        "matrix name templates must be retained for pattern matching")


def test_a_guard_that_is_only_advisory_does_NOT_fail_this_check(guard):
    """The asymmetry that keeps this check promotable.

    `guard_is_required` being false is REPO CONFIGURATION — red on every PR until
    an admin changes a setting no PR can change. `scm-platform-policy` §3.1 says
    a check that is red whenever it is working may never be required, and §3.2
    requires this very check on any repo running a queue, so failing on it would
    make the precondition unsatisfiable by construction.

    The concrete consequence is worse than the principle: `gate_pr_actions.py`
    excludes only `gate-label-guard` from its red/green evaluation, so a red
    guard would drop every PR on the repo into the `ci_red` bucket — handing an
    unfixable-by-design failure to an LLM fix pass and blocking the un-draft
    path. That is config-I4447 repeating with a new check.
    """
    assert guard.should_fail([], guard_gap=True) is False
    assert guard.should_fail([("pytest", "ci.yml")], guard_gap=False) is True
    assert guard.should_fail([("pytest", "ci.yml")], guard_gap=True) is True
    assert guard.should_fail([], guard_gap=False) is False
