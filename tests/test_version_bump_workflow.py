"""Pins the version-release CI contract (config-I2716, inverted 2026-07-16).

History: the original L347 guard (alpha-engine-lib #76 recurrence class)
required every src-touching PR to bump ``pyproject.toml::version``. That
per-PR-bump contract made every pair of concurrently-open PRs conflict on
the version line — the "version-bump conflict treadmill" (evidence:
nousergon-lib #190, 2026-07-14→16). The contract is now INVERTED:

  1. ``version-bump-check.yml`` (job ``version-policy``) FORBIDS PRs from
     touching the version lines, except under the ``release:manual`` label
     (validated: lockstep + monotonic + not already tagged).
  2. ``auto-version-bump.yml`` is the single routine version writer: on
     push to main with src/** changes since the last release tag, it bumps
     the patch version via ``scripts/autobump.py`` and pushes directly to
     main through the AUTOBUMP_DEPLOY_KEY ruleset bypass. ``auto-tag.yml``
     + ``publish.yml`` then release as before.

The #76 defect (src change merges → auto-tag idempotently skips →
downstream pin fails) is prevented structurally: the version writer runs
unconditionally after every src-touching merge, so there is no "forgot to
bump" state. These tests pin BOTH workflows' load-bearing shape so a
refactor can't silently strip an invariant.

What this test does NOT do: execute the workflows (platform concern) or
validate GHA intrinsic-function runtime semantics — shape pinning only,
same scope as the original L347 test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "version-bump-check.yml"
AUTOBUMP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "auto-version-bump.yml"


@pytest.fixture(scope="module")
def policy_text() -> str:
    assert POLICY_WORKFLOW.exists(), (
        f"Missing CI guard: {POLICY_WORKFLOW.relative_to(REPO_ROOT)}. The "
        f"version-policy workflow is the chokepoint that keeps PRs off the "
        f"version lines (config-I2716); without it the conflict treadmill "
        f"returns."
    )
    return POLICY_WORKFLOW.read_text()


@pytest.fixture(scope="module")
def autobump_text() -> str:
    assert AUTOBUMP_WORKFLOW.exists(), (
        f"Missing release writer: {AUTOBUMP_WORKFLOW.relative_to(REPO_ROOT)}. "
        f"With version-bump-check inverted (PRs may not bump), this workflow "
        f"is the ONLY thing that versions/releases src changes — deleting it "
        f"re-opens the alpha-engine-lib #76 class (src change merges, "
        f"auto-tag idempotently skips, downstream pins fail)."
    )
    return AUTOBUMP_WORKFLOW.read_text()


@pytest.fixture(scope="module")
def policy_yaml(policy_text):
    try:
        import yaml
    except ImportError:  # pragma: no cover — defensive
        pytest.skip("PyYAML not installed; the file-text invariants below still pin shape")
    return yaml.safe_load(policy_text)


@pytest.fixture(scope="module")
def autobump_yaml(autobump_text):
    try:
        import yaml
    except ImportError:  # pragma: no cover — defensive
        pytest.skip("PyYAML not installed; the file-text invariants below still pin shape")
    return yaml.safe_load(autobump_text)


def _on_block(workflow_yaml):
    # YAML 1.1 parses bare `on` as boolean True; accept both spellings.
    return workflow_yaml.get("on", workflow_yaml.get(True))


def test_policy_triggers_on_pull_request_and_label_events(policy_yaml):
    """Must fire on PRs including labeled/unlabeled — applying
    ``release:manual`` has to re-evaluate the check without a new push."""
    on = _on_block(policy_yaml)
    assert on is not None and "pull_request" in on
    assert "main" in on["pull_request"].get("branches", [])
    types = on["pull_request"].get("types", [])
    assert {"labeled", "unlabeled"} <= set(types), (
        f"pull_request types must include labeled+unlabeled (got {types}); "
        f"otherwise adding release:manual never re-runs the check and the "
        f"manual-release path is unusable."
    )


def test_policy_job_name_is_the_required_check_contract(policy_yaml):
    """Branch ruleset requires the check context produced by this job's
    ``name:`` — renaming it without coordinating the ruleset silently
    un-gates the workflow."""
    jobs = policy_yaml.get("jobs", {})
    assert "version-policy" in jobs, f"got jobs={list(jobs)}"
    assert jobs["version-policy"].get("name") == (
        "version untouched by PRs (merge-time autobump)"
    )


def test_policy_forbids_version_edits(policy_text):
    """The inverted invariant: a version diff without release:manual fails."""
    assert "this PR changes pyproject.toml::version" in policy_text, (
        "The forbid-version-edit error message is gone — either the "
        "inequality check was removed or its operator guidance was deleted."
    )
    assert "release:manual" in policy_text


def test_policy_manual_release_path_validates(policy_text):
    """release:manual must validate lockstep + not-already-tagged +
    monotonicity rather than skipping validation entirely."""
    assert "must bump pyproject" in policy_text  # lockstep
    assert "already exists as a tag" in policy_text  # monotonic tag check
    assert re.search(r"sort -V", policy_text), (
        "Monotonicity comparison (sort -V) removed from the manual-release "
        "validation."
    )


def test_autobump_triggers_on_push_to_main_with_serialization(autobump_yaml):
    on = _on_block(autobump_yaml)
    assert on is not None and "push" in on
    assert "main" in on["push"].get("branches", [])
    conc = autobump_yaml.get("concurrency", {})
    assert conc.get("cancel-in-progress") is False, (
        "Autobump runs must queue, not cancel — a cancelled bump run is a "
        "missed release."
    )


def test_autobump_pins_load_bearing_invariants(autobump_text):
    """The four things a refactor must not silently drop."""
    assert "AUTOBUMP_DEPLOY_KEY" in autobump_text, (
        "Deploy-key checkout removed — pushes to protected main would fail "
        "(GITHUB_TOKEN has no ruleset bypass)."
    )
    assert re.search(r"refs/tags/v\$\{VERSION\}", autobump_text), (
        "Tag-existence no-op branch removed — the workflow would double-bump "
        "on its own push or fight manual releases."
    )
    assert re.search(r"git diff --quiet .*-- src/", autobump_text), (
        "src/-diff gate removed — docs-only merges would publish no-op "
        "releases."
    )
    assert "scripts/autobump.py" in autobump_text, (
        "Bump delegated away from scripts/autobump.py — the tested, "
        "lockstep-asserting writer — to something unvalidated."
    )
