"""Pins the L347 version-bump CI guard (``.github/workflows/version-bump-check.yml``).

Origin: alpha-engine-lib #76 merged a public-surface src change
without bumping ``pyproject.toml::version``. The existing
``auto-tag.yml`` workflow is keyed off the version field — when it
saw the same ``0.36.0`` it had already tagged, it idempotently
skipped. Downstream alpha-engine-dashboard #129's pin to
``@v0.36.1`` failed pip install for ~10 min until follow-up #77
bumped pyproject. ROADMAP L347.

The fix workflow ``.github/workflows/version-bump-check.yml``
enforces, at PR-open time on every PR touching ``src/**``, that:
  1. ``pyproject.toml::version`` increases vs the base branch.
  2. The new version is NOT already in ``git tag --list``.

This test pins the workflow's existence + the load-bearing
invariants. A future PR that silently removes the workflow OR
strips one of the two assertions would re-open the recurrence
class — this test catches that at PR time.

What this test does NOT do:
  - Execute the workflow itself. That's an integration concern owned
    by GitHub Actions; we trust the YAML parser at the platform level.
  - Validate the GHA-runner runtime semantics of intrinsic functions
    (``${{ steps.detect.outputs.src_touched }}`` etc.) — those are
    schema-validated by GitHub on push, and a malformed workflow
    fails at trigger time, not silently. We only verify the workflow's
    structural shape against the regression class the L347 entry names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "version-bump-check.yml"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    assert WORKFLOW.exists(), (
        f"Missing CI guard: {WORKFLOW.relative_to(REPO_ROOT)}. L347 "
        f"version-bump-check workflow is the chokepoint that prevents "
        f"the alpha-engine-lib #76 recurrence class (public-surface src "
        f"change without pyproject.toml version bump → auto-tag.yml "
        f"silently skips → downstream pin fails pip install)."
    )
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def workflow_yaml(workflow_text):
    # PyYAML is the only YAML lib bundled in the lib's test deps via
    # transitive includes; if it isn't present, fall back to a regex-only
    # check so this test never silently skips. PyYAML IS available in
    # the [dev,rag,arcticdb] extras CI matrix (test.yml line 52) so we
    # default to using it.
    try:
        import yaml
    except ImportError:  # pragma: no cover — defensive
        pytest.skip("PyYAML not installed; the file-text invariants below still pin shape")
    return yaml.safe_load(workflow_text)


def test_workflow_triggers_on_pull_request_to_main(workflow_yaml):
    """Must fire on PRs (not just push) — the whole point is PR-time gating.

    The `on:` key is parsed by PyYAML as the literal Python `True`
    because YAML 1.1's boolean shorthand treats bare `on` / `off` /
    `yes` / `no` as booleans. The pyproject project.requires-python
    pin (>=3.9) lets us use either form; we accept both for safety.
    """
    on = workflow_yaml.get("on", workflow_yaml.get(True))
    assert on is not None, "Workflow missing 'on:' trigger block"
    assert "pull_request" in on, (
        "Workflow does not trigger on pull_request — the L347 chokepoint "
        "MUST fire at PR-open time, not just on post-merge push to main "
        "(where the auto-tag silent-skip already happened)."
    )
    pr_branches = on["pull_request"].get("branches", [])
    assert "main" in pr_branches, (
        f"Workflow's pull_request trigger does not target main "
        f"(got branches={pr_branches}). Branch-protection requires main "
        f"as the target for this to be a meaningful gate."
    )


def test_workflow_has_version_bump_check_job(workflow_yaml):
    """The 'version-bump-check' job is the load-bearing surface — branch
    protection will (or should) be configured against this job's
    success status; renaming it silently degrades the gate."""
    jobs = workflow_yaml.get("jobs", {})
    assert "version-bump-check" in jobs, (
        f"Workflow missing 'version-bump-check' job (got jobs="
        f"{list(jobs)}). The job name is the branch-protection contract; "
        f"renaming it without coordinating with branch-protection settings "
        f"in GitHub would silently un-gate the workflow."
    )


def test_workflow_enforces_version_increase_invariant(workflow_text):
    """The workflow must compare PR version vs base version + fail when
    they're equal. Pinning the specific phrasing 'pyproject.toml::version
    is unchanged' guards against a refactor that accidentally removes
    the equality check (e.g., replacing with a no-op assertion)."""
    assert "pyproject.toml::version is unchanged" in workflow_text, (
        "Workflow does not surface the 'pyproject.toml::version is "
        "unchanged' error message. That phrasing is the load-bearing "
        "operator hint that names what to do — its absence either "
        "means the equality check was removed or the error guidance "
        "was deleted."
    )
    # The actual comparison: equality check between base + PR version.
    assert re.search(r'BASE_VERSION.*=.*PR_VERSION', workflow_text), (
        "Workflow's equality test between BASE_VERSION + PR_VERSION "
        "appears to have been removed. Without it, the version-bump "
        "invariant is unenforced."
    )


def test_workflow_enforces_new_version_not_already_tagged(workflow_text):
    """The workflow must reject a version that's already a git tag.
    Without this, a PR could bump pyproject.toml to an already-tagged
    version, auto-tag would skip (idempotent), and the consumer pin
    would resolve to the OLD commit instead of the PR's HEAD."""
    assert "already exists as a tag" in workflow_text, (
        "Workflow does not surface the 'already exists as a tag' "
        "error message. That phrasing is the operator hint that names "
        "the second-class failure mode — its absence either means the "
        "tag-list check was removed or the error guidance was deleted."
    )
    assert "git tag --list" in workflow_text, (
        "Workflow does not invoke 'git tag --list' — the tag-uniqueness "
        "check is the second invariant the L347 chokepoint enforces."
    )


def test_workflow_skips_when_only_docs_changed(workflow_text):
    """Doc / test / CI / README-only PRs MUST be exempt — they legitimately
    don't bump the version. Failing them would either force spurious
    version bumps on every README typo OR train operators to ignore
    this check.

    The skip path is the early `src_touched=false` exit; verify the
    string is present so the guard logic remains intact."""
    assert "src_touched=false" in workflow_text, (
        "Workflow no longer carries the src_touched=false skip branch. "
        "Without it, doc/test/CI-only PRs would force spurious version "
        "bumps — training operators to ignore the chokepoint."
    )
    # The detection itself uses ``grep -qE '^src/'`` against the diff
    # — pin the regex shape so a future refactor doesn't widen it to
    # include tests/ or docs/ accidentally.
    assert re.search(r"grep -qE '\^src/'", workflow_text), (
        "Workflow's src-detection grep regex no longer matches '^src/'. "
        "Widening the pattern would either over-fire (force version "
        "bumps on test/doc edits) or under-fire (silently allow src "
        "changes via a path the regex misses)."
    )
