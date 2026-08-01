"""Validate that every required status check on this repo has a workflow that
triggers on `merge_group` events — and, if this repo actually runs a merge
queue, that `merge-group-required-check-guard` is itself a required check.

A required context whose workflow fires only on `pull_request` never reports for
a queue entry: the entry sits in `AWAITING_CHECKS` until GitHub's response
timeout, then **auto-dequeues unmerged** with nothing red and nothing logged.
That deadlocked the fleet on 2026-07-24 and forced a same-day revert.

**The logic lives in `nousergon_lib.merge_queue`, not here.** This file is the
single-repo CLI over it. The org-wide sweep in `alpha-engine-config` answers the
same questions about repos it has no checkout of, so the parser and the two API
readers are shared rather than copied (`nous-ergon-ops-I349`,
`shared-code-policy.md` second-adoption trigger).

Run locally:  python3 scripts/validate_merge_group_required_checks.py --repo owner/name
In CI:        called from merge-group-required-check-guard.yml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Self-bootstrap, and why it is not belt-and-braces ────────────────────────
#
# This script is executed from a CHECKOUT of nousergon-lib, not from an
# installed package: the reusable workflow clones this repo beside the caller's
# and runs the file directly. Every consumer pins the reusable workflow at a
# revision of its own choosing, but the script always comes from lib's DEFAULT
# BRANCH — so the moment the script grew a `nousergon_lib` import, every repo
# whose pinned skeleton predated the matching PYTHONPATH step failed with
# `ModuleNotFoundError: No module named 'nousergon_lib'`. Observed within
# minutes on alpha-engine-config (run 30666273877), fleet-wide, on a check that
# scm-platform-policy §3.2 wants promoted to blocking.
#
# Adding the PYTHONPATH to the workflow fixes it only for consumers who bump
# their pin, which is the skew itself, not a fix for it. The script sits at
# `<checkout>/scripts/`, so `<checkout>/src` is unambiguously its sibling —
# resolving it here removes the class: no consumer has to bump anything, ever,
# for the script to keep importing its own library.
if __package__ is None:  # executed as a file, which is the only way this runs
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nousergon_lib.merge_queue import (  # noqa: E402, F401 — bootstrap must precede
    active_merge_queue,
    coverage_gaps,
    guard_is_required,
    job_name_matches,
    parse_workflow,
    required_contexts,
)


def parse_workflow_merge_group(path: Path) -> tuple[str | None, set[str]]:
    """Path-taking wrapper over :func:`nousergon_lib.merge_queue.parse_workflow`.

    Retained because the guard reads files off disk while the fleet sweep reads
    them from the contents API — the shared function takes text so both callers
    are natural, and this keeps the local one a one-liner.
    """
    return parse_workflow(Path(path).read_text())


def _flag(argv: list[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def _resolve_repo(argv: list[str]) -> str:
    return _flag(argv, "--repo") or os.environ.get("GITHUB_REPOSITORY", "")


def should_fail(gaps: list, guard_gap: bool) -> bool:
    """Only a missing `merge_group` trigger fails this check.

    Separated from `main` so the asymmetry is pinned by a test rather than
    resting on one `if`. `guard_gap` is repo configuration no PR can change, and
    failing on it would redden every PR on the repo — see the block in `main`
    for why adding this check to `gate_pr_actions._ADVISORY_CHECK_NAMES` is not
    the escape hatch it looks like.
    """
    return bool(gaps)


def main() -> int:
    repo = _resolve_repo(sys.argv[1:])
    if not repo:
        print("::error::GITHUB_REPOSITORY not set and no --repo flag provided")
        return 1

    print(f"Auditing required checks for merge_group triggers in {repo}")

    contexts = required_contexts(repo)
    if not contexts:
        print("No required status checks found — nothing to validate.")
        return 0

    print(f"Found {len(contexts)} required check context(s):")
    for ctx in contexts:
        print(f"  - {ctx}")

    workflows_dir = Path(_flag(sys.argv[1:], "--workflows-dir") or ".github/workflows")
    if not workflows_dir.is_dir():
        print(f"::error::{workflows_dir} not found", file=sys.stderr)
        return 1

    workflows = {
        str(p): p.read_text()
        for p in sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])
    }
    if not workflows:
        print(f"::error::No workflow files found in {workflows_dir}", file=sys.stderr)
        return 1

    # The scanned directory is printed because `--repo` and the working tree are
    # independent: running `--repo owner/A` from a checkout of B audits A's
    # required checks against B's workflows and reports every context as
    # `UNKNOWN`, which reads as a fleet of gaps rather than as operator error.
    print(f"\nScanning {len(workflows)} workflow file(s) in {workflows_dir.resolve()}...")
    gaps = coverage_gaps(contexts, workflows)

    # Only evaluated when a queue is actually running here. Requiring the guard
    # on a repo with no queue would add a required check that protects nothing,
    # and scm-platform-policy §3.1 is explicit that a required check which cannot
    # fail usefully is a liability rather than a safeguard.
    queue = active_merge_queue(repo)
    guard_gap = queue is not None and not guard_is_required(contexts)

    if gaps:
        print("\n--- MISSING merge_group TRIGGERS ---\n")
        for ctx, wf in gaps:
            print(f"  Required check:  {ctx}")
            if wf:
                print(f"  Produced by:     {wf}")
                print("  Missing:         merge_group: {types: [checks_requested]}\n")
            else:
                print("  Produced by:     UNKNOWN (no matching workflow job found)")
                print("  Action needed:   identify the producing workflow and add the trigger\n")

    # Reported, deliberately NOT failed. Two reasons, both load-bearing:
    #
    #  1. It is a property of the REPO's configuration, not of the PR being
    #     checked — red on every PR until an admin changes a setting no PR can
    #     change. scm-platform-policy §3.1: a check that is red whenever it is
    #     working may never be required, and this check is required wherever a
    #     queue runs.
    #  2. Concretely, it would brick the auto-merge lanes. `gate_pr_actions.py`
    #     excludes only `gate-label-guard` from its red/green evaluation, so a
    #     red guard would put every PR on the repo into the `ci_red` bucket —
    #     handing an unfixable-by-design failure to an LLM fix pass and blocking
    #     the un-draft path. That is config-I4447 repeating with a new check.
    #     Adding this guard to that exclusion set is NOT the fix: once it is
    #     promoted to required it must count, and a permanent exemption for a
    #     required check is worse than the problem.
    #
    # Enforcement lives in the org-wide sweep (`alpha-engine-config`'s
    # merge-queue readiness check), which files a finding on the check-result
    # surface instead of reddening every PR. Detection here, consequence there.
    if guard_gap:
        print("\n--- GUARD IS ADVISORY WHILE A QUEUE IS RUNNING ---\n")
        print(f"  Ruleset:         {queue['_ruleset_name']} (id {queue.ruleset_id})")
        print("  Missing:         merge-group-required-check-guard in required contexts")
        print("  Why it matters:  this check's failure mode is a silent dequeue, and an")
        print("                   advisory check can only report one. scm-platform-policy")
        print("                   §3.2 makes blocking a precondition of running a queue.")
        print("  Not failed here: it is repo configuration, not this PR — the org-wide")
        print("                   readiness sweep raises it as a finding.\n")
        print("::warning::merge-group-required-check-guard is advisory while a merge "
              "queue is active — scm-platform-policy §3.2 precondition unmet.")

    if should_fail(gaps, guard_gap):
        print(f"::error::{len(gaps)} required check(s) lack "
              "merge_group: {types: [checks_requested]} triggers.")
        print("Add the trigger to the on: block of each listed workflow.")
        return 1

    if queue is not None and not guard_gap:
        print(f"\nMerge queue active (ruleset {queue.ruleset_id}); guard is a required check.")
    print("\nAll required checks have merge_group triggers — no gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
