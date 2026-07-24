"""Validate that every required status check (from branch-protection rulesets
AND classic branch protection) has a workflow that triggers on merge_group
events.

Prevents the class of bug that deadlocked the merge queue three times on
2026-07-23: a required check whose workflow only fires on pull_request
but the merge queue fires merge_group — the context never reports, the
queue waits 60 minutes, the PR auto-dequeues unmerged.

Sources consulted:
  a) Active rulesets of type required_status_checks (rulesets-only repos like
     nousergon-lib).
  b) Classic branch protection required_status_checks.contexts (repos still on
     legacy protection, like nousergon-data, crucible-dashboard).
  c) The merge_queue ruleset itself — checks required by the queue are the
     same set as branch protection, but we check the queue ruleset's
     parameters.required_status_checks if present (nousergon-lib does NOT
     require status checks at the queue-ruleset level; they are gated in the
     main-protection ruleset instead).

One required check CAN be served by multiple workflows — it's enough that
AT LEAST ONE has merge_group coverage. The guard reports the first producer
found that lacks it, but the error is a single line per check regardless.

Run locally:  python3 scripts/validate_merge_group_required_checks.py
In CI:        called from merge-group-required-check-guard.yml
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def gh_api(path: str) -> dict | list:
    """Call gh api and return parsed JSON."""
    gh = shutil.which("gh") or "gh"
    result = subprocess.run(
        [gh, "api", path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"::error::gh api {path} failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def get_required_checks(repo: str) -> list[str]:
    """Collect every required status check context from rulesets AND classic
    branch protection.

    Sources (all consulted, deduplicated):
      1. Active rulesets with required_status_checks rules.
      2. Classic branch protection on main.
      3. merge_queue rulesets that list required_status_checks in their params.
    """
    contexts: list[str] = []

    # Source 1 + 3: rulesets API.
    rulesets = gh_api(f"/repos/{repo}/rulesets")
    for rs in rulesets:
        if rs.get("enforcement") != "active":
            continue
        full = gh_api(f"/repos/{repo}/rulesets/{rs['id']}")
        for rule in full.get("rules", []):
            if rule["type"] == "required_status_checks":
                for check in rule["parameters"].get("required_status_checks", []):
                    ctx: str = check["context"]
                    if ctx not in contexts:
                        contexts.append(ctx)

    # Source 2: classic branch protection (used by nousergon-data,
    # crucible-dashboard alongside the merge_queue-only ruleset).
    try:
        prot = gh_api(f"/repos/{repo}/branches/main/protection")
        for ctx in prot.get("required_status_checks", {}).get("contexts", []):
            if ctx not in contexts:
                contexts.append(ctx)
    except (SystemExit, KeyError, json.JSONDecodeError):
        pass  # 404 = no classic protection; rulesets-only repos are common.

    return contexts


def parse_workflow_merge_group(path: Path) -> tuple[str | None, set[str]]:
    """Parse a workflow file and return (has_merge_group, {job_names}).

    Returns has_merge_group as the name of the merge_group types list if
    present, None otherwise. job_names is the set of job name: values from
    the workflow (these are the check context names that appear in CI).
    """
    with open(path) as fh:
        doc = yaml.safe_load(fh)

    if not isinstance(doc, dict):
        return None, set()

    has_mg = None
    on_block = doc.get("on", doc.get(True, {}))
    if isinstance(on_block, dict):
        mg = on_block.get("merge_group")
        if isinstance(mg, dict):
            mg_types = mg.get("types")
            if isinstance(mg_types, list):
                has_mg = str(mg_types)

    jobs = doc.get("jobs", {})
    job_names: set[str] = set()
    if isinstance(jobs, dict):
        for _jid, job_def in jobs.items():
            if isinstance(job_def, dict):
                name = job_def.get("name")
                # Matrix-expanded job names contain ${{ matrix.* }} — skip
                # those for exact match; they're handled via pattern.
                if isinstance(name, str) and "${{" not in name:
                    job_names.add(name)
                elif isinstance(name, str):
                    # Store the raw name for pattern matching later
                    job_names.add(name)

    return has_mg, job_names


def job_name_matches(job_pattern: str, check_context: str) -> bool:
    """Check if a job name pattern (potentially with matrix vars) matches
    a concrete check context name.

    Example: job_pattern="pytest (py${{ matrix.python-version }})"
             matches check_context="pytest (py3.9)"
    """
    # If no template var, exact match
    if "${{" not in job_pattern:
        return job_pattern == check_context

    # Build a regex: replace ${{ ... }} with a capture group for any non-empty chars
    import re
    escaped = re.escape(job_pattern)
    escaped = escaped.replace(r"\${{", "\\${{")
    pattern = re.sub(r"\\\$\\\{\\{[^}]+\\}\\}", r"(.+)", escaped)
    return bool(re.fullmatch(pattern, check_context))


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        # Allow local runs with --repo flag
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--repo" and i < len(sys.argv):
                repo = sys.argv[i + 1]
            elif arg.startswith("--repo="):
                repo = arg.split("=", 1)[1]
    if not repo:
        print("::error::GITHUB_REPOSITORY not set and no --repo flag provided")
        return 1

    print(f"Auditing required checks for merge_group triggers in {repo}")

    required = get_required_checks(repo)
    if not required:
        print("No required status checks found — nothing to validate.")
        return 0

    print(f"Found {len(required)} required check context(s):")
    for ctx in required:
        print(f"  - {ctx}")

    workflows_dir = Path(".github/workflows")
    if not workflows_dir.is_dir():
        print("::error::.github/workflows/ directory not found", file=sys.stderr)
        return 1

    wf_files = sorted(workflows_dir.glob("*.yml"))
    if not wf_files:
        print("::error::No workflow files found in .github/workflows/", file=sys.stderr)
        return 1

    print(f"\nScanning {len(wf_files)} workflow file(s)...")

    gaps: list[tuple[str, str | None]] = []  # (context, workflow_file or None)

    for ctx in required:
        found = False
        for wf_path in wf_files:
            has_mg, job_names = parse_workflow_merge_group(wf_path)
            if has_mg is None:
                continue  # workflow doesn't have merge_group at all
            # Check if any job in this workflow produces this check context
            for jname in job_names:
                if job_name_matches(jname, ctx):
                    found = True
                    break
            if found:
                break

        if not found:
            # Find which workflow produces this context (so we can report it)
            producer = None
            for wf_path in wf_files:
                _, job_names = parse_workflow_merge_group(wf_path)
                for jname in job_names:
                    if job_name_matches(jname, ctx):
                        producer = str(wf_path)
                        break
                if producer:
                    break
            gaps.append((ctx, producer))

    if gaps:
        print("\n--- MISSING merge_group TRIGGERS ---\n")
        for ctx, wf in gaps:
            if wf:
                print(f"  Required check:  {ctx}")
                print(f"  Produced by:     {wf}")
                print("  Missing:         merge_group: {types: [checks_requested]}\n")
            else:
                print(f"  Required check:  {ctx}")
                print("  Produced by:     UNKNOWN (no matching workflow job found)")
                print("  Action needed:   identify the producing workflow and add merge_group trigger\n")

        msg = f"{len(gaps)} required check(s) lack merge_group: {{types: [checks_requested]}} triggers."
        print(f"::error::{msg}")
        print("Add merge_group: {types: [checks_requested]} to the on: block of each listed workflow.")
        return 1

    print("\nAll required checks have merge_group triggers — no gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
