"""Validate that every required status check in branch-protection rulesets
has a workflow that triggers on merge_group events.

Prevents the class of bug that deadlocked the merge queue twice today
(2026-07-23): a required check whose workflow only fires on pull_request
but the merge queue fires merge_group — the context never reports, the
queue waits 60 minutes, the PR auto-dequeues unmerged.

Run locally:  python3 scripts/validate_merge_group_required_checks.py
In CI:        called from merge-group-required-check-guard.yml
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def gh_api(path: str) -> dict | list:
    """Call gh api and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"::error::gh api {path} failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def get_required_checks() -> list[str]:
    """Collect every required status check context from all active rulesets.

    The rulesets LIST endpoint does NOT include the rules array — each
    ruleset must be fetched individually to retrieve its rules.
    """
    repo = os.environ["GITHUB_REPOSITORY"]
    rulesets = gh_api(f"/repos/{repo}/rulesets")
    contexts: list[str] = []
    for rs in rulesets:
        if rs.get("enforcement") != "active":
            continue
        rs_id = rs["id"]
        full = gh_api(f"/repos/{repo}/rulesets/{rs_id}")
        for rule in full.get("rules", []):
            if rule.get("type") != "required_status_checks":
                continue
            for check in rule.get("parameters", {}).get("required_status_checks", []):
                ctx: str = check["context"]
                if ctx not in contexts:
                    contexts.append(ctx)
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
    repo = os.environ["GITHUB_REPOSITORY"]
    print(f"Auditing required checks for merge_group triggers in {repo}")

    required = get_required_checks()
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
