"""Merge-queue readiness primitives, shared by the in-repo guard and the
org-wide sweep.

**Why this is a library module and not a script.** The logic below existed as
`nousergon-lib/scripts/validate_merge_group_required_checks.py`, which can only
audit the repo it is running inside — it globs `.github/workflows` relative to
the working directory. `auto-merge-policy.md` §7.0 and `scm-platform-policy.md`
§3.2 both need the same questions answered *about other repos*, from a sweep
that has no checkout of them. Copying the parser into that sweep would be the
second adoption `shared-code-policy.md` names, so it is lifted here instead and
the script becomes a thin CLI over it (`nous-ergon-ops-I349`).

**The failure this whole module exists to prevent.** A merge queue re-runs
required checks against a synthetic merge commit, delivered as a `merge_group`
event. A workflow triggered only on `pull_request` never fires for that event,
so the required context never reports, so the entry waits until GitHub's
response timeout and **auto-dequeues unmerged** — nothing red, nothing logged,
the PR simply does not merge. That is what deadlocked the fleet on 2026-07-24
and forced a same-day revert (`nous-ergon-ops/retros/17-...`).

**Two mechanisms, always both.** Required checks live in classic branch
protection (`/branches/main/protection`) *or* in a ruleset
(`/rulesets`) — a ruleset-only repo returns 404 from the classic endpoint.
Reading one and not the other produces confident false answers in both
directions, which is how `nousergon-lib` was reported unprotected on
2026-07-28 while being the fleet's best-configured repo.

Every network read goes through an injected `api` callable so the pure logic is
testable with recorded state and no credentials.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

import yaml

__all__ = [
    "GUARD_CONTEXT_MARKER",
    "MergeQueueConfig",
    "active_merge_queue",
    "coverage_gaps",
    "gh_api",
    "guard_is_required",
    "job_name_matches",
    "parse_workflow",
    "required_contexts",
]

#: The guard's identity, matched against a *normalised* context (see
#: :func:`guard_is_required`). GitHub emits this check under at least three
#: shapes across the fleet — the bare slug, `<job-id> / <called job name>` for a
#: reusable-workflow call, and title-cased display names — so a literal
#: substring test on the raw context misses the very shape
#: `alpha-engine-config` emits: `guard / merge-group required-check guard`.
GUARD_CONTEXT_MARKER = "merge-group-required-check-guard"

ApiFn = Callable[[str], Any]


class MergeQueueConfig(dict):
    """The `merge_queue` rule's parameters, plus the ruleset that carried it.

    A dict subclass rather than a dataclass so callers can serialise it straight
    into a check-result envelope without a converter.
    """

    @property
    def ruleset_id(self) -> int:
        return self["_ruleset_id"]

    @property
    def response_timeout_minutes(self) -> int | None:
        return self.get("check_response_timeout_minutes")


def gh_api(path: str) -> Any:
    """Default `api` implementation: shell out to `gh api`.

    Raises on failure rather than returning a sentinel. A sweep that cannot read
    a repo's protection must not silently record that repo as compliant — an
    unreadable repo and a clean repo are the two things this module may never
    confuse (`principles.md` §2.7).
    """
    gh = shutil.which("gh") or "gh"
    result = subprocess.run(
        [gh, "api", path], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _api_or_none(api: ApiFn, path: str) -> Any:
    """Read a path whose absence is a legitimate answer (a 404 from the classic
    protection endpoint means 'this repo is ruleset-only', not 'error')."""
    try:
        return api(path)
    except Exception:
        return None


def required_contexts(repo: str, *, api: ApiFn = gh_api) -> list[str]:
    """Every required status-check context on `repo`'s default branch, from
    rulesets AND classic branch protection, deduplicated, order preserved.

    An empty list means *no required checks*, which is a real state and not an
    error — several fleet repos have none.
    """
    contexts: list[str] = []

    for rs in api(f"/repos/{repo}/rulesets") or []:
        if rs.get("enforcement") != "active":
            continue
        full = api(f"/repos/{repo}/rulesets/{rs['id']}")
        for rule in full.get("rules", []):
            if rule.get("type") != "required_status_checks":
                continue
            for check in rule.get("parameters", {}).get("required_status_checks", []):
                ctx = check.get("context")
                if ctx and ctx not in contexts:
                    contexts.append(ctx)

    prot = _api_or_none(api, f"/repos/{repo}/branches/main/protection")
    if isinstance(prot, dict):
        for ctx in prot.get("required_status_checks", {}).get("contexts", []) or []:
            if ctx not in contexts:
                contexts.append(ctx)

    return contexts


def active_merge_queue(repo: str, *, api: ApiFn = gh_api) -> MergeQueueConfig | None:
    """The active `merge_queue` rule on `repo`, or None.

    **Detected by rule type, never by ruleset name.** After the 2026-07-24
    revert, ~10 repos carried a ruleset literally named `main-merge-queue` whose
    `rules` was `[]` — it enforced nothing. A name-based check would have read
    every one of them as a live queue, which is exactly backwards from what an
    auditor needs.
    """
    for rs in api(f"/repos/{repo}/rulesets") or []:
        if rs.get("enforcement") != "active":
            continue
        full = api(f"/repos/{repo}/rulesets/{rs['id']}")
        for rule in full.get("rules", []):
            if rule.get("type") == "merge_queue":
                cfg = MergeQueueConfig(rule.get("parameters", {}))
                cfg["_ruleset_id"] = rs["id"]
                cfg["_ruleset_name"] = rs.get("name", "")
                return cfg
    return None


def parse_workflow(text: str) -> tuple[str | None, set[str]]:
    """`(merge_group_types_repr_or_None, {context names this workflow emits})`.

    Two GitHub behaviours drive the job-name logic, and getting either wrong
    makes the guard useless in opposite directions:

    * **A job's status context is its `name:` if present, else its job ID.** The
      original parser collected only `name:`, so `jobs: {pytest: {...}}` — the
      most common shape in the fleet — was invisible. Measured 2026-07-29: every
      one of 14 required checks across 7 repos reported `Produced by: UNKNOWN`,
      and worse, a `merge_group` trigger that DID cover such a job could not be
      seen, so the guard would keep reporting a gap after the gap was closed.
    * **Matrix names are templates.** `pytest (py${{ matrix.py }})` emits
      `pytest (py3.11)`. The raw template is retained here and resolved by
      :func:`job_name_matches`.
    """
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        return None, set()

    # `on:` is parsed by PyYAML as the boolean True under YAML 1.1, so both keys
    # must be tried. A workflow whose trigger block silently vanished would read
    # as "no merge_group" — a false gap on every check it produces.
    on_block = doc.get("on", doc.get(True, {}))
    has_mg: str | None = None
    if isinstance(on_block, dict):
        mg = on_block.get("merge_group")
        if isinstance(mg, dict) and isinstance(mg.get("types"), list):
            has_mg = str(mg["types"])

    job_names: set[str] = set()
    for jid, job_def in (doc.get("jobs") or {}).items():
        if not isinstance(job_def, dict):
            continue
        name = job_def.get("name")
        job_names.add(name if isinstance(name, str) else str(jid))

    return has_mg, job_names


def job_name_matches(job_pattern: str, check_context: str) -> bool:
    """Whether a (possibly matrix-templated) job name emits `check_context`."""
    if "${{" not in job_pattern:
        return job_pattern == check_context
    escaped = re.escape(job_pattern).replace(r"\${{", "\\${{")
    pattern = re.sub(r"\\\$\\\{\\{[^}]+\\}\\}", r"(.+)", escaped)
    return bool(re.fullmatch(pattern, check_context))


def coverage_gaps(
    contexts: list[str],
    workflows: dict[str, str],
) -> list[tuple[str, str | None]]:
    """`[(context, producing_workflow_or_None)]` for every required context that
    no `merge_group`-triggered workflow produces.

    `workflows` maps a display name (path or filename) to file text. One context
    may be produced by several workflows; it is covered if **at least one** of
    them declares the trigger. The reported producer is a best-effort name of a
    file to edit, and `None` means no workflow in the set claims the context at
    all — which is itself worth surfacing, because a required context nothing
    produces blocks every PR forever.
    """
    parsed = {name: parse_workflow(text) for name, text in workflows.items()}
    gaps: list[tuple[str, str | None]] = []

    for ctx in contexts:
        covered = any(
            has_mg is not None and any(job_name_matches(j, ctx) for j in jobs)
            for has_mg, jobs in parsed.values()
        )
        if covered:
            continue
        producer = next(
            (name for name, (_, jobs) in parsed.items()
             if any(job_name_matches(j, ctx) for j in jobs)),
            None,
        )
        gaps.append((ctx, producer))

    return gaps


def _normalise_context(ctx: str) -> str:
    """Lowercase, non-alphanumerics collapsed to single hyphens, ends trimmed.

    A status context is a display string, not an identifier: the same workflow
    appears as `merge-group-required-check-guard`, as
    `guard / merge-group required-check guard` when called as a reusable
    workflow, and title-cased when the workflow carries a `name:`. Comparing raw
    strings answers a question about formatting; normalising asks the one that
    was meant.
    """
    return re.sub(r"[^a-z0-9]+", "-", ctx.lower()).strip("-")


def guard_is_required(contexts: list[str]) -> bool:
    """Whether `merge-group-required-check-guard` is itself a required context.

    `scm-platform-policy.md` §3.2 makes this a precondition of running a queue:
    advisory is the wrong authority for a control whose failure mode is a silent
    dequeue. The guard reported the 2026-07-24 gap correctly and could not
    prevent it, because reporting is all an advisory check can do.

    Matched on the normalised context. A literal substring test looks correct
    and is not: `alpha-engine-config` emits this check as
    `guard / merge-group required-check guard`, which does not contain the slug,
    so the naive version would report the guard as advisory on the one repo
    where it is required — a false finding on the exact case the check exists
    for.
    """
    marker = _normalise_context(GUARD_CONTEXT_MARKER)
    return any(marker in _normalise_context(ctx) for ctx in contexts)
