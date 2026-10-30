"""No live-state job on the pull_request path: the fleet guard (alpha-engine-config-I9830).

`scm-platform-policy.md` §3.1 (Brian ruling 2026-08-10): a check grades the
diff. A check whose subject is live infrastructure, or the system's state at a
phase boundary, does not belong on a ``pull_request`` trigger at all. On a PR
it either reports the author's own change as drift, or it reports somebody
else's out-of-band change, which the author cannot clear. The policy says
conformance "is held by a per-repo guard test". Until this module, three
repos carried one, each hand-written, and one of them matched a list of
known drift-script filenames, so a new script walked past it.

**The question is inverted, and that is the whole design.** Every job that
can run on a PR event counts as live-state UNTIL the consuming repo names it
in its own allowlist with a reason. Nothing here scans ``run:`` bodies for
markers like ``aws `` or ``boto3``. crucible's reference guard
(``crucible/tests/test_workflow_triggers.py``, crucible-PR38) recorded why:
six shapes walked past a marker scan (a job-level ``uses:``, a marker in
``with:``, a called script, a tab for the space, a credential step plus a
CLI, ``pull_request_target``), and every per-command splitter fell to the
next shell syntax. Any predicate over a shell body is a partial shell
parser, which is a denylist of the syntax somebody thought of.

What this module does decide, structurally:

* **Which events count.** ``pull_request`` and ``pull_request_target``. The
  second runs with base-repo secrets, so it is held most tightly, not exempted.
* **What excludes a job.** Only a WHOLE ``if:`` expression from
  :func:`exclusions_for`, compared as a string. ``... != 'pull_request' ||
  x`` is a disjunction that runs on PRs, and a substring test waves it
  through. An expression naming one PR event does not exclude the other.
* **What an allowlist entry may cover.** A job-level ``uses:`` delegates to a
  workflow this guard never reads, so allowlisting one needs a second,
  explicit admission (``reusable``), which the consumer backs with a test
  pinning the target. An allowlisted job with no steps and no ``uses:``
  describes nothing.
* **That entries cannot outlive their job.** An allowlist key naming no job
  that still runs on a PR is a finding. Otherwise the list only grows.
* **That a dark guard fails.** No workflow files, an unparseable workflow, or
  a workflow with no ``on:`` block is a finding, never a pass. PyYAML reads a
  bare ``on`` as the boolean True, and a guard that missed that would read
  every workflow as trigger-free and pass them all.

Consumers call :func:`check` (or :func:`assert_no_live_state_on_pr_path`)
from one test, with their own allowlist::

    PR_REACHABLE_JOBS = {"ci.yml:test": "lint + unit tests over the checkout; no credential"}
    def test_no_live_state_job_runs_on_the_pull_request_path():
        assert_no_live_state_on_pr_path(WORKFLOW_DIR, PR_REACHABLE_JOBS)

Standard library plus ``yaml`` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

#: The PR events this guard holds. ``pull_request_target`` is the one someone
#: reaches for when they want secrets on a PR.
PR_EVENTS: frozenset[str] = frozenset({"pull_request", "pull_request_target"})

_BOTH = '!contains(fromJSON(\'["pull_request","pull_request_target"]\'), github.event_name)'


class WorkflowParseError(ValueError):
    """A workflow this guard could not understand. It is never trigger-free."""


@dataclass(frozen=True)
class Job:
    """One job, as much of it as this guard reasons about."""

    name: str
    condition: str
    uses: str
    has_steps: bool


@dataclass(frozen=True)
class Workflow:
    """A workflow's file name, its trigger events and its jobs."""

    name: str
    events: frozenset[str]
    jobs: Mapping[str, Job]

    @property
    def pr_events(self) -> frozenset[str]:
        return self.events & PR_EVENTS


@dataclass(frozen=True)
class Finding:
    """One reason the guard is red. ``key`` is ``<workflow file>:<job>``."""

    key: str
    kind: str
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.key}: {self.message}"


def exclusions_for(events: Iterable[str]) -> frozenset[str]:
    """The ``if:`` expressions that provably keep a job off every PR event in ``events``.

    Both the bare and the ``${{ }}``-wrapped forms are accepted. For two
    events, a form naming only one of them is not an exclusion.
    """
    present = sorted(set(events) & PR_EVENTS)
    if not present:
        return frozenset()
    single = [f"github.event_name != '{event}'" for event in present]
    accepted = {_BOTH, " && ".join(single)}
    return frozenset(accepted | {"${{ " + form + " }}" for form in accepted})


def is_excluded(job: Job, pr_events: Iterable[str]) -> bool:
    """Whole-string comparison against :func:`exclusions_for`, never a substring test."""
    return job.condition.strip() in exclusions_for(pr_events)


def _events(triggers: object) -> frozenset[str]:
    if isinstance(triggers, str):
        return frozenset({triggers})
    if isinstance(triggers, list):
        return frozenset(str(item) for item in triggers)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType] -- YAML list of event names
    if isinstance(triggers, dict):
        return frozenset(str(key) for key in triggers)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType] -- YAML mapping keyed by event name
    raise WorkflowParseError(f"`on:` is a {type(triggers).__name__}, not an event name, list or mapping")


def _job(name: str, raw: object) -> Job:
    if not isinstance(raw, dict):
        raise WorkflowParseError(f"job {name!r} is not a mapping")
    condition = raw.get("if", "")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- raw YAML mapping
    uses = raw.get("uses", "")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- raw YAML mapping
    steps = raw.get("steps") or []  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- raw YAML mapping
    return Job(
        name=name,
        condition=str(condition),  # pyright: ignore[reportUnknownArgumentType] -- coerced to str
        uses=str(uses),  # pyright: ignore[reportUnknownArgumentType] -- coerced to str
        has_steps=isinstance(steps, list) and len(steps) > 0,  # pyright: ignore[reportUnknownArgumentType] -- length only
    )


def parse_workflow(name: str, text: str) -> Workflow:
    """Parse one workflow document. Raises :class:`WorkflowParseError` rather than guess."""
    try:
        document: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkflowParseError(f"not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkflowParseError("not a mapping")
    # PyYAML (YAML 1.1) resolves a bare `on` key to the boolean True.
    if True in document:
        triggers: object = document[True]  # pyright: ignore[reportUnknownVariableType] -- raw YAML value
    elif "on" in document:
        triggers = document["on"]  # pyright: ignore[reportUnknownVariableType] -- raw YAML value
    else:
        raise WorkflowParseError("no `on:` block: unparseable, not trigger-free")
    raw_jobs: object = document.get("jobs") or {}  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- raw YAML mapping
    if not isinstance(raw_jobs, dict):
        raise WorkflowParseError("`jobs:` is not a mapping")
    jobs = {str(job_name): _job(str(job_name), raw) for job_name, raw in raw_jobs.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType] -- raw YAML mapping
    return Workflow(name=name, events=_events(triggers), jobs=jobs)  # pyright: ignore[reportUnknownArgumentType] -- narrowed in _events


def workflow_files(workflow_dir: Path) -> list[Path]:
    """Every ``.yml`` / ``.yaml`` file directly under ``workflow_dir``, sorted. GitHub accepts both suffixes."""
    return sorted(p for p in workflow_dir.glob("*") if p.is_file() and p.suffix in {".yml", ".yaml"})


def check(
    workflow_dir: Path,
    pr_reachable: Mapping[str, str],
    reusable: Iterable[str] = (),
) -> list[Finding]:
    """Every reason ``workflow_dir`` breaks §3.1, given the repo's own allowlist.

    ``pr_reachable`` maps ``<workflow file>:<job>`` to the reason that job's
    subject is the tree or the pull request itself. ``reusable`` names the
    subset of those that are job-level ``uses:`` calls the consumer has
    argued separately, normally in a test pinning the called workflow.
    An empty result means a clean, non-vacuous pass.
    """
    reusable_keys = frozenset(reusable)
    findings: list[Finding] = []
    files = workflow_files(workflow_dir)
    if not files:
        return [
            Finding(str(workflow_dir), "no-workflows", "no workflow files found; a guard that read nothing is dark")
        ]

    reachable: set[str] = set()
    for path in files:
        try:
            workflow = parse_workflow(path.name, path.read_text(encoding="utf-8"))
        except WorkflowParseError as exc:
            findings.append(Finding(path.name, "unparseable", str(exc)))
            continue
        pr_events = workflow.pr_events
        if not pr_events:
            continue
        for job in workflow.jobs.values():
            key = f"{workflow.name}:{job.name}"
            if is_excluded(job, pr_events):
                continue
            reachable.add(key)
            reason = pr_reachable.get(key)
            if reason is None:
                findings.append(
                    Finding(
                        key,
                        "unlisted",
                        f"can run on {sorted(pr_events)} and is not declared as grading the tree. "
                        "scm-platform-policy.md §3.1: move it to `push: [main]` plus a schedule, "
                        f"give it one of these exact `if:` expressions {sorted(exclusions_for(pr_events))}, "
                        "or, if its subject genuinely is the diff, add it to the repo's allowlist with the reason.",
                    )
                )
                continue
            if not reason.strip():
                findings.append(Finding(key, "empty-reason", "allowlisted with an empty reason"))
            if job.uses and key not in reusable_keys:
                findings.append(
                    Finding(
                        key,
                        "reusable-unlisted",
                        f"is allowlisted but is a job-level `uses:` of {job.uses!r}. The called workflow declares "
                        "workflow_call, so nothing here reads it. Admit it in `reusable` with a test pinning the "
                        "target, or inline the steps.",
                    )
                )
            elif not job.uses and not job.has_steps:
                findings.append(
                    Finding(key, "stepless", "is allowlisted and has no steps; an entry must describe what a job does")
                )

    for key in sorted(set(pr_reachable) - reachable):
        findings.append(
            Finding(key, "stale-entry", "is allowlisted but no such job runs on a PR event; remove the entry")
        )
    for key in sorted(reusable_keys - set(pr_reachable)):
        findings.append(Finding(key, "stale-entry", "is in `reusable` but not in the allowlist"))
    return findings


def assert_no_live_state_on_pr_path(
    workflow_dir: Path,
    pr_reachable: Mapping[str, str],
    reusable: Iterable[str] = (),
) -> None:
    """:func:`check`, raising ``AssertionError`` that lists every finding."""
    findings = check(workflow_dir, pr_reachable, reusable)
    if findings:
        raise AssertionError(
            f"{len(findings)} live-state-on-PR-path finding(s):\n" + "\n".join(f"  {f}" for f in findings)
        )
