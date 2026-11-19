"""Synchronize one open GitHub issue per graded subject.

The subject-to-title function is the identity boundary: only exact title
matches are updated, and only titles produced for ``graded_subjects`` are
eligible for closure.  This lets callers share the mechanics without sharing
their presentation or accidentally adopting another caller's issues.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

__all__ = ["DuplicateIssueTitleError", "IssueSyncCounts", "reconcile_issues"]

Subject = TypeVar("Subject")
Finding = TypeVar("Finding")
CommandRunner = Callable[[Sequence[str]], str]


class DuplicateIssueTitleError(ValueError):
    """Raised when a title cannot identify exactly one open issue."""


@dataclass(frozen=True)
class IssueSyncCounts:
    """Numbers of mutations performed (or planned during a dry run)."""

    created: int = 0
    commented: int = 0
    closed: int = 0


@dataclass(frozen=True)
class _Issue:
    number: int
    title: str


def _run(command: Sequence[str]) -> str:
    """Run ``gh`` without hiding its exit status or diagnostic output."""
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603


def _open_issues(*, repo: str | None, runner: CommandRunner) -> list[_Issue]:
    command = ["gh", "issue", "list", "--state", "open", "--limit", "1000", "--json", "number,title"]
    if repo is not None:
        command.extend(("--repo", repo))
    raw = cast(object, json.loads(runner(command)))
    if not isinstance(raw, list):
        raise ValueError("gh issue list did not return a JSON array")

    issues: list[_Issue] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise ValueError("gh issue list returned an invalid issue")
        values = cast(dict[str, object], item)
        number = values.get("number")
        title = values.get("title")
        if not isinstance(number, int) or not isinstance(title, str):
            raise ValueError("gh issue list returned an invalid issue")
        issues.append(_Issue(number=number, title=title))
    return issues


def reconcile_issues(
    findings: Mapping[Subject, Finding],
    *,
    graded_subjects: Iterable[Subject],
    title_for: Callable[[Subject], str],
    body_for: Callable[[Subject, Finding], str],
    labels: Sequence[str] = (),
    repeat_comment_for: Callable[[Subject, Finding], str],
    close_comment_for: Callable[[Subject], str],
    repo: str | None = None,
    dry_run: bool = False,
    runner: CommandRunner = _run,
) -> IssueSyncCounts:
    """Reconcile current ``findings`` with open issues in a repository.

    A missing current finding is created, and a repeated current finding is
    commented on without changing the issue's original body.  An open issue
    is closed only when its exact title belongs to a graded subject that has
    no current finding.  Counts describe the same planned work in dry-run
    mode, while mutation commands are omitted entirely.

    The runner is injectable for callers that already wrap ``gh``.  It must
    return stdout and raise on command failure; the default runner uses
    ``check=True`` and deliberately lets failures propagate.
    """
    subjects = tuple(graded_subjects)
    if len(set(subjects)) != len(subjects):
        raise ValueError("graded_subjects contains a duplicate subject")
    unknown = set(findings).difference(subjects)
    if unknown:
        raise ValueError("findings contains a subject outside graded_subjects")

    titles = {subject: title_for(subject) for subject in subjects}
    if any(not title.strip() for title in titles.values()):
        raise ValueError("issue titles must not be empty")
    if len(set(titles.values())) != len(titles):
        raise DuplicateIssueTitleError("title_for produced the same title for multiple subjects")

    issues_by_title: dict[str, _Issue] = {}
    graded_titles = set(titles.values())
    for issue in _open_issues(repo=repo, runner=runner):
        if issue.title not in graded_titles:
            continue
        if issue.title in issues_by_title:
            raise DuplicateIssueTitleError(f"multiple open issues have title {issue.title!r}")
        issues_by_title[issue.title] = issue

    created = commented = closed = 0
    repo_args = [] if repo is None else ["--repo", repo]
    for subject in subjects:
        title = titles[subject]
        issue = issues_by_title.get(title)
        has_finding = subject in findings
        if has_finding and issue is None:
            finding = findings[subject]
            created += 1
            if not dry_run:
                command = ["gh", "issue", "create", "--title", title, "--body", body_for(subject, finding)]
                for label in labels:
                    command.extend(("--label", label))
                runner([*command, *repo_args])
        elif has_finding and issue is not None:
            finding = findings[subject]
            commented += 1
            if not dry_run:
                runner(
                    ["gh", "issue", "comment", str(issue.number), "--body", repeat_comment_for(subject, finding), *repo_args]
                )
        elif not has_finding and issue is not None:
            closed += 1
            if not dry_run:
                runner(
                    ["gh", "issue", "close", str(issue.number), "--comment", close_comment_for(subject), *repo_args]
                )

    return IssueSyncCounts(created=created, commented=commented, closed=closed)
