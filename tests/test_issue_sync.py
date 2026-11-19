from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

import pytest

from nousergon_lib.issue_sync import DuplicateIssueTitleError, IssueSyncCounts, reconcile_issues


class Commands:
    def __init__(self, issues: list[dict[str, object]], *, fail_on: str | None = None) -> None:
        self.issues = issues
        self.fail_on = fail_on
        self.calls: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        call = list(command)
        self.calls.append(call)
        if self.fail_on in call:
            raise subprocess.CalledProcessError(1, call)
        return json.dumps(self.issues) if call[1:3] == ["issue", "list"] else ""


def sync(findings: dict[str, str], commands: Commands, *, dry_run: bool = False) -> IssueSyncCounts:
    return reconcile_issues(
        findings,
        graded_subjects=("alpha", "beta"),
        title_for=lambda subject: f"Grade: {subject}",
        body_for=lambda subject, finding: f"original {subject}: {finding}",
        labels=("grade", "automated"),
        repeat_comment_for=lambda subject, finding: f"repeat {subject}: {finding}",
        close_comment_for=lambda subject: f"resolved {subject}",
        repo="owner/repo",
        dry_run=dry_run,
        runner=commands,
    )


def test_creates_missing_issue_with_caller_formatting_and_labels() -> None:
    commands = Commands([])
    assert sync({"alpha": "bad"}, commands) == IssueSyncCounts(created=1)
    assert commands.calls[1] == [
        "gh", "issue", "create", "--title", "Grade: alpha", "--body", "original alpha: bad",
        "--label", "grade", "--label", "automated", "--repo", "owner/repo",
    ]


def test_repeated_finding_adds_comment_without_rewriting_original_body() -> None:
    commands = Commands([{"number": 7, "title": "Grade: alpha"}])
    assert sync({"alpha": "still bad"}, commands) == IssueSyncCounts(commented=1)
    assert commands.calls[1] == [
        "gh", "issue", "comment", "7", "--body", "repeat alpha: still bad", "--repo", "owner/repo",
    ]
    assert all("edit" not in call for call in commands.calls)


def test_only_resolved_graded_subject_is_closed() -> None:
    commands = Commands([
        {"number": 7, "title": "Grade: alpha"},
        {"number": 8, "title": "unrelated issue"},
    ])
    assert sync({}, commands) == IssueSyncCounts(closed=1)
    assert commands.calls[1] == [
        "gh", "issue", "close", "7", "--comment", "resolved alpha", "--repo", "owner/repo",
    ]
    assert all("8" not in call for call in commands.calls[1:])


def test_dry_run_enumerates_and_counts_but_does_not_mutate() -> None:
    commands = Commands([{"number": 7, "title": "Grade: beta"}])
    assert sync({"alpha": "bad"}, commands, dry_run=True) == IssueSyncCounts(created=1, closed=1)
    assert len(commands.calls) == 1


def test_duplicate_open_titles_are_refused() -> None:
    commands = Commands([
        {"number": 7, "title": "Grade: alpha"},
        {"number": 9, "title": "Grade: alpha"},
    ])
    with pytest.raises(DuplicateIssueTitleError, match="multiple open issues"):
        sync({"alpha": "bad"}, commands)
    assert len(commands.calls) == 1


def test_github_command_failure_propagates() -> None:
    commands = Commands([], fail_on="create")
    with pytest.raises(subprocess.CalledProcessError):
        sync({"alpha": "bad"}, commands)
