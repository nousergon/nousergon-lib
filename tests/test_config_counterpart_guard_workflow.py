"""The config-counterpart guard answers on merge_group without loosening pull_request.

`.github/workflows/config-counterpart-guard.yml` used to read only
`github.head_ref`, which is empty on a `merge_group` event, so every
merge-queue entry failed the required check (alpha-engine-config-I12754).
nousergon-lib#512 resolves the queued PR's number from
`merge_group.head_ref`, reads its head branch through the pulls API, and asks
the same `git ls-remote` question, failing closed on any parse or API error.

These tests run the workflow step's own `run:` script, read out of the YAML
so they cannot drift from what CI executes, with `gh` and `git` stubbed on
PATH. The stubs record what they were asked, so each case proves which
branch the guard actually checked, not only its exit code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "config-counterpart-guard.yml"

BASH = shutil.which("bash")
JQ = shutil.which("jq")
pytestmark = pytest.mark.skipif(BASH is None or JQ is None, reason="needs bash and jq")

REPO = "nousergon/crucible-research"
QUEUE_REF = "refs/heads/gh-readonly-queue/main/pr-42-0123456789abcdef0123456789abcdef01234567"

# gh: `gh api repos/<repo>/pulls/<n> --jq .head.ref` prints STUB_PR_BRANCH,
# or exits STUB_GH_RC when that is set non-zero.
GH_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG_DIR/gh.calls"
if [ -n "${STUB_GH_RC:-}" ] && [ "$STUB_GH_RC" != 0 ]; then
  echo "gh: Resource not accessible by integration (HTTP 403)" >&2
  exit "$STUB_GH_RC"
fi
printf '%s\\n' "${STUB_PR_BRANCH:-}"
"""

# git: `git ls-remote --exit-code --heads <url> refs/heads/<b>` exits 0 when
# <b> is listed in STUB_CONFIG_BRANCHES, else 2; STUB_LSREMOTE_RC overrides.
GIT_STUB = """#!/usr/bin/env bash
ref="${@: -1}"
echo "$ref" >> "$STUB_LOG_DIR/git.calls"
if [ -n "${STUB_LSREMOTE_RC:-}" ]; then exit "$STUB_LSREMOTE_RC"; fi
for b in ${STUB_CONFIG_BRANCHES:-}; do
  if [ "$ref" = "refs/heads/$b" ]; then exit 0; fi
done
exit 2
"""


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step() -> dict:
    steps = _wf()["jobs"]["check"]["steps"]
    runs = [s for s in steps if "run" in s]
    assert len(runs) == 1, "the guard is exactly one run step"
    return runs[0]


def _run(
    tmp_path: Path,
    *,
    event: str,
    head_ref: str = "",
    queue_ref: str | None = None,
    with_config_token: bool = True,
    pr_branch: str = "",
    gh_rc: int = 0,
    config_branches: tuple[str, ...] = (),
    lsremote_rc: int | None = None,
) -> tuple[subprocess.CompletedProcess, list[str], list[str]]:
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in (("gh", GH_STUB), ("git", GIT_STUB)):
        p = stubs / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    payload: dict = {}
    if queue_ref is not None:
        payload["merge_group"] = {"head_ref": queue_ref}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload), encoding="utf-8")

    # The step's env block, as GitHub would render it for this event.
    rendered = {
        "TOKEN": "stub-value" if with_config_token else "",
        "EVENT_NAME": event,
        "PR_HEAD_REF": head_ref,
        "GH_TOKEN": "gh-token",
        "REPO": REPO,
    }
    assert set(_step()["env"]) == set(rendered), "step env changed; update the test's rendering"
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ.get('PATH', '')}",
        "GITHUB_EVENT_PATH": str(event_path),
        "STUB_LOG_DIR": str(logs),
        "STUB_PR_BRANCH": pr_branch,
        "STUB_GH_RC": str(gh_rc),
        "STUB_CONFIG_BRANCHES": " ".join(config_branches),
        **rendered,
    }
    if lsremote_rc is not None:
        env["STUB_LSREMOTE_RC"] = str(lsremote_rc)
    # GitHub runs a bash `run:` step as `bash --noprofile --norc -eo pipefail {0}`.
    proc = subprocess.run(
        [BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c", _step()["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    def calls(name: str) -> list[str]:
        f = logs / f"{name}.calls"
        return f.read_text(encoding="utf-8").splitlines() if f.exists() else []

    return proc, calls("gh"), calls("git")


def test_the_check_context_and_permissions_are_unchanged_for_sha_pinned_callers():
    wf = _wf()
    assert list(wf["jobs"]) == ["check"]
    assert wf["jobs"]["check"]["name"] == "config-counterpart-merged"
    # A callee that asks for more than its caller grants fails at startup, and
    # crucible-research calls this with `permissions: {}`.
    assert "permissions" not in wf
    assert "permissions" not in wf["jobs"]["check"]


# --- pull_request: the enforcing path must be exactly as strict as before ---


@pytest.mark.parametrize("event", ["pull_request", "pull_request_target"])
def test_pr_with_a_same_named_config_branch_fails(tmp_path, event):
    proc, gh, git = _run(tmp_path, event=event, head_ref="feat-x", config_branches=("feat-x",))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "::error::Branch 'feat-x' still exists" in proc.stdout
    assert git == ["refs/heads/feat-x"]
    assert gh == [], "the PR path reads github.head_ref, never the pulls API"


def test_pr_without_a_counterpart_passes(tmp_path):
    proc, gh, git = _run(tmp_path, event="pull_request", head_ref="feat-x", config_branches=("other",))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK to merge" in proc.stdout
    assert git == ["refs/heads/feat-x"]


def test_pr_with_an_empty_head_ref_fails_closed(tmp_path):
    proc, _, git = _run(tmp_path, event="pull_request", head_ref="")
    assert proc.returncode == 1
    assert "No head branch resolved" in proc.stdout
    assert git == []


def test_pr_ls_remote_error_fails_loud(tmp_path):
    proc, _, _ = _run(tmp_path, event="pull_request", head_ref="feat-x", lsremote_rc=128)
    assert proc.returncode == 1
    assert "git ls-remote exit 128" in proc.stdout


def test_fork_pr_without_the_config_token_is_the_documented_degrade(tmp_path):
    proc, _, git = _run(tmp_path, event="pull_request", head_ref="feat-x", with_config_token=False)
    assert proc.returncode == 0
    assert "::notice::ALPHA_ENGINE_CONFIG_TOKEN unavailable" in proc.stdout
    assert git == []


# --- merge_group: resolve the queued PR's branch, then ask the same question ---


def test_merge_group_with_a_same_named_config_branch_fails(tmp_path):
    """The issue's proof: fails with a counterpart branch..."""
    proc, gh, git = _run(
        tmp_path, event="merge_group", queue_ref=QUEUE_REF, pr_branch="feat-x", config_branches=("feat-x",)
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Resolved merge_group entry to PR #42, head branch 'feat-x'" in proc.stdout
    assert "::error::Branch 'feat-x' still exists" in proc.stdout
    assert gh == [f"api repos/{REPO}/pulls/42 --jq .head.ref"]
    assert git == ["refs/heads/feat-x"]


def test_merge_group_without_a_counterpart_passes(tmp_path):
    """...and passes with none."""
    proc, gh, git = _run(tmp_path, event="merge_group", queue_ref=QUEUE_REF, pr_branch="feat-x")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK to merge" in proc.stdout
    assert git == ["refs/heads/feat-x"]


def test_merge_group_ignores_an_empty_github_head_ref(tmp_path):
    """head_ref is empty on merge_group; the queue ref is what is resolved."""
    proc, _, git = _run(
        tmp_path, event="merge_group", head_ref="", queue_ref=QUEUE_REF, pr_branch="feat-x", config_branches=("feat-x",)
    )
    assert proc.returncode == 1
    assert git == ["refs/heads/feat-x"]


def test_merge_group_on_a_base_branch_with_a_slash(tmp_path):
    ref = "refs/heads/gh-readonly-queue/release/v2/pr-7-abcdef0"
    proc, gh, _ = _run(tmp_path, event="merge_group", queue_ref=ref, pr_branch="feat-y")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert gh == [f"api repos/{REPO}/pulls/7 --jq .head.ref"]


@pytest.mark.parametrize(
    "queue_ref",
    [
        "",
        "refs/heads/feat-x",
        "refs/heads/gh-readonly-queue/main/pr-abc-0123abc",
        "refs/heads/gh-readonly-queue/main/pr-42-NOTHEX",
    ],
)
def test_merge_group_with_an_unparseable_queue_ref_fails_closed(tmp_path, queue_ref):
    proc, gh, git = _run(tmp_path, event="merge_group", queue_ref=queue_ref, pr_branch="feat-x")
    assert proc.returncode == 1
    assert "Could not extract a PR number" in proc.stdout
    assert gh == [] and git == []


def test_merge_group_with_no_merge_group_payload_fails_closed(tmp_path):
    proc, gh, git = _run(tmp_path, event="merge_group", queue_ref=None, pr_branch="feat-x")
    assert proc.returncode == 1
    assert gh == [] and git == []


def test_merge_group_pulls_api_failure_fails_closed_and_names_the_grant(tmp_path):
    proc, _, git = _run(tmp_path, event="merge_group", queue_ref=QUEUE_REF, gh_rc=1)
    assert proc.returncode == 1
    assert "pull-requests: read" in proc.stdout
    assert git == []


def test_merge_group_pulls_api_empty_branch_fails_closed(tmp_path):
    proc, _, git = _run(tmp_path, event="merge_group", queue_ref=QUEUE_REF, pr_branch="")
    assert proc.returncode == 1
    assert "No head branch resolved" in proc.stdout
    assert git == []


# --- any other event is a misuse ---


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "schedule"])
def test_other_events_fail(tmp_path, event):
    proc, gh, git = _run(tmp_path, event=event, head_ref="feat-x")
    assert proc.returncode == 1
    assert "call it only from pull_request or merge_group" in proc.stdout
    assert gh == [] and git == []
