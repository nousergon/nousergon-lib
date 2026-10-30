"""The agent-instruction guard checks BOTH names and the gitignore (alpha-engine-config-I8049).

`.github/workflows/claude-md-guard.yml` is the reusable check public repos
call to keep agent-instruction files out of git. It used to assert only
`git ls-files | grep -qx CLAUDE.md`. Since 2026-07-23, though, `AGENTS.md`
is the canonical file and `CLAUDE.md` a symlink to it, so the guard watched
the pointer and missed the content. On 2026-08-21 `telos` and
`nousergon-docs` gitignored `CLAUDE.md` and not `AGENTS.md`.

These tests run the workflow step's own `run:` script, read out of the YAML
so the test cannot drift from what CI executes, in throwaway git repos.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude-md-guard.yml"

GIT = shutil.which("git")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(GIT is None or BASH is None, reason="needs git and bash")


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _script() -> str:
    steps = _wf()["jobs"]["check"]["steps"]
    runs = [s["run"] for s in steps if "run" in s]
    assert len(runs) == 1, "the guard is exactly one run step"
    return runs[0]


def _repo(tmp_path: Path, *, gitignore: str, tracked: tuple[str, ...] = ()) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": str(Path(GIT).parent)}

    def git(*args: str) -> None:
        subprocess.run([GIT, *args], cwd=repo, check=True, capture_output=True, env=env)

    git("init", "-q")
    (repo / ".gitignore").write_text(gitignore, encoding="utf-8")
    (repo / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    (repo / "CLAUDE.md").symlink_to("AGENTS.md")
    git("add", ".gitignore")
    for name in tracked:
        git("add", "-f", name)
    return repo


def _run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", _script()], cwd=repo, capture_output=True, text=True)


def test_the_job_id_is_unchanged_for_sha_pinned_callers():
    jobs = _wf()["jobs"]
    assert list(jobs) == ["check"]
    assert jobs["check"]["name"] == "claude-md-untracked"


def test_both_names_untracked_and_ignored_passes(tmp_path):
    proc = _run(_repo(tmp_path, gitignore="AGENTS.md\nCLAUDE.md\n"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENTS.md is untracked and gitignored" in proc.stdout
    assert "CLAUDE.md is untracked and gitignored" in proc.stdout


@pytest.mark.parametrize("name", ["AGENTS.md", "CLAUDE.md"])
def test_a_tracked_file_fails(tmp_path, name):
    proc = _run(_repo(tmp_path, gitignore="AGENTS.md\nCLAUDE.md\n", tracked=(name,)))
    assert proc.returncode == 1
    assert f"::error::{name} is committed" in proc.stdout


def test_the_measured_case_claude_ignored_agents_not_fails(tmp_path):
    """telos and nousergon-docs, 2026-08-21."""
    proc = _run(_repo(tmp_path, gitignore="CLAUDE.md\n"))
    assert proc.returncode == 1
    assert "::error::AGENTS.md is not gitignored" in proc.stdout
    assert "CLAUDE.md is untracked and gitignored" in proc.stdout


def test_neither_ignored_reports_both(tmp_path):
    proc = _run(_repo(tmp_path, gitignore=""))
    assert proc.returncode == 1
    assert proc.stdout.count("is not gitignored") == 2


def test_a_glob_ignore_counts(tmp_path):
    proc = _run(_repo(tmp_path, gitignore="/*.md\n!README.md\n"))
    assert proc.returncode == 0, proc.stdout
