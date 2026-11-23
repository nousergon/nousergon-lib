"""Tests for scripts/agent_commit_identity_guard.py (alpha-engine-config-I13060).

Closes-when of that issue: "a PR commit with a `Claude-Session:` trailer and
the owner as author or sign-off fails the check, shown by a test fixture".
``test_closes_when_*`` are those fixtures.

Every test builds a real throwaway git repo (base commit, then the PR's
commits) and runs ``main()`` with a fake GitHub reader, so the git range, the
trailer parsing and the verdict are exercised end to end. The owner is a
fixture identity; nothing real is named here.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "agent_commit_identity_guard.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "agent-commit-identity-guard.yml"
_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="needs git")

OWNER_LOGIN = "owner-login"
OWNER_ID = 4242
OWNER = ("Owner Person", "owner@example.com")
AGENT = ("Agent", "agent@example.invalid")
GITHUB = ("GitHub", "noreply@github.com")
SESSION = "Claude-Session: https://example.invalid/session_x"


def _load():
    spec = importlib.util.spec_from_file_location("agent_commit_identity_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_commit_identity_guard"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


g = _load()


class _Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir()
        self._git("init", "-q")
        self.base = self.commit("base", author=AGENT)

    def _git(self, *args: str, env: dict | None = None) -> str:
        full = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": str(Path(_GIT).parent)}
        full.update(env or {})
        return subprocess.run([_GIT, *args], cwd=self.path, check=True, capture_output=True, text=True, env=full).stdout

    def commit(self, message: str, *, author: tuple[str, str], committer: tuple[str, str] | None = None) -> str:
        committer = committer or author
        (self.path / "f.txt").write_text(message, encoding="utf-8")
        self._git("add", "f.txt")
        env = {
            "GIT_AUTHOR_NAME": author[0],
            "GIT_AUTHOR_EMAIL": author[1],
            "GIT_COMMITTER_NAME": committer[0],
            "GIT_COMMITTER_EMAIL": committer[1],
        }
        self._git("commit", "-q", "-m", message, env=env)
        return self._git("rev-parse", "HEAD").strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()


def _getter(repo: _Repo, *, profile: Any = None, fail: str = "", forged: bool = False) -> Any:
    """GitHub stand-in: the owner's email resolves to the owner login, GitHub's
    own committer address to ``web-flow``; nothing else resolves."""
    logins = {OWNER[1]: OWNER_LOGIN, GITHUB[1]: "web-flow"}
    if profile is None:
        profile = {"login": OWNER_LOGIN, "id": OWNER_ID, "name": OWNER[0], "email": None}

    def get(path: str) -> Any:
        if fail and path.startswith(fail):
            raise g.VerifyError(f"GET /{path}: HTTP 502")
        if path.startswith("users/"):
            return profile
        sha = path.rsplit("/", 1)[-1]
        out = repo._git("show", "-s", "--format=%ae%x00%ce", sha).strip()
        ae, ce = out.split("\x00")
        # GitHub signs what it commits; a local commit claiming its address is unsigned.
        signed = ce == GITHUB[1] and not forged
        return {"author": {"login": logins[ae]} if ae in logins else None,
                "committer": {"login": logins[ce]} if ce in logins else None,
                "commit": {"verification": {"verified": signed,
                                            "reason": "valid" if signed else "unsigned"}}}

    return get


def _run(repo: _Repo, *extra: str, **kw: Any) -> int:
    args = ["--repo", str(repo.path), "--base", repo.base, "--head", repo.head(),
            "--github-repo", "nousergon/example", "--owner-login", OWNER_LOGIN, *extra]
    return g.main(args, get=_getter(repo, **kw))


# ── the Closes-when fixtures ─────────────────────────────────────────────────


def test_closes_when_agent_commit_authored_as_owner_fails(tmp_path, capsys):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix: thing\n\n{SESSION}\n", author=OWNER)
    assert _run(repo) == 1
    out = capsys.readouterr().out
    assert "AUTHORED as the owner" in out and "COMMITTED as the owner" in out


def test_closes_when_agent_commit_signed_off_as_owner_fails(tmp_path, capsys):
    """The 10-04 shape had both; the sign-off alone is enough. The agent is
    the author, and only the profile name identifies the sign-off."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix: thing\n\nSigned-off-by: {OWNER[0]} <private@example.org>\n{SESSION}\n", author=AGENT)
    assert _run(repo) == 1
    assert "Signed-off-by for the owner (it is the owner's name)" in capsys.readouterr().out


# ── the rest of the rule ─────────────────────────────────────────────────────


def test_agent_commit_as_agent_passes(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix: thing\n\nCo-Authored-By: {AGENT[0]} <{AGENT[1]}>\n{SESSION}\n", author=AGENT)
    assert _run(repo) == 0


def test_owner_commit_without_session_trailer_is_not_judged(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"chore: by hand\n\nSigned-off-by: {OWNER[0]} <{OWNER[1]}>\n", author=OWNER)
    assert _run(repo) == 0


def test_agent_author_owner_committer_fails(tmp_path, capsys):
    """`git -c user.name=…` with a preset author still commits as the owner."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=AGENT, committer=OWNER)
    assert _run(repo) == 1
    out = capsys.readouterr().out
    assert "COMMITTED as the owner" in out and "AUTHORED" not in out


def test_noreply_signoff_is_the_owner_without_a_profile_name(tmp_path, capsys):
    repo = _Repo(tmp_path / "r")
    repo.commit(
        f"fix\n\nSigned-off-by: someone <{OWNER_ID}+{OWNER_LOGIN}@users.noreply.github.com>\n{SESSION}\n",
        author=AGENT,
    )
    assert _run(repo, profile={"id": OWNER_ID, "name": None}) == 1
    assert "noreply address" in capsys.readouterr().out


def test_email_resolved_elsewhere_in_the_pr_catches_a_signoff(tmp_path, capsys):
    """One commit resolves the owner's email; another signs off with it under
    a different name."""
    repo = _Repo(tmp_path / "r")
    repo.commit("by hand, no trailer", author=OWNER)
    repo.commit(f"fix\n\nSigned-off-by: B <{OWNER[1]}>\n{SESSION}\n", author=AGENT)
    assert _run(repo, profile={"id": OWNER_ID, "name": None}) == 1
    assert "an email address the owner uses" in capsys.readouterr().out


def test_github_synthesised_commit_is_exempt_but_its_signoff_is_not(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"squash\n\n{SESSION}\n", author=OWNER, committer=GITHUB)
    assert _run(repo) == 0
    repo.commit(f"api commit\n\nSigned-off-by: {OWNER[0]} <x@example.org>\n{SESSION}\n",
                author=OWNER, committer=GITHUB)
    assert _run(repo) == 1


def test_base_commits_are_not_the_prs(tmp_path):
    """Only base..head is judged: an offending commit already on the base is
    not this PR's."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"old\n\n{SESSION}\n", author=OWNER)
    repo.base = repo.head()
    repo.commit(f"new\n\n{SESSION}\n", author=AGENT)
    assert _run(repo) == 0


# ── fails closed ─────────────────────────────────────────────────────────────


def test_unresolvable_commit_fails_closed(tmp_path, capsys):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=AGENT)
    assert _run(repo, fail="repos/") == 2
    assert "failing closed" in capsys.readouterr().out


def test_unreadable_profile_with_an_unjudged_signoff_fails_closed(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\nSigned-off-by: Someone <s@example.org>\n{SESSION}\n", author=AGENT)
    assert _run(repo, fail="users/") == 2


def test_unreadable_profile_without_a_signoff_still_judges_by_login(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=OWNER)
    assert _run(repo, fail="users/") == 1


def test_unreadable_profile_cannot_pass_an_unresolved_author_unjudged(tmp_path):
    """codex 2774b7ac6e9626fe: with no profile there is no owner name, so an
    owner-NAMED author on an email no login resolves, without a sign-off,
    used to pass. It now fails closed rather than unjudged."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=(OWNER[0], "unlinked@example.org"))
    assert _run(repo, fail="users/") == 2
    assert _run(repo) == 1  # with the profile, the name check catches it


def test_a_readable_owner_profile_does_not_mask_an_unread_one(tmp_path):
    """codex a834056337675e84: one owner's profile supplied a name, so the
    unread second owner's name was treated as known and an author carrying
    it on an unlinked email passed unjudged."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=("Second Owner", "unlinked@example.org"))
    assert _run(repo, "--owner-login", "second-owner", fail="users/second-owner") == 2


def test_github_synthesised_commit_is_not_unjudged_by_an_unread_profile(tmp_path):
    """codex bd60ea81f11500a9: findings() exempts a GitHub-synthesised commit's
    author and committer, so an unresolved author on one must not fail the
    unread-profile check that exists only for what findings() judges."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"squash\n\n{SESSION}\n", author=AGENT, committer=GITHUB)
    assert _run(repo, fail="users/") == 0
    assert _run(repo, fail="users/", forged=True) == 2  # a forged GitHub committer is still judged


def test_bad_range_fails_closed(tmp_path):
    repo = _Repo(tmp_path / "r")
    args = ["--repo", str(repo.path), "--base", "0" * 40, "--head", repo.head(),
            "--github-repo", "nousergon/example", "--owner-login", OWNER_LOGIN]
    assert g.main(args, get=_getter(repo)) == 2


# ── the event file ───────────────────────────────────────────────────────────


def _event(tmp_path: Path, repo: _Repo, user_type: str = "User") -> Path:
    ev = {
        "pull_request": {
            "base": {"sha": repo.base},
            "head": {"sha": repo.head()},
            "user": {"login": OWNER_LOGIN, "id": OWNER_ID, "type": user_type},
        },
        "repository": {"full_name": "nousergon/example"},
    }
    p = tmp_path / "event.json"
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_event_opener_is_the_owner(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=OWNER)
    ev = _event(tmp_path, repo)
    assert g.main(["--repo", str(repo.path), "--event", str(ev), "--owner-login", ""], get=_getter(repo)) == 1


def test_bot_opened_pr_names_no_owner_unless_passed(tmp_path):
    repo = _Repo(tmp_path / "r")
    repo.commit(f"fix\n\n{SESSION}\n", author=OWNER)
    ev = _event(tmp_path, repo, user_type="Bot")
    assert g.main(["--repo", str(repo.path), "--event", str(ev)], get=_getter(repo)) == 0
    assert g.main(["--repo", str(repo.path), "--event", str(ev), "--owner-login", f"x, {OWNER_LOGIN}"],
                  get=_getter(repo)) == 1


def test_non_pull_request_event_passes(tmp_path):
    p = tmp_path / "push.json"
    p.write_text(json.dumps({"ref": "refs/heads/main"}), encoding="utf-8")
    assert g.main(["--event", str(p)], get=lambda path: pytest.fail("no API call expected")) == 0


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_signoffs_parse_with_and_without_email():
    msg = "x\n\nSigned-off-by: A B <a@b.c>\nsigned-off-by:   Solo  \n"
    assert g.signoffs(msg) == [("A B", "a@b.c"), ("Solo", "")]


def test_session_trailer_must_start_a_line():
    assert g.is_agent_commit(f"x\n\n{SESSION}")
    assert not g.is_agent_commit("x mentions Claude-Session: inline only")
    assert not g.is_agent_commit("Claude-Session:")


# ── the workflow ─────────────────────────────────────────────────────────────


def test_workflow_runs_the_script_on_pull_requests_read_only():
    wf = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert "pull_request" in on and "workflow_call" in on
    assert wf["permissions"] == {"contents": "read"}
    steps = wf["jobs"]["guard"]["steps"]
    assert steps[0]["with"]["fetch-depth"] == 0
    scan = steps[-1]["run"]
    assert "scripts/agent_commit_identity_guard.py" in scan and '--event "$GITHUB_EVENT_PATH"' in scan
    assert steps[-1]["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_a_forged_github_committer_is_not_exempt(tmp_path, capsys):
    """Codex review 2026-11-03: GIT_COMMITTER_EMAIL=noreply@github.com on a
    local commit resolves to web-flow; without GitHub's valid signature it is
    judged like any other agent commit."""
    repo = _Repo(tmp_path / "r")
    repo.commit(f"forged\n\n{SESSION}\n", author=OWNER, committer=GITHUB)
    assert _run(repo, forged=True) == 1
    assert "AUTHORED as the owner" in capsys.readouterr().out


def test_is_github_synthesised_needs_the_signature():
    base = {"sha": "a" * 40, "author_name": "x", "author_email": "x@example.org",
            "committer_name": "GitHub", "committer_email": "noreply@github.com", "message": ""}
    assert not g.is_github_synthesised(g.Commit(**base, committer_login="web-flow"))
    assert g.is_github_synthesised(g.Commit(**base, committer_login="web-flow", signature_valid=True))
