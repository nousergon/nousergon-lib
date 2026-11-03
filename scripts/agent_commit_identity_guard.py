#!/usr/bin/env python3
"""Refuse an agent-session commit that claims the owner's identity.

alpha-engine-config-I13060, prevention for nous-ergon-ops-I2019. In the cycle
10-04 nightly a worker made a commit authored AND signed off as the owner.
It was never pushed, and the only containment was a worker rule ("commit as
the agent identity"). Worker containers now default to that identity, but
nothing refused a commit that overrode it: ``git -c user.name=…``,
``--author``, or a hand-written ``Signed-off-by:`` line. An instruction to a
model is not a control; this is the control.

THE RULE. A PR commit carrying a ``Claude-Session:`` trailer was made by an
agent session. It fails when its author, its committer or any of its
``Signed-off-by:`` lines names the owner. "Names the owner" is any of:

  * GitHub resolves the author or committer email to an owner login;
  * the email is the owner's GitHub noreply address
    (``<login>@users.noreply.github.com`` or ``<id>+<login>@…``);
  * the email is one the owner's profile publishes, or one another commit in
    the same PR resolved to the owner (so a sign-off reusing it is caught);
  * the name equals the owner's profile name (case and spacing ignored).

WHO THE OWNER IS. The account the PR is opened under (when it is a human
``User``, not a bot), plus any ``--owner-login`` the caller passes. Agent PRs
in this fleet are opened under the owner's account, so the opener IS the
identity an agent must not claim. Nothing personal is hard-coded here: this
repo is public, and the identity is read from the event and the API at run
time.

ONE EXEMPTION. A commit whose committer is GitHub itself (``web-flow``) was
synthesised by GitHub on the token holder's behalf: a contents-API commit, a
squash, a merge queue entry, an "Update branch" merge. Its author is the
token holder by construction, not by choice, so its author and committer are
not judged. Its ``Signed-off-by:`` lines still are: a sign-off is text the
session wrote.

FAILS CLOSED. If GitHub cannot be asked who an agent commit resolves to, the
guard has not checked anything and exits 2 rather than pass.

Usage (CI passes the event file; the rest is for a local run):
    agent_commit_identity_guard.py --repo . --event "$GITHUB_EVENT_PATH"
    agent_commit_identity_guard.py --repo . --base <sha> --head <sha> \\
        --github-repo owner/name --owner-login <login>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_API = "https://api.github.com"
_SESSION = re.compile(r"(?im)^claude-session:[ \t]*\S")
_SIGNOFF = re.compile(r"(?im)^signed-off-by:[ \t]*(.*?)[ \t]*$")
_NAME_EMAIL = re.compile(r"^(.*?)\s*<([^<>]*)>$")
_NOREPLY = re.compile(r"^(?:(\d+)\+)?([^@+]+)@users\.noreply\.github\.com$", re.IGNORECASE)
_WEB_FLOW_EMAIL = "noreply@github.com"
_WEB_FLOW_LOGIN = "web-flow"


class VerifyError(RuntimeError):
    """The guard could not establish a fact it needs; it must not pass."""


@dataclass(frozen=True)
class Commit:
    sha: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    message: str
    author_login: str | None = None
    committer_login: str | None = None
    #: GitHub's ``commit.verification`` reads ``verified`` with reason ``valid``.
    signature_valid: bool = False


@dataclass
class Owner:
    logins: set[str] = field(default_factory=set)
    ids: set[int] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.logins)


# ── pure ─────────────────────────────────────────────────────────────────────


def _norm_name(name: str | None) -> str:
    return " ".join((name or "").split()).casefold()


def _norm_email(email: str | None) -> str:
    return (email or "").strip().lower()


def is_agent_commit(message: str) -> bool:
    """A ``Claude-Session:`` trailer marks a commit made by an agent session."""
    return bool(_SESSION.search(message or ""))


def signoffs(message: str) -> list[tuple[str, str]]:
    """Every ``Signed-off-by:`` line as (name, email); email may be ''."""
    out: list[tuple[str, str]] = []
    for value in _SIGNOFF.findall(message or ""):
        m = _NAME_EMAIL.match(value)
        out.append((m.group(1).strip(), m.group(2).strip()) if m else (value.strip(), ""))
    return out


def is_github_synthesised(c: Commit) -> bool:
    """Committed by GitHub itself on the token holder's behalf.

    The committer email is set by whoever makes the commit (a local
    ``GIT_COMMITTER_EMAIL=noreply@github.com`` resolves to ``web-flow`` too),
    so it is not proof. GitHub signs what it synthesises, and a signature
    GitHub reports as ``valid`` for a ``noreply@github.com`` committer can only
    be GitHub's own: both are required.
    """
    return (
        (c.committer_login or "").lower() == _WEB_FLOW_LOGIN
        and _norm_email(c.committer_email) == _WEB_FLOW_EMAIL
        and c.signature_valid
    )


def names_owner(name: str, email: str, login: str | None, owner: Owner) -> str | None:
    """Why (name, email, login) is the owner, or None when it is not."""
    if login and login.lower() in owner.logins:
        return f"GitHub resolves it to the owner's account `{login}`"
    e = _norm_email(email)
    m = _NOREPLY.match(e)
    if m and (m.group(2).lower() in owner.logins or (m.group(1) and int(m.group(1)) in owner.ids)):
        return "it is the owner's GitHub noreply address"
    if e and e in owner.emails:
        return "it is an email address the owner uses"
    if _norm_name(name) and _norm_name(name) in owner.names:
        return "it is the owner's name"
    return None


def learn_owner_emails(commits: list[Commit], owner: Owner) -> None:
    """An email GitHub resolved to an owner login anywhere in the PR is the
    owner's, so a sign-off elsewhere that reuses it is caught too."""
    for c in commits:
        if c.author_login and c.author_login.lower() in owner.logins:
            owner.emails.add(_norm_email(c.author_email))
        if c.committer_login and c.committer_login.lower() in owner.logins and not is_github_synthesised(c):
            owner.emails.add(_norm_email(c.committer_email))
    owner.emails.discard("")


def findings(commits: list[Commit], owner: Owner) -> list[str]:
    """One line per way an agent-session commit claims the owner. Pure."""
    out: list[str] = []
    for c in commits:
        if not is_agent_commit(c.message):
            continue
        short = c.sha[:12]
        if not is_github_synthesised(c):
            why = names_owner(c.author_name, c.author_email, c.author_login, owner)
            if why:
                out.append(f"{short}: agent-session commit AUTHORED as the owner ({why})")
            why = names_owner(c.committer_name, c.committer_email, c.committer_login, owner)
            if why:
                out.append(f"{short}: agent-session commit COMMITTED as the owner ({why})")
        for name, email in signoffs(c.message):
            why = names_owner(name, email, None, owner)
            if why:
                out.append(f"{short}: agent-session commit carries a Signed-off-by for the owner ({why})")
    return out


# ── I/O ──────────────────────────────────────────────────────────────────────

Getter = Callable[[str], Any]


def api_getter(token: str | None) -> Getter:
    def get(path: str) -> Any:
        req = urllib.request.Request(f"{_API}/{path}")  # noqa: S310 -- literal https host
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "nousergon-agent-commit-identity-guard")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- as above
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise VerifyError(f"GET /{path}: {exc}") from exc

    return get


def git_commits(repo: Path, base: str, head: str) -> list[Commit]:
    """The PR's own commits: reachable from head, not from base."""
    git = shutil.which("git")
    if git is None:
        raise VerifyError("git not found on PATH")
    fmt = "%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x1e"
    try:
        out = subprocess.run(
            [git, "-C", str(repo), "log", f"--format={fmt}", f"{base}..{head}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise VerifyError(f"git log {base}..{head}: {exc.stderr.strip()}") from exc
    commits: list[Commit] = []
    for rec in out.split("\x1e"):
        rec = rec.lstrip("\n")
        if not rec:
            continue
        sha, an, ae, cn, ce, body = rec.split("\x00", 5)
        commits.append(Commit(sha, an, ae, cn, ce, body))
    return commits


def resolve_logins(commits: list[Commit], github_repo: str, get: Getter) -> list[Commit]:
    """Ask GitHub which account each AGENT commit's author and committer
    resolve to. The PR's other commits are looked up only when an agent
    commit carries a sign-off, because then the owner emails they reveal
    (``learn_owner_emails``) are what judges it; otherwise they cost a call
    and decide nothing."""
    every = any(signoffs(c.message) for c in commits if is_agent_commit(c.message))
    out: list[Commit] = []
    for c in commits:
        if not (every or is_agent_commit(c.message)):
            out.append(c)
            continue
        data = get(f"repos/{github_repo}/commits/{c.sha}") or {}
        author = (data.get("author") or {}).get("login")
        committer = (data.get("committer") or {}).get("login")
        verification = (data.get("commit") or {}).get("verification") or {}
        signed = verification.get("verified") is True and verification.get("reason") == "valid"
        out.append(
            Commit(
                c.sha, c.author_name, c.author_email, c.committer_name, c.committer_email, c.message,
                author, committer, signed,
            )
        )
    return out


def enrich_owner(owner: Owner, get: Getter) -> list[str]:
    """Add each owner login's id, profile name and public email. Returns the
    logins whose profile could not be read (the caller decides if that
    matters)."""
    unread: list[str] = []
    for login in sorted(owner.logins):
        try:
            prof = get(f"users/{login}") or {}
        except VerifyError:
            unread.append(login)
            continue
        if isinstance(prof.get("id"), int):
            owner.ids.add(prof["id"])
        if _norm_name(prof.get("name")):
            owner.names.add(_norm_name(prof.get("name")))
        if _norm_email(prof.get("email")):
            owner.emails.add(_norm_email(prof.get("email")))
    return unread


def from_event(path: Path) -> tuple[str | None, str | None, str | None, str | None, int | None]:
    """(base, head, repo, opener login, opener id) for a pull_request event;
    all None for any other event. The opener counts only when it is a human
    ``User``: a bot-opened PR names no owner by itself."""
    ev = json.loads(path.read_text(encoding="utf-8"))
    pr = ev.get("pull_request")
    if not pr:
        return None, None, None, None, None
    user = pr.get("user") or {}
    login = user.get("login") if user.get("type") == "User" else None
    uid = user.get("id") if login and isinstance(user.get("id"), int) else None
    return (
        (pr.get("base") or {}).get("sha"),
        (pr.get("head") or {}).get("sha"),
        (ev.get("repository") or {}).get("full_name"),
        login,
        uid,
    )


def run(
    repo: Path,
    base: str,
    head: str,
    github_repo: str,
    owner: Owner,
    get: Getter,
) -> int:
    commits = git_commits(repo, base, head)
    agent = [c for c in commits if is_agent_commit(c.message)]
    print(f"{len(commits)} PR commit(s), {len(agent)} carrying a Claude-Session trailer; owner login(s): {len(owner.logins)}")
    if not agent:
        print("No agent-session commit to judge — check passes.")
        return 0
    commits = resolve_logins(commits, github_repo, get)
    unread = enrich_owner(owner, get)
    learn_owner_emails(commits, owner)
    found = findings(commits, owner)
    if unread and not found and any(signoffs(c.message) for c in agent):
        raise VerifyError(
            f"could not read the owner profile for {', '.join(unread)}, and an agent-session "
            "commit carries a Signed-off-by line that only the profile can judge"
        )
    for line in found:
        print(f"::error::{line}")
    if found:
        print(
            "::error::An agent session must commit as the agent identity and must never sign off as the owner "
            "(alpha-engine-config-I13060, nous-ergon-ops-I2019). Rewrite the commit with the agent identity "
            "(`git commit --amend --reset-author` under the agent's git config) and drop the owner's sign-off."
        )
        return 1
    print("No agent-session commit claims the owner — check passes.")
    return 0


def main(argv: list[str] | None = None, get: Getter | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--event", type=Path, help="GITHUB_EVENT_PATH; supplies base, head, repo and opener")
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--github-repo")
    ap.add_argument("--owner-login", action="append", default=[], help="repeatable; comma lists accepted")
    args = ap.parse_args(argv)

    base, head, github_repo = args.base, args.head, args.github_repo
    owner = Owner()
    if args.event:
        e_base, e_head, e_repo, e_login, e_id = from_event(args.event)
        if e_base is None:
            print("Not a pull_request event: GitHub synthesises these commits as the opener — nothing to judge.")
            return 0
        base, head, github_repo = base or e_base, head or e_head, github_repo or e_repo
        if e_login:
            owner.logins.add(e_login.lower())
        if e_id is not None:
            owner.ids.add(e_id)
    for raw in args.owner_login:
        owner.logins.update(x.strip().lower() for x in re.split(r"[,\s]+", raw) if x.strip())
    if not (base and head and github_repo):
        print("::error::need --event, or --base, --head and --github-repo", file=sys.stderr)
        return 2
    if not owner:
        print("::notice::No owner login (bot-opened PR and no --owner-login): nothing to judge against.")
        return 0
    getter = get or api_getter(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    try:
        return run(args.repo, base, head, github_repo, owner, getter)
    except VerifyError as exc:
        print(f"::error::agent-commit identity guard could not verify, failing closed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
