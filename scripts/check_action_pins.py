#!/usr/bin/env python3
"""Guard every ``uses:`` reference in ``.github/workflows/`` against a real commit.

**Why this exists.** A ``uses:`` ref that names a git object of the *wrong type*
is not a typo GitHub can report. The workflow fails to resolve the reference and
dies as **startup_failure: zero jobs, zero logs, no annotation naming the line**
— the same silent class ``check_workflow_shape.py`` covers for malformed YAML,
reached by a different route. That file says so itself: *"Deliberately structural
only. It does not validate action versions."* This is the missing half.

Hit live on ``crucible-predictor`` PR #433 (2026-07-31): **all six** cross-repo
``uses: nousergon/...`` refs were unresolvable and four required checks were
dead, with nothing in CI able to say why.

The SHAs were not invented. They were **real, addressable objects in the correct
repository, of the wrong type** — which is exactly why review, and a first
automated diagnosis, both passed over them:

* five were the **blob** SHA of the target ``.yml`` file, the value returned by
  ``GET /repos/{owner}/{repo}/contents/{path}`` in its ``.sha`` field. That
  endpoint describes the *file*, never the commit;
* one was the **annotated tag object** SHA of ``v1`` (``c266a50c``) rather than
  the commit it dereferences to (``6c4bdb83``).

So a finding here does not stop at "does not resolve". Whenever the ref is a
blob or a tag object this names which, and — for a tag — the commit to use
instead. The correct sources are ``GET /repos/{o}/{r}/commits/{ref}`` -> ``.sha``
and, to dereference an annotated tag, ``GET /repos/{o}/{r}/git/tags/{sha}`` ->
``.object.sha``.

Checks, per ``uses:`` value (local ``./`` and ``docker://`` refs are skipped):

1. carries an ``@ref`` at all;
2. that ref is a 40-hex SHA -- the fleet pins actions by commit, never by tag
   or branch, so a mutable ref is a finding in its own right; a short or long
   hexadecimal ref is named as such rather than as a branch;
3. that SHA resolves to a **commit** in the named repository.

Network failure is a hard error, never a pass. A guard that treats an
unreachable API as "no findings" reports green precisely when it has checked
nothing.

Invisibility is reported as invisibility. ``GITHUB_TOKEN`` reads only the repo
running the workflow, so a ``uses:`` into a *different private* repo 404s at
every object endpoint — byte-identical to a deleted SHA. That case names the
access problem instead, because the fix is a token, not a repin.

**The logic lives in ``nousergon_lib.action_pins``, not here.** This file is the
CLI: it walks workflow files, owns the HTTP transport, and words each finding.
Parsing ``uses:`` values, the 40-hex SHA rule, and the commit/blob/tag/unreadable
classification are the shared module's, so a second guard asks the same
question and gets the same answer.

Exit codes: ``0`` clean, ``1`` findings, ``2`` could not complete the check.

Usage::

    python3 scripts/check_action_pins.py
    python3 scripts/check_action_pins.py --dir path/to/workflows
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import yaml

# Self-bootstrap, same reason as validate_merge_group_required_checks.py: this
# file is executed from a CHECKOUT of nousergon-lib (the reusable workflow clones
# it beside the caller's repo), not from an installed package, and the script
# ref floats while each consumer pins the workflow skeleton. Resolving the
# sibling `src/` here means no consumer has to bump a pin for the import to work.
if __package__ is None:  # executed as a file, which is the only way this runs
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nousergon_lib.action_pins import (  # noqa: E402 -- bootstrap must precede
    ObjectKind,
    RefShape,
    ResolutionError,
    UsesKind,
    classify_object,
    parse_uses,
    ref_shape,
)

_API = "https://api.github.com"


# ── GitHub reads ─────────────────────────────────────────────────────────────


def _get(path: str, token: str | None) -> tuple[int, Any]:
    """GET *path* off the API. Returns (status, decoded-body-or-None).

    404 and 422 are answers, not faults: they are how the API says "no such
    object of that type". Everything else raises, because a guard that cannot
    reach GitHub has not checked anything and must not say it has.
    """
    req = urllib.request.Request(f"{_API}/{path}")  # noqa: S310 -- literal https host
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- as above
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            return exc.code, None
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ResolutionError(f"GET /{path} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ResolutionError(f"GET /{path} -> {exc.reason}") from exc


def classify_ref(owner_repo: str, sha: str, token: str | None = None) -> str | None:
    """Return ``None`` when *sha* is a commit in *owner_repo*, else why it is not.

    Resolution is ``nousergon_lib.action_pins.classify_object`` over this
    file's transport; what is left here is the wording of the finding.
    """
    verdict = classify_object(lambda path: _get(path, token), owner_repo, sha)
    if verdict.kind is ObjectKind.COMMIT:
        return None
    if verdict.detail == "tag_object":
        return (
            f"is the ANNOTATED TAG OBJECT for `{verdict.tag_name}`, not a commit. "
            f"GitHub cannot resolve a tag object in `uses:` -- dereference it: use {verdict.target_sha}"
        )
    if verdict.detail == "blob":
        return (
            "is a BLOB SHA (the file object), not a commit. This is what "
            "`GET /repos/{owner}/{repo}/contents/{path}` returns in `.sha` -- "
            "read the commit from `GET /repos/{owner}/{repo}/commits/{ref}` instead"
        )
    if verdict.kind is ObjectKind.UNREADABLE:
        # GITHUB_TOKEN is scoped to the repo running the workflow, so a `uses:`
        # into a DIFFERENT private repo reads exactly like a deleted object.
        return (
            f"could not be checked: `{owner_repo}` is not visible to this token "
            f"(private to another repo, renamed, or deleted). GITHUB_TOKEN only "
            f"reads the repo running the workflow -- a cross-repo private `uses:` "
            f"needs a token with access to both"
        )
    return f"does not exist in `{owner_repo}` as any reachable object"


# ── workflow walking ─────────────────────────────────────────────────────────


def collect_uses(doc: Any) -> list[str]:
    """Every ``uses:`` value in a parsed workflow, job-level and step-level."""
    found: list[str] = []
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    if not isinstance(jobs, dict):
        return found
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        if isinstance(job.get("uses"), str):
            found.append(job["uses"])
        steps = job.get("steps")
        if isinstance(steps, list):
            found.extend(s["uses"] for s in steps if isinstance(s, dict) and isinstance(s.get("uses"), str))
    return found


def line_of(text: str, value: str) -> int:
    """1-indexed line where *value* appears, so the finding lands on the diff."""
    for i, line in enumerate(text.splitlines(), 1):
        if value in line:
            return i
    return 1


def check_uses(value: str, resolver: Callable[[str, str], str | None]) -> str | None:
    """Return a problem string for one ``uses:`` value, or ``None`` when it is fine."""
    parsed = parse_uses(value)
    if parsed.kind is UsesKind.SKIPPED:
        return None  # local action or container image -- no cross-repo ref to resolve
    if parsed.kind is UsesKind.UNPINNED:
        return "is not pinned -- `uses:` must carry `@<40-hex commit SHA>`"
    if parsed.kind is UsesKind.UNPARSEABLE:
        return "is not a parseable `owner/repo[/path]@ref` reference"

    ref = parsed.ref
    shape = ref_shape(ref)
    if shape is RefShape.SHORT_HEX:
        return (
            f"is pinned to `{ref}`, a {len(ref)}-character hexadecimal ref -- an abbreviated SHA is "
            f"mutable (it can become ambiguous, and GitHub does not resolve it in `uses:`). Pin to the "
            f"full 40-hex commit SHA (keep the tag in a trailing comment)"
        )
    if shape is RefShape.LONG_HEX:
        return (
            f"is pinned to `{ref}`, a {len(ref)}-character hexadecimal ref, longer than a 40-hex "
            f"commit SHA. Pin to the exact 40-hex commit SHA (keep the tag in a trailing comment)"
        )
    if shape is not RefShape.SHA:
        return (
            f"is pinned to `{ref}`, a mutable ref. Pin to the 40-hex commit SHA "
            f"it currently resolves to (keep the tag in a trailing comment)"
        )

    problem = resolver(parsed.owner_repo, ref)
    return None if problem is None else f"@{ref} {problem}"


def check_dir(
    workflow_dir: Path,
    resolver: Callable[[str, str], str | None],
) -> list[tuple[Path, int, str, str]]:
    """Findings as (path, line, uses-value, problem), in file then line order."""
    findings: list[tuple[Path, int, str, str]] = []
    seen: dict[str, str | None] = {}  # one API round trip per distinct ref

    for path in sorted(workflow_dir.glob("*.y*ml")):
        text = path.read_text()
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError:
            # check_workflow_shape.py owns unparseable YAML and reports it with
            # the right message. Staying silent here avoids two guards blaming
            # the same line for different reasons.
            continue
        for value in collect_uses(doc):
            if value not in seen:
                seen[value] = check_uses(value, resolver)
            problem = seen[value]
            if problem:
                findings.append((path, line_of(text, value), value, problem))

    return findings


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=".github/workflows", help="workflow directory to check")
    args = parser.parse_args(argv)

    workflow_dir = Path(args.dir)
    if not workflow_dir.is_dir():
        print(f"::error::no workflow directory at {workflow_dir}")
        return 2

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # Unauthenticated is 60 requests/hour by IP -- enough to look like it
        # works locally and to exhaust itself in CI. Say so rather than emit a
        # green run built on a handful of successful calls.
        print("::warning::no GH_TOKEN/GITHUB_TOKEN -- API reads are rate-limited to 60/hour")

    try:
        findings = check_dir(workflow_dir, lambda repo, sha: classify_ref(repo, sha, token))
    except ResolutionError as exc:
        print(f"::error::could not verify action pins, so none are verified: {exc}")
        return 2

    if not findings:
        print(f"All `uses:` references in {workflow_dir} resolve to commits.")
        return 0

    for path, line, value, problem in findings:
        print(f"::error file={path},line={line}::`{value.split('@')[0]}` {problem}")
    print(f"\n{len(findings)} unresolvable or unpinned `uses:` reference(s).", file=sys.stderr)
    print(
        "An unresolvable ref does not fail loudly: the workflow never starts, posts no check, and writes no logs.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
