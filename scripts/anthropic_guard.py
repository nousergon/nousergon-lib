#!/usr/bin/env python3
"""Guard against direct Anthropic-API linkage (alpha-engine-config-I9263).

**Why this exists.** Brian's 2026-08-29 ruling: "I will not fund the
anthropic account, at this point we shouldn't be using the anthropic api at
all." The direct Anthropic API is RETIRED fleet-wide. A call site is
directly linked when it holds its own ``ANTHROPIC_API_KEY`` and addresses
``api.anthropic.com`` (or constructs the ``anthropic`` SDK client) itself,
rather than reaching a model — Anthropic's included — as a router-managed
member behind ``krepis``. Without a guard, a future PR can silently
reintroduce that linkage and nobody notices until the next manual sweep.

**This mirrors ``scripts/openrouter_guard.py`` structurally** (same CLI
shape, same allowlist convention, same exit semantics, same reporting) —
alpha-engine-config-I6564's OpenRouter direct-linkage guard, built for
Brian's 2026-08-03 "no agent directly linked to OpenRouter" ruling (I6367).
Do not invent a parallel design; extend this one the way that guard was
extended (I9092, I9111) if a blind spot is ever found here.

**Four patterns**, matched line-by-line over the caller repo's tracked
code/config files (docs and markdown are excluded by default, same rationale
as the OpenRouter guard: this fleet's policy prose discusses Anthropic
constantly as a topic, and that discussion is not linkage):

  1. ``sdk_client`` — the ``anthropic`` Python SDK client being constructed
     or imported directly: ``anthropic.Anthropic(``, ``anthropic.AsyncAnthropic(``,
     or ``from anthropic import Anthropic``.
  2. ``env_key`` — ``ANTHROPIC_API_KEY`` as a literal (an env-var name being
     read).
  3. ``base_url`` — ``api.anthropic.com`` as a literal (a base URL).
  4. ``base_url_env`` — ``ANTHROPIC_BASE_URL`` as a literal (an env-var name
     naming the base URL).

**Two exclusions, not by path filter but documented here and in the caller
workflow's header** (same convention as the OpenRouter guard's krepis
exclusion):

  - ``krepis`` is EXCLUDED ENTIRELY from this guard fleet-wide — it is the
    router's own repo and holds the one legitimate declared provider
    adapter (``krepis/src/krepis/llm.py`` constructs ``anthropic.Anthropic``
    inside the adapter boundary; principle 8 — substitutability — permits
    exactly this: one adapter behind the router, not a call site shaped
    around the provider). krepis is not a caller of this workflow.
  - ``morning-signal`` carries a POLICY CARVE-OUT
    (``llm-provider-model-policy.md`` §4, "Morning-signal Anthropic
    fallback"). It is **not** excluded from the guard — it still runs it —
    it needs an ``.anthropic-allowlist.yaml`` entry citing that policy
    section and I9263, the same as any other pre-existing match.

**Baseline, not a blank ban.** Mirroring the OpenRouter guard's own
``.openrouter-allowlist.yaml`` convention: an ``.anthropic-allowlist.yaml``
at the repo root pre-clears known matches. A NEW match with no allowlist
entry fails. An allowlist entry whose ``expires`` date has passed fails
LOUDLY (the match must be re-justified or removed, never silently
re-grandfathered). An allowlist entry that no longer matches anything is
also a failure — a stale entry is undetected drift in the other direction.

Router-owned config (e.g. the egress proxy's route contract naming
``api.anthropic.com`` as an upstream host it forwards to) is not exempted by
path — it is allowlisted like anything else, with a reason and an expiry.

Exit codes: ``0`` clean, ``1`` findings, ``2`` could not complete the check.

Usage::

    python3 scripts/anthropic_guard.py --repo /path/to/caller/checkout
    python3 scripts/anthropic_guard.py --repo . --allowlist .anthropic-allowlist.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── patterns ─────────────────────────────────────────────────────────────────

PATTERN_SDK_CLIENT = "sdk_client"
PATTERN_ENV_KEY = "env_key"
PATTERN_BASE_URL = "base_url"
PATTERN_BASE_URL_ENV = "base_url_env"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        PATTERN_SDK_CLIENT,
        re.compile(
            r"anthropic\.Anthropic\(|anthropic\.AsyncAnthropic\("
            r"|from\s+anthropic\s+import\s+Anthropic\b"
        ),
    ),
    (PATTERN_ENV_KEY, re.compile(r"ANTHROPIC_API_KEY")),
    (PATTERN_BASE_URL, re.compile(r"api\.anthropic\.com", re.IGNORECASE)),
    (PATTERN_BASE_URL_ENV, re.compile(r"ANTHROPIC_BASE_URL")),
)
ALL_PATTERN_CLASSES = frozenset(p for p, _ in _PATTERNS)

# Code/config extensions scanned by default. Deliberately excludes markdown,
# rst and plain text — see module docstring. A repo that wants docs covered
# too can pass --include-docs.
DEFAULT_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".mjs", ".sh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".env",
})
DOC_EXTENSIONS = frozenset({".md", ".mdx", ".rst", ".txt"})


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    pattern_class: str
    text: str


@dataclass(frozen=True)
class AllowlistEntry:
    path: str
    pattern_class: str
    reason: str
    expires: _dt.date
    tracking: str | None
    line_index: int  # position in the source list, for stable error ordering


class GuardError(RuntimeError):
    """The check could not be completed -- not a finding, an infrastructure fault."""


# ── scanning ─────────────────────────────────────────────────────────────────


def _tracked_files(repo: Path, extensions: frozenset[str]) -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise GuardError("`git` not found on PATH")
    result = subprocess.run(
        [git, "-C", str(repo), "ls-files"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise GuardError(f"`git ls-files` failed in {repo}: {result.stderr.strip()}")
    out = []
    for rel in result.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if Path(rel).suffix.lower() not in extensions:
            continue
        out.append(repo / rel)
    return out


# ── the scanner never scans itself (mirrors openrouter_guard.py) ─────────────
#
# A guard that flags its own source is a defect in its own right: every
# pattern this module detects is necessarily WRITTEN OUT in this module and
# in the tests that exercise it, so self-scanning generates one allowlist
# entry per pattern class forever and each new pattern silently adds one
# more. Handled as a PROPERTY rather than as allowlist lines: the scanner
# skips its own file and any file that loads it. Both signals are
# structural — a production call site cannot acquire them without literally
# importing the guard, which is visible in a diff.
_SCANNER_SELF_RE = re.compile(
    r"""(?:^|\s)(?:from|import)\s+anthropic_guard\b"""
    r"""|spec_from_file_location\(\s*["']anthropic_guard["']"""
    r"""|["'][^"']*anthropic_guard\.py["']""",
    re.MULTILINE,
)
_SELF_PATH = Path(__file__).resolve()


def _is_scanner_source(fp: Path, text: str) -> bool:
    """This module, or a file whose subject is this module (its tests).

    The basename test is deliberate and is what makes the behaviour
    identical whether the caller runs the checked-out copy of this script or
    the one inside the repo being scanned (the reusable workflow does the
    former, a local pytest run inside nousergon-lib the latter — path
    identity alone would disagree between those two).
    """
    try:
        if fp.resolve() == _SELF_PATH:
            return True
    except OSError:  # pragma: no cover - resolve() on a broken symlink
        pass
    if fp.name == _SELF_PATH.name:
        return True
    return bool(_SCANNER_SELF_RE.search(text))


def scan(repo: Path, extensions: frozenset[str], skip: frozenset[str] = frozenset()) -> list[Match]:
    """Every pattern hit in every tracked, in-scope file.

    ``skip`` holds repo-relative paths to exclude outright — the allowlist
    file itself, whose ``reason`` prose legitimately names these same
    strings and would otherwise have to allowlist itself.
    """
    matches: list[Match] = []
    for fp in _tracked_files(repo, extensions):
        rel_check = str(fp.relative_to(repo))
        if rel_check in skip:
            continue
        try:
            text = fp.read_text(errors="replace")
        except OSError as exc:
            print(f"::warning::could not read {fp}: {exc}", file=sys.stderr)
            continue
        rel = str(fp.relative_to(repo))
        if _is_scanner_source(fp, text):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern_class, regex in _PATTERNS:
                m = regex.search(line)
                if not m:
                    continue
                matches.append(Match(rel, lineno, pattern_class, line.strip()))
    return matches


# ── allowlist ────────────────────────────────────────────────────────────────


def load_allowlist(path: Path) -> list[AllowlistEntry]:
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("entries") or []
    if not isinstance(entries, list):
        raise GuardError(f"{path}: 'entries' must be a list")
    out: list[AllowlistEntry] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise GuardError(f"{path}: entries[{i}] is not a mapping")
        missing = {"path", "pattern", "reason", "expires"} - e.keys()
        if missing:
            raise GuardError(f"{path}: entries[{i}] missing required keys: {sorted(missing)}")
        pattern_class = e["pattern"]
        if pattern_class not in ALL_PATTERN_CLASSES:
            raise GuardError(
                f"{path}: entries[{i}].pattern {pattern_class!r} not in "
                f"{sorted(ALL_PATTERN_CLASSES)}"
            )
        try:
            expires = _dt.date.fromisoformat(str(e["expires"]))
        except ValueError as exc:
            raise GuardError(f"{path}: entries[{i}].expires must be YYYY-MM-DD") from exc
        if not str(e["reason"]).strip():
            raise GuardError(f"{path}: entries[{i}].reason must be non-empty")
        out.append(AllowlistEntry(
            path=e["path"],
            pattern_class=pattern_class,
            reason=str(e["reason"]),
            expires=expires,
            tracking=e.get("tracking"),
            line_index=i,
        ))
    return out


# ── evaluation ───────────────────────────────────────────────────────────────


@dataclass
class Report:
    unallowlisted: list[Match]
    expired: list[AllowlistEntry]
    stale: list[AllowlistEntry]
    covered: int

    @property
    def ok(self) -> bool:
        return not (self.unallowlisted or self.expired or self.stale)


def evaluate(
    matches: list[Match],
    allowlist: list[AllowlistEntry],
    today: _dt.date,
) -> Report:
    by_key: dict[tuple[str, str], list[AllowlistEntry]] = {}
    for e in allowlist:
        by_key.setdefault((e.path, e.pattern_class), []).append(e)

    matched_keys: set[tuple[str, str]] = set()
    unallowlisted: list[Match] = []
    covered = 0

    for m in matches:
        key = (m.path, m.pattern_class)
        entries = by_key.get(key)
        if not entries:
            unallowlisted.append(m)
            continue
        matched_keys.add(key)
        covered += 1

    expired = [
        e for e in allowlist
        if e.expires < today
    ]
    stale = [
        e for e in allowlist
        if e.expires >= today and (e.path, e.pattern_class) not in matched_keys
    ]

    return Report(unallowlisted=unallowlisted, expired=expired, stale=stale, covered=covered)


# ── reporting ────────────────────────────────────────────────────────────────


def render(report: Report, total_matches: int) -> int:
    if report.ok:
        print(
            f"No unexpected direct-Anthropic-API references. "
            f"{total_matches} total match(es), {report.covered} allowlisted."
        )
        return 0

    for m in sorted(report.unallowlisted, key=lambda m: (m.path, m.line)):
        print(
            f"::error file={m.path},line={m.line}::"
            f"unallowlisted direct-Anthropic reference ({m.pattern_class}): {m.text}"
        )
    for e in sorted(report.expired, key=lambda e: (e.path, e.pattern_class)):
        tracking = f" ({e.tracking})" if e.tracking else ""
        print(
            f"::error file={e.path}::allowlist entry EXPIRED {e.expires.isoformat()} "
            f"for pattern={e.pattern_class}{tracking}: {e.reason} — re-justify with a "
            f"new expires date or remove the reference"
        )
    for e in sorted(report.stale, key=lambda e: (e.path, e.pattern_class)):
        print(
            f"::error file={e.path}::stale allowlist entry (pattern={e.pattern_class}) — "
            f"no longer matches anything in the repo. Remove the entry: an allowlist "
            f"that outlives what it covers silently widens over time"
        )

    n = len(report.unallowlisted) + len(report.expired) + len(report.stale)
    print(f"\n{n} finding(s).", file=sys.stderr)
    return 1


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=".", help="repo root to scan")
    ap.add_argument(
        "--allowlist", default=None,
        help="path to allowlist YAML (default: <repo>/.anthropic-allowlist.yaml)",
    )
    ap.add_argument(
        "--include-docs", action="store_true",
        help="also scan markdown/rst/txt files (off by default)",
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    allowlist_path = (
        Path(args.allowlist) if args.allowlist else repo / ".anthropic-allowlist.yaml"
    )
    extensions = DEFAULT_EXTENSIONS | (DOC_EXTENSIONS if args.include_docs else frozenset())

    try:
        allowlist_rel = None
        try:
            allowlist_rel = str(allowlist_path.resolve().relative_to(repo))
        except ValueError:
            pass  # allowlist lives outside repo (e.g. a test fixture) -- nothing to skip
        skip = frozenset({allowlist_rel}) if allowlist_rel else frozenset()
        matches = scan(repo, extensions, skip=skip)
        allowlist = load_allowlist(allowlist_path)
    except GuardError as exc:
        print(f"::error::could not complete anthropic guard scan: {exc}")
        return 2

    today = _dt.date.today()
    report = evaluate(matches, allowlist, today)
    return render(report, len(matches))


if __name__ == "__main__":
    raise SystemExit(main())
