#!/usr/bin/env python3
"""Guard against direct OpenRouter linkage (alpha-engine-config-I6564, epic I6367).

**Why this exists.** Brian's 2026-08-03 ruling (I6367): "I NO LONGER WANT ANY
AGENT DIRECTLY LINKED TO OPENROUTER." A call site is directly linked when it
holds its own ``OPENROUTER_API_KEY`` and addresses ``openrouter.ai`` itself,
rather than reaching OpenRouter (if at all) as a router-managed fallback
member behind ``krepis``. Without a guard, a future PR can silently
reintroduce that linkage and nobody notices until the next manual sweep.

**Three patterns**, matched line-by-line over the caller repo's tracked
code/config files (docs and markdown are excluded by default — this fleet's
policy prose discusses OpenRouter constantly, and that discussion is not
linkage):

  1. ``openrouter.ai`` as a literal (a base URL)
  2. ``OPENROUTER_API_KEY`` as a literal (an env-var name being read)
  3. ``provider: openrouter`` / ``provider = "openrouter"`` style config literals

**Baseline, not a blank ban.** Measured 2026-08-19 across the fleet: the
literal ``openrouter.ai`` / ``OPENROUTER_API_KEY`` strings already appear
dozens of times in ALREADY-LEGITIMATE places — the router's own model
registry declaring OpenRouter as a fallback member's ``upstream_host``, the
egress proxy's fallback-route bootstrap args, guard tests asserting the
string's ABSENCE, and code comments describing the bug this guard exists to
catch. A guard that reds every one of those on rollout day trains everyone to
ignore it. So, mirroring this fleet's own ``gitleaks-scan.yml`` baseline
convention: an ``.openrouter-allowlist.yaml`` at the repo root pre-clears
known matches. A NEW match with no allowlist entry fails. An allowlist entry
whose ``expires`` date has passed fails LOUDLY (the match must be re-justified
or removed, never silently re-grandfathered). An allowlist entry that no
longer matches anything is also a failure — a stale entry is undetected drift
in the other direction, hiding that a linkage was actually removed and letting
the allowance quietly widen unnoticed.

Router-owned config is not exempted by path — it is allowlisted like anything
else, with a reason and an expiry, so a change to it is still visible.

Exit codes: ``0`` clean, ``1`` findings, ``2`` could not complete the check.

Usage::

    python3 scripts/openrouter_guard.py --repo /path/to/caller/checkout
    python3 scripts/openrouter_guard.py --repo . --allowlist .openrouter-allowlist.yaml
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

PATTERN_BASE_URL = "base_url"
PATTERN_ENV_KEY = "env_key"
PATTERN_PROVIDER_LITERAL = "provider_literal"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (PATTERN_BASE_URL, re.compile(r"openrouter\.ai", re.IGNORECASE)),
    (PATTERN_ENV_KEY, re.compile(r"OPENROUTER_API_KEY")),
    (
        PATTERN_PROVIDER_LITERAL,
        re.compile(r"""provider["']?\s*[:=]\s*["']?openrouter["']?\b""", re.IGNORECASE),
    ),
)
ALL_PATTERN_CLASSES = frozenset(p for p, _ in _PATTERNS)

# Code/config extensions scanned by default. Deliberately excludes markdown,
# rst and plain text: this fleet's own policy library and issue trackers
# discuss OpenRouter as a *topic* constantly (the epic ruling itself, this
# script's own docstring elsewhere) and that prose is not linkage. A repo that
# wants docs covered too can pass --include-docs.
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
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern_class, regex in _PATTERNS:
                if regex.search(line):
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
            f"No unexpected direct-OpenRouter references. "
            f"{total_matches} total match(es), {report.covered} allowlisted."
        )
        return 0

    for m in sorted(report.unallowlisted, key=lambda m: (m.path, m.line)):
        print(
            f"::error file={m.path},line={m.line}::"
            f"unallowlisted OpenRouter reference ({m.pattern_class}): {m.text}"
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
        help="path to allowlist YAML (default: <repo>/.openrouter-allowlist.yaml)",
    )
    ap.add_argument(
        "--include-docs", action="store_true",
        help="also scan markdown/rst/txt files (off by default)",
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    allowlist_path = (
        Path(args.allowlist) if args.allowlist else repo / ".openrouter-allowlist.yaml"
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
        print(f"::error::could not complete openrouter guard scan: {exc}")
        return 2

    today = _dt.date.today()
    report = evaluate(matches, allowlist, today)
    return render(report, len(matches))


if __name__ == "__main__":
    raise SystemExit(main())
