#!/usr/bin/env python3
"""Guard against direct OpenRouter linkage (alpha-engine-config-I6564, epic I6367).

**Why this exists.** Brian's 2026-08-03 ruling (I6367): "I NO LONGER WANT ANY
AGENT DIRECTLY LINKED TO OPENROUTER." A call site is directly linked when it
holds its own ``OPENROUTER_API_KEY`` and addresses ``openrouter.ai`` itself,
rather than reaching OpenRouter (if at all) as a router-managed fallback
member behind ``krepis``. Without a guard, a future PR can silently
reintroduce that linkage and nobody notices until the next manual sweep.

**Five patterns**, matched line-by-line over the caller repo's tracked
code/config files (docs and markdown are excluded by default — this fleet's
policy prose discusses OpenRouter constantly, and that discussion is not
linkage):

  1. ``openrouter.ai`` as a literal (a base URL)
  2. ``OPENROUTER_API_KEY`` as a literal (an env-var name being read)
  3. ``provider: openrouter`` / ``provider = "openrouter"`` style config
     literals — an ASSIGNMENT
  4. ``openrouter_api_key`` (any case OTHER than the all-caps literal
     pattern 2 already covers) as an attribute/variable name being read
  5. a runtime EQUALITY comparison against the literal ``"openrouter"`` /
     ``'openrouter'`` (``==`` or ``!=``) — a DIFFERENT token shape than
     pattern 3's assignment, and the one that let
     ``vires/api/services/coach/agent.py``'s
     ``spec.provider == "openrouter"`` ship undetected (alpha-engine-config#
     9092): the call site read the lowercase attribute
     ``settings.openrouter_api_key`` (missed by pattern 2's case-sensitive
     env-var literal) and compared it at runtime rather than assigning it
     (missed by pattern 3's assignment-shaped regex). Patterns 4 and 5 close
     that blind spot — not by loosening 2 or 3 case-insensitively (that would
     also start matching every ``OPENROUTER_API_KEY`` env-var read case
     variance the fleet already allowlists under pattern 2), but by adding
     the two token shapes that were actually missing.

**Pattern 5 fires only on a comparison that GUARDS A CONSTRUCTION
(alpha-engine-config#9111).** The regex alone matched any equality against the
string, which is data handling in 37 of the 38 places the fleet does it — a
registry row, a config value, an expense-table cell, a routing decision the
router itself already made. It reddened seven repos' PRs the evening it
shipped. What is banned is a call site SHAPED AROUND OpenRouter, so the finding
is now decided by ``_linkage_comparison_lines()``: the comparison must be the
test of a branch (not a value being asserted, filtered or assigned) AND the
branch it guards must name a credential, base URL, headers, HTTP client or
model-spec constructor in its CODE (never in a string or a comment). Python
only — every measured instance is Python and there is no honest AST for the
rest. Pattern 5 also stays test-path-exempt, below. An ``==``/``!=`` comparison
against the literal ``"openrouter"`` is the exact shape a test asserts a
value equals a fixture's chosen string — e.g. a table-row assertion like
``row["Provider"] == "openrouter"`` in a display test, which constructs no
outbound linkage at all. Measured across the fleet the morning pattern 5 shipped:
of ~39 matches, all but one (``vires/api/services/coach/agent.py:373``, a
PRODUCTION call site — the one this pattern exists to catch) were either the
fleet's own guard/registry source naming the literal defensively, or a test
asserting equality against a fixture value. Patterns 1-4 stay test-covered —
an ``openrouter.ai`` URL, an ``OPENROUTER_API_KEY``/``openrouter_api_key``
literal, or a ``provider: openrouter`` assignment IN a test file can still
construct a real call if that test is ever run for real (e.g. an
integration test hitting the network), so only pattern 5's specific
comparison shape is exempted, not the whole file class.

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
import ast
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
PATTERN_ATTR_KEY = "attr_key"
PATTERN_PROVIDER_COMPARISON = "provider_comparison"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (PATTERN_BASE_URL, re.compile(r"openrouter\.ai", re.IGNORECASE)),
    (PATTERN_ENV_KEY, re.compile(r"OPENROUTER_API_KEY")),
    (
        PATTERN_PROVIDER_LITERAL,
        re.compile(r"""provider["']?\s*[:=]\s*["']?openrouter["']?\b""", re.IGNORECASE),
    ),
    # Case-insensitive attribute/variable name — e.g. `settings.openrouter_api_key`,
    # `self.openrouter_api_key`. Deliberately NOT the same regex as PATTERN_ENV_KEY
    # widened with re.IGNORECASE: that would double-flag every already-allowlisted
    # all-caps `OPENROUTER_API_KEY` env-var read under a second pattern class. The
    # exact all-caps literal is excluded here in `scan()` and left to PATTERN_ENV_KEY.
    (PATTERN_ATTR_KEY, re.compile(r"openrouter_api_key", re.IGNORECASE)),
    # A runtime equality/inequality comparison against the literal, e.g.
    # `spec.provider == "openrouter"` / `route_name != 'openrouter'` — a different
    # token shape than PATTERN_PROVIDER_LITERAL's assignment (`provider: openrouter`,
    # `provider="openrouter"`).
    #
    # THE REGEX ALONE OVER-MATCHES AND MUST NOT BE USED ALONE (I9111). Comparing a
    # provider NAME THAT ARRIVED AS DATA — a registry row, a config value, a
    # dataframe cell, a function parameter — is not linkage; it is data handling,
    # and it is what 37 of the 38 fleet-wide instances of this shape do. A match is
    # promoted to a finding only by `_linkage_comparison_lines()` below, which
    # decides on what the comparison GUARDS, not on the literal. See the block
    # comment there for the discriminator.
    (
        PATTERN_PROVIDER_COMPARISON,
        re.compile(r"""[!=]=\s*["']openrouter["']"""),
    ),
)
ALL_PATTERN_CLASSES = frozenset(p for p, _ in _PATTERNS)

# PATTERN_ATTR_KEY's regex is deliberately case-insensitive and therefore matches
# the all-caps literal too; a line matching this exact case is already the
# PATTERN_ENV_KEY finding and must not be flagged a second time under a second
# pattern class (which would need a redundant allowlist entry for every existing
# env_key entry in the fleet).
_ATTR_KEY_ENV_LITERAL = "OPENROUTER_API_KEY"

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


# Test-path shapes, fleet-wide (alpha-engine-config-I9111). Kept intentionally
# broad — false negatives here (a test NOT recognized as one) are the safe
# failure mode, since PATTERN_PROVIDER_COMPARISON then just stays strict on
# that file. Covers: a `test_`-prefixed or `_test`-suffixed Python module, a
# `.test.`/`.spec.` JS/TS module, or any path with a `test`/`tests` directory
# segment (`tests/test_expenses_page.py`, `src/__tests__/foo.test.ts`).
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]+\.py$|_test\.py$|\.(test|spec)\.[jt]sx?$"
)


def _is_test_path(rel: str) -> bool:
    return bool(_TEST_PATH_RE.search(rel))


# ── the scanner never scans itself (alpha-engine-config-I9111) ───────────────
#
# A guard that flags its own source is a defect in its own right: every pattern
# this module detects is necessarily WRITTEN OUT in this module and in the tests
# that exercise it, so self-scanning generates one allowlist entry per pattern
# class forever and each new pattern silently adds two more. Handled as a
# PROPERTY rather than as allowlist lines: the scanner skips its own file and any
# file that loads it. Both signals are structural — a production call site cannot
# acquire them without literally importing the guard, which is visible in a diff.
_SCANNER_SELF_RE = re.compile(
    r"""(?:^|\s)(?:from|import)\s+openrouter_guard\b"""
    r"""|spec_from_file_location\(\s*["']openrouter_guard["']"""
    r"""|["'][^"']*openrouter_guard\.py["']""",
    re.MULTILINE,
)
_SELF_PATH = Path(__file__).resolve()


def _is_scanner_source(fp: Path, text: str) -> bool:
    """This module, or a file whose subject is this module (its tests).

    The basename test is deliberate and is what makes the behaviour identical
    whether the caller runs the checked-out copy of this script or the one
    inside the repo being scanned (the reusable workflow does the former, a
    local  in nousergon-lib the latter — path identity alone
    disagreed between those two). A file named  IS a
    guard implementation; it is not a call site.
    """
    try:
        if fp.resolve() == _SELF_PATH:
            return True
    except OSError:  # pragma: no cover - resolve() on a broken symlink
        pass
    if fp.name == _SELF_PATH.name:
        return True
    return bool(_SCANNER_SELF_RE.search(text))


# ── provider_comparison: data handling vs. linkage ───────────────────────────
#
# THE DISCRIMINATOR (alpha-engine-config-I9111). The banned thing (Brian's
# 2026-08-03 ruling, I6367; principle 8) is a call site SHAPED AROUND OpenRouter
# — one that constructs a provider-specific outbound path. An `==` against the
# string "openrouter" is not that by itself. So a comparison is a finding only
# when BOTH hold:
#
#   1. it is the TEST of a branch (`if` / `elif` / conditional expression /
#      `while`) rather than a value being computed. A comparison used as a value
#      — an assertion, a comprehension filter over rows, a dict entry, a boolean
#      assigned to a name — decides nothing about how a request is built; and
#   2. the branch it guards CONSTRUCTS: its code (identifiers only — never a
#      string literal or a comment, which is how a scan turns a file's own
#      rationale into a violation) names a credential, a base URL, request
#      headers, an HTTP client, or a model-spec constructor/mutator.
#
# Measured on the 2026-08-28 fleet population (38 instances, 7 repos): this
# promotes exactly one — `vires/api/services/coach/agent.py:373`, the linkage
# I9092 was filed about, whose guarded branch mutates the request spec via
# `replace(spec, reasoning=...)`. Every other instance is a test assertion, a
# registry/router lookup returning data, or the fleet's own defensive guard
# source. What this still catches that no other pattern does: a branch that
# builds an OpenRouter-specific request from values that are never spelled out
# as a literal on the line (a base URL or key held in a variable), which is
# precisely the dynamic, config-driven variant I9092 recorded as the guard's
# blind spot.
#
# Applied to PYTHON only. Every one of the 38 measured instances is Python, the
# `spec.provider == "openrouter"` shape the rule exists for is Python, and there
# is no honest AST for the other extensions — a line-scoped guess over YAML or
# TypeScript would reintroduce exactly the over-match this replaces. The other
# four patterns remain in force on every extension, including in tests.

_CONSTRUCTION_IDENT_RE = re.compile(
    r"base_url|api_base|api_key|apikey|auth|header|credential|token|secret"
    r"|endpoint|httpx|requests|urllib",
    re.IGNORECASE,
)
# Bare-callable constructors/mutators of a request or a model spec. Deliberately
# call-position and case-exact: `x.replace("-", "_")` is a string method, while
# `replace(spec, ...)` is `dataclasses.replace` on a resolved ModelSpec.
_CONSTRUCTION_CALLS = frozenset({
    "replace", "ModelSpec", "OpenAI", "AsyncOpenAI", "Anthropic",
    "AsyncAnthropic", "Client", "AsyncClient", "urlopen",
})

_OPENROUTER_LITERAL = "openrouter"


def _compares_openrouter(test: ast.AST) -> tuple[bool, bool]:
    """``(has_eq, has_noteq)`` for comparisons against the literal in ``test``."""
    has_eq = has_noteq = False
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        names_literal = any(
            isinstance(o, ast.Constant)
            and isinstance(o.value, str)
            and o.value == _OPENROUTER_LITERAL
            for o in operands
        )
        if not names_literal:
            continue
        for op in node.ops:
            if isinstance(op, ast.Eq):
                has_eq = True
            elif isinstance(op, ast.NotEq):
                has_noteq = True
    return has_eq, has_noteq


def _constructs(body: list[ast.AST]) -> bool:
    """Does this branch build a provider-specific outbound path?

    Identifiers only. String literals and comments are excluded on purpose:
    a message that NAMES the thing being rejected is documentation, not
    construction, and scanning prose is how a guard flags its own rationale.
    """
    for stmt in body:
        for node in ast.walk(stmt):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.keyword):
                name = node.arg
            elif isinstance(node, ast.arg):
                name = node.arg
            if name and _CONSTRUCTION_IDENT_RE.search(name):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _CONSTRUCTION_CALLS
            ):
                return True
    return False


def _linkage_comparison_lines(source: str) -> frozenset[int] | None:
    """Lines carrying a provider comparison that GUARDS a construction.

    ``None`` means the file could not be parsed — the strict fallback, in which
    every regex match is reported rather than silently dropped.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            test = node.test
            eq_body: list[ast.AST] = (
                list(node.body) if isinstance(node, ast.If) else [node.body]
            )
            noteq_body: list[ast.AST] = (
                list(node.orelse) if isinstance(node, ast.If) else [node.orelse]
            )
        elif isinstance(node, ast.While):
            test, eq_body, noteq_body = node.test, list(node.body), []
        else:
            continue

        has_eq, has_noteq = _compares_openrouter(test)
        if not (has_eq or has_noteq):
            continue
        # An `==` puts the provider-specific work in the branch body; a `!=`
        # puts it in the else (`if provider != "openrouter": return` is an
        # early-out guard, and its empty else constructs nothing).
        if not ((has_eq and _constructs(eq_body)) or (has_noteq and _constructs(noteq_body))):
            continue
        for sub in ast.walk(test):
            if isinstance(sub, ast.Compare):
                lines.add(sub.lineno)
    return frozenset(lines)


def _reportable_comparison_lines(rel: str, text: str) -> frozenset[int] | None:
    """Which lines may carry a provider_comparison FINDING in this file.

    ``None`` = no restriction (report every regex match), the strict fallback.
    """
    if _is_test_path(rel):
        # I9111: a comparison in a test asserts a fixture value; it constructs
        # no outbound linkage. The other four patterns stay test-covered — an
        # `openrouter.ai` URL or an `OPENROUTER_API_KEY` read in a test is a
        # real linkage if that test is ever exercised for real.
        return frozenset()
    if not rel.endswith(".py"):
        return frozenset()
    return _linkage_comparison_lines(text)


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
        comparison_lines = _reportable_comparison_lines(rel, text)
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern_class, regex in _PATTERNS:
                if (
                    pattern_class == PATTERN_PROVIDER_COMPARISON
                    and comparison_lines is not None
                    and lineno not in comparison_lines
                ):
                    continue  # data handling, not linkage — see the discriminator
                m = regex.search(line)
                if not m:
                    continue
                if pattern_class == PATTERN_ATTR_KEY and m.group() == _ATTR_KEY_ENV_LITERAL:
                    continue  # the exact all-caps literal is PATTERN_ENV_KEY's finding
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
