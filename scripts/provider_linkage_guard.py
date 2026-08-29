#!/usr/bin/env python3
"""Guard against DIRECT PROVIDER LINKAGE for every LLM provider, not one.

**Why this exists.** Brian's 2026-08-29 ruling, verbatim: "the entire nous
ergon system should now be running through the krepis router... we should
have no other parallel setups, it should all funnel through the krepis
router." A call site is *directly linked* when it constructs a provider SDK
client, reads a provider credential, or addresses a provider hostname
ITSELF, rather than reaching that model as a router-managed member behind
``krepis``. Principle 8 (substitutability) states the same rule from the
other side: address a capability class or a registry model group through the
router -- never a model ID, a base URL, a provider name, or an SDK client
constructed at the call site.

**What this replaces, and why the replacement is the point.** The fleet had
``scripts/openrouter_guard.py`` (I6564/I6367, Brian's 2026-08-03 OpenRouter
ruling) and then ``scripts/anthropic_guard.py`` (I9263, today's Anthropic
ruling) -- the second a near-verbatim copy of the first with the provider's
name swapped. That architecture costs one new script + one new reusable
workflow + one new allowlist file + one new per-repo caller for EVERY
provider, so provider N+1 is always undetected until someone notices by
hand. Direct ``anthropic.Anthropic(...)`` construction in
``crucible-research``'s eval-judge handlers survived every guard the fleet
had for exactly this reason: the guard was scoped to a different vendor.

So the provider is DATA here, not code. Adding a provider is one entry in
``PROVIDERS`` below and nothing else -- no new file, no new workflow, no new
per-repo wiring. That is the whole design.

**Pattern classes**, matched line-by-line over the caller repo's tracked
code/config files. Every finding is namespaced ``<provider>:<class>`` so an
allowlist entry is specific about which vendor linkage it is clearing:

  ``sdk_client``   the provider's SDK client being constructed or imported
                   directly (``anthropic.Anthropic(``, ``ChatOpenAI(``,
                   ``genai.GenerativeModel(``, the npm SDK ...). This is the
                   class that actually catches a new bypass -- the other
                   three catch its supporting furniture.
  ``env_key``      a provider credential env-var name being read.
  ``base_url``     a provider hostname literal.
  ``base_url_env`` an env-var name that REPOINTS a client's base URL
                   (``ANTHROPIC_BASE_URL``, ``OPENAI_BASE_URL``). Addressing
                   a model by base URL is a principle-8 violation in its own
                   right even when the URL happens to be the router's.

Docs and markdown are excluded by default (``--include-docs`` to override),
same rationale both predecessor guards carried: this fleet's policy library
discusses every one of these vendors constantly as a TOPIC, and prose is not
linkage.

**Baseline, not a blank ban.** A ``.provider-linkage-allowlist.yaml`` at the
repo root pre-clears known matches. A NEW match with no entry fails. An
entry whose ``expires`` date has passed fails LOUDLY -- re-justify or remove,
never silently re-grandfather. An entry that no longer matches anything also
fails: a stale entry is undetected drift in the other direction, hiding that
a linkage was actually removed and letting the allowance quietly widen.

Router-owned config is NOT exempted by path. The egress proxy's route table
naming a provider host as an upstream, and a LangChain client bound to
``krepis.router.resolve_group_spec()``'s returned base URL, are both
legitimate -- and both get an allowlist entry with a reason and an expiry, so
a change to either is still visible in a diff. A path-based exemption would
have hidden precisely the call sites this guard exists for.

**``krepis`` is excluded fleet-wide** -- it is the router's own repo and
holds the one legitimate declared provider adapter. Principle 8 permits
exactly one adapter behind the router; what it forbids is a call site shaped
around a vendor. krepis is not a caller of this workflow.

Exit codes: ``0`` clean, ``1`` findings, ``2`` could not complete the check.

Usage::

    python3 scripts/provider_linkage_guard.py --repo /path/to/checkout
    python3 scripts/provider_linkage_guard.py --repo . --providers anthropic,openai
    python3 scripts/provider_linkage_guard.py --repo . --list-providers
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

# -- the provider table: the ONLY thing that changes when a vendor is added --

CLASS_SDK_CLIENT = "sdk_client"
CLASS_ENV_KEY = "env_key"
CLASS_BASE_URL = "base_url"
CLASS_BASE_URL_ENV = "base_url_env"

PATTERN_CLASSES = (CLASS_SDK_CLIENT, CLASS_ENV_KEY, CLASS_BASE_URL, CLASS_BASE_URL_ENV)


@dataclass(frozen=True)
class Provider:
    """One vendor's four linkage shapes. Any field may be ``None``.

    ``sdk_client`` patterns are deliberately QUALIFIED (``anthropic.Anthropic(``,
    ``from anthropic import Anthropic``) rather than bare (``Anthropic(``): a
    bare constructor name collides with unrelated local classes across a fleet
    this size, and a guard whose findings are mostly noise trains everyone to
    ignore it -- the failure mode the OpenRouter guard's own pattern-5 rollout
    recorded (alpha-engine-config-I9111).
    """

    name: str
    sdk_client: str | None = None
    env_key: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="anthropic",
        # Python SDK, LangChain binding, and the npm SDK -- a TypeScript call
        # site bypasses the router exactly as much as a Python one does.
        sdk_client=(
            r"anthropic\.Anthropic\("
            r"|anthropic\.AsyncAnthropic\("
            r"|from\s+anthropic\s+import\s+(?:Anthropic|AsyncAnthropic)\b"
            r"|\bChatAnthropic\("
            r"|@anthropic-ai/sdk"
        ),
        env_key=r"ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN",
        base_url=r"api\.anthropic\.com",
        base_url_env=r"ANTHROPIC_BASE_URL",
    ),
    Provider(
        name="openai",
        sdk_client=(
            r"openai\.OpenAI\("
            r"|openai\.AsyncOpenAI\("
            r"|from\s+openai\s+import\s+(?:OpenAI|AsyncOpenAI)\b"
            r"|\bChatOpenAI\("
        ),
        env_key=r"OPENAI_API_KEY",
        base_url=r"api\.openai\.com",
        base_url_env=r"OPENAI_BASE_URL",
    ),
    Provider(
        name="openrouter",
        env_key=r"OPENROUTER_API_KEY",
        base_url=r"openrouter\.ai",
    ),
    Provider(
        name="deepseek",
        env_key=r"DEEPSEEK_API_KEY",
        base_url=r"api\.deepseek\.com",
    ),
    Provider(
        name="xai",
        env_key=r"XAI_API_KEY|GROK_API_KEY",
        base_url=r"api\.x\.ai",
    ),
    Provider(
        name="google",
        sdk_client=(
            r"google\.generativeai"
            r"|\bgenai\.GenerativeModel\("
            r"|\bChatGoogleGenerativeAI\("
            r"|@google/generative-ai"
        ),
        env_key=r"GEMINI_API_KEY|GOOGLE_GENERATIVE_AI_API_KEY",
        base_url=r"generativelanguage\.googleapis\.com",
    ),
    Provider(
        name="groq",
        env_key=r"GROQ_API_KEY",
        base_url=r"api\.groq\.com",
    ),
    Provider(
        name="mistral",
        env_key=r"MISTRAL_API_KEY",
        base_url=r"api\.mistral\.ai",
    ),
    Provider(
        name="zhipu",
        env_key=r"GLM_API_KEY|ZHIPU_API_KEY",
        base_url=r"open\.bigmodel\.cn",
    ),
    Provider(
        # Bedrock is a rented provider like any other under
        # model-portability-policy: an SDK client constructed at a call site
        # is linkage even when the vendor is AWS. AWS is an accepted fleet
        # lock-in for INFRASTRUCTURE, which is a different question from
        # addressing a model by vendor at a call site.
        name="bedrock",
        sdk_client=r"bedrock-runtime",
    ),
    Provider(
        # The Vercel AI SDK is a provider abstraction of its own -- a second
        # router by another name (a "parallel setup" in Brian's words). It is
        # listed here so a TypeScript surface adopting it is a visible,
        # justified decision rather than a silent one.
        name="vercel_ai_sdk",
        sdk_client=r"@ai-sdk/",
    ),
)

PROVIDERS_BY_NAME = {p.name: p for p in PROVIDERS}
ALL_PROVIDER_NAMES = tuple(p.name for p in PROVIDERS)


def pattern_class(provider: str, klass: str) -> str:
    """The namespaced identifier an allowlist entry names, e.g. ``anthropic:env_key``."""
    return f"{provider}:{klass}"


def compile_patterns(providers: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """``(namespaced_class, regex)`` for every declared shape of every selected provider."""
    out: list[tuple[str, re.Pattern[str]]] = []
    for name in providers:
        p = PROVIDERS_BY_NAME[name]
        for klass in PATTERN_CLASSES:
            raw = getattr(p, klass)
            if raw is None:
                continue
            # Hostnames match case-insensitively (a URL authority is not
            # case-sensitive); env-var names and SDK symbols match EXACTLY,
            # because case is meaning there.
            flags = re.IGNORECASE if klass == CLASS_BASE_URL else 0
            out.append((pattern_class(name, klass), re.compile(raw, flags)))
    return tuple(out)


def all_pattern_classes() -> frozenset[str]:
    return frozenset(
        pattern_class(p.name, k)
        for p in PROVIDERS
        for k in PATTERN_CLASSES
        if getattr(p, k) is not None
    )


# Code/config extensions scanned by default -- see the module docstring on why
# markdown is not among them. Dotenv files are deliberately absent: they are
# gitignored fleet-wide, so `git ls-files` never returns one.
DEFAULT_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".mjs", ".sh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    # Non-code EXECUTION surfaces. A launchd plist or a systemd unit that
    # exports a provider credential is a live call site with no source file,
    # and neither predecessor guard looked at them.
    ".plist", ".service", ".timer",
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
    line_index: int


class GuardError(RuntimeError):
    """The check could not be completed -- not a finding, an infrastructure fault."""


# -- scanning ---------------------------------------------------------------


def _tracked_files(repo: Path, extensions: frozenset[str]) -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise GuardError("`git` not found on PATH")
    result = subprocess.run(
        [git, "-C", str(repo), "ls-files"],
        capture_output=True, text=True, timeout=60,
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


# -- the scanner never scans itself -----------------------------------------
#
# Inherited as a PROPERTY from openrouter_guard.py (alpha-engine-config-I9111):
# every pattern this module detects is necessarily written out in this module
# and in the tests that exercise it, so self-scanning would generate one
# allowlist entry per pattern class forever -- and here that is 33 of them,
# growing with every provider added. Both signals are structural: a production
# call site cannot acquire them without literally importing the guard, which is
# visible in a diff.
_SCANNER_SELF_RE = re.compile(
    r"""(?:^|\s)(?:from|import)\s+provider_linkage_guard\b"""
    r"""|spec_from_file_location\(\s*["']provider_linkage_guard["']"""
    r"""|["'][^"']*provider_linkage_guard\.py["']""",
    re.MULTILINE,
)
_SELF_PATH = Path(__file__).resolve()


def _is_scanner_source(fp: Path, text: str) -> bool:
    """This module, or a file whose subject is this module (its tests)."""
    try:
        if fp.resolve() == _SELF_PATH:
            return True
    except OSError:  # pragma: no cover - resolve() on a broken symlink
        pass
    if fp.name == _SELF_PATH.name:
        return True
    return bool(_SCANNER_SELF_RE.search(text))


def scan(
    repo: Path,
    extensions: frozenset[str],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    skip: frozenset[str] = frozenset(),
) -> list[Match]:
    """Every pattern hit in every tracked, in-scope file.

    ``skip`` holds repo-relative paths excluded outright -- the allowlist file
    itself, whose ``reason`` prose legitimately names these same strings and
    would otherwise have to allowlist itself.
    """
    matches: list[Match] = []
    for fp in _tracked_files(repo, extensions):
        rel = str(fp.relative_to(repo))
        if rel in skip:
            continue
        try:
            text = fp.read_text(errors="replace")
        except OSError as exc:
            print(f"::warning::could not read {fp}: {exc}", file=sys.stderr)
            continue
        if _is_scanner_source(fp, text):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for klass, regex in patterns:
                if regex.search(line):
                    matches.append(Match(rel, lineno, klass, line.strip()))
    return matches


# -- allowlist --------------------------------------------------------------


def load_allowlist(path: Path, known_classes: frozenset[str]) -> list[AllowlistEntry]:
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
        klass = e["pattern"]
        if klass not in known_classes:
            raise GuardError(
                f"{path}: entries[{i}].pattern {klass!r} is not a known "
                f"<provider>:<class> identifier. Known: {sorted(known_classes)}"
            )
        try:
            expires = _dt.date.fromisoformat(str(e["expires"]))
        except ValueError as exc:
            raise GuardError(f"{path}: entries[{i}].expires must be YYYY-MM-DD") from exc
        if not str(e["reason"]).strip():
            raise GuardError(f"{path}: entries[{i}].reason must be non-empty")
        out.append(AllowlistEntry(
            path=e["path"],
            pattern_class=klass,
            reason=str(e["reason"]),
            expires=expires,
            tracking=e.get("tracking"),
            line_index=i,
        ))
    return out


# -- evaluation -------------------------------------------------------------


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
        if key not in by_key:
            unallowlisted.append(m)
            continue
        matched_keys.add(key)
        covered += 1

    expired = [e for e in allowlist if e.expires < today]
    stale = [
        e for e in allowlist
        if e.expires >= today and (e.path, e.pattern_class) not in matched_keys
    ]
    return Report(unallowlisted=unallowlisted, expired=expired, stale=stale, covered=covered)


# -- reporting --------------------------------------------------------------


def render(report: Report, total_matches: int) -> int:
    if report.ok:
        print(
            f"No unallowlisted direct provider linkage. "
            f"{total_matches} total match(es), {report.covered} allowlisted."
        )
        return 0

    for m in sorted(report.unallowlisted, key=lambda m: (m.path, m.line)):
        print(
            f"::error file={m.path},line={m.line}::"
            f"unallowlisted direct provider linkage ({m.pattern_class}): {m.text} "
            f"-- route this through krepis (address a model GROUP), or add a "
            f"justified, expiring entry to .provider-linkage-allowlist.yaml"
        )
    for e in sorted(report.expired, key=lambda e: (e.path, e.pattern_class)):
        tracking = f" ({e.tracking})" if e.tracking else ""
        print(
            f"::error file={e.path}::allowlist entry EXPIRED {e.expires.isoformat()} "
            f"for pattern={e.pattern_class}{tracking}: {e.reason} -- re-justify with a "
            f"new expires date or remove the reference"
        )
    for e in sorted(report.stale, key=lambda e: (e.path, e.pattern_class)):
        print(
            f"::error file={e.path}::stale allowlist entry (pattern={e.pattern_class}) -- "
            f"no longer matches anything in the repo. Remove the entry: an allowlist "
            f"that outlives what it covers silently widens over time"
        )

    n = len(report.unallowlisted) + len(report.expired) + len(report.stale)
    print(f"\n{n} finding(s).", file=sys.stderr)
    return 1


# -- CLI --------------------------------------------------------------------


def _selected_providers(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ALL_PROVIDER_NAMES
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    unknown = [n for n in names if n not in PROVIDERS_BY_NAME]
    if unknown:
        raise GuardError(
            f"unknown provider(s): {unknown}. Known: {list(ALL_PROVIDER_NAMES)}"
        )
    if not names:
        raise GuardError("--providers was given but selected nothing")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--repo", default=".", help="repo root to scan")
    ap.add_argument(
        "--allowlist", default=None,
        help="path to allowlist YAML (default: <repo>/.provider-linkage-allowlist.yaml)",
    )
    ap.add_argument(
        "--providers", default=None,
        help=f"comma-separated subset (default: all -- {','.join(ALL_PROVIDER_NAMES)})",
    )
    ap.add_argument(
        "--include-docs", action="store_true",
        help="also scan markdown/rst/txt files (off by default)",
    )
    ap.add_argument(
        "--list-providers", action="store_true",
        help="print the provider table and exit 0",
    )
    args = ap.parse_args(argv)

    if args.list_providers:
        for p in PROVIDERS:
            shapes = [k for k in PATTERN_CLASSES if getattr(p, k) is not None]
            print(f"{p.name}: {', '.join(shapes)}")
        return 0

    repo = Path(args.repo).resolve()
    allowlist_path = (
        Path(args.allowlist)
        if args.allowlist
        else repo / ".provider-linkage-allowlist.yaml"
    )
    extensions = DEFAULT_EXTENSIONS | (DOC_EXTENSIONS if args.include_docs else frozenset())

    try:
        providers = _selected_providers(args.providers)
        patterns = compile_patterns(providers)
        allowlist_rel = None
        try:
            allowlist_rel = str(allowlist_path.resolve().relative_to(repo))
        except ValueError:
            pass  # allowlist lives outside the repo (a test fixture) -- nothing to skip
        skip = frozenset({allowlist_rel}) if allowlist_rel else frozenset()
        matches = scan(repo, extensions, patterns, skip=skip)
        allowlist = load_allowlist(allowlist_path, all_pattern_classes())
    except GuardError as exc:
        print(f"::error::could not complete provider linkage guard scan: {exc}")
        return 2

    report = evaluate(matches, allowlist, _dt.date.today())
    return render(report, len(matches))


if __name__ == "__main__":
    raise SystemExit(main())
