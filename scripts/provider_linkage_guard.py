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
linkage. **Comments are excluded for the same reason** (I9295): prose after a
``#`` or ``//`` is not an execution surface, and the file it sits in does not
change that. String literals are NOT excluded -- a credential name or a base
URL in a string executes, and must still fail.

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


# -- comments are not linkage ------------------------------------------------
#
# The guard matched line-by-line over raw text, so a COMMENT naming a provider
# credential read as a call site. Measured 2026-08-29 on `alpha-engine-config`:
# 471 findings, the overwhelming majority inside comments -- an allowlist of
# that size is not a baseline, it is the guard being switched off one entry at
# a time. The correct fix is upstream: a comment is not an execution surface,
# for exactly the reason `DOC_EXTENSIONS` are excluded by default. Prose is not
# linkage whether it sits in a .md file or after a `#`.
#
# STRING LITERALS ARE DELIBERATELY LEFT INTACT. A base URL or a credential name
# in a string IS executable and must still fail. Only the comment regions are
# blanked, and blanking preserves line numbering so every reported line number
# still points at the real line.

_COMMENT_HASH_EXTENSIONS = frozenset({
    ".sh", ".bash", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".service", ".timer",
})
_COMMENT_SLASH_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".mjs"})


def _blank_hash_comments(text: str) -> str:
    """Blank `#` comments, respecting single/double quoted strings on the line."""
    out: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        cut: int | None = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def _blank_slash_comments(text: str) -> str:
    """Blank `//` and `/* ... */` comments, respecting quoted strings."""
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        buf: list[str] = []
        quote: str | None = None
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_block:
                if ch == "*" and nxt == "/":
                    in_block = False
                    i += 2
                    continue
                i += 1
                continue
            if quote is not None:
                buf.append(ch)
                if ch == "\\":
                    if nxt:
                        buf.append(nxt)
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in "\"'`":
                quote = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "/" and nxt == "/":
                break
            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def _blank_python_comments(text: str) -> str:
    """Blank `#` comments using the real tokenizer, so `#` inside a string survives.

    Falls back to the quote-aware scanner when the file does not tokenize --
    a syntactically broken file must still be SCANNED, never silently skipped.
    """
    import io
    import tokenize

    lines = text.splitlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _blank_hash_comments(text)
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        if 1 <= row <= len(lines):
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(lines)


def strip_comments(rel: str, text: str) -> str:
    """`text` with comment regions blanked, line numbering preserved.

    Extensions with no comment syntax (``.json``, ``.plist``) are returned
    unchanged -- there is nothing to strip and inventing a rule for them would
    only create a way to hide a real literal.
    """
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        return _blank_python_comments(text)
    if suffix in _COMMENT_HASH_EXTENSIONS:
        return _blank_hash_comments(text)
    if suffix in _COMMENT_SLASH_EXTENSIONS:
        return _blank_slash_comments(text)
    return text


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


# -- declared registries are not call sites ---------------------------------
#
# alpha-engine-config-I9295: a registry of provider linkage necessarily NAMES
# providers -- that is its job, not a violation of it (principle 8 names the
# registry as the one legitimate home for a model id). The remaining findings
# on alpha-engine-config's `main` (360, all measured) are ONE class: the
# declared registry files themselves, plus the validators/tests whose entire
# SUBJECT is one of those registries.
#
# This is the SAME property `_is_scanner_source` already encodes for this
# module, generalized rather than re-invented, and generalized the same way
# PR374 generalized comment-stripping: by a STRUCTURAL marker, never a path
# list. Two structural signals, each visible in a diff:
#
#   1. A file DECLARES itself a registry with a one-line header marker. Any
#      new registry adopts the same marker rather than growing an allowlist
#      or this scanner's code.
#   2. A file whose SUBJECT is a declared registry names that registry's
#      filename literally -- exactly how a test of THIS module names
#      "provider_linkage_guard.py". Verified 2026-08-29 against every
#      offending file on alpha-engine-config's main: every one of
#      scripts/validate_llm_callsite_registry.py,
#      scripts/validate_llm_model_registry.py,
#      scripts/check_llm_custody_conformance.py and their test_* companions
#      references "LLM_CALLSITE_REGISTRY.yaml" or "LLM_MODEL_REGISTRY.yaml"
#      by name.
#
# A path-based exemption would have hidden precisely the call sites this
# guard exists for -- this does not: an UNRELATED file that happens to
# mention a registry filename in passing gets the same file-wide exemption a
# real registry test already earns today, and a genuine new bypass would
# have to either literally reference the registry (visible) or go undetected
# by a different, unrelated mechanism.
_REGISTRY_SELF_RE = re.compile(r"^\s*#\s*provider-linkage-registry:\s*declared\b", re.MULTILINE)
_REGISTRY_FILENAME_RE = re.compile(r"\b[A-Za-z0-9_]*_REGISTRY\.ya?ml\b")


def _is_declared_registry(fp: Path, text: str) -> bool:
    """A file that IS a provider-linkage registry, by structural marker."""
    if fp.suffix.lower() not in {".yaml", ".yml"}:
        return False
    return bool(_REGISTRY_SELF_RE.search(text))


def _registry_filenames(repo: Path, extensions: frozenset[str]) -> frozenset[str]:
    """Basenames of every declared registry tracked in the repo.

    A separate pass, not a hardcoded list: any file anywhere in the tree that
    carries the ``provider-linkage-registry: declared`` marker counts, so a
    new registry needs only that one line, never an edit here.
    """
    names: set[str] = set()
    for fp in _tracked_files(repo, extensions):
        if fp.suffix.lower() not in {".yaml", ".yml"}:
            continue
        try:
            text = fp.read_text(errors="replace")
        except OSError:
            continue
        if _is_declared_registry(fp, text):
            names.add(fp.name)
    return frozenset(names)


def _is_registry_subject(text: str, registry_names: frozenset[str]) -> bool:
    """A file whose subject is a declared registry -- it names the registry."""
    return any(name in text for name in registry_names)


def scan(
    repo: Path,
    extensions: frozenset[str],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    skip: frozenset[str] = frozenset(),
    *,
    strip: bool = True,
    registry_aware: bool = True,
) -> list[Match]:
    """Every pattern hit in every tracked, in-scope file.

    ``skip`` holds repo-relative paths excluded outright -- the allowlist file
    itself, whose ``reason`` prose legitimately names these same strings and
    would otherwise have to allowlist itself.

    ``registry_aware`` gates the declared-registry exemption (see
    ``_is_declared_registry`` / ``_is_registry_subject``) exactly the way
    ``strip`` gates comment-stripping, and for the identical reason
    (alpha-engine-config-I9295): this is a RELAXATION, so it must be evaluated
    only in the findings scan, never in the raw scan staleness is computed
    from. Applying it to both would let it silently retire an existing
    allowlist entry into "stale" the moment a repo's registry earns the
    marker -- turning a guard-side relaxation into a red consumer `main` with
    no commit there, the same failure mode comment-stripping had to avoid.
    """
    registry_names = _registry_filenames(repo, extensions) if registry_aware else frozenset()
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
        if registry_aware and _is_declared_registry(fp, text):
            continue
        if registry_names and _is_registry_subject(text, registry_names):
            continue
        body = strip_comments(rel, text) if strip else text
        for lineno, line in enumerate(body.splitlines(), 1):
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
    raw_matches: list[Match] | None = None,
) -> Report:
    """Findings from ``matches``; STALENESS from ``raw_matches``.

    The two are deliberately different match sets (alpha-engine-config-I9295).
    ``matches`` is comment-stripped, because a comment is not a call site.
    Staleness asks a different question -- *is this entry still about anything
    in this repo* -- and is answered against the RAW text, so that teaching the
    guard to ignore comments does not, by itself, convert a live allowlist
    entry into a failure.

    That distinction is load-bearing for the fleet, not a nicety. The reusable
    guard workflow checks the script out UNPINNED (a deliberate decision: a
    script fix must not wait on every consumer's pin bump), so a guard-side
    change re-verdicts every consumer repo's `main` with no commit in that
    repo -- measured twice on 2026-08-28. A change that makes the guard
    strictly LESS sensitive must therefore be incapable of reddening anyone.
    Without this split, ignoring comments would have turned three entries on
    `crucible-research` main stale on merge.
    """
    by_key: dict[tuple[str, str], list[AllowlistEntry]] = {}
    for e in allowlist:
        by_key.setdefault((e.path, e.pattern_class), []).append(e)

    seen_keys = {
        (m.path, m.pattern_class)
        for m in (matches if raw_matches is None else raw_matches)
    }
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

    matched_keys |= {k for k in seen_keys if k in by_key}

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
        raw_matches = scan(repo, extensions, patterns, skip=skip, strip=False, registry_aware=False)
        allowlist = load_allowlist(allowlist_path, all_pattern_classes())
    except GuardError as exc:
        print(f"::error::could not complete provider linkage guard scan: {exc}")
        return 2

    report = evaluate(matches, allowlist, _dt.date.today(), raw_matches=raw_matches)
    return render(report, len(matches))


if __name__ == "__main__":
    raise SystemExit(main())
