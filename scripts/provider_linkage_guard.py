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
  ``dist``         a PEP 503 distribution name in a dependency file
                   (``anthropic==0.34.0`` in ``requirements.txt``, a ``name =
                   "anthropic"`` package table in ``uv.lock``, a bare
                   ``"openai>=1.0"`` string in ``pyproject.toml``). See
                   "DEPENDENCY FILES" below -- this class grades what the
                   resolved environment can REACH, not what a call site
                   spells out (alpha-engine-config-I10032). Gated behind
                   ``--include-dist`` (see that section) -- OFF by default,
                   same opt-in shape as ``--include-docs``.

**DEPENDENCY FILES (alpha-engine-config-I10032).** The four classes above are
call-site-shaped: they grade what a repo's *source* addresses. They cannot
see a provider SDK that enters the resolved environment as a plain
dependency and is never imported anywhere in that repo's tree --
``alpha-engine-config-I7723`` is exactly this: the ``anthropic`` SDK sat in
the trading box's venv, pulled transitively through ``krepis``'s old
``flow-doctor[diagnosis]`` extra, with zero ``import anthropic`` in
``crucible-executor``'s tree, for three months, invisible to every guard the
fleet had because every guard looked at source.

``dist`` closes that gap by reading dependency/lock files for a bare
distribution name. Two design choices, both deliberate:

  1. **Matched by FILENAME, not extension** (``DEPENDENCY_FILENAME_RE``):
     ``requirements*.txt``, ``requirements*.in``, ``pyproject.toml``,
     ``uv.lock``, ``poetry.lock``, ``Pipfile.lock``. The alternative --
     adding ``.in``/``.lock`` to ``DEFAULT_EXTENSIONS`` and moving ``.txt``
     out of ``DOC_EXTENSIONS`` -- was rejected: ``.txt`` genuinely is a doc
     extension everywhere else this guard looks (a runbook, a changelog), and
     widening it fleet-wide would scan every ``.txt`` file in every repo as
     code. Filename matching reaches exactly the dependency-resolution
     surface and nothing else, independent of ``--include-docs``.
  2. **The pattern is a bare PEP 503 name, boundary-matched** (``_dist_boundary``),
     not a "looks like a requirements line" heuristic: a distribution name is
     free to appear as ``anthropic==0.34.0``, ``anthropic[bedrock]>=0.30``,
     ``name = "anthropic"`` (a TOML lock package table), or a bare
     ``"anthropic"`` JSON key (``Pipfile.lock``) -- one boundary-aware regex
     per name covers all four shapes without parsing each format, matching
     this module's existing architecture (a line-by-line regex scanner, not a
     format-aware parser). PEP 503 normalizes ``-``/``_``/``.`` to a single
     separator and lowercases, so the pattern treats those three characters
     as interchangeable and matches case-insensitively, same as ``base_url``.

A distribution appearing only in a **compiled lock file**, never in
``requirements.in`` itself (a transitive dependency), is still caught --
``uv.lock``/``poetry.lock``/``Pipfile.lock`` are all in scope for exactly
this reason: a bare ``requirements.in`` scan alone would recreate the exact
blind spot ``I7723`` occupied.

**Rollout is warn-only until each caller's baseline is allowlisted.** The
reusable workflow (``.github/workflows/provider-linkage-guard.yml``) invokes
this script WITHOUT ``--include-dist`` for now -- merging this class lands it
in the library without re-verdicting any of the fleet's callers on merge (the
workflow is checked out unpinned; see the module-level note on
``evaluate()`` for why a guard-side widening must not by itself redden a
consumer). The enforcement flip -- adding ``--include-dist`` to the reusable
workflow step -- is a separate, later change, once each caller's baseline is
measured and any legitimate match is allowlisted with an expiry. Tracked:
alpha-engine-config-I10032 (this class), alpha-engine-config-I10225 (the
flip -- baseline measured 2026-09-08 against all 15 current callers, 6 with
legitimate pre-existing matches, 9 clean).

Docs and markdown are excluded by default (``--include-docs`` to override),
same rationale both predecessor guards carried: this fleet's policy library
discusses every one of these vendors constantly as a TOPIC, and prose is not
linkage. **Comments are excluded for the same reason** (I9295): prose after a
``#`` or ``//`` is not an execution surface, and the file it sits in does not
change that. String literals are NOT excluded -- a credential name or a base
URL in a string executes, and must still fail, **except** two narrow,
structural carve-outs (I9263), neither a text heuristic over what a string
says:

  - a Python **docstring** -- a bare string-constant statement in first
    position of a module/class/function body (``_docstring_spans``) -- is
    prose under the identical rationale as a comment, identified the same
    structural way comments already are (a real syntax position, never
    content-sniffed);
  - a file that opts in with a ``# provider-linkage-guard: asserts-absence``
    marker (``_is_asserts_absence``) -- the same declared-marker discipline
    ``_is_declared_registry`` already uses -- because a test's literal tuple
    of forbidden strings, compared with ``not in`` to prove a retired
    pattern is GONE, is not a call site either. Both are file-wide and
    apply only to the findings scan, never the staleness scan (see
    ``scan()``'s ``registry_aware``/``absence_aware`` docstring).

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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# -- the provider table: the ONLY thing that changes when a vendor is added --

CLASS_SDK_CLIENT = "sdk_client"
CLASS_ENV_KEY = "env_key"
CLASS_BASE_URL = "base_url"
CLASS_BASE_URL_ENV = "base_url_env"
CLASS_DIST = "dist"

PATTERN_CLASSES = (CLASS_SDK_CLIENT, CLASS_ENV_KEY, CLASS_BASE_URL, CLASS_BASE_URL_ENV, CLASS_DIST)

# Pattern classes scanned by DEFAULT, without `--include-dist`. `dist` is
# excluded here -- see the module docstring's DEPENDENCY FILES /
# "warn-only until baseline is allowlisted" sections. `all_pattern_classes()`
# still reports `dist` as a KNOWN class (an allowlist entry naming it is
# valid), independent of whether a given run scans for it.
DEFAULT_PATTERN_CLASSES = tuple(k for k in PATTERN_CLASSES if k != CLASS_DIST)


def _dist_boundary(*names: str) -> str:
    """A PEP 503 distribution-name regex: boundary-matched, separator-flexible.

    A distribution name can appear as ``anthropic==0.34.0``,
    ``anthropic[bedrock]>=0.30``, a TOML lock's ``name = "anthropic"``, or a
    bare ``"anthropic"`` JSON key -- one regex per name covers all of those
    shapes without parsing each format (this module scans line-by-line, not
    format-aware). ``-``/``_``/``.`` are treated as interchangeable, per PEP
    503 normalization, so ``google-generativeai`` also matches
    ``google_generativeai``. The lookaround boundary (neither a word char nor
    one of those three separators on either side) is what stops a real match
    on ``anthropic==1.0`` while still refusing a false one on an unrelated
    compound name that merely CONTAINS this name as a substring.
    """
    alts = []
    for n in names:
        parts = re.split(r"[-_.]", n)
        alts.append("[-_.]".join(re.escape(p) for p in parts))
    return rf"(?<![\w.-])(?:{'|'.join(alts)})(?![\w.-])"


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
    dist: str | None = None


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
        # PyPI dist "anthropic" -- I7723: the SDK sat in a resolved venv with
        # zero call sites anywhere in the tree.
        dist=_dist_boundary("anthropic"),
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
        dist=_dist_boundary("openai"),
    ),
    Provider(
        # No `dist`: OpenRouter has no dedicated PyPI SDK -- it is reached
        # through the openai-compatible client, whose dist is already
        # covered under the "openai" provider above.
        name="openrouter",
        env_key=r"OPENROUTER_API_KEY",
        base_url=r"openrouter\.ai",
    ),
    Provider(
        # No `dist`: same reasoning as openrouter -- reached via the
        # openai-compatible client, no dedicated PyPI package of its own.
        name="deepseek",
        env_key=r"DEEPSEEK_API_KEY",
        base_url=r"api\.deepseek\.com",
    ),
    Provider(
        name="xai",
        env_key=r"XAI_API_KEY|GROK_API_KEY",
        base_url=r"api\.x\.ai",
        dist=_dist_boundary("xai-sdk"),
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
        dist=_dist_boundary("google-generativeai", "google-genai"),
    ),
    Provider(
        name="groq",
        env_key=r"GROQ_API_KEY",
        base_url=r"api\.groq\.com",
        dist=_dist_boundary("groq"),
    ),
    Provider(
        name="mistral",
        env_key=r"MISTRAL_API_KEY",
        base_url=r"api\.mistral\.ai",
        # The PyPI dist is "mistralai", not "mistral" -- the bare name
        # collides with unrelated packages (mistral-common et al. are a
        # different concern; "mistral" bare is too generic a token to
        # boundary-match safely).
        dist=_dist_boundary("mistralai"),
    ),
    Provider(
        name="zhipu",
        env_key=r"GLM_API_KEY|ZHIPU_API_KEY",
        base_url=r"open\.bigmodel\.cn",
        dist=_dist_boundary("zhipuai"),
    ),
    Provider(
        # Bedrock is a rented provider like any other under
        # model-portability-policy: an SDK client constructed at a call site
        # is linkage even when the vendor is AWS. AWS is an accepted fleet
        # lock-in for INFRASTRUCTURE, which is a different question from
        # addressing a model by vendor at a call site.
        # No `dist`: the SDK is `boto3`, generic across every AWS service --
        # boundary-matching "boto3" as a distribution name would fire on
        # every repo using AWS for anything, which is not this guard's
        # question. The `sdk_client` pattern above stays call-site-shaped
        # (the literal "bedrock-runtime" client name) for that reason.
        name="bedrock",
        sdk_client=r"bedrock-runtime",
    ),
    Provider(
        # The Vercel AI SDK is a provider abstraction of its own -- a second
        # router by another name (a "parallel setup" in Brian's words). It is
        # listed here so a TypeScript surface adopting it is a visible,
        # justified decision rather than a silent one.
        # No `dist`: npm-only, no PyPI distribution.
        name="vercel_ai_sdk",
        sdk_client=r"@ai-sdk/",
    ),
)

PROVIDERS_BY_NAME = {p.name: p for p in PROVIDERS}
ALL_PROVIDER_NAMES = tuple(p.name for p in PROVIDERS)


def pattern_class(provider: str, klass: str) -> str:
    """The namespaced identifier an allowlist entry names, e.g. ``anthropic:env_key``."""
    return f"{provider}:{klass}"


def compile_patterns(
    providers: tuple[str, ...],
    classes: tuple[str, ...] = PATTERN_CLASSES,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """``(namespaced_class, regex)`` for every declared shape of every selected
    provider, restricted to `classes` (default: all -- pass
    `DEFAULT_PATTERN_CLASSES` to exclude `dist` the way the CLI does without
    `--include-dist`)."""
    out: list[tuple[str, re.Pattern[str]]] = []
    for name in providers:
        p = PROVIDERS_BY_NAME[name]
        for klass in classes:
            raw = getattr(p, klass)
            if raw is None:
                continue
            # Hostnames match case-insensitively (a URL authority is not
            # case-sensitive), and so does `dist` (PEP 503 normalization
            # lowercases a distribution name); env-var names and SDK symbols
            # match EXACTLY, because case is meaning there.
            flags = re.IGNORECASE if klass in (CLASS_BASE_URL, CLASS_DIST) else 0
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

# Dependency/lock files, matched by FILENAME rather than extension -- see the
# module docstring's DEPENDENCY FILES section for why. Only consulted when
# `--include-dist` is passed (the `dist` pattern class is the only one
# meaningfully found in these files). ``requirements.txt`` shares its suffix
# with every other doc ``.txt`` file in a repo, so this is deliberately NOT
# a change to `DOC_EXTENSIONS`.
DEPENDENCY_FILENAME_RE = re.compile(
    r"^(requirements[\w.-]*\.(?:txt|in)|pyproject\.toml|uv\.lock|poetry\.lock|Pipfile\.lock)$"
)


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
    # requirements*.txt/.in and the TOML-shaped lock files all use `#` for
    # comments (pip's requirements format, and TOML). These only enter the
    # scan at all via DEPENDENCY_FILENAME_RE (--include-dist), never through
    # DEFAULT_EXTENSIONS/DOC_EXTENSIONS -- listing the suffixes here only
    # controls comment-stripping once a file is in scope, so a bare ``.txt``
    # doc file scanned under ``--include-docs`` is unaffected in practice
    # (comments there are prose either way).
    ".in", ".lock", ".txt",
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


def _docstring_spans(text: str) -> list[tuple[int, int, int, int]]:
    """``(start_line, start_col, end_line, end_col)`` for every real docstring.

    A docstring is identified STRUCTURALLY -- a bare string-constant
    ``Expr`` statement in FIRST position of a module, class or function body
    -- exactly the definition Python itself uses (``ast.get_docstring``), not
    a heuristic over content. That is deliberate: it cannot be fooled by a
    string that merely *looks* like documentation, and it cannot hide a
    string that is not in first-statement position, because that string is
    not a docstring at all -- it is a normal expression statement (dead code
    in practice, but not this guard's concern to adjudicate).

    Returns an empty list -- never raises -- on a file that fails to parse:
    a syntactically broken file must still be SCANNED in full, the same
    fallback discipline ``_blank_python_comments`` already applies.
    """
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    spans: list[tuple[int, int, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            v = first.value
            if v.end_lineno is not None and v.end_col_offset is not None:
                spans.append((v.lineno, v.col_offset, v.end_lineno, v.end_col_offset))
    return spans


def _blank_python_docstrings(text: str) -> str:
    """Blank real docstrings (see ``_docstring_spans``), preserving line numbers.

    **Why this exists (alpha-engine-config-I9263).** Five allowlist entries
    on ``crucible-research`` expired covering the exact same shape comment-
    stripping already fixed once for ``#``: a migration's own module
    docstring narrating the retired ``anthropic.Anthropic(api_key=...)``
    construction it replaced is prose, not a call site -- a string is not a
    client construction any more than a comment is. Re-dating those entries
    forever is a suppression renewal, not a fix; the guard's own module
    docstring above already states the rule this closes the gap on:
    "Comments are excluded... prose is not linkage." A module/class/function
    docstring is prose under the identical rationale, identified the same
    structural way (a real syntax position), never a text heuristic over
    what the string says.

    **What this does NOT weaken.** String literals used as CODE -- a base
    URL assigned to a variable, a credential name passed to
    ``os.environ.get(...)``, a test's tuple of forbidden literals compared
    with ``in``/``not in`` -- are untouched: none of those occupy the
    first-statement-of-a-scope position a docstring requires, so none are in
    ``_docstring_spans``' output. Detection of every one of those shapes is
    unchanged.
    """
    spans = _docstring_spans(text)
    if not spans:
        return text
    lines = text.splitlines()
    for l1, c1, l2, c2 in spans:
        if l1 == l2:
            line = lines[l1 - 1]
            lines[l1 - 1] = line[:c1] + " " * (c2 - c1) + line[c2:]
        else:
            lines[l1 - 1] = lines[l1 - 1][:c1]
            for i in range(l1, l2 - 1):
                lines[i] = ""
            lines[l2 - 1] = " " * c2 + lines[l2 - 1][c2:]
    return "\n".join(lines)


def strip_comments(rel: str, text: str) -> str:
    """`text` with comment AND docstring regions blanked, line numbers preserved.

    Extensions with no comment syntax (``.json``, ``.plist``) are returned
    unchanged -- there is nothing to strip and inventing a rule for them would
    only create a way to hide a real literal.
    """
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        return _blank_python_docstrings(_blank_python_comments(text))
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


def _tracked_files(
    repo: Path,
    extensions: frozenset[str],
    extra_filename_re: re.Pattern[str] | None = None,
) -> list[Path]:
    """Tracked files matching `extensions` by suffix, plus, if given, any
    tracked file whose BASENAME matches `extra_filename_re` regardless of
    suffix (see DEPENDENCY_FILENAME_RE -- ``requirements.txt`` must enter
    scope on its filename, independent of whatever `extensions`/`--include-
    docs` says about ``.txt``).
    """
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
        p = Path(rel)
        by_ext = p.suffix.lower() in extensions
        by_name = extra_filename_re is not None and extra_filename_re.match(p.name) is not None
        if not (by_ext or by_name):
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


# -- a structural test literal asserting a pattern's ABSENCE is not linkage -
#
# alpha-engine-config-I9263: the docstring fix above does not reach a string
# literal that is ordinary CODE, not a docstring -- e.g. a test's tuple of
# forbidden strings compared with ``not in`` against a module's source, the
# exact shape ``tests/test_eval_judge_batch_transport.py`` and
# ``tests/test_no_anthropic_sdk_construction`` use to prove the retired
# ``anthropic.Anthropic(...)``/``ANTHROPIC_API_KEY`` pattern is GONE. That
# literal is not a call site any more than the guard's own PROVIDERS table
# (``_is_scanner_source``) or a declared registry's schema
# (``_is_declared_registry``) are -- and it gets the SAME treatment: an
# explicit, opt-in, structural marker a file declares about ITSELF, never a
# path list and never inferred from the string's content.
#
# This is deliberately NARROWER than "any string in a _test_ file is exempt"
# -- that would blind the guard to a genuine accidental construction sitting
# in an unrelated test (a copy-pasted fixture, a live-smoke helper that
# actually calls the SDK). Only a file that explicitly declares its subject
# is proving absence earns the exemption, exactly the discipline
# ``_is_declared_registry`` already established for "a file that IS a
# registry" versus "a file that merely mentions one."
_ASSERTS_ABSENCE_RE = re.compile(
    r"^\s*#\s*provider-linkage-guard:\s*asserts-absence\b", re.MULTILINE
)


def _is_asserts_absence(text: str) -> bool:
    """A file that structurally asserts a retired pattern's ABSENCE, by marker."""
    return bool(_ASSERTS_ABSENCE_RE.search(text))


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
    absence_aware: bool = True,
    extra_filename_re: re.Pattern[str] | None = None,
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

    ``absence_aware`` gates the ``asserts-absence`` marker exemption (see
    ``_is_asserts_absence``) the same way, for the same reason: it is a
    relaxation and must never retroactively stale an existing allowlist
    entry on ``main`` with no commit in that repo.
    """
    registry_names = _registry_filenames(repo, extensions) if registry_aware else frozenset()

    # `dist` is meaningful ONLY inside a dependency/lock file -- a bare
    # distribution name like "anthropic" is a real word that shows up
    # constantly in ordinary prose, JSON test fixtures and shell scripts (a
    # gitleaks baseline naming "anthropic" as a credential-prefix vendor, a
    # DLP script's help text). Applying the `dist` regex fleet-wide over
    # every file `extensions` already selects would flood findings with
    # exactly that noise -- measured, not hypothetical: an early version of
    # this class applied uniformly and produced 79 findings on
    # claude-code-config alone, nearly all inside .sh/.json prose. So `dist`
    # patterns are matched ONLY against files that ALSO match
    # `extra_filename_re` (dependency-file shaped by name); every other
    # pattern class still applies to the full `extensions`-selected set, same
    # as before this class existed.
    dist_suffix = f":{CLASS_DIST}"
    other_patterns = tuple((k, r) for k, r in patterns if not k.endswith(dist_suffix))
    dist_patterns = tuple((k, r) for k, r in patterns if k.endswith(dist_suffix))

    matches: list[Match] = []
    for fp in _tracked_files(repo, extensions, extra_filename_re):
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
        if absence_aware and _is_asserts_absence(text):
            continue
        is_dependency_file = (
            extra_filename_re is not None and extra_filename_re.match(fp.name) is not None
        )
        applicable = other_patterns + (dist_patterns if is_dependency_file else ())
        body = strip_comments(rel, text) if strip else text
        for lineno, line in enumerate(body.splitlines(), 1):
            for klass, regex in applicable:
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
    disabled: list[AllowlistEntry] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unallowlisted or self.expired or self.stale)


def evaluate(
    matches: list[Match],
    allowlist: list[AllowlistEntry],
    today: _dt.date,
    raw_matches: list[Match] | None = None,
    *,
    enabled_classes: frozenset[str] | None = None,
) -> Report:
    """Findings from ``matches``; STALENESS from ``raw_matches``.

    ``enabled_classes`` is the set of bare pattern-class names (``sdk_client``,
    ``env_key``, ... -- NOT the namespaced ``<provider>:<class>`` form) actually
    scanned in this invocation, e.g. `DEFAULT_PATTERN_CLASSES` without
    `--include-dist`. Pass ``None`` (the default) to mean "every class is
    enabled" -- the historical behaviour, still correct for a caller that
    scans unconditionally.

    An allowlist entry for a class this invocation did NOT scan is neither
    matched nor genuinely unmatched -- its usefulness is unknowable, because
    the class it names was never looked at. alpha-engine-config-I10225:
    reporting "unknowable" as `stale` deadlocked six consumer PRs against
    `nousergon-lib-PR397`, which enables the `dist` class fleet-wide -- their
    new `dist` allowlist entries were reported stale by the reusable
    workflow's pre-flip run (no `--include-dist`), a red required check no PR
    may merge past, while the flip that would make the class scanned could
    not itself land first without reddening those six repos' `main` with
    unallowlisted findings the instant it merged. Such an entry moves to
    ``disabled`` (a distinct, non-erroring bucket) instead of ``stale``, and
    is reported as skipped rather than silently dropped. This does NOT relax
    detection for a class this invocation DOES scan: an unused entry for an
    ENABLED class is still `stale`, unchanged.

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
    unmatched = [
        e for e in allowlist
        if e.expires >= today and (e.path, e.pattern_class) not in matched_keys
    ]
    disabled = [
        e for e in unmatched
        if enabled_classes is not None
        and e.pattern_class.rsplit(":", 1)[-1] not in enabled_classes
    ]
    disabled_keys = {(e.path, e.pattern_class) for e in disabled}
    stale = [e for e in unmatched if (e.path, e.pattern_class) not in disabled_keys]
    return Report(
        unallowlisted=unallowlisted, expired=expired, stale=stale,
        covered=covered, disabled=disabled,
    )


# -- reporting --------------------------------------------------------------


def render(report: Report, total_matches: int) -> int:
    for e in sorted(report.disabled, key=lambda e: (e.path, e.pattern_class)):
        print(
            f"::notice file={e.path}::skipping staleness check for pattern="
            f"{e.pattern_class} -- this pattern class is not enabled for this "
            f"invocation, so whether the entry still matches anything is "
            f"unknowable here, not stale"
        )
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
        "--include-dist", action="store_true",
        help=(
            "also scan requirements/lock/pyproject files for a bare "
            "distribution-name pin (the `dist` pattern class; off by "
            "default -- see the module docstring's DEPENDENCY FILES section)"
        ),
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
    classes = PATTERN_CLASSES if args.include_dist else DEFAULT_PATTERN_CLASSES
    dep_re = DEPENDENCY_FILENAME_RE if args.include_dist else None

    try:
        providers = _selected_providers(args.providers)
        patterns = compile_patterns(providers, classes)
        allowlist_rel = None
        try:
            allowlist_rel = str(allowlist_path.resolve().relative_to(repo))
        except ValueError:
            pass  # allowlist lives outside the repo (a test fixture) -- nothing to skip
        skip = frozenset({allowlist_rel}) if allowlist_rel else frozenset()
        matches = scan(repo, extensions, patterns, skip=skip, extra_filename_re=dep_re)
        raw_matches = scan(
            repo, extensions, patterns, skip=skip,
            strip=False, registry_aware=False, absence_aware=False,
            extra_filename_re=dep_re,
        )
        allowlist = load_allowlist(allowlist_path, all_pattern_classes())
    except GuardError as exc:
        print(f"::error::could not complete provider linkage guard scan: {exc}")
        return 2

    report = evaluate(
        matches, allowlist, _dt.date.today(), raw_matches=raw_matches,
        enabled_classes=frozenset(classes),
    )
    return render(report, len(matches))


if __name__ == "__main__":
    raise SystemExit(main())
