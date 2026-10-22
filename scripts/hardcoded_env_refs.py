#!/usr/bin/env python3
"""Shrink-only guard against hard-coded environment references (plan S-04).

Clause ``ENV-NO-LITERALS`` of ``nous-ergon-ops`` ``policies/delivery-policy.md``:
code, workflows and infrastructure files resolve every account and bucket
through one resolver (``nousergon_lib.env``) and name none directly. A literal
production account number or bucket name is what stops the same code from
running in staging (epic ``alpha-engine-config-I12488``), so each one is a
defect to be retired. Tracker: ``alpha-engine-config-I12492``; the flip to
enforcing is ``alpha-engine-config-I12504`` (S-16).

WHAT IT COUNTS
--------------
Per tracked file, the occurrences of two KINDS of literal:

* ``account`` -- the production AWS account number;
* ``bucket``  -- the production bucket names (research, crucible v2).

Only executable code and infrastructure files are scanned (by extension, or a
shebang on an extensionless file). Docs, changelogs and history are exempt:
``*.md``/``*.txt`` and friends never match the extension list, and the
``EXEMPT_DIRS`` / ``EXEMPT_STEMS`` below cover doc trees, changelog
fragments and audit evidence. Git history is never read -- the scan surface is
``git ls-files``. The registry the literals belong in
(``governance/environments.yaml`` and its packaged copy) is exempt by path.

SHRINK-ONLY
-----------
Each repo commits ``.hardcoded-env-refs-baseline.json``: the per-file counts
measured when it adopted the guard. That file is a CEILING:

* a file whose count goes UP (or a new file with any hit) is a ``new`` finding;
* a baseline entry that no longer reproduces -- the file's count went DOWN or
  the file is gone -- is a ``stale`` finding, so the fix that removes a literal
  also lowers the ceiling (``--update``), and the list only ever shrinks.

``--update`` only ever LOWERS the baseline: it drops entries that shrank and
never adds or raises one. There is deliberately no flag that raises a ceiling;
a reviewer sees any hand edit that does.

A repo with no baseline file is treated as having a ceiling of zero.

OBSERVE, THEN ENFORCE
---------------------
``--mode observe`` (the default) reports every finding -- stdout, GitHub
``::warning`` annotations and the job summary -- and exits 0. ``--mode
enforce`` reports identically and exits 1 on any finding. S-16
(``alpha-engine-config-I12504``) flips the call sites once the v2 baselines
reach zero. Nothing else differs between the modes (the ``guard_mode.py``
convention: observation and enforcement report identically).

COST
----
Stdlib only, no install. One ``git ls-files`` plus a regex pass over the
matching files: about a second on the largest fleet repo. Consumers run it as
one step inside an existing CI job (no new workflow), fetching this file the
same way ``provider-linkage-guard.yml`` fetches its script.

Usage::

    python3 hardcoded_env_refs.py --repo .                 # observe (default)
    python3 hardcoded_env_refs.py --repo . --mode enforce
    python3 hardcoded_env_refs.py --repo . --init          # first adoption
    python3 hardcoded_env_refs.py --repo . --update        # lower the ceiling
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

GUARD = "hardcoded-env-refs"
TRACKER = "alpha-engine-config-I12492"
ENFORCE_TRACKER = "alpha-engine-config-I12504"
BASELINE_NAME = ".hardcoded-env-refs-baseline.json"
SCHEMA_VERSION = 1

#: The production literals. Kept in one table so a new bucket is a row, not a
#: code change. Boundaries stop ``alpha-engine-research-foo`` (a different
#: bucket) and a longer digit run from matching.
PATTERNS: dict[str, re.Pattern[str]] = {
    "account": re.compile(r"(?<!\d)711398986525(?!\d)"),
    "bucket": re.compile(r"(?<![\w-])alpha-engine-(?:research|crucible-v2)(?![\w-])"),
}
KINDS: tuple[str, ...] = tuple(PATTERNS)

FIX_HINT = (
    "resolve it through nousergon_lib.env "
    '(env.resolve().account / .bucket("research") / .s3_uri(...)); '
    "the values live in nous-ergon-ops governance/environments.yaml (ENV-NO-LITERALS)"
)

#: Executable code and infrastructure. Docs (.md, .rst, .txt, .html), data
#: (.csv, .jsonl, .parquet) and notebooks are not in this list on purpose.
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".lua",
        ".sql",
        ".yml",
        ".yaml",
        ".json",
        ".jsonc",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".tf",
        ".tfvars",
        ".hcl",
        ".service",
        ".timer",
        ".socket",
        ".path",
        ".plist",
        ".env",
        ".j2",
        ".jinja",
        ".tpl",
        ".tmpl",
    }
)
CODE_BASENAMES = frozenset({"Dockerfile", "Makefile", "Procfile", "Justfile", "Containerfile"})
#: ``foo.yaml.example`` is scanned as ``foo.yaml``.
TEMPLATE_SUFFIXES = (".example", ".sample", ".dist", ".template")

#: Path segments whose subtree is docs, changelog or history.
EXEMPT_DIRS = frozenset(
    {"docs", "doc", "changelog", "changelog.d", "changelogs", "history", "archive", "archived", "audits"}
)
#: Upper-case basename stems that are changelogs/history wherever they sit
#: (``CHANGELOG.json``). Case-sensitive on purpose: ``.github/workflows/
#: changelog.yml`` is a workflow, not a changelog.
EXEMPT_STEMS = frozenset({"CHANGELOG", "HISTORY", "CHANGES"})

#: The one place the literals BELONG (ENV-REGISTRY), plus this guard's own
#: files (they must name what they look for).
EXEMPT_PATHS = frozenset(
    {
        "governance/environments.yaml",
        "src/nousergon_lib/environments.yaml",
        "scripts/hardcoded_env_refs.py",
        BASELINE_NAME,
    }
)

#: Larger files are generated data, not code a person edits.
MAX_BYTES = 5 * 1024 * 1024

Counts = dict[str, dict[str, int]]  # path -> kind -> occurrences


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------


def _effective_suffix(name: str) -> str:
    for t in TEMPLATE_SUFFIXES:
        if name.endswith(t) and len(name) > len(t):
            name = name[: -len(t)]
            break
    return PurePosixPath(name).suffix.lower()


def in_scope(path: str, head: bytes | None = None) -> bool:
    """Whether ``path`` (repo-relative, POSIX) is code/infra the guard reads.

    ``head`` is the file's first bytes, consulted only for an extensionless
    file (a shebang makes it a script).
    """
    if path in EXEMPT_PATHS:
        return False
    p = PurePosixPath(path)
    if any(part in EXEMPT_DIRS for part in p.parts[:-1]):
        return False
    if p.name.split(".", 1)[0] in EXEMPT_STEMS:
        return False
    if p.name in CODE_BASENAMES:
        return True
    suffix = _effective_suffix(p.name)
    if suffix:
        return suffix in CODE_EXTENSIONS
    return bool(head) and head.startswith(b"#!")


def tracked_files(repo: Path) -> list[str]:
    git = shutil.which("git") or "git"
    out = subprocess.run(
        [git, "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def count_text(text: str) -> dict[str, int]:
    return {k: n for k, rx in PATTERNS.items() if (n := len(rx.findall(text)))}


def hit_lines(text: str) -> list[tuple[int, str]]:
    """``(line_no, kind)`` for every matching line, for annotations."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for k, rx in PATTERNS.items():
            if rx.search(line):
                out.append((i, k))
    return out


def _read(repo: Path, rel: str) -> bytes | None:
    fp = repo / rel
    try:
        if not fp.is_file() or fp.is_symlink() or fp.stat().st_size > MAX_BYTES:
            return None
        return fp.read_bytes()
    except OSError:
        return None


def scan(repo: Path) -> Counts:
    counts: Counts = {}
    for rel in tracked_files(repo):
        p = PurePosixPath(rel)
        needs_head = not _effective_suffix(p.name) and p.name not in CODE_BASENAMES
        if not needs_head and not in_scope(rel):
            continue
        data = _read(repo, rel)
        if data is None or b"\0" in data[:8192]:
            continue
        if needs_head and not in_scope(rel, data[:2]):
            continue
        c = count_text(data.decode("utf-8", "replace"))
        if c:
            counts[rel] = c
    return counts


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def totals(counts: Counts) -> dict[str, int]:
    t = dict.fromkeys(KINDS, 0)
    files = dict.fromkeys(KINDS, 0)
    for c in counts.values():
        for k, n in c.items():
            t[k] += n
            files[k] += 1
    out: dict[str, int] = {}
    for k in KINDS:
        out[f"{k}_hits"] = t[k]
        out[f"{k}_files"] = files[k]
    out["files"] = len(counts)
    return out


def load_baseline(path: Path) -> Counts:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != SCHEMA_VERSION or doc.get("guard") != GUARD:
        raise SystemExit(f"{path}: not a {GUARD} baseline (schema_version {SCHEMA_VERSION})")
    files = doc.get("files")
    if not isinstance(files, dict):
        raise SystemExit(f"{path}: 'files' must be an object")
    out: Counts = {}
    for rel, c in files.items():
        if not isinstance(c, dict) or any(k not in KINDS or not isinstance(n, int) or n < 1 for k, n in c.items()):
            raise SystemExit(f"{path}: bad entry for {rel!r}: {c!r}")
        out[rel] = dict(c)
    return out


def render_baseline(counts: Counts) -> str:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "guard": GUARD,
        "tracker": TRACKER,
        "note": (
            "Shrink-only ceiling of hard-coded account/bucket literals per file. "
            "Never raise an entry; lower it with "
            "`python3 hardcoded_env_refs.py --repo . --update` when a literal is removed."
        ),
        "totals": totals(counts),
        "files": {rel: {k: counts[rel][k] for k in KINDS if k in counts[rel]} for rel in sorted(counts)},
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def compare(current: Counts, baseline: Counts) -> tuple[list[dict], list[dict]]:
    """``(new, stale)`` findings, each ``{path, kind, baseline, current}``."""
    new, stale = [], []
    for rel in sorted(set(current) | set(baseline)):
        cur, base = current.get(rel, {}), baseline.get(rel, {})
        for k in KINDS:
            c, b = cur.get(k, 0), base.get(k, 0)
            row = {"path": rel, "kind": k, "baseline": b, "current": c}
            if c > b:
                new.append(row)
            elif c < b:
                stale.append(row)
    return new, stale


def lowered(current: Counts, baseline: Counts) -> Counts:
    """The baseline with every entry lowered to what reproduces; never raised."""
    out: Counts = {}
    for rel, base in baseline.items():
        cur = current.get(rel, {})
        c = {k: min(n, cur.get(k, 0)) for k, n in base.items()}
        c = {k: n for k, n in c.items() if n}
        if c:
            out[rel] = c
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def report(repo: Path, current: Counts, new: list[dict], stale: list[dict], mode: str) -> list[str]:
    level = "error" if mode == "enforce" else "warning"
    lines: list[str] = []
    t = totals(current)
    lines.append(
        f"{GUARD} ({mode}): {t['account_hits']} account / {t['bucket_hits']} bucket "
        f"literal(s) in {t['files']} file(s); {len(new)} new, {len(stale)} stale vs baseline."
    )
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    annotated = 0
    for row in new:
        msg = (
            f"{row['path']}: {row['kind']} literal count {row['baseline']} -> {row['current']} "
            f"(new hard-coded environment reference) -- {FIX_HINT}"
        )
        lines.append("NEW   " + msg)
        if in_actions and annotated < 10:
            data = _read(repo, row["path"]) or b""
            lns = [n for n, k in hit_lines(data.decode("utf-8", "replace")) if k == row["kind"]]
            loc = f",line={lns[0]}" if lns else ""
            print(f"::{level} file={row['path']}{loc},title={GUARD}::{_esc(msg)}")
            annotated += 1
    for row in stale:
        lines.append(
            f"STALE {row['path']}: {row['kind']} baseline {row['baseline']} but {row['current']} now "
            "-- lower the ceiling: `python3 hardcoded_env_refs.py --repo . --update`"
        )
    if in_actions and stale:
        print(
            f"::{level} title={GUARD}::{len(stale)} baseline entr(y/ies) no longer reproduce; "
            f"run --update to lower {BASELINE_NAME} (shrink-only)"
        )
    if mode == "observe" and (new or stale):
        lines.append(f"observe mode: reporting only; enforcement is {ENFORCE_TRACKER} (S-16).")
    return lines


def write_summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"### {GUARD} ({TRACKER})\n\n```\n")
            fh.write("\n".join(lines[:200]) + "\n```\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".", help="repo root (default .)")
    ap.add_argument("--baseline", help=f"baseline path (default <repo>/{BASELINE_NAME})")
    ap.add_argument("--mode", choices=("observe", "enforce"), default="observe")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--init", action="store_true", help="write the first baseline; refuses if one exists")
    g.add_argument("--update", action="store_true", help="lower the baseline to what reproduces; never raises")
    ap.add_argument("--json", action="store_true", help="print the current scan as JSON")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    bpath = Path(args.baseline) if args.baseline else repo / BASELINE_NAME
    current = scan(repo)

    if args.json:
        print(json.dumps({"totals": totals(current), "files": current}, indent=2, sort_keys=True))
        return 0
    if args.init:
        if bpath.exists():
            print(f"{bpath} exists; use --update (shrink-only)", file=sys.stderr)
            return 2
        bpath.write_text(render_baseline(current))
        t = totals(current)
        print(f"wrote {bpath}: {t}")
        return 0

    baseline = load_baseline(bpath)
    if args.update:
        nb = lowered(current, baseline)
        bpath.write_text(render_baseline(nb))
        print(f"wrote {bpath}: {totals(nb)}")
        return 0

    new, stale = compare(current, baseline)
    lines = report(repo, current, new, stale, args.mode)
    print("\n".join(lines))
    write_summary(lines)
    if args.mode == "enforce" and (new or stale):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
