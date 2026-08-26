#!/usr/bin/env python3
"""The `pr-prose-gate-guard` predicate — a PR that says "do not merge" in
prose while nothing machine-readable says so.

WHY THIS EXISTS
===============
alpha-engine-config-I8683. On 2026-08-26 ``crucible-predictor-PR328`` was
merged under a blanket "merge PRs when ready" instruction. It passed every
machine-readable gate in ``auto-merge-policy.md`` §2 Gate A — not a draft,
ZERO labels, ``mergeable_state`` CLEAN, 12/12 checks green, no unresolved
CodeQL review threads — while its body opened with:

    **DRAFT — validated on synthetic data; the closes-#1524 real-cohort run
    is DATA-GATED.** ... **Do not merge until that real-cohort run reports.**

The blocking condition existed ONLY in prose. Every evaluator that decides
whether a PR is landable — the groom sweep, an auto-merge lane, an agent
session, a human reading ``gh pr list`` — reads the draft flag, the labels
and the checks. None of them read the body.

THE PREDICATE IS A CONJUNCTION
==============================
Blocking prose alone is not a failure. Dependabot bodies, PR templates and
retrospectives contain these phrases legitimately. It fails only on:

    not a draft  AND  no ``gate:*`` label  AND  a blocking phrase in the body

A draft is already unmergeable and a ``gate:*`` label already blocks the
merge, so in both cases the prose and the machine-readable state already
agree — which is all this guard asks for.

THE PHRASE LIST IS A FLOOR, NOT A CLAIM OF COMPLETENESS
======================================================
It cannot enumerate every way a human writes "not yet". It catches the
shapes observed in the fleet, and the failure message says what to do rather
than asserting the list is exhaustive. A blocking condition phrased in prose
nobody enumerated remains possible; that residual is itself the argument for
labelling gates, which is what the message tells authors to do.
"""

from __future__ import annotations

import json
import re
import sys

# Observed shapes. Matched against a NORMALISED body (see `_normalise`).
#
# A bare "draft" is anchored to the start of a line: the word appears in
# ordinary prose ("drafted the schema", "draft the migration first") and in
# PR templates, and only a leading DRAFT marker is a merge-blocking claim.
BLOCKING_PATTERNS: tuple[str, ...] = (
    r"do not merge",
    r"don'?t merge",
    r"not for merge",
    r"do not land",
    r"hold (this|off)\b",
    r"^draft\b[^a-z0-9]",
    r"data-gated",
)

_COMPILED = tuple(re.compile(p, re.MULTILINE) for p in BLOCKING_PATTERNS)

# Leading markdown furniture a blocking line is usually wrapped in:
# blockquote markers, emphasis, headings, list bullets, whitespace. Stripped
# per line so `> **DRAFT — ...` still matches an anchored `^draft`.
_LEAD = re.compile(r"^[\s>*_#\-+]+", re.MULTILINE)


def _normalise(body: str) -> str:
    """Lowercase, and strip leading markdown furniture from every line."""
    return _LEAD.sub("", (body or "").lower())


def find_blocking_phrase(body: str) -> str | None:
    """The first blocking pattern present in ``body``, or ``None``."""
    normalised = _normalise(body)
    for pattern, compiled in zip(BLOCKING_PATTERNS, _COMPILED):
        if compiled.search(normalised):
            return pattern
    return None


def evaluate(body: str, labels: list[str], is_draft: bool) -> tuple[bool, str]:
    """``(passes, human_reason)`` for one pull request.

    Fails ONLY on the conjunction. Every early return is a pass with a stated
    reason, so a green result is never silent about which arm produced it.
    """
    if is_draft:
        return True, (
            "Draft PR — a draft cannot merge, so its prose and its "
            "machine-readable state already agree. This guard re-runs on "
            "ready_for_review."
        )
    gates = sorted(lbl for lbl in labels if str(lbl).startswith("gate:"))
    if gates:
        return True, (
            f"PR carries a gate:* label ({', '.join(gates)}) — the merge "
            "block is machine-readable, which is all this guard asks for."
        )
    hit = find_blocking_phrase(body)
    if hit is None:
        return True, "No unlabelled merge-blocking prose found."
    return False, (
        f"This PR is NOT a draft and carries NO gate:* label, but its body "
        f"contains a merge-blocking statement (matched: /{hit}/). Every "
        f"evaluator that decides whether a PR is landable — the groom sweep, "
        f"an auto-merge lane, an agent session, a human reading 'gh pr list' "
        f"— reads the draft flag, the labels and the checks. None of them "
        f"read the body. Fix it one of two ways: (1) if the block is real, "
        f"apply the correct gate:* label with a 'Verified-when:' predicate "
        f"that will clear it, per gate-taxonomy-policy; or (2) if the block "
        f"is stale, delete the prose. Precedent: crucible-predictor-PR328, "
        f"merged 2026-08-26 while its body said 'Do not merge' "
        f"(alpha-engine-config-I8683)."
    )


def main(argv: list[str]) -> int:
    """Read one JSON object — ``{body, labels, isDraft}`` — from the path in
    ``argv[1]``, print the reason, exit 0 (pass) or 1 (fail).

    A malformed or unreadable input fails CLOSED: this check can only ever
    BLOCK a merge, so refusing to read leaves the PR exactly as blocked as an
    outright failure would, and never silently green.
    """
    if len(argv) != 2:
        print("::error::usage: pr_prose_gate_predicate.py <pr.json>")
        return 1
    try:
        with open(argv[1], encoding="utf-8") as handle:
            payload = json.load(handle)
        body = payload.get("body") or ""
        labels = [
            (lbl.get("name") if isinstance(lbl, dict) else str(lbl))
            for lbl in (payload.get("labels") or [])
        ]
        is_draft = bool(payload.get("isDraft", payload.get("draft", False)))
    except Exception as exc:  # noqa: BLE001 — fail closed, with the reason
        print(f"::error::could not read {argv[1]!r}: {exc}. Failing closed.")
        return 1

    passes, reason = evaluate(body, [lbl for lbl in labels if lbl], is_draft)
    if passes:
        print(f"::notice::{reason}")
        return 0
    print(f"::error::{reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
