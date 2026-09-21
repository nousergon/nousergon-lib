"""``nousergon-cost-gate`` — the pre-merge cost gate, as a console entry.

EXIT CODES ARE THE CONTRACT::

    0  graded, no findings
    1  graded, findings
    2  COULD NOT GRADE, with the cause named

2 is not 1. "I could not grade this" and "I graded it and it is unbudgeted"
are different states with different remedies, and a required check that
conflates them teaches the reader to ignore both.

``--warn-only`` downgrades *findings* (1 -> 0) for a repo mid-adoption. It
deliberately does NOT downgrade 2: a gate that could not run at all is a
broken control, not a soft finding, and silencing that is how a check becomes
warn-only forever.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nousergon_lib.cost_gate import grade
from nousergon_lib.cost_gate import resource_map as resource_map_mod


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nousergon-cost-gate",
        description=(
            "Price the diff: refuse a change that adds spend with no budget "
            "line. Grades against the prefix-only cost SSoT packaged with "
            "nousergon-lib unless --ssot names another."
        ),
    )
    ap.add_argument("--base", default="origin/main", help="the base ref (default: origin/main)")
    ap.add_argument("--head", default="HEAD", help="the head ref (default: HEAD)")
    ap.add_argument(
        "--repo-visibility", choices=("public", "private"), default="private",
        help=(
            "PRIVATE fails closed: Actions-minute crons are graded unless the "
            "caller states the repo is public, where minutes are free and "
            "unlimited"
        ),
    )
    ap.add_argument(
        "--warn-only", action="store_true",
        help="report findings and exit 0. Does NOT downgrade exit 2.",
    )
    ap.add_argument(
        "--ssot",
        help=(
            "a budget-prefix document to grade against, instead of the one "
            "packaged with this library"
        ),
    )
    ap.add_argument(
        "--resource-map",
        help="a resource-map document, instead of the one packaged with this library",
    )
    ap.add_argument(
        "--root", default=".",
        help="the head checkout the diff applies to; schedules are graded from it",
    )
    ap.add_argument(
        "--diff-file", help="read a unified diff from a file instead of from git",
    )
    return ap


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    doc = grade.load_ssot(args.ssot) if args.ssot else grade.load_ssot()
    rmap = resource_map_mod.load(args.resource_map) if args.resource_map else resource_map_mod.load()

    try:
        diff = (
            Path(args.diff_file).read_text()
            if args.diff_file
            else grade.git_diff(args.base, args.head)
        )
    except grade.MergeBaseUnreachable as exc:
        print(f"::error title=pre-merge cost gate could not grade this diff::{exc}")
        return 2

    issues, counts = grade.findings(
        diff, doc,
        resource_map=rmap,
        root=args.root,
        repo_visibility=args.repo_visibility,
    )

    budgeted, free = grade.approved_prefixes(doc)
    source = args.ssot or "the map packaged with nousergon-lib"
    print(
        f"Pre-merge cost gate — {len(budgeted)} budgeted prefix(es), "
        f"{len(free)} declared free, "
        f"{len(rmap['resource_types'])} mapped resource type(s), from {source}."
    )
    print(
        f"  graded  {counts['clients']} client construction(s), "
        f"{counts['actions']} entitlement string(s), "
        f"{counts['resources']} always-on resource(s), "
        f"{counts['schedules']} schedule target(s), "
        f"over {counts['lines']} added line(s)"
    )
    # Counted, never silently: a class this gate could not grade must not read
    # the same as a class it graded and passed.
    if counts["ungraded_schedules"] or counts["ungraded_targets"]:
        print(
            f"  NOT GRADED  {counts['ungraded_schedules']} schedule(s) whose "
            f"definition could not be read, {counts['ungraded_targets']} target(s) "
            f"whose billing service could not be resolved. These are UNKNOWN, "
            f"not approved."
        )

    if not issues:
        print("\n0 findings. Every service this diff reaches has a budget line.")
        return 0

    print(f"\n{len(issues)} finding(s):\n")
    for f in issues:
        print(f"  x {f}")
    print("\n" + grade.REMEDY)
    if args.warn_only:
        print(
            "\n--warn-only: exiting 0 with findings above. This repo is "
            "mid-adoption; a gate left warn-only forever is indistinguishable "
            "from not having it."
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
