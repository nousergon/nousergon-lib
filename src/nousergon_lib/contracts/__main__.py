"""CLI veneer over the slot contracts (precursor to the ``ne validate`` verb).

Usage:
    python -m nousergon_lib.contracts validate <slot> <path.json>

Exit 0 on conformance; exit 1 with one error per line on violation.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import SCHEMA_VERSIONS, SLOT_SCHEMAS, conformance_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nousergon_lib.contracts")
    sub = parser.add_subparsers(dest="command", required=True)
    p_val = sub.add_parser("validate", help="validate a slot artifact against its contract")
    p_val.add_argument("slot", choices=sorted(SLOT_SCHEMAS), help="slot artifact name")
    p_val.add_argument("path", help="path to the JSON artifact")
    args = parser.parse_args(argv)

    with open(args.path, encoding="utf-8") as f:
        payload = json.load(f)
    errors = conformance_errors(args.slot, payload)
    if errors:
        print(
            f"FAIL: {args.path} violates {args.slot} contract "
            f"v{SCHEMA_VERSIONS[args.slot]} ({len(errors)} error(s)):",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"OK: {args.path} conforms to {args.slot} contract v{SCHEMA_VERSIONS[args.slot]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
