"""Pre-merge cost gate: price the diff, refuse spend with no budget line.

The grading logic and the prefix-only cost SSoT it grades against, packaged so
every repo in the fleet runs the SAME gate from the library it already pins —
no cross-repo fetch, no credential, and identical behaviour for a public and a
private caller.

Console entry: ``nousergon-cost-gate``. See :mod:`nousergon_lib.cost_gate.cli`
for the flags and the exit-code contract, :mod:`nousergon_lib.cost_gate.grade`
for what each class grades, and ``data/README.md`` for where the packaged
documents come from.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

from nousergon_lib.cost_gate.grade import (
    MergeBaseUnreachable,
    SsotError,
    added_lines,
    added_with_lines,
    approved_prefixes,
    cron_interval_minutes,
    findings,
    git_diff,
    load_ssot,
    merge_base,
)
from nousergon_lib.cost_gate.resource_map import (
    CfnResource,
    Intrinsic,
    ResourceMapError,
    parse_cfn_resources,
    resolve_target_service,
)
from nousergon_lib.cost_gate.resource_map import load as load_resource_map

__all__ = [
    "CfnResource",
    "Intrinsic",
    "MergeBaseUnreachable",
    "ResourceMapError",
    "SsotError",
    "added_lines",
    "added_with_lines",
    "approved_prefixes",
    "cron_interval_minutes",
    "findings",
    "git_diff",
    "load_resource_map",
    "load_ssot",
    "merge_base",
    "parse_cfn_resources",
    "resolve_target_service",
]
