"""Regression coverage for rehearsal run manifests."""

from __future__ import annotations

import json
from pathlib import Path

from nousergon_lib import contracts
from nousergon_lib.run_manifest import LocalDirManifestSink, UnitRun, run_unit


def test_rehearsal_trigger_writes_a_conforming_manifest(tmp_path: Path) -> None:
    def body(ctx: UnitRun) -> None:
        ctx.rows_in = 1

    result = run_unit(
        "D19",
        body,
        sink=LocalDirManifestSink(root=str(tmp_path)),
        trigger="rehearsal",
        trading_day="2026-11-21",
        log_location="rehearsal.log",
        code_sha="a" * 40,
    )

    assert result.key is not None
    manifest = json.loads((tmp_path / result.key).read_text())
    assert manifest["trigger"] == "rehearsal"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []
