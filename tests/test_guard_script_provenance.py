"""Every reusable guard that runs a FLOATING-ref script records what it ran.

alpha-engine-config-I9118, option (c): the script checkout stays unpinned so a
fix reaches every consumer without a pin bump, and each run names the
nousergon-lib commit and the script's sha256. Without that record a merge to
main re-verdicts the whole fleet with nothing saying why (I9111).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
SCRIPT_CHECKOUT_PATH = "_nousergon-lib-script"


def _floating_script_guards() -> list[Path]:
    found = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        if SCRIPT_CHECKOUT_PATH in wf.read_text():
            found.append(wf)
    return found


def test_the_floating_script_guards_are_found():
    names = {p.name for p in _floating_script_guards()}
    assert {
        "action-pin-guard.yml",
        "merge-group-required-check-guard.yml",
        "openrouter-guard.yml",
        "provider-linkage-guard.yml",
    } <= names


@pytest.mark.parametrize("wf", _floating_script_guards(), ids=lambda p: p.name)
def test_provenance_is_recorded_between_checkout_and_scan(wf: Path):
    steps = yaml.safe_load(wf.read_text())["jobs"]["guard"]["steps"]
    checkout = next(
        i for i, s in enumerate(steps)
        if (s.get("with") or {}).get("path") == SCRIPT_CHECKOUT_PATH
    )
    scan = next(
        i for i, s in enumerate(steps)
        if f"{SCRIPT_CHECKOUT_PATH}/scripts/" in (s.get("run") or "")
        and "sha256sum" not in (s.get("run") or "")
    )
    record = [
        i for i, s in enumerate(steps)
        if "sha256sum" in (s.get("run") or "") and "rev-parse HEAD" in (s.get("run") or "")
    ]
    assert record, f"{wf.name}: no step records the script commit + sha256"
    assert checkout < record[0] < scan, f"{wf.name}: provenance must be recorded after checkout, before the scan"
    run = steps[record[0]]["run"]
    assert "GITHUB_STEP_SUMMARY" in run and "::notice" in run
    # The recorded script is the one the scan executes.
    script = steps[record[0]]["env"]["SCRIPT"]
    assert script.split("/")[-1] in steps[scan]["run"]
