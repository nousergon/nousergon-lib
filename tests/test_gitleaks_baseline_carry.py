"""Pins the squash-surviving baseline match in gitleaks-scan.yml (alpha-engine-config-I12293).

gitleaks matches a baseline record on its commit (Commit, Date, Author,
Message and line). A record accepted at a branch commit therefore stopped
matching once the PR was squash-merged, and main went red. The detect step now
builds an effective baseline: the committed records, plus every current finding
whose commit-independent identity (rule, file, start and end column, entropy)
equals a committed record's.

The helper is executed straight out of the workflow file, so these tests cover
exactly the text the workflow runs. They do not run gitleaks itself; the
end-to-end squash repro is recorded in the PR body.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gitleaks-scan.yml"


def _detect_run() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text())
    steps = doc["jobs"]["scan"]["steps"]
    return next(s for s in steps if s.get("id") == "detect")["run"]


def _helper_source() -> str:
    m = re.search(r"<<'PY'\n(.*?)\nPY\n", _detect_run(), re.S)
    assert m, "baseline-carry heredoc not found in the detect step"
    return m.group(1)


def _finding(commit: str, line: int, *, rule="generic-api-key", file="tests/fx.py",
             start_col=16, end_col=40, entropy=3.6168747) -> dict:
    return {
        "RuleID": rule, "Description": "d", "StartLine": line, "EndLine": line,
        "StartColumn": start_col, "EndColumn": end_col, "Match": "REDACTED",
        "Secret": "REDACTED", "File": file, "SymlinkFile": "", "Commit": commit,
        "Entropy": entropy, "Author": "a", "Email": "e", "Date": "2026-10-13T22:43:47Z",
        "Message": "m", "Tags": [], "Fingerprint": f"{commit}:{file}:{rule}:{line}",
    }


def _carry(tmp_path: Path, baseline, findings) -> tuple[subprocess.CompletedProcess, list]:
    script = tmp_path / "carry.py"
    script.write_text(_helper_source())
    b, r, out = tmp_path / "baseline.json", tmp_path / "report.json", tmp_path / "eff.json"
    b.write_text(json.dumps(baseline))
    r.write_text(json.dumps(findings))
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(b), str(r), str(out)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    eff = json.loads(out.read_text()) if out.exists() else []
    return proc, eff


def test_squash_commit_is_carried_from_branch_record(tmp_path):
    """The fa1edc12 case: accepted at branch commit 211fc175, re-found at the squash commit, at a moved line."""
    branch = _finding("211fc1758740872d692fafd63d3c597bb612d6c2", 124)
    squash = _finding("fa1edc12dbd4cdccb0c98f4dbd6a54ca489005c1", 131)
    proc, eff = _carry(tmp_path, [branch], [squash])
    assert proc.returncode == 0, proc.stderr
    assert eff == [branch, squash]
    assert "1 carried forward" in proc.stdout
    assert "fa1edc12dbd4" in proc.stdout and "211fc1758740" in proc.stdout


@pytest.mark.parametrize("change", [
    {"rule": "github-pat"},
    {"file": "src/other.py"},
    {"end_col": 41},
    {"start_col": 15, "end_col": 39},
    {"entropy": 3.9},
])
def test_finding_with_a_different_identity_is_not_carried(tmp_path, change):
    """A different secret or place stays new, so the real scan still fails on it."""
    baseline = [_finding("211fc175", 124)]
    proc, eff = _carry(tmp_path, baseline, [_finding("fa1edc12", 124, **change)])
    assert proc.returncode == 0, proc.stderr
    assert eff == baseline
    assert "0 carried forward" in proc.stdout


def test_exact_match_is_left_to_gitleaks(tmp_path):
    rec = _finding("211fc175", 124)
    proc, eff = _carry(tmp_path, [rec], [dict(rec)])
    assert proc.returncode == 0, proc.stderr
    assert eff == [rec]


def test_record_missing_an_identity_field_carries_nothing(tmp_path):
    rec = _finding("211fc175", 124)
    del rec["Entropy"]
    proc, eff = _carry(tmp_path, [rec], [_finding("fa1edc12", 124)])
    assert proc.returncode == 0, proc.stderr
    assert eff == [rec]


def test_empty_report_is_accepted(tmp_path):
    rec = _finding("211fc175", 124)
    proc, eff = _carry(tmp_path, [rec], None)
    assert proc.returncode == 0, proc.stderr
    assert eff == [rec]


@pytest.mark.parametrize("bad", [{"a": 1}, "x"])
def test_malformed_baseline_fails_closed(tmp_path, bad):
    proc, _ = _carry(tmp_path, bad, [])
    assert proc.returncode != 0


def test_detect_step_shape():
    run = _detect_run()
    pass1, _, rest = run.partition("<<'PY'")
    # Pass 1 is redacted and never fails on findings; its report stays on the runner.
    assert re.search(r"gitleaks detect --source \. --redact --exit-code 0", pass1)
    assert '"$RUNNER_TEMP/gitleaks-unfiltered.json"' in pass1
    # The enforcing scan reads the effective baseline and keeps --redact and SARIF.
    assert 'BASELINE_ARGS="--baseline-path $RUNNER_TEMP/gitleaks-effective-baseline.json"' in rest
    final = rest.split("END baseline-carry", 1)[1]
    assert "gitleaks detect --source . $BASELINE_ARGS" in final
    assert "--redact" in final and "--report-format sarif" in final
    assert "--exit-code" not in final
