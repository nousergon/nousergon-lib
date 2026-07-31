"""The action-pin guard must name the wrong-type SHA, not just reject it.

Every bad pin this guard exists for was a **real object in the correct repo**
(``crucible-predictor`` PR #433, 2026-07-31): five blob SHAs read out of
``GET /contents/{path}``'s ``.sha``, and one annotated-tag-object SHA. A guard
that says "does not resolve" sends the next person back to the same endpoint
that produced the value. The failure-message assertions below are therefore
load-bearing, not cosmetic.

The other half is the fail-loud contract. A transport fault must surface as
exit 2 with nothing claimed, never as a clean run -- an unreachable API is the
one case where "no findings" and "checked nothing" look identical from outside.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_action_pins.py"

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("action_pin_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["action_pin_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ok(_repo, _sha):
    return None


def _write(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / ".github" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body)
    return p


# ── the wrong-type diagnosis ─────────────────────────────────────────────────


def test_blob_sha_is_named_as_a_blob_and_points_at_the_right_endpoint(guard, monkeypatch):
    """The five-of-six case: `.sha` from the contents endpoint is the file, not the commit."""
    monkeypatch.setattr(
        guard,
        "_get",
        lambda path, token: (200, {}) if path.startswith("repos/o/r/git/blobs/") else (404, None),
    )
    problem = guard.classify_ref("o/r", SHA_A)
    assert "BLOB SHA" in problem
    assert "contents/{path}" in problem
    assert "commits/{ref}" in problem


def test_annotated_tag_object_is_dereferenced_to_the_commit_to_use(guard, monkeypatch):
    """The sixth: `v1`'s tag object, where the fix is the SHA it points to."""

    def fake(path, token):
        if path.startswith("repos/o/r/git/tags/"):
            return 200, {"tag": "v1", "object": {"sha": SHA_B, "type": "commit"}}
        return 404, None

    monkeypatch.setattr(guard, "_get", fake)
    problem = guard.classify_ref("o/r", SHA_A)
    assert "ANNOTATED TAG OBJECT" in problem
    assert "v1" in problem
    assert SHA_B in problem, "must name the commit to use, not merely reject the tag"


def test_a_commit_sha_is_accepted(guard, monkeypatch):
    monkeypatch.setattr(guard, "_get", lambda path, token: (200, {"sha": SHA_A}))
    assert guard.classify_ref("o/r", SHA_A) is None


def test_object_absent_entirely_says_so_without_guessing(guard, monkeypatch):
    monkeypatch.setattr(guard, "_get", lambda path, token: (404, None))
    assert "does not exist" in guard.classify_ref("o/r", SHA_A)


# ── ref-shape rules ──────────────────────────────────────────────────────────


def test_mutable_ref_is_a_finding(guard):
    problem = guard.check_uses("nousergon/nousergon-lib/.github/workflows/x.yml@main", _ok)
    assert "mutable ref" in problem


def test_missing_ref_is_a_finding(guard):
    assert "not pinned" in guard.check_uses("actions/checkout", _ok)


def test_local_and_docker_refs_are_skipped(guard):
    assert guard.check_uses("./.github/actions/thing", _ok) is None
    assert guard.check_uses("docker://alpine:3.20", _ok) is None


def test_reusable_workflow_path_is_stripped_before_resolving(guard):
    """The repo to query is `owner/repo`, never `owner/repo/.github/workflows/x.yml`."""
    seen = []

    def resolver(repo, sha):
        seen.append((repo, sha))
        return None

    guard.check_uses(f"nousergon/nousergon-lib/.github/workflows/x.yml@{SHA_A}", resolver)
    assert seen == [("nousergon/nousergon-lib", SHA_A)]


# ── directory walk ───────────────────────────────────────────────────────────


def test_finding_is_anchored_to_the_line_it_is_on(guard, tmp_path):
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        "  build:\n    runs-on: ubuntu-latest\n    steps:\n"
        f"      - uses: actions/checkout@{SHA_A}\n",
    )
    findings = guard.check_dir(tmp_path / ".github" / "workflows", lambda r, s: "is a BLOB SHA")
    assert len(findings) == 1
    _path, line, value, _problem = findings[0]
    assert line == 8, "annotation must land on the uses: line so it renders on the diff"
    assert value == f"actions/checkout@{SHA_A}"


def test_job_level_reusable_workflow_calls_are_checked(guard, tmp_path):
    """A thin caller has no `steps:` at all -- its only `uses:` is on the job."""
    _write(
        tmp_path,
        "guard.yml",
        "name: G\non:\n  pull_request:\njobs:\n"
        f"  g:\n    uses: nousergon/nousergon-lib/.github/workflows/g.yml@{SHA_A}\n",
    )
    findings = guard.check_dir(tmp_path / ".github" / "workflows", lambda r, s: "nope")
    assert len(findings) == 1


def test_each_distinct_ref_is_resolved_once(guard, tmp_path):
    """`actions/checkout` appears in most jobs; the API should see it once."""
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n"
        f"  b:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )
    calls = []
    guard.check_dir(tmp_path / ".github" / "workflows", lambda r, s: calls.append((r, s)) or None)
    assert len(calls) == 1


def test_unparseable_yaml_is_left_to_check_workflow_shape(guard, tmp_path):
    _write(tmp_path, "broken.yml", "jobs:\n  - [unbalanced\n")
    assert guard.check_dir(tmp_path / ".github" / "workflows", _ok) == []


# ── fail-loud contract ───────────────────────────────────────────────────────


def test_transport_failure_exits_2_and_claims_nothing(guard, tmp_path, monkeypatch, capsys):
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )

    def boom(_repo, _sha, _token=None):
        raise guard.ResolutionError("rate limit exceeded")

    monkeypatch.setattr(guard, "classify_ref", boom)
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.chdir(tmp_path)

    assert guard.main([]) == 2
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "none are verified" in out


def test_clean_tree_exits_0(guard, tmp_path, monkeypatch):
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )
    monkeypatch.setattr(guard, "classify_ref", lambda repo, sha, token=None: None)
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.chdir(tmp_path)
    assert guard.main([]) == 0


def test_findings_exit_1_with_a_github_annotation(guard, tmp_path, monkeypatch, capsys):
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )
    monkeypatch.setattr(guard, "classify_ref", lambda repo, sha, token=None: "is a BLOB SHA")
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.chdir(tmp_path)

    assert guard.main([]) == 1
    assert "::error file=" in capsys.readouterr().out


def test_missing_workflow_dir_exits_2(guard, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert guard.main([]) == 2
