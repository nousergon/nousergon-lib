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
import os
import shutil
import subprocess  # noqa: S404 -- runs this repo's own script under the test interpreter
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_action_pins.py"

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    return _load("action_pin_guard", _SCRIPT)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ok(_repo: str, _sha: str) -> None:
    return None


def _write(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / ".github" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body)
    return p


# ── the wrong-type diagnosis ─────────────────────────────────────────────────


def test_blob_sha_is_named_as_a_blob_and_points_at_the_right_endpoint(guard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_annotated_tag_object_is_dereferenced_to_the_commit_to_use(guard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sixth: `v1`'s tag object, where the fix is the SHA it points to."""

    def fake(path: str, token: str | None) -> tuple[int, object]:
        if path.startswith("repos/o/r/git/tags/"):
            return 200, {"tag": "v1", "object": {"sha": SHA_B, "type": "commit"}}
        return 404, None

    monkeypatch.setattr(guard, "_get", fake)
    problem = guard.classify_ref("o/r", SHA_A)
    assert "ANNOTATED TAG OBJECT" in problem
    assert "v1" in problem
    assert SHA_B in problem, "must name the commit to use, not merely reject the tag"


def test_a_commit_sha_is_accepted(guard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "_get", lambda path, token: (200, {"sha": SHA_A}))
    assert guard.classify_ref("o/r", SHA_A) is None


def test_object_absent_from_a_visible_repo_says_so_without_guessing(guard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repo readable, object genuinely gone — the only case that earns this wording."""
    monkeypatch.setattr(
        guard,
        "_get",
        lambda path, token: (200, {"full_name": "o/r"}) if path == "repos/o/r" else (404, None),
    )
    assert "does not exist" in guard.classify_ref("o/r", SHA_A)


def test_invisible_repo_is_not_reported_as_a_missing_object(guard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """GITHUB_TOKEN reads only the repo running the workflow.

    A `uses:` into a different PRIVATE repo 404s at every object endpoint, which
    is byte-identical to a deleted SHA. Reporting it as a bad pin sends someone
    hunting a SHA that is correct — the fix is a token, not a repin.
    """
    monkeypatch.setattr(guard, "_get", lambda path, token: (404, None))
    problem = guard.classify_ref("o/private", SHA_A)
    assert "not visible to this token" in problem
    assert "does not exist" not in problem


# ── ref-shape rules ──────────────────────────────────────────────────────────


def test_mutable_ref_is_a_finding(guard: ModuleType) -> None:
    problem = guard.check_uses("nousergon/nousergon-lib/.github/workflows/x.yml@main", _ok)
    assert "mutable ref" in problem


def test_missing_ref_is_a_finding(guard: ModuleType) -> None:
    assert "not pinned" in guard.check_uses("actions/checkout", _ok)


def test_local_and_docker_refs_are_skipped(guard: ModuleType) -> None:
    assert guard.check_uses("./.github/actions/thing", _ok) is None
    assert guard.check_uses("docker://alpine:3.20", _ok) is None


def test_reusable_workflow_path_is_stripped_before_resolving(guard: ModuleType) -> None:
    """The repo to query is `owner/repo`, never `owner/repo/.github/workflows/x.yml`."""
    seen: list[tuple[str, str]] = []

    def resolver(repo: str, sha: str) -> None:
        seen.append((repo, sha))
        return None

    guard.check_uses(f"nousergon/nousergon-lib/.github/workflows/x.yml@{SHA_A}", resolver)
    assert seen == [("nousergon/nousergon-lib", SHA_A)]


# ── directory walk ───────────────────────────────────────────────────────────


def test_finding_is_anchored_to_the_line_it_is_on(guard: ModuleType, tmp_path: Path) -> None:
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


def test_job_level_reusable_workflow_calls_are_checked(guard: ModuleType, tmp_path: Path) -> None:
    """A thin caller has no `steps:` at all -- its only `uses:` is on the job."""
    _write(
        tmp_path,
        "guard.yml",
        "name: G\non:\n  pull_request:\njobs:\n"
        f"  g:\n    uses: nousergon/nousergon-lib/.github/workflows/g.yml@{SHA_A}\n",
    )
    findings = guard.check_dir(tmp_path / ".github" / "workflows", lambda r, s: "nope")
    assert len(findings) == 1


def test_each_distinct_ref_is_resolved_once(guard: ModuleType, tmp_path: Path) -> None:
    """`actions/checkout` appears in most jobs; the API should see it once."""
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n"
        f"  b:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )
    calls: list[tuple[str, str]] = []
    guard.check_dir(tmp_path / ".github" / "workflows", lambda r, s: calls.append((r, s)) or None)
    assert len(calls) == 1


def test_unparseable_yaml_is_left_to_check_workflow_shape(guard: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path, "broken.yml", "jobs:\n  - [unbalanced\n")
    assert guard.check_dir(tmp_path / ".github" / "workflows", _ok) == []


# ── fail-loud contract ───────────────────────────────────────────────────────


def test_transport_failure_exits_2_and_claims_nothing(guard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )

    def boom(_repo: str, _sha: str, _token: str | None = None) -> str | None:
        raise guard.ResolutionError("rate limit exceeded")

    monkeypatch.setattr(guard, "classify_ref", boom)
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.chdir(tmp_path)

    assert guard.main([]) == 2
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "none are verified" in out


def test_clean_tree_exits_0(guard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_findings_exit_1_with_a_github_annotation(guard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_missing_workflow_dir_exits_2(guard: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert guard.main([]) == 2


# ── short and long hexadecimal refs ──────────────────────────────────────────


def test_short_hex_ref_is_diagnosed_as_an_abbreviated_sha(guard: ModuleType) -> None:
    problem = guard.check_uses("actions/checkout@3d3c42e", _ok)
    assert problem is not None
    assert "7-character hexadecimal" in problem
    assert "full 40-hex commit SHA" in problem


def test_long_hex_ref_is_diagnosed_as_longer_than_a_sha(guard: ModuleType) -> None:
    problem = guard.check_uses(f"actions/checkout@{SHA_A}0", _ok)
    assert problem is not None
    assert "41-character hexadecimal" in problem


def test_hex_shaped_refs_never_reach_the_resolver(guard: ModuleType) -> None:
    seen: list[tuple[str, str]] = []
    guard.check_uses("actions/checkout@abc1234", lambda r, s: seen.append((r, s)))
    assert seen == []


# ── the migration must never loosen a verdict ────────────────────────────────

_LEGACY = Path(__file__).resolve().parent / "_legacy_check_action_pins.py"

_GetResult = tuple[int, object]
_Getter = Callable[[str, str | None], _GetResult]


def _scenarios() -> dict[str, _Getter]:
    """API worlds, keyed by what the target repo is hiding or showing."""

    def commit(path: str, token: str | None) -> _GetResult:
        return (200, {"sha": SHA_A}) if "/commits/" in path else (404, None)

    def tag_object(path: str, token: str | None) -> _GetResult:
        if "/git/tags/" in path:
            return 200, {"tag": "v1", "object": {"sha": SHA_B, "type": "commit"}}
        return 404, None

    def tag_without_target(path: str, token: str | None) -> _GetResult:
        return (200, {"tag": "v1"}) if "/git/tags/" in path else (404, None)

    def blob(path: str, token: str | None) -> _GetResult:
        return (200, {}) if "/git/blobs/" in path else (404, None)

    def missing_in_visible_repo(path: str, token: str | None) -> _GetResult:
        return (200, {}) if path.count("/") == 2 and path.startswith("repos/") else (404, None)

    def invisible_repo(path: str, token: str | None) -> _GetResult:
        return 404, None

    def unprocessable(path: str, token: str | None) -> _GetResult:
        return (422, None) if "/commits/" in path else (404, None)

    return {
        "commit": commit,
        "tag_object": tag_object,
        "tag_without_target": tag_without_target,
        "blob": blob,
        "missing_in_visible_repo": missing_in_visible_repo,
        "invisible_repo": invisible_repo,
        "unprocessable": unprocessable,
    }


_USES_FIXTURES = [
    # skipped
    "./.github/actions/thing",
    ".\\.github\\actions\\thing",
    "docker://alpine:3.20",
    # no ref / unparseable
    "actions/checkout",
    "checkout@v4",
    "justaword",
    "@",
    "actions/@v1",
    "a/b/c@",
    # mutable names
    "actions/checkout@v4",
    "actions/checkout@main",
    "actions/checkout@refs/heads/main",
    "nousergon/nousergon-lib/.github/workflows/x.yml@main",
    # hex-shaped but not a 40-char lowercase SHA
    "actions/checkout@3d3c42e",
    "actions/checkout@a",
    "actions/checkout@" + "a" * 39,
    "actions/checkout@" + "a" * 41,
    "actions/checkout@" + "a" * 64,
    "actions/checkout@" + "A" * 40,
    "actions/checkout@" + "A" * 7,
    "actions/checkout@" + "g" * 40,
    "actions/checkout@" + "a" * 39 + "g",
    "actions/checkout@1234567",
    "actions/checkout@2024",
    # 40-char lowercase SHAs, plain and path forms
    f"actions/checkout@{SHA_A}",
    f"nousergon/nousergon-lib/.github/workflows/x.yml@{SHA_A}",
    f"o/r/.github/actions/y@{SHA_A}",
    f"o/r@{SHA_B}  ",
    f"o/r@{SHA_A} # v4",
]


def _hex_shaped_non_sha(value: str) -> bool:
    ref = value.rpartition("@")[2]
    return ref.isalnum() and all(c in "0123456789abcdefABCDEF" for c in ref) and not (
        len(ref) == 40 and ref == ref.lower()
    )


@pytest.mark.parametrize("scenario", sorted(_scenarios()))
@pytest.mark.parametrize("value", _USES_FIXTURES)
def test_new_verdict_is_never_looser_than_the_old_one(
    guard: ModuleType, monkeypatch: pytest.MonkeyPatch, scenario: str, value: str
) -> None:
    """Run the frozen pre-migration guard and the migrated one over the same world.

    The new verdict may be stricter (a finding where the old one passed). It may
    never be a pass where the old one found something. Where the old wording is
    still right the message is identical; the only intended rewording is the
    short/long hexadecimal diagnosis, which was previously a generic mutable-ref
    finding and is still a finding.
    """
    legacy = _load("legacy_action_pin_guard", _LEGACY)
    get = _scenarios()[scenario]
    monkeypatch.setattr(legacy, "_get", get)
    monkeypatch.setattr(guard, "_get", get)

    old = legacy.check_uses(value, lambda r, s: legacy.classify_ref(r, s, None))
    new = guard.check_uses(value, lambda r, s: guard.classify_ref(r, s, None))

    if old is not None:
        assert new is not None, f"loosened: {value!r} in {scenario}: was {old!r}"
    if not _hex_shaped_non_sha(value):
        assert new == old, f"wording drifted for {value!r} in {scenario}"


@pytest.mark.parametrize("scenario", sorted(_scenarios()))
def test_directory_findings_match_the_old_guard_line_for_line(
    guard: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    steps = "".join(f"      - uses: {v}\n" for v in _USES_FIXTURES if "\n" not in v and v.strip() == v)
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n" + steps,
    )
    legacy = _load("legacy_action_pin_guard", _LEGACY)
    get = _scenarios()[scenario]
    monkeypatch.setattr(legacy, "_get", get)
    monkeypatch.setattr(guard, "_get", get)
    wf = tmp_path / ".github" / "workflows"

    old = legacy.check_dir(wf, lambda r, s: legacy.classify_ref(r, s, None))
    new = guard.check_dir(wf, lambda r, s: guard.classify_ref(r, s, None))

    assert [(p, ln, v) for p, ln, v, _ in new] == [(p, ln, v) for p, ln, v, _ in old]


@pytest.mark.parametrize("scenario", sorted(_scenarios()))
def test_exit_codes_match_the_old_guard(
    guard: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    _write(
        tmp_path,
        "ci.yml",
        "name: CI\non:\n  pull_request:\njobs:\n"
        f"  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@{SHA_A}\n",
    )
    legacy = _load("legacy_action_pin_guard", _LEGACY)
    get = _scenarios()[scenario]
    monkeypatch.setattr(legacy, "_get", get)
    monkeypatch.setattr(guard, "_get", get)
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.chdir(tmp_path)

    assert guard.main([]) == legacy.main([])
    out = capsys.readouterr().out
    assert out.count("::error file=") % 2 == 0, "both guards must annotate identically"


# ── the script resolves its own library when run as a file ───────────────────


def test_script_imports_the_shared_module_from_a_foreign_cwd(tmp_path: Path) -> None:
    """The reusable workflow runs this file from the CALLER's checkout.

    No PYTHONPATH: the script must find its sibling ``src/`` itself, or every
    consumer pinned to an older workflow skeleton breaks the day it grows an
    import. Reaching the "no workflow directory" exit proves the import ran.
    """
    root = tmp_path / "lib"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "nousergon_lib").mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    shutil.copy(_SCRIPT, root / "scripts" / _SCRIPT.name)
    for name in ("__init__.py", "action_pins.py"):
        shutil.copy(repo / "src" / "nousergon_lib" / name, root / "src" / "nousergon_lib" / name)
    caller = tmp_path / "caller"
    caller.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(  # noqa: S603 -- fixed argv, this repo's own script
        [sys.executable, str(root / "scripts" / _SCRIPT.name)],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "no workflow directory" in proc.stdout
