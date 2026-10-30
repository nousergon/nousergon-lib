"""The fleet's no-live-state-on-the-PR-path guard (alpha-engine-config-I9830).

Each case writes a small `.github/workflows` tree and grades it, so these tests
pin behaviour on the workflow shapes the reference guard
(`crucible/tests/test_workflow_triggers.py`) recorded as having walked past
earlier versions: a bare `on` read as boolean True, `pull_request_target`, a
one-event exclusion on a two-event workflow, a disjunctive `if:`, a job-level
`uses:`, and an allowlist entry that outlived its job.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from nousergon_lib import pr_path_guard as g

STEPS = "    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"


def _tree(tmp_path: Path, **files: str) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    for name, text in files.items():
        (wf / name.replace("__", ".")).write_text(text, encoding="utf-8")
    return wf


def _kinds(findings: list[g.Finding]) -> list[tuple[str, str]]:
    return [(f.key, f.kind) for f in findings]


def _wf(on: str, *jobs: str) -> str:
    return f"name: x\non: {on}\njobs:\n" + "".join(jobs)


def _job(name: str, cond: str | None = None, body: str = STEPS) -> str:
    head = f"  {name}:\n"
    if cond is not None:
        head += f"    if: {cond}\n"
    return head + body


class TestTriggers:
    def test_a_bare_on_key_is_read_not_dropped(self, tmp_path):
        """PyYAML reads `on` as True; a guard that missed it would pass everything."""
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("test")))
        assert _kinds(g.check(wf, {})) == [("ci.yml:test", "unlisted")]

    @pytest.mark.parametrize(
        "on",
        ["pull_request", "[push, pull_request]", "{pull_request: {branches: [main]}, push: {}}"],
    )
    def test_every_trigger_shape_is_understood(self, tmp_path, on):
        wf = _tree(tmp_path, ci__yml=_wf(on, _job("test")))
        assert _kinds(g.check(wf, {})) == [("ci.yml:test", "unlisted")]

    def test_a_push_only_workflow_is_not_graded(self, tmp_path):
        wf = _tree(tmp_path, ci__yml=_wf("{push: {branches: [main]}, schedule: [{cron: '0 0 * * *'}]}", _job("x")))
        assert g.check(wf, {}) == []

    def test_pull_request_target_is_held_too(self, tmp_path):
        wf = _tree(tmp_path, t__yml=_wf("pull_request_target", _job("deploy")))
        assert _kinds(g.check(wf, {})) == [("t.yml:deploy", "unlisted")]

    def test_the_yaml_suffix_is_scanned(self, tmp_path):
        wf = _tree(tmp_path, ci__yaml=_wf("pull_request", _job("test")))
        assert _kinds(g.check(wf, {})) == [("ci.yaml:test", "unlisted")]


class TestExclusions:
    @pytest.mark.parametrize(
        "cond",
        [
            "github.event_name != 'pull_request'",
            "${{ github.event_name != 'pull_request' }}",
        ],
    )
    def test_the_exact_single_event_exclusion_takes_a_job_off_the_pr_path(self, tmp_path, cond):
        wf = _tree(tmp_path, ci__yml=_wf("[push, pull_request]", _job("drift", f'"{cond}"')))
        assert g.check(wf, {}) == []

    def test_a_disjunction_is_not_an_exclusion(self, tmp_path):
        cond = "\"github.event_name != 'pull_request' || github.actor == 'x'\""
        wf = _tree(tmp_path, ci__yml=_wf("[push, pull_request]", _job("drift", cond)))
        assert _kinds(g.check(wf, {})) == [("ci.yml:drift", "unlisted")]

    def test_naming_one_event_does_not_exclude_the_other(self, tmp_path):
        """On pull_request_target, `event_name != 'pull_request'` is TRUE: the job runs."""
        cond = "\"github.event_name != 'pull_request'\""
        wf = _tree(tmp_path, ci__yml=_wf("[pull_request, pull_request_target]", _job("drift", cond)))
        assert _kinds(g.check(wf, {})) == [("ci.yml:drift", "unlisted")]
        wf2 = _tree(tmp_path / "b", t__yml=_wf("pull_request_target", _job("drift", cond)))
        assert _kinds(g.check(wf2, {})) == [("t.yml:drift", "unlisted")]

    @pytest.mark.parametrize(
        "cond",
        [
            "github.event_name != 'pull_request' && github.event_name != 'pull_request_target'",
            '!contains(fromJSON(\'["pull_request","pull_request_target"]\'), github.event_name)',
        ],
    )
    def test_both_event_exclusions_work_on_a_two_event_workflow(self, tmp_path, cond):
        wf = _tree(tmp_path, ci__yml=_wf("[pull_request, pull_request_target]", _job("drift", json.dumps(cond))))
        assert g.check(wf, {}) == []

    def test_the_accepted_set_matches_the_reference_guard(self):
        both = '!contains(fromJSON(\'["pull_request","pull_request_target"]\'), github.event_name)'
        one = "github.event_name != 'pull_request'"
        assert g.exclusions_for({"pull_request"}) == frozenset({both, one, "${{ " + both + " }}", "${{ " + one + " }}"})
        assert g.exclusions_for({"push"}) == frozenset()


class TestAllowlist:
    def test_an_allowlisted_job_with_steps_passes(self, tmp_path):
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("test")))
        assert g.check(wf, {"ci.yml:test": "unit tests over the checkout"}) == []

    def test_an_empty_reason_is_a_finding(self, tmp_path):
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("test")))
        assert _kinds(g.check(wf, {"ci.yml:test": "  "})) == [("ci.yml:test", "empty-reason")]

    def test_a_reusable_call_needs_its_own_admission(self, tmp_path):
        body = "    uses: nousergon/nousergon-lib/.github/workflows/x.yml@" + "a" * 40 + "\n"
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("guard", body=body)))
        reason = {"ci.yml:guard": "tree-only lib guard"}
        assert _kinds(g.check(wf, reason)) == [("ci.yml:guard", "reusable-unlisted")]
        assert g.check(wf, reason, reusable={"ci.yml:guard"}) == []

    def test_a_stepless_entry_describes_nothing(self, tmp_path):
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("noop", body="    runs-on: ubuntu-latest\n")))
        assert _kinds(g.check(wf, {"ci.yml:noop": "nothing"})) == [("ci.yml:noop", "stepless")]

    def test_an_entry_cannot_outlive_its_job(self, tmp_path):
        wf = _tree(
            tmp_path,
            ci__yml=_wf("pull_request", _job("test")),
            drift__yml=_wf("{push: {}}", _job("drift")),
        )
        allow = {"ci.yml:test": "unit tests", "drift.yml:drift": "was on PRs once", "gone.yml:x": "deleted"}
        assert _kinds(g.check(wf, allow)) == [("drift.yml:drift", "stale-entry"), ("gone.yml:x", "stale-entry")]

    def test_an_excluded_job_that_is_still_allowlisted_is_stale(self, tmp_path):
        cond = "\"github.event_name != 'pull_request'\""
        wf = _tree(tmp_path, ci__yml=_wf("[push, pull_request]", _job("drift", cond)))
        assert _kinds(g.check(wf, {"ci.yml:drift": "x"})) == [("ci.yml:drift", "stale-entry")]

    def test_a_reusable_admission_without_an_allowlist_entry_is_stale(self, tmp_path):
        wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("test")))
        assert _kinds(g.check(wf, {"ci.yml:test": "t"}, reusable={"ci.yml:other"})) == [("ci.yml:other", "stale-entry")]


class TestDarkIsRed:
    def test_no_workflows_is_a_finding(self, tmp_path):
        wf = tmp_path / "empty"
        wf.mkdir()
        assert [f.kind for f in g.check(wf, {})] == ["no-workflows"]

    def test_a_workflow_with_no_on_block_is_unparseable(self, tmp_path):
        wf = _tree(tmp_path, ci__yml="name: x\njobs:\n" + _job("test"))
        assert _kinds(g.check(wf, {})) == [("ci.yml", "unparseable")]

    def test_invalid_yaml_is_unparseable_and_the_rest_still_graded(self, tmp_path):
        wf = _tree(tmp_path, bad__yml="on: [\n", ci__yml=_wf("pull_request", _job("test")))
        assert _kinds(g.check(wf, {})) == [("bad.yml", "unparseable"), ("ci.yml:test", "unlisted")]


def test_the_assert_helper_lists_every_finding(tmp_path):
    wf = _tree(tmp_path, ci__yml=_wf("pull_request", _job("a"), _job("b")))
    with pytest.raises(AssertionError) as info:
        g.assert_no_live_state_on_pr_path(wf, {})
    text = str(info.value)
    assert "2 live-state-on-PR-path finding(s)" in text
    assert "ci.yml:a" in text and "ci.yml:b" in text
    g.assert_no_live_state_on_pr_path(wf, {"ci.yml:a": "x", "ci.yml:b": "y"})


def test_the_module_imports_nothing_beyond_the_standard_library_and_yaml():
    """Kept adoptable by any repo that can install pyyaml."""
    tree = ast.parse(Path(g.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots - set(sys.stdlib_module_names) - {"__future__"} == {"yaml"}


class TestAllowlistFileAndCli:
    CALLER = (
        "name: PR-path live-state guard\n"
        "on:\n  pull_request:\n    branches: [main]\n  push:\n    branches: [main]\n"
        "permissions:\n  contents: read\n"
        "jobs:\n  guard:\n    uses: nousergon/nousergon-lib/.github/workflows/pr-path-guard.yml@" + "b" * 40 + "\n"
    )

    def _repo(self, tmp_path: Path, allowlist: str | None) -> Path:
        _tree(tmp_path, ci__yml=_wf("pull_request", _job("test")), pr_path_guard__yml=self.CALLER)
        if allowlist is not None:
            (tmp_path / ".pr-reachable-jobs.yaml").write_text(allowlist, encoding="utf-8")
        return tmp_path

    GOOD = (
        "schema_version: 1\n"
        "pr_reachable:\n"
        "  ci.yml:test: unit tests over the checkout; no credential\n"
        "  pr_path_guard.yml:guard: this guard; reads only the checkout\n"
        "reusable:\n  - pr_path_guard.yml:guard\n"
    )

    def _main(self, repo: Path) -> int:
        return g.main(
            [
                "--workflow-dir",
                str(repo / ".github" / "workflows"),
                "--allowlist",
                str(repo / ".pr-reachable-jobs.yaml"),
            ]
        )

    def test_a_documented_caller_with_its_allowlist_passes(self, tmp_path, capsys):
        assert self._main(self._repo(tmp_path, self.GOOD)) == 0
        assert "2 PR-reachable job(s)" in capsys.readouterr().out

    def test_a_missing_allowlist_starts_red(self, tmp_path, capsys):
        assert self._main(self._repo(tmp_path, None)) == 1
        out = capsys.readouterr().out
        assert "::error title=pr-path-guard unlisted::ci.yml:test" in out
        assert "pr_path_guard.yml:guard" in out

    @pytest.mark.parametrize(
        "bad",
        ["pr_reachable: {}\n", "schema_version: 1\npr_reachable: [a]\n", "schema_version: 1\nreusable: x\n", "[\n"],
    )
    def test_a_malformed_allowlist_is_exit_2_never_empty(self, tmp_path, bad):
        assert self._main(self._repo(tmp_path, bad)) == 2

    def test_load_allowlist_returns_both_sets(self, tmp_path):
        repo = self._repo(tmp_path, self.GOOD)
        reachable, reusable = g.load_allowlist(repo / ".pr-reachable-jobs.yaml")
        assert set(reachable) == {"ci.yml:test", "pr_path_guard.yml:guard"}
        assert reusable == frozenset({"pr_path_guard.yml:guard"})


def test_the_reusable_workflow_runs_this_module_against_the_callers_allowlist():
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pr-path-guard.yml"
    doc = __import__("yaml").safe_load(wf.read_text(encoding="utf-8"))
    assert set(doc[True]) == {"workflow_call"}, "the reusable guard must not itself run on a PR event here"
    runs = " ".join(s.get("run", "") for s in doc["jobs"]["guard"]["steps"])
    assert "_nousergon-lib-script/src/nousergon_lib/pr_path_guard.py" in runs
    assert "--allowlist .pr-reachable-jobs.yaml" in runs
    assert doc["permissions"] == {"contents": "read"}
