"""`provider-linkage-guard.yml` must follow the fleet runner switch, for private callers only.

alpha-engine-config-I12850: the reusable guard hardcoded `ubuntu-latest`, so a
`CI_RUNNER_MODE=codebuild` flip moved every private caller's own jobs and left
this one on the hosted allowance. Measured 2026-11-02 it was one of the last
two jobs in nous-ergon-ops that `ci_runner_mode_guard` counted as unreachable
by a flip. The fix is the rule `notify-ci-failure.yml` adopted in
nousergon-lib-PR517 (alpha-engine-config-I11697), and the same two ways to get
it wrong are pinned here:
  * route PUBLIC callers (crucible-*, nousergon-data, this repo's own runs) to
    CodeBuild: they have no runner there, and `vars` resolves in the caller's
    context, so the org-level flip reaches them too. Their guard would queue
    forever and block their merges;
  * land on CodeBuild with `actions/setup-python`, which has no amzn2023/arm64
    build (alpha-engine-config-I7846), so the guard dies before it scans.

The `runs-on` expression is evaluated with the same translator PR517's test
uses, so the cases exercise the expression as written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.test_notify_ci_failure_runner import PRIVATE_RUNNER_PREFIX, _eval_runs_on

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "provider-linkage-guard.yml"


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["guard"]


@pytest.mark.parametrize(
    ("mode", "repository", "expected"),
    [
        ("codebuild", {"name": "nous-ergon-ops", "private": True}, f"{PRIVATE_RUNNER_PREFIX}123-2"),
        ("codebuild", {"name": "metron-ops", "private": True}, f"{PRIVATE_RUNNER_PREFIX}123-2"),
        ("gha", {"name": "nous-ergon-ops", "private": True}, "ubuntu-latest"),
        (None, {"name": "nous-ergon-ops", "private": True}, "ubuntu-latest"),
        ("codebuild", {"name": "nousergon-data", "private": False}, "ubuntu-latest"),
        ("codebuild", {"name": "nousergon-lib", "private": False}, "ubuntu-latest"),
        ("codebuild", {"name": "crucible-executor"}, "ubuntu-latest"),
        ("codebuild", None, "ubuntu-latest"),
    ],
    ids=[
        "private-codebuild",
        "private-codebuild-shared-project",
        "private-gha",
        "mode-unset",
        "public-codebuild-stays-hosted",
        "own-runs-stay-hosted",
        "private-field-absent-stays-hosted",
        "no-repository-payload-stays-hosted",
    ],
)
def test_runs_on_follows_switch_for_private_callers_only(mode, repository, expected):
    assert _eval_runs_on(_job()["runs-on"], mode=mode, repository=repository) == expected


def test_runs_on_is_the_same_rule_as_notify_ci_failure():
    notify = yaml.safe_load(
        (WORKFLOW.parent / "notify-ci-failure.yml").read_text()
    )["jobs"]["notify"]["runs-on"]
    assert _job()["runs-on"] == notify


def test_setup_python_never_runs_on_the_codebuild_lane():
    uses = [s for s in _job()["steps"] if str(s.get("uses", "")).startswith("actions/setup-python@")]
    assert len(uses) == 1, uses
    assert uses[0]["if"].replace(" ", "") == "${{runner.environment=='github-hosted'}}", uses[0]


def test_codebuild_lane_installs_python_and_later_steps_use_it():
    steps = _job()["steps"]
    cb = [s for s in steps if "runner.environment != 'github-hosted'" in str(s.get("if", ""))]
    assert len(cb) == 1, cb
    assert "dnf install" in cb[0]["run"] and "PY=python3.12" in cb[0]["run"]
    idx = steps.index(cb[0])
    for step in steps[idx + 1:]:
        run = step.get("run", "")
        if "-m pip install" in run or ("provider_linkage_guard.py" in run and "sha256sum" not in run):
            assert run.startswith('"${PY:-python3}" '), run
    for step in steps:
        run = step.get("run", "")
        assert not re.search(r"(^|\n)\s*(pip|python3?) ", run), run


def test_the_hosted_lane_still_runs_the_scan_with_the_setup_python_interpreter():
    # With $PY unset every step must still name `python3`, the interpreter
    # actions/setup-python put first on PATH -- the command the guard ran
    # before this change.
    scan = next(s for s in _job()["steps"] if "provider_linkage_guard.py --repo" in s.get("run", ""))
    assert scan["run"].strip() == (
        '"${PY:-python3}" _nousergon-lib-script/scripts/provider_linkage_guard.py --repo . --include-dist'
    )
