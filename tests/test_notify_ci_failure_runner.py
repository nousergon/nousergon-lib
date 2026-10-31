"""`notify-ci-failure.yml` must follow the fleet runner switch, for private callers only.

alpha-engine-config-I11697: the reusable notifier hardcoded `ubuntu-latest`,
so a `CI_RUNNER_MODE=codebuild` flip moved every private caller's own jobs and
left this one on the hosted allowance. Measured 2026-10-31 it was the largest
single cause of unreachable jobs in alpha-engine-config and nous-ergon-ops.

Two ways to get the fix wrong, both pinned here:
  * route PUBLIC callers (crucible-*, nousergon-data, krepis, this repo) to
    CodeBuild: they have no runner there, and `vars` resolves in the caller's
    context, so the org-level flip reaches them too. Their failure alert would
    queue forever;
  * land on CodeBuild with `actions/setup-python`, which has no amzn2023/arm64
    build (alpha-engine-config-I7846), so the fallback alert dies there.

The `runs-on` expression is evaluated by translating GitHub's `&&`/`||`
(value-returning, like Python's `and`/`or`) into Python, so the cases below
exercise the expression as written rather than a restatement of it.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "notify-ci-failure.yml"
PRIVATE_RUNNER_PREFIX = "codebuild-nousergon-private-runner-"


def _job() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text())
    return data["jobs"]["notify"]


def _eval_runs_on(expr: str, *, mode: str, repository: dict | None) -> str:
    body = expr.strip()
    assert body.startswith("${{") and body.endswith("}}"), body
    body = body[3:-2].strip()
    py = body.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"\btrue\b", "True", py)
    py = re.sub(r"\bfalse\b", "False", py)

    def fmt(template: str, *args: object) -> str:
        for i, a in enumerate(args):
            template = template.replace(f"{{{i}}}", str(a))
        return template

    class _Null:
        """GitHub's null: falsy, equal to nothing, and its properties are null too."""

        def __bool__(self) -> bool:
            return False

        def __eq__(self, _other: object) -> bool:
            return False

        def __getattr__(self, _name: str) -> _Null:
            return self

    class _Absent(SimpleNamespace):
        def __getattr__(self, _name: str) -> _Null:  # missing payload field reads as null
            return _Null()

    repo = _Absent(**repository) if repository is not None else None
    event = _Absent(repository=repo) if repo is not None else _Absent()
    github = SimpleNamespace(event=event, run_id=123, run_attempt=2)
    vars_ = _Absent(CI_RUNNER_MODE=mode) if mode is not None else _Absent()
    py = py.replace("vars.", "vars_.").replace("format(", "fmt(")
    return eval(py, {"__builtins__": {}}, {"vars_": vars_, "github": github, "fmt": fmt})  # noqa: S307


@pytest.mark.parametrize(
    ("mode", "repository", "expected"),
    [
        ("codebuild", {"name": "nous-ergon-ops", "private": True}, f"{PRIVATE_RUNNER_PREFIX}123-2"),
        ("codebuild", {"name": "alpha-engine-config", "private": True}, f"{PRIVATE_RUNNER_PREFIX}123-2"),
        ("gha", {"name": "nous-ergon-ops", "private": True}, "ubuntu-latest"),
        (None, {"name": "nous-ergon-ops", "private": True}, "ubuntu-latest"),
        ("codebuild", {"name": "nousergon-data", "private": False}, "ubuntu-latest"),
        ("codebuild", {"name": "crucible"}, "ubuntu-latest"),
        ("codebuild", None, "ubuntu-latest"),
    ],
    ids=[
        "private-codebuild",
        "aec-codebuild-shared-project",
        "private-gha",
        "mode-unset",
        "public-codebuild-stays-hosted",
        "private-field-absent-stays-hosted",
        "no-repository-payload-stays-hosted",
    ],
)
def test_runs_on_follows_switch_for_private_callers_only(mode, repository, expected):
    assert _eval_runs_on(_job()["runs-on"], mode=mode, repository=repository) == expected


def test_runs_on_reads_the_fleet_switch():
    runs_on = _job()["runs-on"]
    assert "vars.CI_RUNNER_MODE == 'codebuild'" in runs_on
    assert "github.event.repository.private" in runs_on
    assert runs_on.rstrip().endswith("|| 'ubuntu-latest' }}")


def test_setup_python_never_runs_on_the_codebuild_lane():
    for step in _job()["steps"]:
        if str(step.get("uses", "")).startswith("actions/setup-python@"):
            assert "runner.environment == 'github-hosted'" in step["if"], step


def test_codebuild_lane_installs_python_and_later_steps_use_it():
    steps = _job()["steps"]
    cb = [s for s in steps if "runner.environment != 'github-hosted'" in str(s.get("if", ""))]
    assert len(cb) == 1, cb
    assert "dnf install" in cb[0]["run"] and "PY=python3.12" in cb[0]["run"]
    for step in steps:
        run = step.get("run", "")
        if "-m pip install" in run or "krepis.alerts" in run:
            assert '"${PY:-python}" -m' in run, run
        assert not re.search(r"(^|\n)\s*pip install", run), run
