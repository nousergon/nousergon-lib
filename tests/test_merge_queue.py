"""Contract tests for `nousergon_lib.merge_queue`.

Every test here is fixture-driven with an injected `api` — no credentials, no
network. The module's whole point is being callable about a repo the caller has
no checkout of, so the reads are the part most worth pinning.
"""

from __future__ import annotations

import pytest

from nousergon_lib import merge_queue as mq


def _api(responses: dict):
    """An `api` callable over a recorded `{path: response}` map.

    Raises on an unrecorded path rather than returning `{}` — a sweep that
    silently reads nothing for a repo would record that repo as compliant, which
    is the one answer this module may never invent.
    """
    def call(path: str):
        if path not in responses:
            raise RuntimeError(f"unrecorded path: {path}")
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return value
    return call


# ---------------------------------------------------------------- workflows


def test_job_id_is_the_context_when_the_job_has_no_name():
    """The most common shape in the fleet, and the one the original parser missed."""
    has_mg, jobs = mq.parse_workflow("""
on:
  pull_request:
  merge_group:
    types: [checks_requested]
jobs:
  pytest:
    runs-on: ubuntu-latest
""")
    assert has_mg is not None
    assert "pytest" in jobs


def test_explicit_name_wins_over_the_job_id():
    _, jobs = mq.parse_workflow("""
on: {merge_group: {types: [checks_requested]}}
jobs:
  build:
    name: iam-policy-change-guard
""")
    assert jobs == {"iam-policy-change-guard"}, (
        "when name: is present it IS the context; adding the ID too would let a "
        "guard match a context GitHub never emits"
    )


def test_yaml_parses_on_as_the_boolean_true_and_the_trigger_still_reads():
    """`on:` is YAML 1.1's `True`. If only the string key were tried, every
    workflow would read as having no merge_group trigger — a false gap on the
    entire fleet, reported with total confidence."""
    doc = """
'on':
  merge_group:
    types: [checks_requested]
jobs: {test: {}}
"""
    has_mg, _ = mq.parse_workflow(doc)
    assert has_mg is not None


def test_absent_trigger_is_still_absent():
    has_mg, jobs = mq.parse_workflow("on: {pull_request: {}}\njobs: {pytest: {}}")
    assert has_mg is None
    assert "pytest" in jobs, "the producer must stay nameable so the report is actionable"


def test_matrix_templates_are_retained_and_matched():
    _, jobs = mq.parse_workflow("""
on: {merge_group: {types: [checks_requested]}}
jobs:
  test:
    name: pytest (py${{ matrix.py }})
""")
    pattern = next(iter(jobs))
    assert mq.job_name_matches(pattern, "pytest (py3.11)")
    assert not mq.job_name_matches(pattern, "ruff (py3.11)")


def test_non_mapping_workflow_is_not_a_crash():
    assert mq.parse_workflow("just a string") == (None, set())


# ------------------------------------------------------------ coverage gaps


def test_a_covered_job_id_is_not_a_gap():
    """The blocking half: the guard must be able to confirm its own fix, or it
    keeps reporting a gap that has already been closed."""
    wfs = {"ci.yml": "on: {merge_group: {types: [checks_requested]}}\njobs: {pytest: {}}"}
    assert mq.coverage_gaps(["pytest"], wfs) == []


def test_an_uncovered_context_names_its_producer():
    wfs = {"ci.yml": "on: {pull_request: {}}\njobs: {pytest: {}}"}
    assert mq.coverage_gaps(["pytest"], wfs) == [("pytest", "ci.yml")]


def test_a_context_no_workflow_produces_reports_an_unknown_producer():
    """A required context nothing emits blocks every PR forever. It is a gap of a
    different kind and must not be silently folded into the covered set."""
    wfs = {"ci.yml": "on: {merge_group: {types: [checks_requested]}}\njobs: {pytest: {}}"}
    assert mq.coverage_gaps(["iam-policy-change-guard"], wfs) == [
        ("iam-policy-change-guard", None)
    ]


def test_one_covering_workflow_is_enough_when_several_claim_the_context():
    wfs = {
        "noop.yml": "on: {pull_request: {}}\njobs: {test: {}}",
        "real.yml": "on: {merge_group: {types: [checks_requested]}}\njobs: {test: {}}",
    }
    assert mq.coverage_gaps(["test"], wfs) == []


# -------------------------------------------------------------- queue reads


_RULESET_LIST = "/repos/o/r/rulesets"


def test_active_merge_queue_is_detected_by_rule_type():
    api = _api({
        _RULESET_LIST: [{"id": 1, "name": "main-merge-queue", "enforcement": "active"}],
        "/repos/o/r/rulesets/1": {
            "rules": [{"type": "merge_queue", "parameters": {
                "check_response_timeout_minutes": 20, "min_entries_to_merge": 1,
            }}]
        },
    })
    cfg = mq.active_merge_queue("o/r", api=api)
    assert cfg is not None
    assert cfg.ruleset_id == 1
    assert cfg.response_timeout_minutes == 20


def test_an_empty_ruleset_named_main_merge_queue_is_NOT_a_queue():
    """The 2026-07-24 revert stripped the rule and left ~10 rulesets named
    `main-merge-queue` with `rules: []`. Detecting by name would have read every
    one of them as a live queue — exactly backwards for an auditor."""
    api = _api({
        _RULESET_LIST: [{"id": 2, "name": "main-merge-queue", "enforcement": "active"}],
        "/repos/o/r/rulesets/2": {"rules": []},
    })
    assert mq.active_merge_queue("o/r", api=api) is None


def test_an_evaluate_mode_ruleset_is_not_active():
    api = _api({
        _RULESET_LIST: [{"id": 3, "name": "fleet-baseline", "enforcement": "evaluate"}],
    })
    assert mq.active_merge_queue("o/r", api=api) is None


def test_required_contexts_unions_rulesets_and_classic_protection():
    api = _api({
        _RULESET_LIST: [{"id": 4, "name": "main-protection", "enforcement": "active"}],
        "/repos/o/r/rulesets/4": {
            "rules": [{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": "from-ruleset"}, {"context": "shared"}],
            }}]
        },
        "/repos/o/r/branches/main/protection": {
            "required_status_checks": {"contexts": ["shared", "from-classic"]}
        },
    })
    assert mq.required_contexts("o/r", api=api) == ["from-ruleset", "shared", "from-classic"]


def test_a_ruleset_only_repo_404s_on_classic_protection_and_still_reads():
    """`nousergon-lib` was reported UNPROTECTED on 2026-07-28 by a sweep that
    queried only the classic endpoint. A 404 there is 'ruleset-only', not
    'no protection'."""
    api = _api({
        _RULESET_LIST: [{"id": 5, "name": "main-protection", "enforcement": "active"}],
        "/repos/o/r/rulesets/5": {
            "rules": [{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": "tests"}],
            }}]
        },
        "/repos/o/r/branches/main/protection": RuntimeError("404 Branch not protected"),
    })
    assert mq.required_contexts("o/r", api=api) == ["tests"]


def test_an_unreadable_ruleset_list_raises_rather_than_reporting_clean():
    """The single most dangerous failure mode: an unreadable repo recorded as
    compliant. `principles.md` §2.7 — no data is never rendered as green."""
    api = _api({_RULESET_LIST: RuntimeError("403 forbidden")})
    with pytest.raises(RuntimeError):
        mq.required_contexts("o/r", api=api)
    with pytest.raises(RuntimeError):
        mq.active_merge_queue("o/r", api=api)


# ------------------------------------------------------------------- guard


@pytest.mark.parametrize("ctx", [
    "merge-group-required-check-guard",
    # The shape alpha-engine-config actually emits, observed on config-PR5904:
    # <job id> / <called workflow's job name>. It does NOT contain the slug, so a
    # literal substring test reports the guard as advisory on the one repo where
    # it is required.
    "guard / merge-group required-check guard",
    "Merge-group required-check guard",
    "merge-group-required-check-guard / guard",
])
def test_guard_is_recognised_under_every_shape_the_fleet_emits(ctx):
    assert mq.guard_is_required([ctx]) is True


@pytest.mark.parametrize("contexts", [
    ["pytest", "secrets"],
    ["gate-label-guard / gate-label-guard"],
    ["required-check-guard"],  # a different guard; the full slug must be present
])
def test_guard_absent_is_reported_absent(contexts):
    assert mq.guard_is_required(contexts) is False
