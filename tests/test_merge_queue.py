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


# ------------------------------------------------- reusable-workflow calls

_CALLED = """
on: {workflow_call: {}}
jobs:
  guard:
    name: merge-group required-check guard
    runs-on: ubuntu-latest
"""

_CALLER_REMOTE = """
on:
  pull_request: {}
  merge_group: {types: [checks_requested]}
jobs:
  guard:
    uses: nousergon/nousergon-lib/.github/workflows/merge-group-required-check-guard.yml@6c4bdb8
"""

_GUARD_CTX = "guard / merge-group required-check guard"


def test_remote_reusable_call_resolves_to_caller_slash_called_job():
    """Regression, 2026-09-30: `alpha-engine-config` ruleset 20141600 requires
    `guard / merge-group required-check guard` and the guard reported its own
    context as `Produced by: UNKNOWN`, failing every PR on the repo."""
    wfs = {"guard.yml": _CALLER_REMOTE}
    assert mq.coverage_gaps([_GUARD_CTX], wfs, resolve=lambda uses: _CALLED) == []


def test_remote_call_is_a_gap_when_the_caller_lacks_merge_group():
    """The called workflow's own `on:` is irrelevant — the caller's trigger
    decides whether the context reports for a queue entry."""
    caller = _CALLER_REMOTE.replace("  merge_group: {types: [checks_requested]}\n", "")
    wfs = {"guard.yml": caller}
    assert mq.coverage_gaps([_GUARD_CTX], wfs, resolve=lambda uses: _CALLED) == [
        (_GUARD_CTX, "guard.yml")
    ]


def test_unresolvable_remote_call_never_counts_as_coverage():
    """Conservative: without the called file, `guard / <anything>` is unproven."""
    wfs = {"guard.yml": _CALLER_REMOTE}
    assert mq.coverage_gaps([_GUARD_CTX], wfs) == [(_GUARD_CTX, None)]
    assert mq.coverage_gaps([_GUARD_CTX], wfs, resolve=lambda uses: None) == [
        (_GUARD_CTX, None)
    ]


def test_remote_call_does_not_match_a_job_the_called_workflow_lacks():
    wfs = {"guard.yml": _CALLER_REMOTE}
    assert mq.coverage_gaps(["guard / something else"], wfs, resolve=lambda u: _CALLED) == [
        ("guard / something else", None)
    ]


def test_local_reusable_call_resolves_from_the_workflow_set():
    wfs = {
        ".github/workflows/ci.yml": """
on: {pull_request: {}, merge_group: {types: [checks_requested]}}
jobs:
  tests:
    name: Tests
    uses: ./.github/workflows/_pytest.yml
""",
        ".github/workflows/_pytest.yml": """
on: {workflow_call: {}}
jobs:
  pytest:
    name: pytest (py${{ matrix.py }})
""",
    }
    assert mq.coverage_gaps(["Tests / pytest (py3.11)"], wfs) == []
    assert mq.coverage_gaps(["tests / pytest (py3.11)"], wfs) == [
        ("tests / pytest (py3.11)", None)
    ], "an explicit caller name: replaces the job id in the context"


def test_nested_reusable_calls_compose():
    wfs = {
        "a.yml": "on: {merge_group: {types: [checks_requested]}}\n"
                 "jobs: {outer: {uses: ./.github/workflows/b.yml}}",
        "b.yml": "on: {workflow_call: {}}\njobs: {mid: {uses: ./.github/workflows/c.yml}}",
        "c.yml": "on: {workflow_call: {}}\njobs: {leaf: {}}",
    }
    assert mq.coverage_gaps(["outer / mid / leaf"], wfs) == []


def test_reusable_call_cycles_terminate():
    wfs = {"a.yml": "on: {merge_group: {types: [checks_requested]}}\n"
                    "jobs: {loop: {uses: ./.github/workflows/a.yml}}"}
    assert mq.coverage_gaps(["nope"], wfs) == [("nope", None)]


def test_plain_job_semantics_are_unchanged_with_a_resolver():
    wfs = {"ci.yml": "on: {merge_group: {types: [checks_requested]}}\njobs: {pytest: {}}"}
    assert mq.coverage_gaps(["pytest"], wfs, resolve=lambda u: None) == []


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
