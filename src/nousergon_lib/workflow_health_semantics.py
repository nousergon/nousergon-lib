"""Finding-surface semantics, declared by a workflow in its own file
(alpha-engine-config-I9169).

Lifted out of ``alpha-engine-config/scripts/`` into this library
(alpha-engine-config-I9707), so that every repo which carries a declaration
runs the SAME validator as a pull-request check, from the library version it
already pins, instead of a copy per repo or a cross-checkout of a private
repo:

    workflow-health-semantics --workflow-dir .github/workflows

It exits non-zero on any declaration that names a step the workflow does not
have, omits its tracking issue or reason, or uses a semantics value outside
the closed vocabulary. The reconciler (``check_scheduled_workflow_health.py``)
imports the same functions, so the guard and the grader can never disagree
about what a declaration means.

Dependencies stay at **yaml and the standard library**: no AWS, no GitHub
credentials, and nothing else from this package. A guard job that installs
only ``pyyaml`` can still run this file directly by path, and
``tests/test_workflow_health_semantics.py`` pins that in a subprocess.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# DECLARED FINDING SURFACES (alpha-engine-config-I9553, I9169)
# --------------------------------------------------------------------------
# A workflow whose CONTRACT is "exit non-zero when it finds something" (an
# alarm-drift comparator, a latched-alarm flagger, a registry completeness
# scan, a sweep that failed on one of several targets, a guard job, an
# aggregator summarizing per-step outcomes) is not broken when it exits 1 —
# it is doing its job. Grading purely on run CONCLUSION cannot tell "found
# something" apart from "cannot run", which is why REPORTING_FINDINGS exists.
#
# WHERE THE DECLARATION LIVES, AND WHY IT MOVED (alpha-engine-config-I9169)
# ------------------------------------------------------------------------
# I9553 shipped this as a hand-maintained dict keyed by (repo, workflow) in
# THIS file. That is a second inventory of the covered repos' workflows, and
# this module's own docstring forbids exactly that shape: "coverage is
# derived from a registry, never hand-listed ... a hand-maintained allowlist
# drifts and its drift is invisible because the missing rows produce no
# signal" (observability-policy.md 2.2). A mute list is the worst possible
# thing to keep in a second inventory: when it drifts, it drifts toward
# silence.
#
# So the declaration now lives in the WORKFLOW FILE ITSELF, co-located with
# the steps it describes, and is discovered by the same read that already
# parses every workflow for its `schedule:` trigger. In the workflow's
# top-level environment block:
#
#     WORKFLOW_HEALTH_SEMANTICS: finding-surface
#     WORKFLOW_HEALTH_FINDING_STEPS: |
#       Flag alarms latched in ALARM more than 7 days
#     WORKFLOW_HEALTH_FINDING_ISSUE: alpha-engine-config-I8063
#     WORKFLOW_HEALTH_FINDING_REASON: >-
#       Exits 1 whenever an alarm has been latched in ALARM for more than
#       7 days - the non-zero exit IS the latched-alarm finding.
#
# Consequences that are the point rather than side effects:
#   - Deleting the workflow deletes its declaration. The I9553 registry could
#     outlive the file it muted; this cannot.
#   - Renaming a step is caught by `declared_finding_surfaces(strict=True)`,
#     which reads the step names out of the SAME file - a check with no
#     second list to agree with.
#   - A repo owns the semantics of its own workflows. `nous-ergon-ops` no
#     longer needs a pull request against `alpha-engine-config` to declare
#     that one of its detectors reports findings.
#   - STEPS IS A LIST. `iam-drift-check.yml` has two finding steps ("Fail on
#     real IAM drift (CONTENT/COVERAGE)" and "Scan the complete roster for
#     scheduled human identities"), measured 2026-08-31 across its five most
#     recent failing runs on `main`. The single-`step` field of the I9553
#     registry could not express it, so that workflow could not be declared
#     at all without muting one of its two real finding steps.

#: The only legal values of `WORKFLOW_HEALTH_SEMANTICS`. A closed vocabulary,
#: because a typo (`finding_surface`, `findings-surface`) must fail loudly
#: rather than silently fall through to execution-health grading - the reader
#: would have declared the semantics and got the old behaviour with no signal.
SEMANTICS_EXECUTION_HEALTH = "execution-health"
SEMANTICS_FINDING_SURFACE = "finding-surface"
#: alpha-engine-config-I9874. A workflow implementing pull-request-policy.md
#: §4.2 form 3 - an operator-gated post-merge security boundary, e.g.
#: `crucible-v2-stack-apply.yml` - whose contract is "stay RED until a human
#: runs the command this merge printed". See the OPERATOR-GATED SEMANTICS
#: section below for the full rationale.
SEMANTICS_OPERATOR_GATED = "operator-gated"
LEGAL_SEMANTICS = (SEMANTICS_EXECUTION_HEALTH, SEMANTICS_FINDING_SURFACE,
                    SEMANTICS_OPERATOR_GATED)

#: Keys read out of a workflow file's top-level environment block.
SEMANTICS_KEY = "WORKFLOW_HEALTH_SEMANTICS"
FINDING_STEPS_KEY = "WORKFLOW_HEALTH_FINDING_STEPS"
FINDING_ISSUE_KEY = "WORKFLOW_HEALTH_FINDING_ISSUE"
FINDING_REASON_KEY = "WORKFLOW_HEALTH_FINDING_REASON"
#: operator-gated declaration keys (I9874) - deliberately separate names
#: from the FINDING_* keys above, even though the shape (steps/issue/reason)
#: is the same: the two semantics answer different questions ("did this find
#: something" vs "is this waiting on an operator") and a workflow declaring
#: one must never be silently read as declaring the other.
GATE_STEPS_KEY = "WORKFLOW_HEALTH_GATE_STEPS"
GATE_ISSUE_KEY = "WORKFLOW_HEALTH_GATE_ISSUE"
GATE_REASON_KEY = "WORKFLOW_HEALTH_GATE_REASON"

#: Steps a finding surface deliberately does NOT declare, listed by name so
#: the omission is a decision on the page rather than an oversight. Required
#: by the reverse check below: inside a `finding-surface` workflow, every
#: `continue-on-error` step is either a declared finding step or named here.
FINDING_EXEMPT_KEY = "WORKFLOW_HEALTH_NOT_A_FINDING"


@dataclass(frozen=True)
class FindingSurface:
    """One workflow's declared finding-surface semantics, read from its own file."""
    workflow: str
    steps: tuple[str, ...]
    issue: str
    reason: str


@dataclass(frozen=True)
class OperatorGated:
    """One workflow's declared operator-gated (§4.2 form 3) semantics, read
    from its own file (alpha-engine-config-I9874).

    ``steps`` are the step(s) whose failure IS the "waiting on the operator"
    signal - e.g. crucible-v2-stack-apply.yml's ``Live template matches
    main``, inside the job named ``the stack is applied and matches main
    (RED until the command runs)``. Same shape and same reasoning as
    ``FindingSurface.steps``: a form-3 detector commonly has exactly one such
    step, but the field is a tuple so a detector checking more than one
    gated resource is not forced to under-declare.
    """
    workflow: str
    steps: tuple[str, ...]
    issue: str
    reason: str


def _as_dict(value: object) -> dict[str, object] | None:
    """``value`` as a YAML mapping, or ``None`` when it is anything else."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _as_list(value: object) -> list[object] | None:
    """``value`` as a YAML sequence, or ``None`` when it is anything else."""
    return cast("list[object]", value) if isinstance(value, list) else None


def _named_steps(doc: dict[str, object]) -> list[dict[str, object]]:
    """Every step mapping, in every job, that carries a string ``name``."""
    out: list[dict[str, object]] = []
    jobs = _as_dict(doc.get("jobs"))
    if jobs is None:
        return out
    for job in jobs.values():
        job_map = _as_dict(job)
        steps = _as_list(job_map.get("steps")) if job_map is not None else None
        if steps is None:
            continue
        for step in steps:
            step_map = _as_dict(step)
            if step_map is not None and isinstance(step_map.get("name"), str):
                out.append(step_map)
    return out


def _declared_lines(value: object) -> list[str]:
    """A declared name list: a YAML block string (one name per line) or a
    YAML sequence, stripped, with blank entries dropped."""
    if isinstance(value, str):
        return [s.strip() for s in value.splitlines() if s.strip()]
    items = _as_list(value)
    if items is None:
        return []
    return [str(s).strip() for s in items if str(s).strip()]


def _workflow_step_names(doc: dict[str, object]) -> set[str]:
    """Every `steps[].name` declared anywhere in a parsed workflow document.

    The denominator a declaration's finding steps are checked against.
    Unnamed steps contribute nothing: GitHub synthesises a display name for
    them, so a declaration can never legitimately match one.
    """
    return {str(step["name"]).strip() for step in _named_steps(doc)}


def _tolerated_step_names(doc: dict[str, object]) -> set[str]:
    """Every named step carrying ``continue-on-error: true``.

    These are the steps whose non-zero exit does NOT fail their job, which in
    a finding surface is exactly the shape of "this red is a finding". They
    are therefore the population the declaration must account for — see
    :func:`_parse_declaration`'s reverse check.
    """
    return {
        str(step["name"]).strip()
        for step in _named_steps(doc)
        if step.get("continue-on-error") is True
    }


def _parse_declaration(
    name: str, doc: dict[str, object],
) -> tuple[FindingSurface | None, list[str]]:
    """``(declaration, errors)`` for one parsed workflow document.

    Returns ``(None, [])`` for a workflow that declares nothing or declares
    `execution-health` - the default, and not a finding surface. Returns
    ``(None, errors)`` for a MALFORMED declaration: never a partially-applied
    one, because every degenerate reading of a broken declaration is a mute.
    """
    declared = _as_dict(doc.get("env"))
    if declared is None:
        return None, []
    raw = declared.get(SEMANTICS_KEY)
    if raw is None:
        # No declaration. Not an error, and not a finding surface: a workflow
        # that says nothing is graded on execution health, which is the
        # correct default for the overwhelming majority.
        return None, []
    if not isinstance(raw, str) or raw.strip() not in LEGAL_SEMANTICS:
        return None, [
            f"{name}: {SEMANTICS_KEY}={raw!r} is not one of {LEGAL_SEMANTICS} - "
            f"a typo here would silently grade a declared finding surface as "
            f"execution health, which is the defect the declaration exists to fix"
        ]
    if raw.strip() in (SEMANTICS_EXECUTION_HEALTH, SEMANTICS_OPERATOR_GATED):
        # Not a finding surface. operator-gated is a DIFFERENT declared
        # semantics entirely - see _parse_operator_gate_declaration - so this
        # function (finding-surface only) has nothing further to validate;
        # validating it here too would mean the same field could fail twice
        # under two different error messages for one typo.
        return None, []

    errors: list[str] = []
    steps: tuple[str, ...] = tuple(_declared_lines(declared.get(FINDING_STEPS_KEY)))
    if not steps:
        errors.append(
            f"{name}: {SEMANTICS_KEY}={SEMANTICS_FINDING_SURFACE} with no "
            f"{FINDING_STEPS_KEY} - a finding surface that names no finding "
            f"step would mute EVERY failure of this workflow, including a "
            f"revoked credential")
    step_names = _workflow_step_names(doc)

    # ── THE REVERSE DIRECTION (alpha-engine-config-I9702) ────────────────────
    # The check below asserts declared -> exists: a declared step must be a
    # real step, which catches a RENAME. It did not assert exists -> declared,
    # so ADDING a checker step and forgetting to declare it passed silently —
    # and its red would then be graded as a broken check rather than as the
    # finding it is.
    #
    # Measured 2026-09-01, on this very guard: `Lambda timeout drift (codified
    # table vs live)` was added to nousergon-data's sf-arn-drift-check.yml,
    # given an id, wired into the terminal gate, and left undeclared. CI was
    # green. The asymmetry is the same one that arc found everywhere else: a
    # guard that checks one direction reads as coverage for both.
    #
    # The population is the `continue-on-error` steps, because in a finding
    # surface that flag IS the statement "a non-zero exit here is a finding,
    # not a break". A step that is deliberately not a finding — remediation,
    # credential setup, a publish — is named in WORKFLOW_HEALTH_NOT_A_FINDING,
    # so the omission is a decision someone wrote down rather than one nobody
    # noticed.
    exempt = set(_declared_lines(declared.get(FINDING_EXEMPT_KEY)))
    for bogus in sorted(exempt - step_names):
        errors.append(
            f"{name}: {FINDING_EXEMPT_KEY} names step {bogus!r}, which is "
            f"not a step in this workflow")
    undeclared = sorted(_tolerated_step_names(doc) - set(steps) - exempt)
    if undeclared:
        errors.append(
            f"{name}: step(s) {undeclared} carry continue-on-error but are "
            f"neither in {FINDING_STEPS_KEY} nor in {FINDING_EXEMPT_KEY}. A "
            f"tolerated step's red is graded as a broken check instead of as "
            f"a finding. Declare it, or name it as deliberately not a finding.")

    for step in steps:
        if step not in step_names:
            errors.append(
                f"{name}: {FINDING_STEPS_KEY} names step {step!r}, which is "
                f"not a step in this workflow. Renaming a step without "
                f"updating the declaration would silently reclassify a broken "
                f"monitor as a finding - or leave a working one paging")
    issue = declared.get(FINDING_ISSUE_KEY)
    if not isinstance(issue, str) or not issue.strip():
        errors.append(
            f"{name}: {SEMANTICS_FINDING_SURFACE} requires {FINDING_ISSUE_KEY} "
            f"- the findings themselves need a destination that is not this "
            f"reconciler's own row (observability-policy.md 7.3)")
    reason = declared.get(FINDING_REASON_KEY)
    if not isinstance(reason, str) or not reason.strip():
        errors.append(
            f"{name}: {SEMANTICS_FINDING_SURFACE} requires {FINDING_REASON_KEY} "
            f"- one line saying why this workflow's non-zero exit is a finding")
    if errors:
        return None, errors
    return FindingSurface(
        workflow=name, steps=steps,
        issue=str(issue).strip(), reason=" ".join(str(reason).split()),
    ), []


# --------------------------------------------------------------------------
# OPERATOR-GATED SEMANTICS (alpha-engine-config-I9874)
# --------------------------------------------------------------------------
# pull-request-policy.md §4.2 sanctions exactly one manual post-merge step -
# form 3, an operator-gated security boundary - and it requires BOTH halves:
# the merge emits the exact command, AND a detector stays RED until that
# command runs. `scheduled-workflow-health` graded that detector on plain
# execution health and filed it `chronic` (alpha-engine-config-I9856):
# measured 2026-09-02 on `crucible-v2-stack-apply.yml`, 9 of 12 graded runs
# failed, every one of them in the job "the stack is applied and matches
# main (RED until the command runs)" - the detector working exactly as
# designed. The longer a form-3 detector correctly holds the line, the worse
# its old grade got: a control penalised in proportion to how well it works.
#
# The declaration follows the exact shape and placement FindingSurface
# already proved out (I9169): in the workflow's own top-level `env:` block,
# discovered by the same read, so a repo owns the semantics of its own
# workflow and deleting the file deletes the declaration.
#
#     env:
#       WORKFLOW_HEALTH_SEMANTICS: operator-gated
#       WORKFLOW_HEALTH_GATE_STEPS: |
#         Live template matches main
#       WORKFLOW_HEALTH_GATE_ISSUE: alpha-engine-config-I9874
#       WORKFLOW_HEALTH_GATE_REASON: >-
#         pull-request-policy.md §4.2 form 3 - stays red until the operator
#         runs the apply command this merge printed.
#
# WHAT THE DECLARATION CHANGES, AND WHAT IT NEVER CAN
# -----------------------------------------------------
# A form-3 detector is graded HEALTHY only when BOTH hold, checked in
# `check_scheduled_workflow_health._reclassify_operator_gated`:
#   (a) it has gone green at least once in the grading window - proving the
#       gate mechanism and the check both work;
#   (b) EVERY failing run in the window failed at one of the declared gate
#       step(s) - the exact anti-mute guard `_reclassify_declared_findings`
#       already uses, so a broken credential or a checkout failure can never
#       be laundered into "waiting on the operator".
#
# (a) is the load-bearing property. A workflow that has NEVER gone green,
# even under this declaration, keeps its original ungraded status - loud,
# unmuted, exactly as before. Dropping that check would turn the
# declaration into a suppression, which AGENTS.md rule 4 forbids by name:
# "Fail loud and fast. No silent swallows." A control nobody has ever seen
# clear is never "expected", whatever it declares about itself.
def _parse_operator_gate_declaration(
    name: str, doc: dict[str, object],
) -> tuple[OperatorGated | None, list[str]]:
    """``(declaration, errors)`` for one parsed workflow document's
    `operator-gated` semantics - the counterpart to `_parse_declaration`.

    Returns ``(None, [])`` for a workflow declaring nothing, `execution-
    health`, or `finding-surface` (that vocabulary is validated once, by
    `_parse_declaration`, so an invalid `WORKFLOW_HEALTH_SEMANTICS` value is
    reported exactly once rather than by both parsers for the same typo).
    """
    declared = _as_dict(doc.get("env"))
    if declared is None:
        return None, []
    raw = declared.get(SEMANTICS_KEY)
    if not isinstance(raw, str) or raw.strip() != SEMANTICS_OPERATOR_GATED:
        return None, []

    errors: list[str] = []
    steps: tuple[str, ...] = tuple(_declared_lines(declared.get(GATE_STEPS_KEY)))
    if not steps:
        errors.append(
            f"{name}: {SEMANTICS_OPERATOR_GATED} with no {GATE_STEPS_KEY} - "
            f"a declaration naming no gate step would mute EVERY failure of "
            f"this workflow, including a genuinely broken one")
    step_names = _workflow_step_names(doc)
    for step in steps:
        if step not in step_names:
            errors.append(
                f"{name}: {GATE_STEPS_KEY} names step {step!r}, which is not "
                f"a step in this workflow. Renaming a step without updating "
                f"the declaration would silently reclassify a broken control "
                f"as healthy - or leave a working one paging")
    issue = declared.get(GATE_ISSUE_KEY)
    if not isinstance(issue, str) or not issue.strip():
        errors.append(
            f"{name}: {SEMANTICS_OPERATOR_GATED} requires {GATE_ISSUE_KEY} - "
            f"the security-boundary decision this gate implements needs a "
            f"citation that is not this reconciler's own row")
    reason = declared.get(GATE_REASON_KEY)
    if not isinstance(reason, str) or not reason.strip():
        errors.append(
            f"{name}: {SEMANTICS_OPERATOR_GATED} requires {GATE_REASON_KEY} - "
            f"one line saying why this workflow's red is an operator gate, "
            f"not a defect")
    if errors:
        return None, errors
    return OperatorGated(
        workflow=name, steps=steps,
        issue=str(issue).strip(), reason=" ".join(str(reason).split()),
    ), []


def declared_operator_gated(
    workflow_dir: Path, *, strict: bool = False,
) -> dict[str, OperatorGated]:
    """Every workflow in ``workflow_dir`` declaring `operator-gated`
    semantics, keyed by filename. Same strict/non-strict contract as
    :func:`declared_finding_surfaces` - see that docstring."""
    d = Path(workflow_dir)
    found: dict[str, OperatorGated] = {}
    all_errors: list[str] = []
    for path in sorted(d.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            all_errors.append(f"{path.name}: unparseable workflow ({exc})")
            continue
        doc_map = _as_dict(doc)
        if doc_map is None:
            continue
        decl, errors = _parse_operator_gate_declaration(path.name, doc_map)
        all_errors.extend(errors)
        if decl is not None:
            found[path.name] = decl
    if all_errors:
        if strict:
            raise ValueError(
                "invalid operator-gated declaration(s) in "
                f"{d}:\n  " + "\n  ".join(all_errors))
        for msg in all_errors:
            logger.error(
                "operator-gated declaration REJECTED (graded on execution "
                "health instead, un-muted): %s", msg)
    return found


def declared_finding_surfaces(
    workflow_dir: Path, *, strict: bool = False,
) -> dict[str, FindingSurface]:
    """Every workflow in ``workflow_dir`` declaring `finding-surface`
    semantics, keyed by filename.

    ``strict=True`` RAISES on any malformed declaration - this is the CI
    guard (`--validate-declarations`), run in the repo that owns the workflow
    files, where a wrong declaration is a merge-blocking defect.

    ``strict=False`` (the grading path) DROPS a malformed declaration and
    logs it at ERROR. That is deliberate and is not a silent swallow:

      (a) the failure mode swallowed is "one repo's declaration is broken";
      (b) the primary deliverable survives - the workflow keeps its ORIGINAL,
          un-muted grade, which is the safe direction: a broken declaration
          can only ever make this reconciler page MORE, never less;
      (c) the recording surface is the ERROR log line plus the un-muted row
          itself, which stays on the console and in the artifact.

    Raising here instead would take down grading for all 27 workflows in a
    repo because one of them was renamed - collateral silence, in a script
    whose whole purpose is to notice silence.
    """
    d = Path(workflow_dir)
    found: dict[str, FindingSurface] = {}
    all_errors: list[str] = []
    for path in sorted(d.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            all_errors.append(f"{path.name}: unparseable workflow ({exc})")
            continue
        doc_map = _as_dict(doc)
        if doc_map is None:
            continue
        decl, errors = _parse_declaration(path.name, doc_map)
        all_errors.extend(errors)
        if decl is not None:
            found[path.name] = decl
    if all_errors:
        if strict:
            raise ValueError(
                "invalid workflow-health declaration(s) in "
                f"{d}:\n  " + "\n  ".join(all_errors))
        for msg in all_errors:
            logger.error(
                "workflow-health declaration REJECTED (graded on execution "
                "health instead, un-muted): %s", msg)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow-dir", default=".github/workflows", metavar="PATH",
                    help="the directory of workflow files to validate "
                         "(default: .github/workflows)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wf_dir = Path(args.workflow_dir)
    if not wf_dir.is_dir():
        raise SystemExit(f"--workflow-dir {wf_dir} is not a directory")
    try:
        surfaces = declared_finding_surfaces(wf_dir, strict=True)
        gates = declared_operator_gated(wf_dir, strict=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name, decl in sorted(surfaces.items()):
        steps = ", ".join(repr(s) for s in decl.steps)
        print(f"{name}: finding-surface, steps {steps}, tracked at {decl.issue}")
    for name, decl in sorted(gates.items()):
        steps = ", ".join(repr(s) for s in decl.steps)
        print(f"{name}: operator-gated, steps {steps}, tracked at {decl.issue}")
    print(f"{len(surfaces)} declared finding surface(s), {len(gates)} declared "
          f"operator-gated detector(s) in {wf_dir}: all valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
