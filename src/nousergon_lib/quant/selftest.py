"""Shared known-answer SELF-TEST runner — the taxonomy every stage's battery reuses.

WHY THIS EXISTS
----------------
``crucible-backtester/analysis/self_test.py`` (v1, config-I7236/I7237) and
``crucible-evaluator/grading/self_test.py`` were built from the same design in
one arc; a second and third adoption (``crucible-predictor/inference/self_test.py``,
``crucible-research/scoring/self_test.py``) copied the same scaffolding rather
than importing it, growing four independent copies of the outcome taxonomy, the
``Case`` shape, the SIGALRM budget, and the provenance helpers
(alpha-engine-config-I7238 — SECOND ADOPTION is ``shared-code-policy``'s lift
trigger, and the count had already reached four).

This module is that lift. It carries ONLY the scaffolding that must not be
reimplemented per stage — the part where "could not measure" silently collapses
into "found a defect" if anyone gets it wrong twice: the ``PASS``/``FAIL``/
``UNKNOWN`` taxonomy, the ``Case`` record, the wall-clock budget, and the
provenance header. It deliberately does NOT carry ``build_cases()`` — the case
batteries are domain-specific and stay in each consuming repo — nor
``write_self_test`` / the console-envelope publishing helpers, which remain
repo-local for now (see ``crucible-predictor/inference/self_test.py``'s own
module docstring on why a lib-import there is staged rather than immediate: a
pin bump into a production image can itself move the very numbers a self-test
measures, so migrating those call sites is deliberately a separate, later step,
tracked in alpha-engine-config-I7274 for the console-envelope piece specifically).

CONTRACT
--------
``run_self_test()`` **never raises**, and the caller writes its output
unconditionally. A case that DISAGREED with its expectation is ``FAIL``
(evidence the numbers are wrong); a case that could not RUN is ``UNKNOWN``
(absence of evidence). Collapsing the two would make a broken venv/image read as
a correctness regression. Per Brian's ruling 2026-08-13, **a case that exceeds
its time budget is FAIL, never UNKNOWN** — a self-test that cannot finish in
seconds is itself a defect, and "the check timed out" must not buy the same
benefit of the doubt as "the check could not be constructed".

This module never introduces a hard-fail path. The caller writes the artifact
and, on a non-PASS verdict, raises the run's existing degraded flag. Withholding
a guarantee beats failing the run (``sf-pipeline-policy.md`` §2.3a).

SUPERSET NOTE (alpha-engine-config-I7238)
------------------------------------------
Diffing the four copies found two capabilities present in only SOME of them.
Both are carried here unconditionally rather than dropped, so no adopter loses
ground by switching to the shared runner:

- ``Case.known_gap`` / ``Case.gap_issue`` (predictor, research): marks a case
  that deliberately PINS behaviour believed incorrect, so a PASS on it is never
  misread as an endorsement. Defaults to ``False`` / ``None``, so backtester and
  evaluator batteries — which never set it — are unaffected; ``n_known_gaps`` is
  always present in the body (0 where unused).
- ``extra_case_providers`` (predictor only): composes an additional named scope
  of cases at call time, because ``training/`` is deliberately absent from the
  predictor's Lambda image and its cases can only run where that module is
  reachable (CI, the spot trainer). Optional and ``None`` by default for every
  other adopter.
- The malformed-case guard (predictor, research): a case provider that returns a
  well-formed list containing a non-``Case`` item is recorded as an ``UNKNOWN``
  row rather than raising past every handler. Carried unconditionally — it only
  ever fires on a genuinely malformed provider.
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

logger = logging.getLogger(__name__)

PASS = "PASS"  # noqa: S105 — a verdict constant, not a credential
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: Per-case wall-clock budget default. Callers may override per invocation.
CASE_TIMEOUT_SECONDS = 30.0


class SelfTestTimeout(Exception):
    """Raised when a case exceeds its wall-clock budget."""


class Case(NamedTuple):
    """One known-answer, metamorphic, degenerate or convention check.

    ``expected`` is derived by hand from the scenario's or metric's definition
    (or is an exact 0.0 residual, for metamorphic relations); ``compute`` drives
    the production path and returns the comparable observed number. They are
    kept apart — rather than ``compute`` returning a bool — so the artifact
    carries both numbers and a later divergence is diagnosable from the artifact
    alone.

    ``inputs`` is published verbatim in the artifact: a reader must be able to
    re-derive ``expected`` on paper without opening the battery's source.

    ``known_gap`` marks a case that pins behaviour believed WRONG, so a reader
    never mistakes its PASS for an endorsement — see the ``run_self_test``
    docstring. Unused adopters leave it at its default.

    ``tolerance`` defaults to ``1e-9`` — the float64-identity band three of the
    four original adopters (evaluator, predictor, research) already used as
    their own default; the backtester always states its tolerance explicitly.
    """

    name: str
    description: str
    inputs: dict
    expected: float
    compute: Callable[[], float]
    tolerance: float = 1e-9
    known_gap: bool = False
    gap_issue: str | None = None


# ════════════════════════════════════════════════════════════════════════════
# Provenance header — the reason this is an INSTRUMENT check, not a code check
# ════════════════════════════════════════════════════════════════════════════

def resolved_library_versions(distributions: tuple[str, ...]) -> dict[str, str]:
    """The installed version of every tracked distribution loaded at runtime.

    ``importlib.metadata.version`` reads the DISTRIBUTION metadata pip actually
    resolved — the thing that moves between the CI runner and the deployed
    instrument (a spot box's fresh ``pip install``, a Lambda image). A missing
    distribution is recorded explicitly, never omitted — an absent key and a
    missing library must not look the same.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for dist in distributions:
        try:
            resolved[dist] = version(dist)
        except PackageNotFoundError:
            resolved[dist] = "<not installed>"
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            resolved[dist] = f"<unavailable: {type(exc).__name__}>"
    return resolved


def code_sha(package_root: str | os.PathLike[str] | None = None) -> str:
    """The SHA of the code that ran, without shelling out.

    Deploy-time stamps first (``GIT_SHA`` env, then the fleet's
    ``/var/task/GIT_SHA.txt`` Lambda-image convention), then the checkout's own
    ``.git`` refs for a spot box or a laptop. ``package_root`` is the directory
    whose ``../.git`` should be walked — each caller passes its own repo root
    (typically ``Path(__file__).resolve().parents[1]``), since the running code
    and this library live in different checkouts. ``unknown`` is a legitimate
    answer and is recorded as one — a fabricated SHA is worse than an absent one.
    """
    for env_key in ("GIT_SHA", "CODE_SHA", "GITHUB_SHA"):
        stamped = os.environ.get(env_key)
        if stamped:
            return stamped.strip()
    try:
        lambda_stamp = Path("/var/task/GIT_SHA.txt")
        if lambda_stamp.is_file():
            stamped = lambda_stamp.read_text().strip()
            if stamped:
                return stamped
    except Exception:  # noqa: BLE001, S110 — provenance never blocks the battery
        pass
    if package_root is None:
        return "unknown"
    try:
        git_dir = Path(package_root) / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0].strip()
        return "unknown"
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _call_with_timeout(fn: Callable[[], float], seconds: float) -> float:
    """Run ``fn`` under a wall-clock budget.

    A SIGALRM budget is installed when this is the main thread of a platform
    that has one (a spot box's or Lambda's handler thread qualifies). Where it
    is not available, the elapsed time is checked after the call instead — that
    cannot interrupt a hang, but it does catch an overrun, and the caller
    distinguishes neither: both are FAIL.
    """
    can_interrupt = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        started = time.monotonic()
        value = fn()
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            raise SelfTestTimeout(
                f"case exceeded its {seconds:g}s budget "
                f"({elapsed:.1f}s, detected after the fact)"
            )
        return value

    def _fire(_signum, _frame):
        raise SelfTestTimeout(f"case exceeded its {seconds:g}s budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def run_self_test(
    run_date: str | None = None,
    *,
    case_provider: Callable[[], list[Case]],
    component: str,
    schema: str,
    resolved_libraries: dict[str, str],
    code_sha_value: str,
    case_timeout_seconds: float = CASE_TIMEOUT_SECONDS,
    extra_case_providers: dict[str, Callable[[], list[Case]]] | None = None,
    primary_scope: str | None = None,
    extra_header: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a known-answer battery on the DEPLOYED instrument and return the artifact body.

    Never raises — see the module docstring's CONTRACT section. ``component`` and
    ``schema`` are the caller's own identity strings (e.g. ``"backtester"`` /
    ``"backtest_self_test-1.0.0"``); ``resolved_libraries`` and ``code_sha_value``
    are passed in rather than resolved here, since each caller tracks a different
    set of distributions and a different repo root — call
    :func:`resolved_library_versions` / :func:`code_sha` beforehand and pass the
    results through.

    ``extra_case_providers`` maps a SCOPE NAME to an additional case builder, for
    an adopter (the predictor) whose battery is split across a Lambda-reachable
    scope and a training-only scope. ``primary_scope`` names the battery's own
    scope (e.g. ``"inference"``) and, when given, the artifact's ``scope`` field
    always lists it plus whichever ``extra_case_providers`` keys actually ran — a
    partial-scope PASS must never be readable as full coverage. Leave
    ``primary_scope`` ``None`` (the default) for a single-scope battery that
    never publishes a ``scope`` field at all, matching the earlier adopters.

    ``extra_header`` merges additional caller-specific fields into the header
    (e.g. a ``scope_note``) without this module needing to know about them.
    """
    started = time.monotonic()
    scopes = [primary_scope] + sorted(extra_case_providers or {}) if primary_scope else None
    header: dict[str, Any] = {
        "schema": schema,
        "component": component,
        "run_date": run_date,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha": code_sha_value,
        "libraries": resolved_libraries,
        "case_timeout_seconds": case_timeout_seconds,
    }
    if scopes is not None:
        header["scope"] = scopes
    if extra_header:
        header.update(extra_header)

    def _provider() -> list[Case]:
        collected = list(case_provider())
        for _scope, extra in sorted((extra_case_providers or {}).items()):
            collected.extend(extra())
        return collected

    try:
        # Materialised inside the try: a provider returning a lazy or broken
        # iterable would otherwise raise at the FOR loop below, outside every
        # handler, and take down the stage this must never be able to fail.
        cases = list(_provider())
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "self-test: the battery could not be constructed (%s: %s) — verdict "
            "UNKNOWN. No correctness guarantee is granted this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {**header, "status": "error", "verdict": UNKNOWN, "cases": [],
                "n_cases": 0, "n_failed": 0, "n_errored": 0, "n_known_gaps": 0,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": round(time.monotonic() - started, 3)}

    records: list[dict] = []
    for index, case in enumerate(cases):
        try:
            record: dict[str, Any] = {
                "case": case.name,
                "description": case.description,
                "inputs": case.inputs,
                "expected": case.expected,
                "actual": None,
                "abs_error": None,
                "tolerance": case.tolerance,
                "verdict": UNKNOWN,
            }
        except Exception as exc:  # noqa: BLE001 — see CONTRACT: never raises
            # A provider that returned a well-formed LIST of malformed items gets
            # past the materialisation guard above and would otherwise raise
            # here, outside every handler — taking down the stage this module
            # must never be able to fail. Recorded as UNKNOWN, never as a pass.
            logger.error(
                "self-test: case %d is not a Case (%s: %s) => UNKNOWN",
                index, type(exc).__name__, exc,
            )
            records.append({
                "case": f"<malformed case {index}>",
                "description": "the case provider returned a non-Case object",
                "inputs": {"units": "n/a"},
                "expected": None, "actual": None, "abs_error": None,
                "tolerance": None, "verdict": UNKNOWN, "errored": True,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": 0.0,
            })
            continue

        if case.known_gap:
            # Stated in the artifact, in words, so a reader never reads this
            # row's PASS as an endorsement of the behaviour it pins.
            record["known_gap"] = True
            record["gap_issue"] = case.gap_issue
            record["known_gap_note"] = (
                "This case PINS behaviour believed incorrect at its MEASURED "
                "value so further drift goes red. PASS means 'unchanged since "
                f"recorded', NOT 'this is correct'. Tracked in {case.gap_issue}."
            )
        case_started = time.monotonic()
        try:
            actual = float(_call_with_timeout(case.compute, case_timeout_seconds))
            error = abs(actual - case.expected)
            record["actual"] = actual
            record["abs_error"] = error
            record["verdict"] = PASS if error <= case.tolerance else FAIL
            if record["verdict"] == FAIL:
                logger.error(
                    "self-test case FAILED: %s expected=%r actual=%r abs_error=%r "
                    "tolerance=%r", case.name, case.expected, actual, error,
                    case.tolerance,
                )
        except SelfTestTimeout as exc:
            # Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN. A
            # known-answer case that cannot finish in its budget is itself
            # evidence something is wrong with the instrument — it must not buy
            # the benefit of the doubt that "the battery could not be
            # constructed" gets.
            record["verdict"] = FAIL
            record["timed_out"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error("self-test case TIMED OUT (=> FAIL): %s (%s)", case.name, exc)
        except Exception as exc:  # noqa: BLE001 — a case that could not RUN is UNKNOWN
            record["verdict"] = UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "self-test case ERRORED (=> UNKNOWN): %s (%s: %s)",
                case.name, type(exc).__name__, exc, exc_info=True,
            )
        record["wall_clock_seconds"] = round(time.monotonic() - case_started, 3)
        records.append(record)

    n_failed = sum(1 for r in records if r["verdict"] == FAIL)
    n_errored = sum(1 for r in records if r["verdict"] == UNKNOWN)
    n_known_gaps = sum(1 for r in records if r.get("known_gap"))
    if n_failed:
        verdict = FAIL
    elif n_errored or not records:
        verdict = UNKNOWN
    else:
        verdict = PASS

    body = {**header, "status": "ok", "verdict": verdict, "cases": records,
            "n_cases": len(records), "n_failed": n_failed, "n_errored": n_errored,
            "n_known_gaps": n_known_gaps,
            "wall_clock_seconds": round(time.monotonic() - started, 3)}

    if verdict == PASS:
        logger.info(
            "self-test PASS — %d/%d known-answer cases agreed on %s (%s); "
            "%d of them PIN a known gap rather than endorse it",
            len(records), len(records), header["python"],
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
            n_known_gaps,
        )
    elif verdict == UNKNOWN:
        logger.error(
            "self-test UNKNOWN — %d/%d cases could not run. The correctness "
            "guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(records),
        )
    else:
        logger.error(
            "self-test FAIL — %d/%d known-answer cases DISAGREE with their "
            "hand-derived expectation on the DEPLOYED libraries (%s). THIS "
            "CYCLE'S NUMBERS ARE NOT TRUSTWORTHY.",
            n_failed, len(records),
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
        )
    return body


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS — ``None`` and ``"ok"`` withhold the guarantee."""
    return verdict == PASS
