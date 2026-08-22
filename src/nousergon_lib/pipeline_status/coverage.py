"""The reader of the stage-coverage surface — expected set versus what landed.

``krepis.stage_coverage`` makes every stage record its own verdict. Nothing
read the resulting surface. This is that reader.

## The defect this module exists to close (``alpha-engine-config-I8154``)

``krepis/src/krepis/stage_coverage.py`` documents its own metric as *"the
absence of the datapoint is itself visible as a gap — ``no data`` is never
rendered as green"*. Measured 2026-08-22, there was **no alarm on it**::

    $ aws cloudwatch describe-alarms \\
        --query 'MetricAlarms[?MetricName==`StageCoverage`].AlarmName'
    (empty)

and no consumer anywhere compared the expected stage set against the verdicts
that actually landed. All three things reading the mechanism read a *single
stage's own* verdict, in observe mode:

- ``crucible-dashboard/health_checker.py::_assert_stage_coverage`` — per-stage,
  self-reported, explicitly "can never fail the stage".
- ``nousergon-data/infrastructure/spot_*.sh`` — each box stage asserting itself.
- ``alpha-engine-config/scripts/check_stage_coverage_drift.py`` — compares the
  registry's ``pipeline_stages`` against the SF *definition*. Both sides are
  static files; it never reads S3.

So nothing asked *"did every stage of run X report, and what did they say?"*
— the only question that catches either of the two classes that blindness hid:

1. **Four stages had never recorded a verdict at all** (``Director``,
   ``ReportCard``, ``EvaluatorDeployDriftCheck``,
   ``EvaluatorDirectorDeployDriftCheck``) — denied by IAM since the prefix was
   created, for eight days (``alpha-engine-config-I8152``). Their CloudWatch
   datapoints published normally, so the metric surface and the S3 surface
   disagreed and nothing compared them.
2. **Three stages reported ``MISSING``/``is_finding: true`` and nothing paged**
   — ``Backtester`` (8 of 14 declared artifacts absent), ``Scanner`` (1),
   ``RegimeSubstrate`` (1, on two consecutive runs).

## Why this reads S3 and not the metric

**The metric cannot distinguish "denied" from "never entered."** A stage whose
role lacks ``s3:PutObject`` publishes its CloudWatch datapoint normally and
writes no verdict object; a stage the graph never entered publishes nothing
and writes nothing. On the metric surface the first looks healthy and the
second looks like a gap; on the S3 surface both are the same absence — and it
is the *expected set* that tells them apart. So the sweep reads the objects.

## The three counts, and why ABSENT is the serious one

:class:`CoverageSweep` reports ``covered`` / ``findings`` / ``absent``.

``absent`` — expected, no verdict object — is the most serious of the three
and pages **at all**, with no threshold. It is the one state that looks
identical to a stage that never ran, and it is the state the recording
mechanism itself produces when it is broken. A findings count has a
distribution; an absent count must be zero by construction.

``findings`` pages above a declared threshold (:data:`DEFAULT_FINDING_THRESHOLD`,
zero — any finding pages). The threshold exists so it can be *raised* with a
written reason, never so the default is permissive.

## The denominator, and the one thing that must not be guessed

Expected = the artifact registry's ``pipeline_stages`` INTERSECTED with the
states the cycle's contributing executions actually entered
(:func:`~.cycle_shape.read_cycle_shape`). Both halves are load-bearing:

- **Registry alone** over-counts. The weekly graph has ~49 declared stages and
  a given run enters a subset — conditional parity stages, the first-Saturday
  judge submit, the degraded-path twins. Reporting 38 absent on a healthy run
  is the noise that trains a reader to ignore the surface.
- **Entered set alone** under-counts, and worse, it cannot see the
  ``I8152`` class at all: a stage that entered, ran, and was denied its write
  is exactly a stage in the entered set with no verdict.

When the entered set cannot be read, the denominator falls back to the
declared set and :attr:`CoverageSweep.denominator_source` says so. That is
**not** a silent degrade: a sweep running on the declared-only denominator is
itself reported and pages, because a coverage reader that cannot establish its
own denominator is unobserved, not healthy.

Public surface:

- :class:`RowState` / :class:`StageRow` / :class:`CoverageSweep`
- :func:`sweep_coverage` — the pure core.
- :func:`read_coverage_sweep` — the boto3 front door.
- :func:`publish_sweep` — metrics + the S3 console artifact.
- :func:`render_rows` — the console/CLI table.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from .cycle_shape import CycleShape, read_cycle_shape

if TYPE_CHECKING:  # pragma: no cover
    from mypy_boto3_stepfunctions.client import SFNClient
else:  # pragma: no cover
    SFNClient = Any

logger = logging.getLogger(__name__)

__all__ = [
    "COVERED_STATUSES",
    "DEFAULT_FINDING_THRESHOLD",
    "METRIC_NAMESPACE",
    "SWEEP_ARTIFACT_PREFIX",
    "VERDICT_PREFIX",
    "CoverageSweep",
    "RowState",
    "StageRow",
    "publish_sweep",
    "read_coverage_sweep",
    "render_rows",
    "sweep_coverage",
]

#: Prefix ``krepis.stage_coverage`` writes per-stage verdicts under.
VERDICT_PREFIX = "_stage_coverage"

#: Prefix this sweep's own per-run artifact lands under — the console's row
#: source and the durable record of what the sweep saw.
SWEEP_ARTIFACT_PREFIX = "_stage_coverage/_sweep"

#: CloudWatch namespace. Same as the per-stage metric so one dashboard covers
#: the mechanism and its reader.
METRIC_NAMESPACE = "AlphaEngine"

#: Verdict statuses that are a pass. ``COVERED_NO_OUTPUT`` is a pass by
#: POSITIVE declaration — the stage said it writes nothing — which is the
#: distinction the registry's stage section exists to draw.
COVERED_STATUSES = frozenset({"COVERED", "COVERED_NO_OUTPUT"})

#: Findings above this count page. Zero: any finding pages. Raising it is a
#: declaration with a written reason, never a default.
DEFAULT_FINDING_THRESHOLD = 0


class RowState(str, Enum):
    """What the sweep concluded about one expected stage. Closed, no default."""

    #: A verdict landed and it was a pass.
    COVERED = "covered"
    #: A verdict landed and it declared itself a finding.
    FINDING = "finding"
    #: The stage was expected and NO verdict object exists. The serious one.
    ABSENT = "absent"
    #: A verdict landed but establishes no claim — the module's own
    #: ``UNMEASURED``, or a status that is neither a pass nor a declared
    #: finding (a ``STALE`` still carrying ``is_finding: false``). Never a
    #: pass: not-covered-and-not-declared-a-finding is an absence of evidence.
    UNMEASURED = "unmeasured"
    #: Declared in the registry, not entered by any contributing execution.
    #: Outside the denominator, reported so the reader can see the shape of
    #: the run rather than a silently smaller world.
    NOT_ENTERED = "not_entered"


@dataclass(frozen=True)
class StageRow:
    """One declared stage, as the sweep found it."""

    stage: str
    state: RowState
    stage_class: str = ""
    declared_output: str = ""
    verdict_status: str | None = None
    reason: str = ""
    covered_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    stale_artifacts: tuple[str, ...] = ()
    unmeasured_artifacts: tuple[str, ...] = ()
    verdict_key: str | None = None
    recorded_at: str = ""

    @property
    def is_finding(self) -> bool:
        return self.state is RowState.FINDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state": self.state.value,
            "stage_class": self.stage_class,
            "declared_output": self.declared_output,
            "verdict_status": self.verdict_status,
            "reason": self.reason,
            "covered_artifacts": list(self.covered_artifacts),
            "missing_artifacts": list(self.missing_artifacts),
            "stale_artifacts": list(self.stale_artifacts),
            "unmeasured_artifacts": list(self.unmeasured_artifacts),
            "verdict_key": self.verdict_key,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class CoverageSweep:
    """The per-run answer to "did every stage report, and what did it say?"."""

    pipeline: str
    run_date: str
    rows: tuple[StageRow, ...]
    #: ``entered_states`` (registry ∩ what the cycle entered) or
    #: ``declared_only`` (the entered set could not be read).
    denominator_source: str
    denominator_reason: str = ""
    cycle: CycleShape | None = field(default=None, repr=False)
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD
    swept_at: str = ""

    def _in_denominator(self) -> tuple[StageRow, ...]:
        return tuple(r for r in self.rows if r.state is not RowState.NOT_ENTERED)

    @property
    def expected(self) -> tuple[str, ...]:
        return tuple(r.stage for r in self._in_denominator())

    @property
    def covered(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.COVERED)

    @property
    def findings(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.FINDING)

    @property
    def absent(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.ABSENT)

    @property
    def unmeasured(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.UNMEASURED)

    @property
    def not_entered(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.NOT_ENTERED)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "expected": len(self.expected),
            "covered": self.covered,
            "findings": self.findings,
            "absent": self.absent,
            "unmeasured": self.unmeasured,
            "not_entered": self.not_entered,
        }

    @property
    def alert_conditions(self) -> tuple[str, ...]:
        """Every reason this sweep pages, named. Empty ⇒ it does not page."""
        conditions: list[str] = []
        if self.denominator_source != "entered_states":
            conditions.append(
                f"denominator_unestablished: {self.denominator_reason or 'entered set unreadable'}"
            )
        if self.absent:
            names = ", ".join(r.stage for r in self.rows if r.state is RowState.ABSENT)
            conditions.append(
                f"absent_verdicts={self.absent} ({names}) — expected, entered, and no "
                "verdict object; indistinguishable from a stage that never ran"
            )
        if self.findings > self.finding_threshold:
            names = ", ".join(r.stage for r in self.rows if r.state is RowState.FINDING)
            conditions.append(
                f"findings={self.findings} > threshold {self.finding_threshold} ({names})"
            )
        return tuple(conditions)

    @property
    def should_alert(self) -> bool:
        return bool(self.alert_conditions)

    def explain(self) -> str:
        c = self.counts
        head = (
            f"{self.pipeline} coverage {self.run_date}: "
            f"{c['covered']} covered / {c['findings']} findings / {c['absent']} absent "
            f"of {c['expected']} expected"
        )
        extra = f" ({c['unmeasured']} unmeasured, {c['not_entered']} not entered)"
        if self.denominator_source != "entered_states":
            extra += f" [denominator: {self.denominator_source}]"
        if not self.should_alert:
            return head + extra + " — no finding"
        return head + extra + " — PAGES: " + " | ".join(self.alert_conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_date": self.run_date,
            "swept_at": self.swept_at,
            "denominator_source": self.denominator_source,
            "denominator_reason": self.denominator_reason,
            "finding_threshold": self.finding_threshold,
            "counts": self.counts,
            "should_alert": self.should_alert,
            "alert_conditions": list(self.alert_conditions),
            "rows": [r.to_dict() for r in self.rows],
            "cycle": self.cycle.to_dict() if self.cycle else None,
            "explain": self.explain(),
        }


def _strs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return tuple(out)


def sweep_coverage(
    *,
    pipeline: str,
    run_date: str,
    registry: Mapping[str, Any],
    verdicts: Mapping[str, Mapping[str, Any]],
    entered_states: Iterable[str] | None,
    entered_reason: str = "",
    cycle: CycleShape | None = None,
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD,
    now: datetime | None = None,
) -> CoverageSweep:
    """Compare the declared stage set against the verdicts that landed. Pure.

    ``verdicts`` maps stage name to the parsed verdict object. A stage in
    ``verdicts`` but absent from the registry is **reported**, not dropped —
    a verdict for a stage nobody declared means the two declarations have
    drifted, and silently ignoring it is how the drift stays invisible.
    """
    now = now or datetime.now(timezone.utc)
    declared = [row for row in (registry.get("pipeline_stages") or []) if row.get("stage")]

    if entered_states is None:
        entered: set[str] | None = None
        denominator_source = "declared_only"
        denominator_reason = entered_reason or "entered-state set could not be read"
    else:
        entered = {str(s) for s in entered_states}
        denominator_source = "entered_states"
        denominator_reason = ""

    rows: list[StageRow] = []
    for row in declared:
        stage = str(row["stage"])
        stage_class = str(row.get("stage_class", ""))
        declared_output = str(row.get("output", ""))
        verdict = verdicts.get(stage)

        if verdict is None:
            if entered is not None and stage not in entered:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.NOT_ENTERED,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason="declared, not entered by any contributing execution",
                    )
                )
            else:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.ABSENT,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason=(
                            "no verdict object under "
                            f"{VERDICT_PREFIX}/{run_date}/ — the stage "
                            + (
                                "entered and recorded nothing"
                                if entered is not None
                                else "may have entered and recorded nothing"
                            )
                        ),
                    )
                )
            continue

        status = str(verdict.get("status", "")) or None
        if status in COVERED_STATUSES:
            state = RowState.COVERED
        elif bool(verdict.get("is_finding")):
            state = RowState.FINDING
        else:
            state = RowState.UNMEASURED

        rows.append(
            StageRow(
                stage=stage,
                state=state,
                stage_class=stage_class or str(verdict.get("stage_class", "")),
                declared_output=declared_output or str(verdict.get("declared_output", "")),
                verdict_status=status,
                reason=str(verdict.get("reason", "")),
                covered_artifacts=_strs(verdict.get("covered")),
                missing_artifacts=_strs(verdict.get("missing")),
                stale_artifacts=_strs(verdict.get("stale")),
                unmeasured_artifacts=_strs(verdict.get("unmeasured")),
                verdict_key=f"{VERDICT_PREFIX}/{run_date}/{stage}.json",
                recorded_at=str(verdict.get("recorded_at", "")),
            )
        )

    declared_names = {str(r["stage"]) for r in declared}
    for stage in sorted(set(verdicts) - declared_names):
        verdict = verdicts[stage]
        rows.append(
            StageRow(
                stage=stage,
                state=RowState.UNMEASURED,
                verdict_status=str(verdict.get("status", "")) or None,
                reason=(
                    "a verdict landed for a stage with no pipeline_stages row — the "
                    "registry and the state machine have drifted; see "
                    "alpha-engine-config/scripts/check_stage_coverage_drift.py"
                ),
                verdict_key=f"{VERDICT_PREFIX}/{run_date}/{stage}.json",
                recorded_at=str(verdict.get("recorded_at", "")),
            )
        )

    return CoverageSweep(
        pipeline=pipeline,
        run_date=run_date,
        rows=tuple(rows),
        denominator_source=denominator_source,
        denominator_reason=denominator_reason,
        cycle=cycle,
        finding_threshold=finding_threshold,
        swept_at=now.astimezone(timezone.utc).isoformat(),
    )


# ── S3 front door ────────────────────────────────────────────────────────────


def _load_verdicts(
    s3_client: Any, *, bucket: str, run_date: str, prefix: str = VERDICT_PREFIX
) -> dict[str, dict[str, Any]]:
    """Read every verdict object under ``<prefix>/<run_date>/``.

    A key that lists but will not parse is kept with an explicit
    ``status: UNREADABLE``: dropping it would make a corrupt verdict
    indistinguishable from an absent one, and only the second is the
    ``I8152`` class.
    """
    out: dict[str, dict[str, Any]] = {}
    token: str | None = None
    base = f"{prefix}/{run_date}/"
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": base}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3_client.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            stage = key[len(base) : -len(".json")]
            if not stage or "/" in stage:
                continue
            try:
                body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
                parsed = json.loads(body)
                if isinstance(parsed, str):  # defensively unwrap double-encoding
                    parsed = json.loads(parsed)
            except Exception as exc:  # noqa: BLE001 — recorded, never dropped
                logger.error("coverage sweep: verdict %s unreadable", key, exc_info=True)
                out[stage] = {
                    "status": "UNREADABLE",
                    "is_finding": False,
                    "reason": f"verdict object unreadable: {type(exc).__name__}: {exc}",
                }
                continue
            if isinstance(parsed, dict):
                out[stage] = parsed
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return out


def read_coverage_sweep(
    *,
    pipeline: str,
    run_date: str,
    state_machine_arn: str | None = None,
    registry: Mapping[str, Any] | None = None,
    bucket: str = "alpha-engine-research",
    s3_client: Any = None,
    sfn_client: SFNClient | None = None,
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD,
    now: datetime | None = None,
) -> CoverageSweep:
    """Read the registry, the verdicts and the cycle, and sweep them."""
    if s3_client is None:  # pragma: no cover — production path
        import boto3

        from krepis.aws_region import resolve_region

        s3_client = boto3.client("s3", region_name=resolve_region())

    if registry is None:
        from krepis import stage_coverage as sc

        registry = sc.load_registry(s3_client, bucket=bucket)

    verdicts = _load_verdicts(s3_client, bucket=bucket, run_date=run_date)

    cycle: CycleShape | None = None
    entered: list[str] | None = None
    entered_reason = ""
    if state_machine_arn:
        try:
            cycle = read_cycle_shape(state_machine_arn, run_date, client=sfn_client)
        except Exception as exc:  # noqa: BLE001 — degrades LOUDLY, never silently
            entered_reason = f"{type(exc).__name__}: {exc}"
            logger.error(
                "coverage sweep: could not read the cycle's entered states for %s %s "
                "— falling back to the DECLARED denominator, which pages",
                pipeline,
                run_date,
                exc_info=True,
            )
        else:
            seen: list[str] = []
            for execution in cycle.executions:
                seen.extend(execution.all_states_entered)
            entered = seen
    else:
        entered_reason = "no state machine ARN supplied — cannot read the entered-state set"

    return sweep_coverage(
        pipeline=pipeline,
        run_date=run_date,
        registry=registry,
        verdicts=verdicts,
        entered_states=entered,
        entered_reason=entered_reason,
        cycle=cycle,
        finding_threshold=finding_threshold,
        now=now,
    )


# ── Emission ─────────────────────────────────────────────────────────────────


def sweep_artifact_key(sweep: CoverageSweep, *, prefix: str = SWEEP_ARTIFACT_PREFIX) -> str:
    return f"{prefix}/{sweep.pipeline}/{sweep.run_date}.json"


def publish_sweep(
    sweep: CoverageSweep,
    *,
    s3_client: Any = None,
    cloudwatch_client: Any = None,
    bucket: str = "alpha-engine-research",
    prefix: str = SWEEP_ARTIFACT_PREFIX,
) -> None:
    """Persist the sweep and publish its counts. Never raises.

    ``StageCoverageSweepRan`` is published unconditionally and FIRST. The
    surface has to be observable itself (``observability-policy.md`` §9): a
    sweep that stops running publishes no count at all, and an alarm on the
    absence of *that* datapoint is the only thing that can tell a clean week
    from a dead reader.
    """
    if cloudwatch_client is not None:
        dims = [
            {"Name": "Pipeline", "Value": sweep.pipeline},
        ]
        data = [{"MetricName": "StageCoverageSweepRan", "Dimensions": dims, "Value": 1.0, "Unit": "None"}]
        for name, value in (
            ("StageCoverageSweepExpected", sweep.counts["expected"]),
            ("StageCoverageSweepCovered", sweep.covered),
            ("StageCoverageSweepFindings", sweep.findings),
            ("StageCoverageSweepAbsent", sweep.absent),
            ("StageCoverageSweepUnmeasured", sweep.unmeasured),
        ):
            data.append(
                {"MetricName": name, "Dimensions": dims, "Value": float(value), "Unit": "Count"}
            )
        try:
            cloudwatch_client.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=data)
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error("coverage sweep: FAILED to publish sweep metrics", exc_info=True)

    if s3_client is not None:
        key = sweep_artifact_key(sweep, prefix=prefix)
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(sweep.to_dict(), indent=2, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error(
                "coverage sweep: FAILED to write the sweep artifact to s3://%s/%s",
                bucket,
                key,
                exc_info=True,
            )


_STATE_GLYPH = {
    RowState.COVERED: "OK  ",
    RowState.FINDING: "FIND",
    RowState.ABSENT: "GONE",
    RowState.UNMEASURED: "????",
    RowState.NOT_ENTERED: "----",
}


def render_rows(sweep: CoverageSweep, *, include_not_entered: bool = True) -> str:
    """Per-stage table — the console surface and the CLI's stdout.

    ``NOT_ENTERED`` rows are rendered by default. A surface that hides the
    stages a run did not reach reports a true number about a smaller world
    than its name implies.
    """
    order = {
        RowState.ABSENT: 0,
        RowState.FINDING: 1,
        RowState.UNMEASURED: 2,
        RowState.COVERED: 3,
        RowState.NOT_ENTERED: 4,
    }
    rows = [r for r in sweep.rows if include_not_entered or r.state is not RowState.NOT_ENTERED]
    rows.sort(key=lambda r: (order[r.state], r.stage))
    width = max((len(r.stage) for r in rows), default=5)
    lines = [sweep.explain(), ""]
    for row in rows:
        status = row.verdict_status or "-"
        lines.append(f"  {_STATE_GLYPH[row.state]}  {row.stage:<{width}}  {status:<18}  {row.reason}")
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m nousergon_lib.pipeline_status.coverage``.

    Exit codes: 0 clean · 2 the sweep pages · 3 the sweep itself could not run.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="nousergon_lib.pipeline_status.coverage")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--state-machine-arn", default=None)
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--registry-path", default=None, help="read the registry from disk")
    parser.add_argument("--finding-threshold", type=int, default=DEFAULT_FINDING_THRESHOLD)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--publish", action="store_true", help="write the sweep artifact and publish its metrics"
    )
    parser.add_argument(
        "--alert", action="store_true", help="publish an alert when the sweep pages"
    )
    parser.add_argument(
        "--augment-marker",
        action="store_true",
        help=(
            "merge the cycle's real shape into the SF completion marker "
            "(alpha-engine-config-I8186) — the sweep already read the cycle to "
            "establish its own denominator, so this costs one GET and one PUT"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import boto3

    from krepis.aws_region import resolve_region

    region = resolve_region()
    s3_client = boto3.client("s3", region_name=region)

    registry = None
    if args.registry_path:
        from krepis import stage_coverage as sc

        registry = sc.load_registry(None, local_path=args.registry_path)

    try:
        sweep = read_coverage_sweep(
            pipeline=args.pipeline,
            run_date=args.run_date,
            state_machine_arn=args.state_machine_arn,
            registry=registry,
            bucket=args.bucket,
            s3_client=s3_client,
            finding_threshold=args.finding_threshold,
        )
    except Exception as exc:  # noqa: BLE001 — a sweep that cannot run says so
        print(
            f"ERROR: coverage sweep {args.pipeline} {args.run_date} COULD NOT RUN — "
            f"{type(exc).__name__}: {exc}. This is not a clean result.",
            file=__import__("sys").stderr,
        )
        return 3

    print(json.dumps(sweep.to_dict(), indent=2, default=str) if args.json else render_rows(sweep))

    if args.publish:
        publish_sweep(
            sweep,
            s3_client=s3_client,
            cloudwatch_client=boto3.client("cloudwatch", region_name=region),
            bucket=args.bucket,
        )

    if args.augment_marker:
        if sweep.cycle is None:
            print(
                "WARNING: --augment-marker was asked for but the cycle could not be "
                "read, so the marker keeps its bare envelope claim. A marker with no "
                "cycle block resolves to UNKNOWN, never to a pass.",
                file=__import__("sys").stderr,
            )
        else:
            from .completion_marker import augment_marker

            augment_marker(sweep.cycle, s3_client=s3_client, bucket=args.bucket)

    if args.alert and sweep.should_alert:
        from krepis import alerts

        alerts.publish(
            sweep.explain(),
            severity="error",
            source=f"stage-coverage-sweep/{args.pipeline}",
            dedup_key=f"stage-coverage-sweep/{args.pipeline}/{args.run_date}",
        )

    return 2 if sweep.should_alert else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
