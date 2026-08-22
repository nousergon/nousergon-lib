"""The coverage sweep, replayed against the 2026-08-22 weekly run.

``alpha-engine-config-I8154``. Every fixture under
``tests/fixtures/coverage_sweep_260822/`` is a VERBATIM capture of live state
taken 2026-08-22 — the registry's ``pipeline_stages`` section, the verdict
objects the run itself wrote to ``s3://alpha-engine-research/_stage_coverage/``,
and the entered-state union of the cycle's four contributing executions read
from ``GetExecutionHistory``. Nothing here is invented, so a test that passes
is a claim about a real run rather than about a mock.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from nousergon_lib.pipeline_status.coverage import (
    DEFAULT_FINDING_THRESHOLD,
    CoverageSweep,
    RowState,
    _load_verdicts,
    publish_sweep,
    render_rows,
    sweep_artifact_key,
    sweep_coverage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coverage_sweep_260822"
PIPELINE = "ne-weekly-freshness-pipeline"
RUN_DATE = "2026-08-22"
NOW = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    return yaml.safe_load((FIXTURES / "registry_pipeline_stages.yaml").read_text())


@pytest.fixture(scope="module")
def entered() -> list[str]:
    return json.loads((FIXTURES / "entered_states.json").read_text())


@pytest.fixture(scope="module")
def verdicts_unified() -> dict[str, Any]:
    """Every verdict the 2026-08-22 cycle wrote, keyed by stage.

    What the surface looks like once ``alpha-engine-config-I8155`` (krepis
    PR 180) has landed and one execution's verdicts share one prefix.
    """
    return json.loads((FIXTURES / "verdicts_unified.json").read_text())


@pytest.fixture(scope="module")
def verdicts_as_recorded() -> dict[str, Any]:
    """Only what landed under ``_stage_coverage/2026-08-22/`` on the day."""
    return json.loads((FIXTURES / "verdicts_as_recorded.json").read_text())


def _sweep(registry, verdicts, entered, **kw) -> CoverageSweep:
    return sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=registry,
        verdicts=verdicts,
        entered_states=entered,
        now=NOW,
        **kw,
    )


# ── The closes-when demonstration ────────────────────────────────────────────


def test_a_missing_verdict_pages_and_names_the_stage(registry, verdicts_unified, entered):
    """A stage that produced a MISSING verdict pages, by name.

    This is the class that ran unread on 2026-08-22: ``Backtester`` (8 of 14
    declared artifacts absent), ``Scanner`` (1) and ``RegimeSubstrate`` (1,
    on two consecutive runs) all reported ``is_finding: true`` and nothing
    anywhere paged.
    """
    sweep = _sweep(registry, verdicts_unified, entered)
    finders = {r.stage for r in sweep.rows if r.state is RowState.FINDING}
    assert {"Backtester", "Scanner", "RegimeSubstrate"} <= finders

    assert sweep.should_alert
    conditions = " ".join(sweep.alert_conditions)
    for stage in ("Backtester", "Scanner", "RegimeSubstrate"):
        assert stage in conditions, f"the page must NAME {stage}"


def test_a_stage_with_no_verdict_at_all_pages_as_absent(registry, verdicts_unified, entered):
    """``ReportCard`` entered, wrote nothing, and recorded no verdict.

    The ``alpha-engine-config-I8152`` class: four stages were denied
    ``s3:PutObject`` on the verdict prefix and recorded nothing for eight
    days while their CloudWatch datapoints published normally. Three of the
    four were repaired on 2026-08-22 and appear in the fixture; ``ReportCard``
    was not, so it is the live instance this test pins.
    """
    sweep = _sweep(registry, verdicts_unified, entered)
    absent = {r.stage for r in sweep.rows if r.state is RowState.ABSENT}
    assert "ReportCard" in absent

    assert sweep.absent > 0
    assert sweep.should_alert
    assert "ReportCard" in " ".join(sweep.alert_conditions)
    assert any("absent_verdicts" in c for c in sweep.alert_conditions)


def test_absent_pages_at_all_with_no_threshold(registry, entered):
    """One absent verdict pages. There is no absent threshold, by design."""
    sweep = sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=registry,
        verdicts={},
        entered_states=["ReportCard"],
        finding_threshold=10_000,
        now=NOW,
    )
    assert sweep.absent == 1
    assert sweep.findings == 0
    assert sweep.should_alert, "absent must page even with the finding threshold wide open"


# ── The denominator ──────────────────────────────────────────────────────────


def test_stages_the_run_never_entered_are_not_counted_absent(registry, verdicts_unified, entered):
    """A conditional stage that did not enter is NOT_ENTERED, never ABSENT.

    Reporting the eleven parity/judge stages a normal Saturday does not reach
    as coverage failures is the noise that trains a reader to ignore the
    surface — the outcome ``alpha-engine-config-I7180`` was filed about.
    """
    sweep = _sweep(registry, verdicts_unified, entered)
    not_entered = {r.stage for r in sweep.rows if r.state is RowState.NOT_ENTERED}
    assert "PitParityCompare" in not_entered
    assert "EvalJudgeSubmitFirstSaturday" in not_entered
    assert not_entered.isdisjoint({r.stage for r in sweep.rows if r.state is RowState.ABSENT})
    assert sweep.counts["expected"] + sweep.not_entered == len(sweep.rows)


def test_an_unreadable_denominator_pages_rather_than_passing(registry, verdicts_unified):
    """No entered set ⇒ ``declared_only`` ⇒ the sweep pages about itself.

    A coverage reader that cannot establish its own denominator is
    unobserved, not healthy (``observability-policy.md`` §9).
    """
    sweep = sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=registry,
        verdicts=verdicts_unified,
        entered_states=None,
        entered_reason="AccessDeniedException on GetExecutionHistory",
        now=NOW,
    )
    assert sweep.denominator_source == "declared_only"
    assert sweep.should_alert
    assert any("denominator_unestablished" in c for c in sweep.alert_conditions)
    assert "AccessDenied" in " ".join(sweep.alert_conditions)


def test_the_split_prefix_defect_is_visible_as_absence(registry, verdicts_as_recorded, entered):
    """Replaying only ``_stage_coverage/2026-08-22/`` reports 28 absent.

    ``alpha-engine-config-I8155``: one execution's verdicts landed under TWO
    date prefixes. Read at the run's own prefix, 28 stages that DID record a
    verdict are indistinguishable from stages that recorded nothing — which
    is precisely why the sweep's denominator is only trustworthy once
    krepis PR 180 has merged. The sweep is correct here; the surface it reads
    is not, and it says so loudly rather than reporting a smaller world.
    """
    sweep = _sweep(registry, verdicts_as_recorded, entered)
    assert sweep.absent == 28
    assert sweep.should_alert
    for stage in ("Backtester", "DataPhase1", "MorningEnrich", "Director"):
        assert stage in " ".join(sweep.alert_conditions)


# ── Classification ───────────────────────────────────────────────────────────


def test_covered_no_output_is_a_pass_by_positive_declaration(registry, verdicts_unified, entered):
    sweep = _sweep(registry, verdicts_unified, entered)
    row = next(r for r in sweep.rows if r.stage == "WeeklyRunDayGate")
    assert row.verdict_status == "COVERED_NO_OUTPUT"
    assert row.state is RowState.COVERED


def test_a_non_finding_stale_is_never_counted_covered(registry, verdicts_unified, entered):
    """``ReplayConcordance`` refreshed NOTHING and reported ``is_finding: false``.

    Until ``alpha-engine-config-I8166`` promotes a wholly-stale verdict to a
    finding, the sweep must still refuse to call it covered: not-covered and
    not-declared-a-finding is an absence of evidence, never a pass.
    """
    sweep = _sweep(registry, verdicts_unified, entered)
    row = next(r for r in sweep.rows if r.stage == "ReplayConcordance")
    assert row.verdict_status == "STALE"
    assert row.state is RowState.UNMEASURED
    assert row.state is not RowState.COVERED


def test_a_verdict_for_an_undeclared_stage_is_reported_not_dropped(registry, entered):
    sweep = sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=registry,
        verdicts={"GhostStage": {"status": "COVERED", "is_finding": False}},
        entered_states=entered,
        now=NOW,
    )
    row = next(r for r in sweep.rows if r.stage == "GhostStage")
    assert row.state is RowState.UNMEASURED
    assert "no pipeline_stages row" in row.reason


def test_a_clean_run_does_not_page(registry, entered):
    verdicts = {
        stage: {"status": "COVERED", "is_finding": False, "reason": "ok"}
        for stage in entered
    }
    sweep = _sweep(registry, verdicts, entered)
    assert sweep.absent == 0
    assert sweep.findings == 0
    assert not sweep.should_alert
    assert "no finding" in sweep.explain()


def test_finding_threshold_default_is_zero():
    assert DEFAULT_FINDING_THRESHOLD == 0


# ── Emission ─────────────────────────────────────────────────────────────────


class _FakeCW:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def put_metric_data(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        return {}


def test_publish_emits_the_three_counts_and_its_own_liveness(registry, verdicts_unified, entered):
    sweep = _sweep(registry, verdicts_unified, entered)
    cw, s3 = _FakeCW(), _FakeS3()
    publish_sweep(sweep, s3_client=s3, cloudwatch_client=cw)

    names = {m["MetricName"] for m in cw.calls[0]["MetricData"]}
    assert {
        "StageCoverageSweepRan",
        "StageCoverageSweepCovered",
        "StageCoverageSweepFindings",
        "StageCoverageSweepAbsent",
    } <= names

    body = json.loads(s3.puts[0]["Body"])
    assert body["counts"]["absent"] == sweep.absent
    assert body["should_alert"] is True
    assert s3.puts[0]["Key"] == sweep_artifact_key(sweep)
    assert sweep_artifact_key(sweep).endswith(f"{PIPELINE}/{RUN_DATE}.json")


def test_publish_never_raises_on_a_dead_client(registry, verdicts_unified, entered):
    class _Dead:
        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError("boom")

    sweep = _sweep(registry, verdicts_unified, entered)
    publish_sweep(sweep, s3_client=_Dead(), cloudwatch_client=_Dead())


def test_render_shows_every_row_including_the_ones_not_entered(registry, verdicts_unified, entered):
    sweep = _sweep(registry, verdicts_unified, entered)
    rendered = render_rows(sweep)
    assert "ReportCard" in rendered
    assert "PitParityCompare" in rendered
    assert rendered.index("GONE") < rendered.index("OK  ")


# ── The S3 reader ────────────────────────────────────────────────────────────


class _ListingS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = kwargs["Prefix"]
        return {
            "Contents": [{"Key": k} for k in self.objects if k.startswith(prefix)],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        body = self.objects[kwargs["Key"]]
        if body is None:
            raise RuntimeError("corrupt")

        class _Body:
            @staticmethod
            def read() -> bytes:
                return body

        return {"Body": _Body()}


def test_a_corrupt_verdict_is_unreadable_never_absent():
    """A verdict that will not parse must not masquerade as one that never landed."""
    s3 = _ListingS3(
        {
            f"_stage_coverage/{RUN_DATE}/Scanner.json": b'{"stage":"Scanner","status":"COVERED"}',
            f"_stage_coverage/{RUN_DATE}/Director.json": None,
        }
    )
    got = _load_verdicts(s3, bucket="b", run_date=RUN_DATE)
    assert got["Scanner"]["status"] == "COVERED"
    assert got["Director"]["status"] == "UNREADABLE"


def test_a_double_encoded_verdict_still_parses():
    s3 = _ListingS3(
        {f"_stage_coverage/{RUN_DATE}/Scanner.json": json.dumps(json.dumps({"status": "COVERED"})).encode()}
    )
    got = _load_verdicts(s3, bucket="b", run_date=RUN_DATE)
    assert got["Scanner"]["status"] == "COVERED"
