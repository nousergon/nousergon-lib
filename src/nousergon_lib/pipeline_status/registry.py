"""SF-state → archive-page registry + substantive-state filter primitives.

This module is the single source of truth for two cross-consumer questions:

1. **Which SF states are substantive?** — the `Wait*` polling companions and
   bare `Pass` / `Choice` / `Succeed` plumbing should not appear as their
   own rows on the operator console; they're internal control flow. The
   :data:`SUBSTANTIVE_RESOURCES` set + :data:`WAIT_GROUPING` map define
   the filter.

2. **Where does each state's persisted artifact live on the dashboard?** —
   each substantive Task state either produces an artifact that has a
   dedicated archive page (deep-link target) OR it's substrate-only. Per
   ``feedback_no_silent_fails`` the registry never returns a generic "no
   artifact" placeholder — substrate-only states carry an explicit
   :class:`ArtifactReason` string the page renders verbatim.

The registry is materialized as a flat dict-of-dataclasses rather than a
walked-from-SF-JSON projection because (a) the SF JSONs live in
``alpha-engine-data`` (cross-repo coupling we want to avoid at the lib
layer), and (b) the operator-meaningful labels + page slugs are editorial
choices that don't belong in the SF JSON anyway. A CI test in the
consuming repo (alpha-engine-dashboard or alpha-engine-data) asserts every
substantive Task state in the live SF JSONs has a registry entry; that's
how the two stay in sync without a runtime coupling.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ── Substantive-state filtering (§3.2 of the plan doc) ────────────────────


SUBSTANTIVE_RESOURCES: Final[frozenset[str]] = frozenset(
    {
        # Lambda invokes
        "arn:aws:states:::lambda:invoke",
        # SSM sendCommand (EC2 spot + trading instance commands)
        "arn:aws:states:::aws-sdk:ssm:sendCommand",
        # SNS publish (terminal-state emails — kept substantive so the
        # console shows whether the success/failure email actually fired)
        "arn:aws:states:::sns:publish",
        # EC2 lifecycle (StartExecutorEC2 + StartTradingInstance +
        # StopTradingInstance + ForceStopInstance)
        "arn:aws:states:::aws-sdk:ec2:startInstances",
        "arn:aws:states:::aws-sdk:ec2:stopInstances",
    }
)


# Every ``Wait*`` state in the SF JSONs is the polling companion to a parent
# ``sendCommand`` Task — the parent fires the SSM command and returns
# instantly; the wait state polls ``getCommandInvocation`` until terminal.
# For console rendering we want one row per logical step, durations measured
# parent_entry → wait_exit (see ``read._materialize_tasks`` for the math).
#
# This map is intentionally exhaustive (every Wait* state across all 3 SF
# JSONs) so the read layer can absorb wait companions without a runtime
# fallback. New Wait* states added in future SF edits must be added here AND
# to the registry below; the CI test (planned in dashboard Phase 2) asserts
# this round-trip.
WAIT_GROUPING: Final[dict[str, str]] = {
    # Weekly Freshness SF
    "WaitForMorningEnrich": "MorningEnrich",
    "WaitForDataPhase1": "DataPhase1",
    "WaitForRAGIngestion": "RAGIngestion",
    "WaitForPredictorTraining": "PredictorTraining",
    "WaitForBacktester": "Backtester",
    "WaitForPredictorBacktest": "PredictorBacktest",
    "WaitForPortfolioOptimizerBacktest": "PortfolioOptimizerBacktest",
    "WaitForParity": "Parity",
    "WaitForEvaluator": "Evaluator",
    "WaitForSaturdayHealthCheck": "SaturdayHealthCheck",
    "WaitForWeeklySubstrateHealthCheck": "WeeklySubstrateHealthCheck",
    "WaitForModelZoo": "ModelZooRotation",  # L4544 weekly model-zoo rotation
    # Pre-open Trading SF
    "WaitForMorningPlanner": "RunMorningPlanner",
    "WaitForDailyNews": "RunDailyNews",  # secondary daily news pull (fail-soft)
    "WaitForChronicGap": "ChronicGapSelfHeal",  # L4604 fail-soft heal split
    "WaitForMorningArcticAppend": "MorningArcticAppend",  # L4608 daily_append split
    "WaitForInstanceReady": "StartExecutorEC2",
    # Note: weekday SF's MorningEnrich shares its WaitForMorningEnrich with
    # the Saturday map above (same state name). Lookup-by-name is OK because
    # the parent name is the same in both SFs.
    # Post-close Trading SF
    "WaitForPostMarketData": "PostMarketData",
    # Top-of-pipeline executor-checkout refresh chokepoint (config#1549) —
    # the whole EOD run executes latest origin/main by construction.
    "WaitForRefreshExecutorDeploy": "RefreshExecutorDeploy",
    "WaitForPostMarketArcticAppend": "PostMarketArcticAppend",  # data #... EOD append split (2026-06-16)
    "WaitForCaptureSnapshot": "CaptureSnapshot",
    "WaitForEOD": "EODReconcile",
    "WaitForDailySubstrateHealthCheck": "DailySubstrateHealthCheck",
}


# ── Pretty-label registry (mirrors sf-telegram-notifier verbatim) ─────────


PIPELINE_LABELS: Final[dict[str, str]] = {
    "ne-weekly-freshness-pipeline": "Weekly Freshness SF",
    "ne-preopen-trading-pipeline": "Pre-open Trading SF",
    "ne-postclose-trading-pipeline": "Post-close Trading SF",
}


# ── Artifact registry types ───────────────────────────────────────────────


class ArchivePageRef(BaseModel):
    """Deep-link target for a substantive Task state that produces an
    operator-readable artifact.

    The ``page`` slug is the dashboard page module name (e.g.
    ``"19_EOD_Reconcile_Archive"`` — corresponds to
    ``alpha-engine-dashboard/pages/19_EOD_Reconcile_Archive.py``). The
    dashboard consumer constructs the full URL from its base host +
    page slug at render time; the lib does not bake URL hosts because the
    same page is reachable at ``console.nousergon.ai`` (private) and may
    or may not be reachable at ``live.nousergon.ai`` (public) depending
    on the page.

    ``artifact_label`` is the human-readable label for the deep-link cell
    on page 25 — e.g. "Morning briefing" rather than the bare page slug.

    ``kind`` is the discriminator field for the tagged-union round-trip
    in :class:`nousergon_lib.pipeline_status.read.TaskRow.archive`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["archive_page_ref"] = "archive_page_ref"
    page: str
    artifact_label: str


class ArtifactReason(BaseModel):
    """Explicit non-generic reason a substantive Task state has no archive
    page deep-link.

    Per ``feedback_no_silent_fails`` — substrate-only states must surface
    a specific reason ("Substrate refresh; no per-run artifact"), never
    a generic "no artifact" placeholder. The reason text is what the
    page 25 cell renders.

    ``kind`` is the discriminator field for the tagged-union round-trip
    in :class:`nousergon_lib.pipeline_status.read.TaskRow.archive`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["artifact_reason"] = "artifact_reason"
    reason: str


# Discriminated union for :class:`TaskRow.archive` — Pydantic V2 routes
# dict input to the right variant via the ``kind`` tag, so
# ``model_dump(mode="json")`` → ``model_validate`` round-trips reconstruct
# the typed instance (instead of leaving the dict raw, which the page-25
# ``isinstance`` checks would mis-classify as registry drift).
RegistryEntry = Annotated[
    Union[ArchivePageRef, ArtifactReason],
    Field(discriminator="kind"),
]


# ── The registry ──────────────────────────────────────────────────────────


# Every substantive Task state across the 3 SF JSONs maps to either an
# ArchivePageRef (operator-readable artifact has a dedicated page) or an
# ArtifactReason (substrate-only — explicit reason rendered verbatim).
#
# Sourced from plan doc §2.1 inventory + a `jq` walk of all 3 SF JSONs
# (Saturday 89 / Weekday 36 / EOD 21 total states; nested Parallel branches
# walked). Reviewed against ROADMAP L3050 + the post-2026-05-15
# artifact-archive pages (dashboard #86: pages 16-22).
STATE_TO_ARCHIVE_PAGE: Final[dict[str, Union[ArchivePageRef, ArtifactReason]]] = {
    # ── Weekly Freshness SF (24 substantive Task steps) ──────────────────────────
    "MorningEnrich": ArtifactReason(
        reason="Daily OHLCV write to predictor/daily_closes/{date}.parquet; "
        "no per-run rendered artifact — substrate for downstream stages."
    ),
    "DataPhase1": ArtifactReason(
        reason="Bulk weekly write to predictor/price_cache/, archive/macro/, "
        "ArcticDB universe library; no per-run rendered artifact — "
        "substrate refresh."
    ),
    "Scanner": ArtifactReason(
        reason="Standalone scanner Lambda (ROADMAP L1995 Phase 1-2, "
        "alpha-engine-research #235): writes candidates.json for the "
        "run_date as observe-only output, gated by "
        "$.enable_standalone_scanner. No consumer reads it today (Phase "
        "4 will flip RAG to read it; Phase 5 will flip Research). "
        "Failure is non-blocking — the SF Catch routes forward to "
        "CheckSkipRAGIngestion. Once Phase 4/5 lands, swap this entry "
        "for an ArchivePageRef pointing at the scanner-candidates page."
    ),
    "RAGIngestion": ArtifactReason(
        reason="SEC/8-K/earnings/theses corpus refresh in rag/corpus/; "
        "substrate-only — consumed at Research time."
    ),
    "RegimeSubstrate": ArchivePageRef(
        page="15_Regime",
        artifact_label="Regime substrate",
    ),
    "RegimeRetrospectiveEval": ArchivePageRef(
        page="15_Regime",
        artifact_label="Regime retrospective eval",
    ),
    "Research": ArchivePageRef(
        page="17_Research_Briefing_Archive",
        artifact_label="Morning research briefing",
    ),
    "DataPhase2": ArtifactReason(
        reason="Alt-data + fundamentals refresh; substrate-only, no per-run "
        "rendered artifact."
    ),
    "EvalJudgeSubmitFirstSaturday": ArchivePageRef(
        page="8_Eval_Quality",
        artifact_label="Eval judge (first Saturday batch)",
    ),
    "EvalJudgeSubmitWeekly": ArchivePageRef(
        page="8_Eval_Quality",
        artifact_label="Eval judge (weekly batch)",
    ),
    "EvalJudgePoll": ArtifactReason(
        reason="Polling state for the EvalJudge batch job; no per-run artifact — "
        "see EvalJudgeProcess for the materialized rubric output."
    ),
    "EvalJudgeProcess": ArchivePageRef(
        page="8_Eval_Quality",
        artifact_label="Eval judge processed rubrics",
    ),
    "EvalRollingMean": ArchivePageRef(
        page="8_Eval_Quality",
        artifact_label="Eval 4-week rolling mean",
    ),
    "RationaleClustering": ArtifactReason(
        reason="Rationale cluster artifact written to S3; no dedicated page yet "
        "(P3 follow-up — backlog)."
    ),
    "ReplayConcordance": ArtifactReason(
        reason="Concordance metric written to backtest/{date}/; surfaced inline "
        "in Backtester evaluator report (page 21)."
    ),
    "Counterfactual": ArtifactReason(
        reason="Counterfactual artifact written to backtest/{date}/; surfaced "
        "inline in Backtester evaluator report (page 21)."
    ),
    "AggregateCosts": ArchivePageRef(
        page="23_LLM_Cost",
        artifact_label="LLM cost telemetry (daily aggregate)",
    ),
    "PredictorTraining": ArchivePageRef(
        page="20_Predictor_Training_Archive",
        artifact_label="Predictor training summary",
    ),
    # L4544/L4571: after PredictorTraining, the model-zoo rotation trains the
    # base champion-arch + variant(s) and runs leak-free-CPCV selection, writing
    # predictor/model_zoo/leaderboard/{date}.json. Surfaced on the Predictor
    # console's "Model Zoo — Weekly Selection" panel (dashboard #170).
    "ModelZooRotation": ArchivePageRef(
        page="7_Predictor",
        artifact_label="Model-zoo selection leaderboard",
    ),
    # config#1083 PARALLEL fan-out (decomposes the monolithic ModelZooRotation):
    # ResolveZooSpecs (dispatch — list the budget-N spec ids) → TrainSpecDispatch
    # (Map: one spot per spec via spot_train.sh) → ModelZooSelect (one spot after
    # the Map joins, runs leak-free CPCV over whatever spec-* challengers
    # registered, writes predictor/model_zoo/leaderboard/{date}.json).
    "ResolveZooSpecs": ArtifactReason(
        reason="Model-zoo fan-out step 1 (config#1083): a dispatcher SSM command "
        "that resolves the spec ids to train this rotation, emitting a JSON array "
        "on stdout consumed by the TrainSpecDispatch Map. No per-run rendered "
        "artifact — the leaderboard is produced downstream by ModelZooSelect.",
    ),
    "TrainSpecDispatch": ArtifactReason(
        reason="Model-zoo fan-out step 2 (config#1083): per-spec training dispatch "
        "(Map) — launches a dedicated spot per spec via spot_train.sh "
        "--model-zoo-spec. The trained spec-* challengers register for the date; "
        "the rendered artifact is the ModelZooSelect leaderboard, not this "
        "dispatch step.",
    ),
    "ModelZooSelect": ArchivePageRef(
        page="7_Predictor",
        artifact_label="Model-zoo selection leaderboard",
    ),
    # L4517: preventive cross-repo nousergon-lib pin-drift gate — runs before
    # any spot launch (asserts backtester-pin == predictor-pin co-install parity
    # + every Saturday-SF repo pin >= MIN_LIB_VERSION). A guard; blocks the run
    # on drift. No per-run rendered artifact.
    "LibPinDriftCheck": ArtifactReason(
        reason="Cross-repo lib-pin drift gate (L4517) — asserts co-install pin "
        "parity + the MIN_LIB_VERSION floor before any spot launch; blocks the "
        "run on drift. No per-run rendered artifact — a preventive guard.",
    ),
    "Backtester": ArchivePageRef(
        page="21_Backtester_Evaluator_Archive",
        artifact_label="Backtester consolidated report",
    ),
    # L4472 phase-split (2026-05-31): the monolithic Backtester state was
    # decomposed into Backtester (simulate) → PredictorBacktest → Portfolio-
    # OptimizerBacktest. All three write into the same backtest/{date}/ prefix
    # surfaced on the consolidated evaluator archive page.
    "PredictorBacktest": ArchivePageRef(
        page="21_Backtester_Evaluator_Archive",
        artifact_label="Predictor backtest + Phase 4 report",
    ),
    "PortfolioOptimizerBacktest": ArchivePageRef(
        page="21_Backtester_Evaluator_Archive",
        artifact_label="Portfolio-optimizer / cov / gamma sweep report",
    ),
    "Parity": ArchivePageRef(
        page="3_Analysis",
        artifact_label="Parity replay diff",
    ),
    "Evaluator": ArchivePageRef(
        page="21_Backtester_Evaluator_Archive",
        artifact_label="Backtester evaluator report",
    ),
    "DriftDetection": ArchivePageRef(
        page="4_System_Health",
        artifact_label="SF-vs-CFN drift report",
    ),
    "SaturdayHealthCheck": ArchivePageRef(
        page="4_System_Health",
        artifact_label="Saturday per-repo health check",
    ),
    "WeeklySubstrateHealthCheck": ArchivePageRef(
        page="4_System_Health",
        artifact_label="Weekly substrate health check",
    ),
    "ReportCard": ArchivePageRef(
        page="Report_Card",
        artifact_label="System Report Card",
    ),
    "Director": ArchivePageRef(
        page="Director_Plan",
        artifact_label="Director weekly action plan",
    ),
    "NotifyComplete": ArtifactReason(
        reason="Terminal success SNS publish to alpha-engine-alerts; "
        "no persisted artifact (the email IS the surface)."
    ),
    "NotifyShellRunComplete": ArtifactReason(
        reason="Friday-PM shell-run dry-pass terminal SNS publish; "
        "no persisted artifact (the email IS the surface)."
    ),
    "HandleFailure": ArtifactReason(
        reason="Terminal failure SNS publish to alpha-engine-alerts; "
        "no persisted artifact (the email IS the surface)."
    ),
    "PublishResearchFailureImmediate": ArtifactReason(
        reason="Early-signal SNS publish fired the moment the Research branch "
        "fails inside ResearchPredictorParallel — BEFORE the sibling "
        "PredictorTraining branch completes its work and the parallel "
        "aggregation joins. No persisted artifact (the email IS the "
        "surface). Salvage-at-join semantics preserved: the branch still "
        "terminates via BranchAFailed Pass and the SF fails at "
        "CheckBranchOutcomes."
    ),
    "PublishPredictorFailureImmediate": ArtifactReason(
        reason="Early-signal SNS publish fired the moment the PredictorTraining "
        "branch fails inside ResearchPredictorParallel — BEFORE the "
        "sibling Research branch's eval-judge / RollingMean / "
        "Counterfactual chain completes. No persisted artifact (the "
        "email IS the surface). Salvage-at-join semantics preserved: "
        "the branch still terminates via BranchBFailed Pass and the SF "
        "fails at CheckBranchOutcomes."
    ),
    "PublishModelZooFailureImmediate": ArtifactReason(
        reason="SNS failure alert fired when the model-zoo rotation fails/times-out "
        "(config#1083). Required for fallback safety: PredictorTraining exports "
        "PREDICTOR_DEFER_TRAINING_EMAIL=1, so the base champion-arch retrain still "
        "serves while the zoo arc is degraded. No persisted artifact (the email IS "
        "the surface)."
    ),
    # ── Pre-open Trading SF (13 substantive Task steps) ───────────────────────────
    "DeployDriftCheck": ArchivePageRef(
        page="4_System_Health",
        artifact_label="Deploy-drift assertions",
    ),
    "StartExecutorEC2": ArtifactReason(
        reason="EC2 startInstances on the trading instance; no artifact — "
        "operational only."
    ),
    "DescribeInstanceInfo": ArtifactReason(
        reason="Boot diagnostic call against the trading instance; "
        "no artifact — operational only."
    ),
    # config#1430 (2026-06-30): replaced the on-box SSM trading_calendar check
    # (CheckTradingDay / TradingDayCheckFailed, whose stdout was unreliably
    # captured on a just-cold-booted instance) with a pre-boot Lambda gate
    # invoked BEFORE StartExecutorEC2 — the box is never booted on a holiday.
    "TradingDayGate": ArtifactReason(
        reason="NYSE-holiday gate — predictor Lambda invoke (action="
        "check_trading_day), pure calendar math, run before StartExecutorEC2; "
        "no artifact — gate outcome is encoded in the SF branch taken."
    ),
    "TradingDayGateFailed": ArtifactReason(
        reason="Trading-day gate Lambda failed for infrastructure reasons "
        "(not a holiday detection) — SNS alert, pipeline proceeds as a "
        "trading day; no persisted artifact (the email IS the surface)."
    ),
    "NotifyHolidaySkip": ArtifactReason(
        reason="Holiday-skip SNS publish; no persisted artifact (the email IS "
        "the surface)."
    ),
    # MorningEnrich (weekday) — same state name as Saturday; same entry above wins.
    "PredictorInference": ArchivePageRef(
        page="18_Predictor_Briefing_Archive",
        artifact_label="Predictor morning briefing",
    ),
    "CheckPredictorCoverage": ArtifactReason(
        reason="Coverage-gate Lambda; outcome encoded in the SF branch taken — "
        "see PredictorHealthCheck for any persisted health JSON."
    ),
    "ReinvokePredictor": ArtifactReason(
        reason="Re-invocation Lambda when CheckPredictorCoverage finds a gap; "
        "no per-run artifact — replaces the PredictorInference output."
    ),
    "RecheckCoverage": ArtifactReason(
        reason="Second coverage-gate Lambda after ReinvokePredictor; outcome "
        "encoded in the SF branch taken."
    ),
    "PredictorHealthCheck": ArchivePageRef(
        page="4_System_Health",
        artifact_label="Predictor health check",
    ),
    "RunMorningPlanner": ArchivePageRef(
        page="16_Order_Book_Rationale",
        artifact_label="Order book + rationale",
    ),
    "RunDaemon": ArchivePageRef(
        page="22_Intraday_Surveillance",
        artifact_label="Intraday surveillance (daemon)",
    ),
    # Secondary daily news pull for the held + tracked universe; writes
    # data/news_aggregates_daily/ for the robodashboard morning brief + AE
    # consumers. Runs AFTER RunDaemon (never delays trading) and is FAIL-SOFT.
    # Substrate for a separate app (robodashboard) — no AE-console artifact.
    "RunDailyNews": ArtifactReason(
        reason="Secondary daily news pull → data/news_aggregates_daily/ for the "
        "robodashboard morning brief + AE consumers; runs after RunDaemon, "
        "fail-soft. Substrate for a separate app — no AE-console artifact.",
    ),
    "ChronicGapSelfHeal": ArtifactReason(
        reason="Weekday fail-soft chronic-polygon-gap self-heal (L4604, data "
        "#398): split out of MorningEnrich after the 2026-06-11 SIGKILL "
        "incident; its Catch and Choice default both route forward to "
        "PredictorInference so a heal hang can never fail the morning. "
        "Substrate-only — no per-run rendered artifact.",
    ),
    "MorningArcticAppend": ArtifactReason(
        reason="Weekday ArcticDB daily_append split into its own skip-gated "
        "state (L4608, data #405) so reruns resume without re-paying the "
        "append. Writes the ArcticDB universe library; substrate-only.",
    ),
    # ── Post-close Trading SF (6 substantive Task steps) ────────────────────────────────
    "PostMarketData": ArtifactReason(
        reason="Polygon T+1 daily aggregate write to predictor/daily_closes/; "
        "substrate-only — consumed by EODReconcile."
    ),
    "PostMarketArcticAppend": ArtifactReason(
        reason="EOD ArcticDB daily_append split into its own state (2026-06-16) "
        "— writes today's post-market OHLCV row + recomputed features to the "
        "ArcticDB universe library, read next by EODReconcile + predictor "
        "inference. Substrate-only; the slow append separated from PostMarketData "
        "so reruns resume without re-paying it (mirrors MorningArcticAppend)."
    ),
    "CaptureSnapshot": ArchivePageRef(
        page="1_Portfolio",
        artifact_label="NAV + positions snapshot",
    ),
    "EODReconcile": ArchivePageRef(
        page="19_EOD_Reconcile_Archive",
        artifact_label="EOD reconcile briefing",
    ),
    "DailySubstrateHealthCheck": ArchivePageRef(
        page="4_System_Health",
        artifact_label="Daily substrate health check",
    ),
    "StopTradingInstance": ArtifactReason(
        reason="EC2 stopInstances on the trading instance; no artifact — "
        "operational only."
    ),
    "RefreshExecutorDeploy": ArtifactReason(
        reason="Top-of-pipeline executor-checkout git refresh chokepoint "
        "(config#1549) — hoists deploy-freshness above postmarket → arctic → "
        "snapshot → reconcile so the whole EOD run executes latest "
        "origin/main by construction; no artifact — operational only."
    ),
    "StartTradingInstance": ArtifactReason(
        reason="EC2 startInstances re-runnability guard on the EOD SF "
        "(nousergon-data#576) — ensures the trading instance is up + "
        "SSM-registered before EOD sendCommands on an operator recovery "
        "rerun; no artifact — operational only."
    ),
    "ForceStopInstance": ArtifactReason(
        reason="EC2 stopInstances fallback on a non-graceful EOD; no artifact — "
        "operational only."
    ),
}


def lookup_registry(state_name: str) -> Optional[Union[ArchivePageRef, ArtifactReason]]:
    """Return the registry entry for ``state_name`` (None if absent).

    ``None`` here signals "this state is not in the registry" — distinct
    from :class:`ArtifactReason` ("registered as substrate-only with this
    reason"). The dashboard consumer should treat ``None`` as a CI-time
    test failure (registry drift); it should NEVER render in production.
    """
    return STATE_TO_ARCHIVE_PAGE.get(state_name)
