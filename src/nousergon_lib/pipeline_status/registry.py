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

from typing import Annotated, Final, Literal, Union

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
    # alpha-engine-config-I5759: DataPhase2 became a spot stage on 2026-07-31,
    # so it acquired a poll companion like its siblings. Without this row the
    # companion renders as its own console row instead of rolling up into
    # DataPhase2, and alpha-engine-data's registry-source check fails.
    "WaitForDataPhase2": "DataPhase2",
    "WaitForPredictorTraining": "PredictorTraining",
    "WaitForBacktester": "Backtester",
    "WaitForPredictorBacktest": "PredictorBacktest",
    "WaitForPortfolioOptimizerBacktest": "PortfolioOptimizerBacktest",
    "WaitForParity": "Parity",
    # alpha-engine-config#6030 (2026-08-09): the bundled Parity stage was
    # split into a ParityParallel of three fail-open branch quartets + a
    # PitParityCompare join, each with its own poll companion. The legacy
    # WaitForParity row above is retained until the SF cutover
    # (nousergon-data PR) merges everywhere and the transition window
    # closes (cleanup rides alpha-engine-config-I6725).
    "WaitForPitParityLookahead": "PitParityLookahead",
    "WaitForPitParityWalkforward": "PitParityWalkforward",
    "WaitForParityReplay": "ParityReplay",
    "WaitForPitParityCompare": "PitParityCompare",
    # alpha-engine-config-I7267 (2026-08-19): a trivial `aws s3api
    # head-object` marker check, dispatched on the same instance only after
    # PitParityLookahead/PitParityWalkforward already exited non-zero, to
    # classify a resource-killed pass (halts the SF) distinctly from one
    # that merely could not conclude (fail-open). Its own poll companion
    # must roll up too.
    "WaitForPitParityLookaheadResourceKillCheck": "PitParityLookaheadResourceKillCheck",
    "WaitForPitParityWalkforwardResourceKillCheck": "PitParityWalkforwardResourceKillCheck",
    # alpha-engine-config-I3112 deliverable 3 (2026-08-11): the single
    # Evaluator state became EvaluatorDiagnostics -> EvaluatorOptimize, each
    # with its own dispatch/poll pair. Both companions must roll up, or each
    # renders as its own dashboard row.
    "WaitForEvaluatorDiagnostics": "EvaluatorDiagnostics",
    "WaitForEvaluatorOptimize": "EvaluatorOptimize",
    "WaitForSaturdayHealthCheck": "SaturdayHealthCheck",
    "WaitForWeeklySubstrateHealthCheck": "WeeklySubstrateHealthCheck",
    "WaitForModelZoo": "ModelZooRotation",  # L4544 weekly model-zoo rotation
    # alpha-engine-config-I5758: ThinkTankCoverage moved off a direct
    # lambda:invoke (TimeoutSeconds 900 — the AWS Lambda MAXIMUM, which the
    # agent loop does not fit in) onto the spot dispatcher, so it gained the
    # standard dispatch/poll companion. Without this row the wait renders as
    # its own dashboard row instead of rolling up into its parent.
    "WaitForThinkTank": "ThinkTankCoverage",
    # Per-execution ephemeral dispatch box (nousergon-data#975) — retires the
    # always-on dashboard-box SPOF that every weekly SSM stage dispatched from.
    "WaitForWeeklyFreshnessSpotBootstrap": "DispatchWeeklyFreshnessSpot",
    # Pre-open Trading SF
    "WaitForMorningPlanner": "RunMorningPlanner",
    "WaitForDailyNews": "RunDailyNews",  # secondary daily news pull (fail-soft)
    "WaitForChronicGap": "ChronicGapSelfHeal",  # L4604 fail-soft heal split
    "WaitForMorningArcticAppend": "MorningArcticAppend",  # L4608 daily_append split
    "WaitForInstanceReady": "StartExecutorEC2",
    # config#1811 code-freshness poll loop (ssm-liveness-poller iterations).
    "WaitForCodeFreshness": "CodeFreshnessGate",
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

    The ``page`` value is a **console URL path** relative to the console
    host root — either a registered ``st.Page`` slug (a pinned
    ``url_path`` like ``"eod-report"`` / ``"fleet-status"``, or a
    filename-derived default like ``"Report_Card"``) or a lazy-host tab
    deep-link (``"host_execution?tab=Order+Book"``). This replaced the
    pre-nav-collapse "page module name" contract (console-IA sync,
    alpha-engine-config#1990): after crucible-dashboard#273 collapsed
    most pages into ``view_host`` tabs, bare module names stopped
    resolving as URLs. The dashboard consumer renders the value as a
    relative markdown link; the lib does not bake URL hosts because the
    same console serves at ``console.nousergon.ai`` behind Cloudflare
    Access. A dashboard-side guard test pins every ``page`` value
    against the live nav registration + host tab lists.

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
STATE_TO_ARCHIVE_PAGE: Final[dict[str, ArchivePageRef | ArtifactReason]] = {
    # ── Weekly Freshness SF (27 substantive Task steps) ──────────────────────────
    "DispatchWeeklyFreshnessSpot": ArtifactReason(
        reason="Launches the per-execution ephemeral spot box that every "
        "downstream weekly SSM stage dispatches from, then bootstraps repos "
        "+ venv on it; writes no run artifact — its output is the "
        "instance id threaded into $.ec2_instance_id."
    ),
    # alpha-engine-config-I7119: the same dispatcher, invoked to REPLACE a
    # launcher box AWS reclaimed mid-run (spot no-capacity). Registered as its
    # own state rather than folded into DispatchWeeklyFreshnessSpot because the
    # two answer different operational questions: this one appearing in a run
    # means the run survived an infrastructure event, which is the whole signal.
    "RelaunchWeeklyFreshnessSpot": ArtifactReason(
        reason="Replaces the ephemeral launcher spot after a mid-run reclaim, "
        "forced on-demand so the replacement cannot itself be reclaimed; "
        "writes no run artifact — its output is the new instance id, which "
        "REPLACES $.ec2_instance_id. Its presence in an execution is the "
        "run's substrate-loss signal; the count also rides the SF completion "
        "marker as substrate_relaunches."
    ),
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
    "ThinkTankCoverage": ArchivePageRef(
        page="Think_Tank",
        artifact_label="Think Tank coverage fill",
    ),
    "RAGIngestion": ArtifactReason(
        reason="SEC/8-K/earnings/theses corpus refresh in rag/corpus/; "
        "substrate-only — consumed at Research time."
    ),
    "RegimeSubstrate": ArchivePageRef(
        page="host_predictor?tab=Regime",
        artifact_label="Regime substrate",
    ),
    "RegimeRetrospectiveEval": ArchivePageRef(
        page="host_predictor?tab=Regime",
        artifact_label="Regime retrospective eval",
    ),
    "Research": ArchivePageRef(
        page="host_research_signals?tab=Briefing+Archive",
        artifact_label="Morning research briefing",
    ),
    # config#2515 Phase B: thin envelope producer that replaced the
    # multi-agent Research graph runner as the signals.json producer for
    # the champion path (scanner -> predictor direct, config#1580 ruling).
    # LOAD-BEARING — the executor hard-fails Monday without signals.json,
    # so unlike ThinkTankCoverage/ChallengerShadow this state's SF Catch
    # is NOT non-blocking. Writes the same S3 key as the old signals.json
    # for contract safety; no dedicated archive page of its own yet, but
    # its output is the input the morning research briefing page renders.
    "SignalsEnvelope": ArtifactReason(
        reason="signals.json envelope producer (config#2515 Phase B) — "
        "sources scanner-board + regime-substrate fields into the "
        "production signals.json; load-bearing (executor hard-fails "
        "without it). No dedicated archive page yet; see Research/"
        "ThinkTankCoverage for the surrounding briefing artifacts."
    ),
    # config#2515 Phase B: keeps the no_agent champion-baseline shadow
    # (signals_shadow/no_agent/{trading_day}/signals.json) alive for the
    # producer leaderboard now that the weekly SF no longer hosts the
    # multi-agent Research Lambda invocation that used to fire this mode
    # as a side effect. Observe-only — never read by the executor or the
    # live gate; non-blocking Catch mirrors ThinkTankCoverage's.
    "ChallengerShadow": ArtifactReason(
        reason="Non-blocking observe-only shadow feed (config#2515 Phase "
        "B) — writes signals_shadow/no_agent/{trading_day}/signals.json "
        "for the producer leaderboard (scoring/leaderboard_producers.py); "
        "never read by the executor or the live trading gate. No "
        "dedicated archive page — the leaderboard that consumes this is "
        "hosted in the eval_rolling_mean Lambda."
    ),
    # alpha-engine-config-I7726 — the research module's §2.3a CORRECTNESS
    # VERDICT: a known-answer battery over the DEPLOYED instrument
    # (crucible-research scoring/self_test.py -> nousergon_lib.quant.selftest).
    # The producer was always correct and NO LIVE TRIGGER REACHED IT — measured
    # 2026-08-19, research/{trading_day}/self_test.json had ZERO instances in
    # the bucket ever, since its registry row was created 2026-08-13. This state
    # is the trigger.
    #
    # ArtifactReason, not ArchivePageRef: the verdict's consumer surface is the
    # console check row (ops/checks/ae-research-self-test/latest.json) plus the
    # freshness monitor's page on the artifact's absence, not an archive page.
    "ResearchSelfTest": ArtifactReason(
        reason="Research module self-test (config-I7726) — writes "
        "research/{trading_day}/self_test.json, a known-answer battery over "
        "the deployed scoring instrument. Surfaced on the console self-test "
        "check row and paged by the freshness monitor when absent; no "
        "dedicated archive page."
    ),
    "DataPhase2": ArtifactReason(
        reason="Alt-data + fundamentals refresh; substrate-only, no per-run "
        "rendered artifact."
    ),
    "EvalJudgeSubmitFirstSaturday": ArchivePageRef(
        page="evaluator",
        artifact_label="Eval judge (first Saturday batch)",
    ),
    "EvalJudgeSubmitWeekly": ArchivePageRef(
        page="evaluator",
        artifact_label="Eval judge (weekly batch)",
    ),
    "EvalJudgePoll": ArtifactReason(
        reason="Polling state for the EvalJudge batch job; no per-run artifact — "
        "see EvalJudgeProcess for the materialized rubric output."
    ),
    "EvalJudgeProcess": ArchivePageRef(
        page="evaluator",
        artifact_label="Eval judge processed rubrics",
    ),
    "EvalRollingMean": ArchivePageRef(
        page="evaluator",
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
        page="host_cost_usage?tab=API",
        artifact_label="LLM cost telemetry (daily aggregate)",
    ),
    # alpha-engine-config-I8336: the named WARNING alert on AggregateCosts'
    # fail-open Catch, mirroring PublishScannerLeaderboardDegraded's shape.
    # sf-pipeline-policy.md §5 lets the cost-aggregation stage fail open (it is
    # secondary observability; the primary deliverable survives) — but a
    # fail-open that only surfaces in the terminal email is silent for the
    # whole run. This is what makes it loud at the moment it fires.
    "PublishAggregateCostsDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when AggregateCosts fails "
        "and the cycle's LLM cost record (cost.parquet) is not written "
        "(alpha-engine-config-I8336) — non-fatal, no trading or research "
        "artifact depends on it, but the cycle has no cost attribution and "
        "the run terminates DEGRADED. Reached both by a Lambda failure and "
        "by a fan-in coverage breach, which the alert deliberately does not "
        "try to tell apart. No persisted artifact (the email IS the surface)."
    ),
    "PredictorTraining": ArchivePageRef(
        page="host_predictor?tab=Archives",
        artifact_label="Predictor training summary",
    ),
    # L4544/L4571: after PredictorTraining, the model-zoo rotation trains the
    # base champion-arch + variant(s) and runs leak-free-CPCV selection, writing
    # predictor/model_zoo/leaderboard/{date}.json. Surfaced on the Predictor
    # console's "Model Zoo — Weekly Selection" panel (dashboard #170).
    "ModelZooRotation": ArchivePageRef(
        page="model-zoo",
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
        page="model-zoo",
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
    # L4595 / config#693: PIPELINE_CONTRACT.yaml preflight — sibling to
    # LibPinDriftCheck, runs right after it in the pre-spend preflight chain.
    # Re-runs the validate_pipeline_contract.py invariants at SF start so a
    # contract break made outside PR CI halts in seconds, before spot spend.
    "PipelineContractCheck": ArtifactReason(
        reason="Pre-spend PIPELINE_CONTRACT.yaml self-consistency gate "
        "(L4595/config#693) — asserts the contract parses and every "
        "artifact_id resolves in ARTIFACT_REGISTRY.yaml before any spot "
        "launch. No per-run rendered artifact — a preventive guard "
        "(fail-open on fetch/parse misses, mirroring LibPinDriftCheck).",
    ),
    # config#2348: THIRD pre-spend sibling gate, composed after
    # PipelineContractGate — checks each evaluator Lambda's :live alias SHA
    # against origin/main via the Lambda's own action=check_deploy_drift
    # dispatch. The two aliases share one image but can drift apart if only
    # one got promoted, so they are probed independently.
    "EvaluatorDeployDriftCheck": ArtifactReason(
        reason="Evaluator Lambda-SHA deploy-drift gate (config#2348) — "
        "asserts alpha-engine-evaluator:live is built from origin/main "
        "before any spot launch; halts the run on confirmed drift. No "
        "per-run rendered artifact — a preventive guard, fail-open on "
        "probe failure (mirrors LibPinDriftCheck).",
    ),
    "EvaluatorDirectorDeployDriftCheck": ArtifactReason(
        reason="Evaluator-director Lambda-SHA deploy-drift gate "
        "(config#2348) — the sibling probe for "
        "alpha-engine-evaluator-director:live, run independently because "
        "the two aliases share one image but are promoted separately. No "
        "per-run rendered artifact — a preventive guard, fail-open on "
        "probe failure.",
    ),
    # I4494: FOURTH pre-spend sibling gate, composed after the evaluator-
    # director deploy-drift gate and immediately before CheckMutexRole (the
    # last pre-spend probe of all). Runs the alpha-engine-weekly-preflight
    # Lambda's sf_preflight checks (IAM reachability, tool contracts,
    # definition input coherence, Lambda memory headroom) against live AWS
    # state before the pipeline spends on any spot or mutex acquisition.
    # Fail-hard (fail-closed) on a failing or malformed check — the opposite
    # posture to the advisory LibPinDriftCheck/PipelineContractCheck gates —
    # because it is the last check before spend. Routes failures through
    # ExtractWeeklyPreflightError -> NormalizeFailureContext like the other
    # pre-spend gates.
    "WeeklyPreflight": ArtifactReason(
        reason="Pre-spend I4494 weekly-preflight gate (alpha-engine-"
        "config#4494) — runs sf_preflight.py checks (IAM reachability, "
        "tool contracts, definition input coherence, Lambda memory "
        "headroom) against live AWS before any spot or mutex acquisition; "
        "fail-hard on violation, fail-closed on a malformed verdict. No "
        "per-run rendered artifact — a blocking preventive guard.",
    ),
    # config#2249: fast pre-dispatch substrate health gate, immediately
    # before MorningEnrich (the first state to actually dispatch work to
    # the Saturday box). Replaces discovering a dead dispatch box (disk
    # full, or an unresponsive SSM agent) only after MorningEnrich burns
    # its full retry ladder (config#2279) with a fast, named
    # SubstrateUnhealthy failure — a blocking guard, not a fail-open one
    # (a dead box means the run genuinely cannot proceed).
    "SubstrateHealthGate": ArtifactReason(
        reason="Pre-dispatch substrate health check (config#2249) — "
        "issues a quick SSM disk-headroom probe against the Saturday "
        "dispatch box immediately before MorningEnrich and fails fast "
        "(named SubstrateUnhealthy: disk_full / ssm_unresponsive / "
        "ssm_command_never_registered) rather than burning MorningEnrich's "
        "full retry ladder against a dead box. No per-run rendered "
        "artifact — a blocking preventive guard.",
    ),
    # config#2278: fail-open + alert SNS publishes fired when LibPinDriftCheck
    # / PipelineContractCheck itself can't check (Lambda failed after
    # retries, or returned a malformed payload) — the run proceeds fail-open
    # rather than halting, and this notifier is the alert that it did.
    "PublishMutexAcquireDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when AcquireMutex hit a DynamoDB "
        "infra error other than ConditionalCheckFailedException (which the "
        "separate MutexConflict hard-fail already handles) — the run "
        "proceeds fail-open without the run-slot mutex, so a concurrent "
        "duplicate execution would no longer be excluded. No persisted "
        "artifact (the email IS the surface); sets $.gate_degraded, carried "
        "to the terminal degraded notifier (config#6722).",
    ),
    "PublishLibPinGateDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when LibPinDriftCheck could not "
        "verify cross-repo pin parity (gate Lambda failed after retries, or "
        "malformed payload) — the run proceeds fail-open, unprotected "
        "against the co-install break the gate exists to catch. No "
        "persisted artifact (the email IS the surface); see "
        "NotifyCompleteGatesDegraded / NotifyCompleteGatesAndHealthDegraded "
        "for the terminal marker this alert sets up.",
    ),
    "PublishPipelineContractGateDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when PipelineContractCheck could "
        "not verify PIPELINE_CONTRACT.yaml self-consistency (gate Lambda "
        "failed after retries, or malformed payload) — the run proceeds "
        "fail-open, unprotected against the contract break the gate exists "
        "to catch. No persisted artifact (the email IS the surface); see "
        "NotifyCompleteGatesDegraded / NotifyCompleteGatesAndHealthDegraded "
        "for the terminal marker this alert sets up.",
    ),
    # config#2348: the same fail-open alert shape as the two above, for the
    # third pre-spend sibling gate pair. Both publish independently — the
    # grading probe's degraded chain proceeds INTO the director probe
    # (config#2278's "don't silently skip the sibling gate"), so a run can
    # fire either, or both.
    "PublishEvaluatorGateDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when EvaluatorDeployDriftCheck "
        "could not verify alpha-engine-evaluator:live against origin/main "
        "(gate Lambda failed after retries, or malformed payload) — the "
        "run proceeds fail-open, unprotected against the stale-deploy "
        "break the gate exists to catch. No persisted artifact (the email "
        "IS the surface); see NotifyCompleteGatesDegraded / "
        "NotifyCompleteGatesAndHealthDegraded for the terminal marker this "
        "alert sets up.",
    ),
    "PublishEvaluatorDirectorGateDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when "
        "EvaluatorDirectorDeployDriftCheck could not verify "
        "alpha-engine-evaluator-director:live against origin/main (gate "
        "Lambda failed after retries, or malformed payload) — the run "
        "proceeds fail-open, unprotected against the stale-deploy break "
        "the gate exists to catch. No persisted artifact (the email IS the "
        "surface); see NotifyCompleteGatesDegraded / "
        "NotifyCompleteGatesAndHealthDegraded for the terminal marker this "
        "alert sets up.",
    ),
    # config#1824 (2026-07-06): run-day gate mirroring the weekday
    # TradingDayGate (config#1430) — predictor Lambda action=
    # check_weekly_run_day, pure NYSE-calendar math, true iff yesterday was
    # the week's last trading session (Sat normally; Fri/Thu on
    # holiday-shortened weeks, real precedent Fri 2026-07-03).
    "WeeklyRunDayGate": ArtifactReason(
        reason="Weekly run-day gate — predictor Lambda invoke (action="
        "check_weekly_run_day), pure calendar math run before preflight; "
        "no artifact — gate outcome is encoded in the SF branch taken."
    ),
    "WeeklyRunDayGateFailed": ArtifactReason(
        reason="Weekly run-day gate Lambda failed for infrastructure reasons "
        "(not a run-day determination) — SNS alert, pipeline proceeds as a "
        "run day; no persisted artifact (the email IS the surface)."
    ),
    "Backtester": ArchivePageRef(
        page="analysis",
        artifact_label="Backtester consolidated report",
    ),
    # L4472 phase-split (2026-05-31): the monolithic Backtester state was
    # decomposed into Backtester (simulate) → PredictorBacktest → Portfolio-
    # OptimizerBacktest. All three write into the same backtest/{date}/ prefix
    # surfaced on the consolidated evaluator archive page.
    "PredictorBacktest": ArchivePageRef(
        page="analysis",
        artifact_label="Predictor backtest + Phase 4 report",
    ),
    "PortfolioOptimizerBacktest": ArchivePageRef(
        page="analysis",
        artifact_label="Portfolio-optimizer / cov / gamma sweep report",
    ),
    "Parity": ArchivePageRef(
        page="analysis",
        artifact_label="Parity replay diff",
    ),
    # alpha-engine-config#6030 (2026-08-09): the bundled Parity stage was
    # split per sf-pipeline-policy §2.1 into three ParityParallel branches
    # (each its own spot + script + timeout + skip flag) and a
    # PitParityCompare join. The legacy "Parity" row above is retained for
    # the transition window (pre-cutover executions still render it);
    # cleanup rides alpha-engine-config-I6725. Each pass publishes a
    # versioned per-pass stats artifact
    # (parity/{run_date}/pit_stats_{pass}.json, crucible-backtester
    # contracts/pit_stats_pass.schema.json); the compare reads both and
    # writes backtest/{run_date}/pit_parity.json — verdict UNKNOWN when a
    # pass artifact is missing (§2.3a).
    "PitParityLookahead": ArchivePageRef(
        page="analysis",
        artifact_label="Pit-parity lookahead pass stats",
    ),
    "PitParityWalkforward": ArchivePageRef(
        page="analysis",
        artifact_label="Pit-parity walkforward (PIT) pass stats",
    ),
    "ParityReplay": ArchivePageRef(
        page="analysis",
        artifact_label="Parity replay diff",
    ),
    "PitParityCompare": ArchivePageRef(
        page="analysis",
        artifact_label="Pit-parity contamination report (delta + verdict)",
    ),
    # alpha-engine-config-I7267 (Brian's 2026-08-13 ruling, sf-pipeline-
    # policy §3 SFP-3-resource-kill-halts-and-is-named): runs only after
    # the pass already exited non-zero; heads a well-known S3 marker key
    # (written by the pass's own commands when the just-published
    # pit_stats_{pass}.json read back timed_out:true) to distinguish a
    # RESOURCE KILL from a generic "could not conclude" — writes no run
    # artifact of its own, and its Catch/non-Success routing falls back to
    # the pre-existing *Degraded fail-open, so this can only ADD the
    # RESOURCE_KILL classification, never remove existing coverage.
    "PitParityLookaheadResourceKillCheck": ArtifactReason(
        reason="Trivial `aws s3api head-object` check for the "
        "RESOURCE_KILL marker written by PitParityLookahead's own commands "
        "when the just-published pit_stats_lookahead.json read back "
        "timed_out:true; writes no run artifact of its own."
    ),
    "PitParityWalkforwardResourceKillCheck": ArtifactReason(
        reason="Trivial `aws s3api head-object` check for the "
        "RESOURCE_KILL marker written by PitParityWalkforward's own "
        "commands when the just-published pit_stats_walkforward.json read "
        "back timed_out:true; writes no run artifact of its own."
    ),
    "PublishParityCompareDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when the PitParityCompare "
        "join stage (reads both pit_stats pass artifacts, writes "
        "backtest/{run_date}/pit_parity.json) does not complete — SSM "
        "timeout, crash, or dispatch/poll failure (alpha-engine-config#6030). "
        "Non-fatal: the weekly run continues, but no parity verdict was "
        "produced this run — absence of a verdict must NOT be read as a "
        "clean pass (sf-pipeline-policy §2.3a). No persisted artifact (the "
        "email IS the surface)."
    ),
    # alpha-engine-config-I6025: WARNING-severity alert fired when Parity
    # (pit_parity look-ahead/walk-forward contamination check + backtester
    # ↔executor replay) does not complete — SSM timeout/crash (terminal
    # non-Success via CheckParityStatus.Default), or send/poll infra failure
    # (the Parity + WaitForParity Task Catches). Degrade-not-fail: the run
    # continues to Evaluator/ReportCard/Director. Mirrors
    # PublishReportCardDegraded / PublishLibPinGateDegraded's shape.
    "PublishParityDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when the Parity stage "
        "(pit_parity + replay) does not complete — SSM timeout, crash, or "
        "dispatch/poll failure (alpha-engine-config-I6025). Non-fatal: the "
        "weekly run's real trading artifacts are unaffected, but no "
        "parity verdict was produced this run — absence of a verdict must "
        "NOT be read as a clean pass (alpha-engine-config-I6044). No "
        "persisted artifact (the email IS the surface)."
    ),
    # alpha-engine-config-I3112 deliverable 3 (2026-08-11): one Evaluator
    # state became two. Both halves point at the same archive page — they are
    # two stages of one report, not two reports: the diagnostics half writes
    # the signal-quality and component-diagnostics results, the optimize half
    # consumes them (via the S3 handoff snapshot) and produces the optimizer
    # verdicts and champion promotion that the page renders. Distinct labels
    # so a reader can tell which half a row is without opening it.
    "EvaluatorDiagnostics": ArchivePageRef(
        page="analysis",
        artifact_label="Backtester evaluator report (diagnostics)",
    ),
    "EvaluatorOptimize": ArchivePageRef(
        page="analysis",
        artifact_label="Backtester evaluator report (optimizers)",
    ),
    "DriftDetection": ArchivePageRef(
        page="fleet-status",
        artifact_label="SF-vs-CFN drift report",
    ),
    "SaturdayHealthCheck": ArchivePageRef(
        page="fleet-status",
        artifact_label="Saturday per-repo health check",
    ),
    "WeeklySubstrateHealthCheck": ArchivePageRef(
        page="fleet-status",
        artifact_label="Weekly substrate health check",
    ),
    # alpha-engine-config-I7620: RunScope runs immediately before
    # CheckSkipReportCard and derives THIS execution's own scope — one row per
    # gated stage, each DISABLED / ENABLED_COMPLETED / ENABLED_FAILED /
    # NOT_REACHED, with the skip_* flag that decided it. It exists because which
    # stages a run dispatched was never recorded anywhere a consumer could read:
    # on 2026-08-14 the Director reported the DELIBERATE absence of pit_parity
    # (skip_parity set on the Saturday trigger since 2026-08-13, a recorded
    # ruling) as "the producer never ran this cycle" and withheld issue_filing +
    # loop_verification on the strength of it. Derived from the definition plus
    # the execution history rather than kept in a registry, so a flag flip in the
    # CFN preset moves the pipeline, the artifact and the Director's purview
    # together.
    "RunScope": ArtifactReason(
        reason="Run-scope derivation (alpha-engine-config-I7620) — writes "
        "backtest/{date}/run_scope.json, the per-stage record of what this "
        "execution dispatched and what an operator flag switched off. Not an "
        "archive page of its own: it is the DENOMINATOR the Report Card and "
        "Director render their grades against, so it surfaces there rather "
        "than as a page nobody would open on its own.",
    ),
    # alpha-engine-config-I8214: the stage-coverage sweep at the tail of the
    # weekly run, immediately after WriteCompletionMarker. The reader shipped
    # here and nothing called it, so it detected nothing; the Lambda these two
    # states invoke is its caller. It also augments the completion marker with
    # the cycle's real shape — the 2026-08-22 marker was written by an
    # execution that entered 1 of 16 declared stages and said nothing about it
    # (alpha-engine-config-I8186).
    "WeeklyCoverageSweep": ArtifactReason(
        reason="Stage-coverage sweep (alpha-engine-config-I8214) — writes "
        "_stage_coverage/_sweep/{pipeline}/{date}.json and merges the cycle's "
        "real shape into _sf_completion/{pipeline}/{date}.json. Not an archive "
        "page of its own: it is the VERDICT other surfaces carry, so it "
        "surfaces on the completion marker and the Report Card rather than as "
        "a page nobody would open on its own.",
    ),
    "WeeklyCoverageSweepUnavailable": ArtifactReason(
        reason="The sweep DID NOT RUN — an SNS page, no artifact. It exists "
        "because a sweep that never started cannot page for itself, and "
        "'found nothing' and 'did not run' are different facts (principles.md "
        "§2.7). Writing an artifact here would be the defect: an object whose "
        "presence implies the surface was observed.",
    ),
    "ReportCard": ArchivePageRef(
        page="Report_Card",
        artifact_label="System Report Card",
    ),
    # config#2302: WARNING-severity alert fired from ReportCard's Catch(States.ALL)
    # — the 2026-07 incident where the evaluator Lambda died silently for 9 days
    # because that Catch used to route straight to CheckShellRunNotify with no
    # alert. Non-fatal (advisory grading only); mirrors PublishLibPinGateDegraded's
    # shape.
    "PublishReportCardDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when ReportCard (Evaluator "
        "Report Card v2 Lambda) fails (config#2302) — non-fatal, the weekly "
        "run's real trading artifacts are unaffected, but the advisory "
        "grading layer (report_card.json, downstream Director) did not "
        "run. No persisted artifact (the email IS the surface)."
    ),
    "Director": ArchivePageRef(
        page="director",
        artifact_label="Director weekly action plan",
    ),
    # alpha-engine-config-I7813: the observe-only scanner champion/challenger
    # board left the Scanner Lambda's inline post-steps and became its own
    # weekly-SF leaf state placed AFTER ReportCard/Director — the blast-radius
    # split sf-pipeline-policy.md §2.1 asks for. Same Lambda as "Scanner",
    # invoked with mode=scanner_leaderboard. No archive page renders this board
    # today; crucible-dashboard's Experiments view (views/46_Experiments.py)
    # reads the prefix directly, so swap this for an ArchivePageRef if that
    # view ever becomes a first-class archive page.
    "ScannerLeaderboard": ArtifactReason(
        reason="Observe-only scanner champion/challenger board — writes "
        "s3://alpha-engine-research/scanner/leaderboard/{run_date}.json "
        "(alpha-engine-config-I7813). No pipeline stage branches on it: "
        "the dashboard Experiments view renders it and gate predicates "
        "poll its n_dates. A failure degrades the weekly run (the run "
        "terminates DEGRADED naming this state) but cannot reach any "
        "stage that does not consume its output."
    ),
    # alpha-engine-config-I7813: mirrors PublishReportCardDegraded's shape —
    # the named WARNING alert on the leaf's fail-open Catch, so a board that
    # was not written can never be silent.
    "PublishScannerLeaderboardDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when the ScannerLeaderboard "
        "leaf fails and scanner/leaderboard/{run_date}.json is not written "
        "(alpha-engine-config-I7813) — non-fatal for trading (nothing "
        "branches on the board), but the scanner champion/challenger "
        "comparison has no observation for the cycle. No persisted "
        "artifact (the email IS the surface)."
    ),
    # alpha-engine-config-I7813: mirrors NotifyCompleteReportCardDegraded —
    # the terminal notifier for a run whose ONLY degradation is the leaf, so
    # it can never reach NotifyComplete's "All steps completed successfully"
    # while terminating in DegradedRun (sf-pipeline-policy.md §2.3 Option A).
    "NotifyCompleteScannerLeaderboardDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run whose only "
        "degradation was the observe-only ScannerLeaderboard leaf "
        "(alpha-engine-config-I7813) — a PublishScannerLeaderboardDegraded "
        "alert fired earlier in the run. No persisted artifact (the email "
        "IS the surface)."
    ),
    # config#2302: same silent-swallow class as ReportCard's pre-fix Catch —
    # WARNING-severity alert fired from Director's Catch(States.ALL).
    "PublishDirectorDegraded": ArtifactReason(
        reason="WARNING-severity SNS alert fired when Director (Evaluator "
        "Director Lambda) fails (config#2302) — non-fatal, the weekly "
        "run's real trading artifacts are unaffected, but the advisory "
        "Director layer (action_plan.json, carry-over ledger update) did "
        "not run. No persisted artifact (the email IS the surface)."
    ),
    "NotifyComplete": ArtifactReason(
        reason="Terminal success SNS publish to alpha-engine-alerts; "
        "no persisted artifact (the email IS the surface)."
    ),
    # config#2278: SUCCESS-path notifier for a run whose pre-spend gates
    # (LibPinDriftCheck / PipelineContractCheck) failed open unchecked —
    # mirrors NotifyComplete exactly, Subject flags the gates-degraded run.
    "NotifyCompleteGatesDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run whose pre-spend "
        "gates (LibPinDriftCheck / PipelineContractCheck) failed open "
        "unchecked (config#2278) — a PublishLibPinGateDegraded / "
        "PublishPipelineContractGateDegraded alert fired earlier in the "
        "run. No persisted artifact (the email IS the surface)."
    ),
    # config#2276: SUCCESS-path notifier for a run whose tail health checks
    # (SaturdayHealthCheck / WeeklySubstrateHealthCheck) degraded — crashed,
    # timed out, or finished non-Success.
    "NotifyCompleteHealthDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run whose tail health "
        "checks (SaturdayHealthCheck freshness and/or "
        "WeeklySubstrateHealthCheck substrate/constituents-drift) could not "
        "run to Success (config#2276) — freshness validation is NOT "
        "confirmed for this run. No persisted artifact (the email IS the "
        "surface)."
    ),
    # config#2278 + config#2276: both degradations fired in the same run —
    # Subject reflects both; mirrors the other two Degraded notifiers.
    "NotifyCompleteGatesAndHealthDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run where BOTH the "
        "pre-spend gates failed open unchecked AND the tail health checks "
        "degraded (config#2278 + config#2276). No persisted artifact (the "
        "email IS the surface)."
    ),
    # config#6685: SUCCESS-path notifier for a run whose ReportCard advisory
    # grading (Evaluator Report Card v2 Lambda) degraded, with neither of
    # the other two flag families also set — mirrors NotifyCompleteGatesDegraded
    # / NotifyCompleteHealthDegraded exactly. Pre-fix, a ReportCard failure
    # on an otherwise-clean run terminated at plain NotifyComplete ("All
    # steps completed successfully") with no mention of the degradation —
    # weekly-sf-policy.md §2.3 violation closed by this notifier.
    "NotifyCompleteReportCardDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run whose ReportCard "
        "advisory grading (Evaluator Report Card v2 Lambda) failed "
        "(config#6685) — a PublishReportCardDegraded alert fired earlier "
        "in the run. No persisted artifact (the email IS the surface)."
    ),
    # config#6685 + alpha-engine-config-I6025: folded combined notifier —
    # fires when TWO OR MORE of the four degraded-flag families (pre-spend
    # gates, tail health checks, report card advisory grading, parity
    # verdict) are true together. Deliberately generic (names all four
    # categories, points at the execution record for the exact combination)
    # rather than enumerating every additional per-combination Task state —
    # the CheckGateDegradedNotify Choice's own config#2276 comment predicted
    # exactly this: a third (and now fourth) flag family should fold into a
    # data-driven notifier, not an N-way enumeration.
    "NotifyCompleteMultipleDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run where two or more "
        "of {pre-spend gates, tail health checks, report card advisory "
        "grading, parity verdict} degraded together (config#6685, "
        "alpha-engine-config-I6025) — the exact combination is on the "
        "execution record ($.gate_degraded / $.health_check_degraded / "
        "$.report_card_degraded / $.parity_degraded). No persisted "
        "artifact (the email IS the surface)."
    ),
    # alpha-engine-config-I6025/I6044: SUCCESS-path notifier for a run whose
    # Parity verdict degraded, with none of the other three flag families
    # also set — mirrors NotifyCompleteReportCardDegraded /
    # NotifyCompleteGatesDegraded / NotifyCompleteHealthDegraded exactly.
    # Pre-fix, a Parity degradation on an otherwise-clean run set
    # $.parity_degraded but never threaded it into CheckGateDegradedNotify,
    # so the terminal email still said plain "SUCCESS" — the
    # absence-of-verdict-read-as-pass shape alpha-engine-config-I6044 rules
    # against, at the terminal-notification surface.
    "NotifyCompleteParityDegraded": ArtifactReason(
        reason="Terminal success SNS publish for a run whose Parity verdict "
        "degraded (alpha-engine-config-I6025) — a PublishParityDegraded "
        "alert fired earlier in the run. No persisted artifact (the email "
        "IS the surface)."
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
    # ── Pre-open Trading SF (17 substantive Task steps) ───────────────────────────
    "DeployDriftCheck": ArchivePageRef(
        page="fleet-status",
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
    # alpha-engine-config-I7111 (Brian ruling 2026-08-13, option c): the
    # "no trading-pipeline start during NYSE market hours" boundary, sited on
    # the PIPELINE rather than on one caller's IAM role. These five states are
    # the FIRST thing both weekday definitions do — shared by name across
    # step_function_daily.json and step_function_eod.json, so one entry each
    # serves both (same convention as MorningEnrich / PublishDeployDriftDegraded
    # above). None of them writes an artifact: the gate's whole output is which
    # SF branch is taken, and it is preserved in $.market_hours_gate in
    # execution history plus the SNS alert.
    "MarketHoursGate": ArtifactReason(
        reason="NYSE regular-session gate — predictor Lambda invoke (action="
        "check_market_hours), pure calendar math judged on "
        "$$.Execution.StartTime, run before the mutex and before anything "
        "that spends. Refuses a start inside [09:30, 16:00) ET from every "
        "principal (scheduler, overseer-sf-watch, operator); the config#2932 "
        "IAM-toggle mechanism it replaces bound only one caller and was never "
        "deployed. No artifact — the verdict is encoded in the SF branch "
        "taken and in $.market_hours_gate.",
    ),
    "RecordMarketHoursOverride": ArtifactReason(
        reason="Audit publish for an authorised in-session start: an explicit, "
        "expiring market_hours_override (reason + authorized_by + expires_at, "
        "bounded to 24h) crossed the market-hours boundary. Announced rather "
        "than silent; no persisted artifact (the alert plus "
        "$.market_hours_gate in execution history ARE the record).",
    ),
    "NotifyMarketHoursBlocked": ArtifactReason(
        reason="Refusal SNS publish — a trading pipeline was started inside "
        "the NYSE regular session with no override. Precedes the "
        "MarketHoursBlocked Fail terminal, so the run does not read green to "
        "a status-keyed watcher (sf-pipeline-policy §2.3). The message "
        "carries the override shape needed to authorise one start. No "
        "persisted artifact (the alert IS the surface).",
    ),
    "NotifyMarketHoursOverrideMalformed": ArtifactReason(
        reason="Refusal SNS publish — a market_hours_override was offered and "
        "is not usable (missing/blank field, unparseable or back-dated "
        "expires_at, or one beyond the 24h ceiling). Distinct from "
        "NotifyMarketHoursBlocked because it calls for a different operator "
        "action: fix the override, not wait for the close. No persisted "
        "artifact (the alert IS the surface).",
    ),
    "NotifyMarketHoursUnverified": ArtifactReason(
        reason="The market-hours gate Lambda did not answer after retries. "
        "The two weekday pipelines diverge here on purpose: preopen fails "
        "CLOSED (this alert precedes a Fail — the same Lambda serves "
        "PredictorInference, so nothing was left to rescue), postclose fails "
        "OPEN and DEGRADED (NAV continuity, sf-pipeline-policy §1.3, and its "
        "only automated caller applies the identical predicate earlier). No "
        "persisted artifact (the alert IS the surface).",
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
    # config#1811 Part A (2026-07-06 pre-open wedge incident): verify the
    # three repo checkouts on ae-trading are at origin/main's SHA before any
    # pipeline work runs, with one in-place self-heal attempt.
    "CodeFreshnessGate": ArtifactReason(
        reason="Box code-freshness gate — SSM check that the ae-trading "
        "checkouts (executor/data/config) are on main @ origin/main before "
        "pipeline work, one self-heal retry; no artifact — gate outcome is "
        "encoded in the SF branch taken."
    ),
    # config#1807 (2026-07-06): pre-open data phase decoupled onto a daily
    # data spot; launch is fire-and-forget from the dashboard dispatcher.
    "LaunchDailyDataSpot": ArtifactReason(
        reason="Fire-and-forget spot_data_weekly.sh --launch-only dispatch "
        "for the daily data spot (krepis.ec2_spot rotation + dead-man "
        "watchdog); instance id lands in ops/daily_data_spot/{date}.json — "
        "operational only, no rendered artifact."
    ),
    # config#1811 Part B: the external ssm-liveness-poller's unresponsive-box
    # branch (≥3 consecutive SSM ping misses) force-stops the wedged instance.
    "ForceStopUnresponsiveInstance": ArtifactReason(
        reason="EC2 stopInstances on an SSM-unresponsive (wedged) trading "
        "box, dispatched by the liveness-watchdog branch before HandleFailure; "
        "no artifact — operational only."
    ),
    # MorningEnrich (weekday) — same state name as Saturday; same entry above wins.
    "PredictorInference": ArchivePageRef(
        page="predictor",
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
        page="fleet-status",
        artifact_label="Predictor health check",
    ),
    # config#1853 (closing config#1282 / crucible-predictor#305): daily
    # prediction-health producer — action=check_drift AFTER PredictorHealthCheck
    # so trading_day's predictions exist to score. Fail-soft observability
    # producer, never a trading gate.
    "PredictorDriftCheck": ArtifactReason(
        reason="Daily prediction-drift producer (config#1853) — writes "
        "predictor/metrics/drift_{trading_day}.json (artifact_id "
        "predictor_drift_detection; freshness tracked on page 26). Fail-soft "
        "observability, never blocks the morning planner — no dedicated "
        "archive page yet.",
    ),
    "RunMorningPlanner": ArchivePageRef(
        page="host_execution?tab=Order+Book",
        artifact_label="Order book + rationale",
    ),
    "RunDaemon": ArchivePageRef(
        page="fleet-status",
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
    # config#1687 spot migration: the weekday data phases dispatch to per-stage
    # data spots via the alpha-engine-data-spot-dispatcher Lambda. The launch
    # states are fire-and-forget dispatches; the phase outputs themselves are
    # substrate (see MorningEnrich / MorningArcticAppend).
    "LaunchMorningEnrichSpot": ArtifactReason(
        reason="Fire-and-forget spot dispatch (alpha-engine-data-spot-"
        "dispatcher Lambda, workload=morning-enrich) for the weekday "
        "MorningEnrich phase — data-spot migration config#1687/#1807. "
        "Operational only, no rendered artifact; the enrich output is "
        "substrate (see MorningEnrich).",
    ),
    "LaunchMorningArcticAppendSpot": ArtifactReason(
        reason="Fire-and-forget spot dispatch (alpha-engine-data-spot-"
        "dispatcher Lambda, workload=morning-arctic-append) for the weekday "
        "ArcticDB daily_append phase — data-spot migration config#1687/#1807. "
        "Operational only, no rendered artifact; the append output is "
        "substrate (see MorningArcticAppend).",
    ),
    # Shared state name across the pre-open AND post-close SFs (the registry
    # is keyed by state name; one entry serves both, like MorningEnrich).
    # alpha-engine-config-I6615 / sf-pipeline-policy §3 (2026-08-09): the
    # morning DeployDriftGate now distinguishes drift families — SF-definition
    # drift halts, CloudFormation/stack-state drift degrades loudly.
    "PublishDeployDriftDegraded": ArtifactReason(
        reason="Degraded-path SNS alert fired when the morning DeployDriftGate "
        "finds CloudFormation/stack-state drift WITHOUT SF-definition drift "
        "(alpha-engine-config-I6615): trading proceeds on the last verified "
        "frozen SHA with the degraded flag set, and the day terminates "
        "DEGRADED rather than halting. No persisted artifact (the alert IS "
        "the surface).",
    ),
    # alpha-engine-config-I5569 (2026-08-09): CaptureSnapshot is the EOD
    # pipeline's one non-re-runnable stage (live-capture-only) — bounded
    # same-day retry with paging on first failure and on budget exhaustion.
    "PageCaptureSnapshotFailureImmediate": ArtifactReason(
        reason="Same-day SNS page fired on CaptureSnapshot's FIRST failure, "
        "before its single bounded retry (alpha-engine-config-I5569) — "
        "snapshots are live-capture-only, so the operator response window "
        "closes at the date boundary. No persisted artifact (the page IS the "
        "surface).",
    ),
    "PageCaptureSnapshotIrreversibleFailure": ArtifactReason(
        reason="SNS page fired when CaptureSnapshot's bounded retry budget is "
        "exhausted (alpha-engine-config-I5569): the day's snapshot becomes "
        "permanently unrecoverable at the date boundary; the execution still "
        "hard-fails via HandleFailure (no silent fallback to live IB). No "
        "persisted artifact (the page IS the surface).",
    ),
    "PublishDataSpotFailureImmediate": ArtifactReason(
        reason="Fail-open SNS alert fired when a data-spot phase fails — the "
        "pipeline continues past it (daemon / EOD reconcile proceed on prior "
        "data) per the fail-open contract of the config#1687 spot migration. "
        "No persisted artifact (the email IS the surface).",
    ),
    "PublishScannerFailureImmediate": ArtifactReason(
        reason="Fail-open SNS alert fired when the weekday Scanner fails — the "
        "pipeline continues to predictor inference and daemon start "
        "(alpha-engine-config#6722). Added by config-I6903: this was the only "
        "one of the five degraded paths in the weekday definition with no "
        "alert at all, so on 2026-08-11 the Scanner timed out twice at 300s "
        "and nothing said so for 37 minutes — until the terminal, past the "
        "window where a rerun before the 06:30 open was still possible. "
        "No persisted artifact (the alert IS the surface); what it carries is "
        "that the day's universe board, membership and leaderboard were not "
        "written and the optimizer is running without per-name ADV.",
    ),
    "PublishDaemonFailureImmediate": ArtifactReason(
        reason="Fail-open SNS alert fired when the RunDaemon systemd restart "
        "fails and the pipeline continues on the assumption that the boot "
        "trigger starts the daemon instead (config-I6903). "
        "sf-pipeline-policy.md §5 names this fail-open explicitly as one that "
        "must page immediately, and the paging half was never built. That the "
        "boot trigger is a backup is WHY this stage fails open; it is not a "
        "reason to stay quiet, because a backup that also missed leaves the "
        "book unmanaged. No persisted artifact (the alert IS the surface).",
    ),
    "WeeklyExerciseLaunchFailed": ArtifactReason(
        reason="Fail-open SNS alert fired when postclose cannot start the "
        "weekly pipeline's daily EXERCISE run (alpha-engine-config-I5489). "
        "Postclose's own deliverable is already complete at this point, so "
        "the alert routes onward to the normal terminal rather than failing "
        "the execution. No persisted artifact (the alert IS the surface) — "
        "the signal it carries is that the weekly pipeline went un-exercised "
        "that day, which is otherwise indistinguishable from a day nobody "
        "looked at.",
    ),
    # alpha-engine-config-I6689: the declared-cadence read/gate inserted
    # ahead of LaunchWeeklyExerciseRun (ReadExerciseCadence ->
    # CheckExerciseCadence). Both alerts are fail-open + best-effort SNS
    # publishes, same shape as WeeklyExerciseLaunchFailed / PublishLibPinGateDegraded
    # above — no persisted artifact, the alert IS the surface.
    "PublishCadenceReadDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when ReadExerciseCadence cannot "
        "read /alpha-engine/weekly-sf/exercise-cadence from SSM (most "
        "likely: the IAM grant for alpha-engine-step-functions-role has not "
        "landed yet — alpha-engine-config#6689). The run proceeds fail-open "
        "as cadence=daily, i.e. the pre-config#6689 default behavior — the "
        "exercise launch is NOT skipped. No persisted artifact (the alert "
        "IS the surface).",
    ),
    "PublishCadenceUnknownValueDegraded": ArtifactReason(
        reason="Fail-open SNS alert fired when ReadExerciseCadence's read "
        "succeeded but returned a value CheckExerciseCadence does not "
        "recognize (not one of daily / weekly-only / off — a manifest/SSM "
        "typo, or an undeployed edit to infrastructure/weekly_cadence.json). "
        "The run proceeds fail-open as cadence=daily (alpha-engine-config#6689). "
        "No persisted artifact (the alert IS the surface).",
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
    # config#1687 spot migration, EOD side (see the weekday Launch* entries
    # above; PublishDataSpotFailureImmediate is shared by state name).
    "LaunchPostMarketDataSpot": ArtifactReason(
        reason="Fire-and-forget spot dispatch (alpha-engine-data-spot-"
        "dispatcher Lambda, workload=post-market-data) for the EOD "
        "PostMarketData phase — data-spot migration config#1687. Operational "
        "only, no rendered artifact; the phase output is substrate (see "
        "PostMarketData).",
    ),
    "LaunchPostMarketArcticAppendSpot": ArtifactReason(
        reason="Fire-and-forget spot dispatch (alpha-engine-data-spot-"
        "dispatcher Lambda, workload=post-market-arctic-append) for the EOD "
        "ArcticDB daily_append phase — data-spot migration config#1687. "
        "Operational only, no rendered artifact; the append output is "
        "substrate (see PostMarketArcticAppend).",
    ),
    "CaptureSnapshot": ArchivePageRef(
        page="eod-report",
        artifact_label="NAV + positions snapshot",
    ),
    "EODReconcile": ArchivePageRef(
        page="eod-report",
        artifact_label="EOD reconcile briefing",
    ),
    "SkipEODReconcileDataGap": ArtifactReason(
        reason="Loud, distinctly-labeled fail-open SNS alert (nousergon-data "
        "#813) fired when the post-market data-spot phase exhausts its retry "
        "budget (most likely a spot interruption) and EODReconcile is "
        "skipped rather than crashing into the generic HandleFailure path — "
        "today's close is genuinely absent from ArcticDB. Recovery is "
        "automatic/operator-driven: chronic-gap-heal backfills the missing "
        "bar, then an operator-replay execution with "
        "skip_post_market_data=true reruns EODReconcile alone. No persisted "
        "artifact (the email IS the surface); see EODReconcile for the "
        "briefing this alert stands in for."
    ),
    # config-I2702 EOD self-heal loop (deliverable #1): verify-by-artifact
    # precondition probe that replaced the old $.data_spot_error launch-
    # phase flag test with a fresh S3 readback of the macro-freshness
    # sentinel — the actual precondition EODReconcile's _spy_close hard-
    # depends on. Called fresh at every reconcile-gate entry, including
    # each HealReProbe iteration below; never cached.
    "ProbeEODReconcilePrecondition": ArtifactReason(
        reason="Verify-by-artifact EOD-reconcile precondition probe "
        "(config-I2702) — invokes alpha-engine-eod-precondition-probe to "
        "readback-verify run_date's SPY close is queryable in ArcticDB "
        "before gating EODReconcile/the heal loop. No persisted artifact "
        "— outcome is encoded in the SF branch taken (CheckSkipEODReconcile)."
    ),
    # config-I2702 deliverable #3(a): heal-loop dispatch of the missing
    # post-market-data workload, force_on_demand=true unconditionally
    # (already in a degraded-recovery context past the earlier
    # single-retry-on-spot, so on-demand is preferred over a second spot
    # gamble). Mirrors LaunchPostMarketDataSpot's registry entry.
    "HealLaunchPostMarketDataSpot": ArtifactReason(
        reason="Heal-loop fire-and-forget spot/on-demand re-dispatch "
        "(config-I2702) of the missing post-market-data workload via "
        "alpha-engine-data-spot-dispatcher, force_on_demand=true. "
        "Operational only, no rendered artifact; the phase output is "
        "substrate (see PostMarketData)."
    ),
    # config-I2702 deliverable #3(a): heal-loop dispatch of the missing
    # post-market-arctic-append workload — the actual SPY-close WRITE
    # ProbeEODReconcilePrecondition/HealReProbe check for.
    "HealLaunchArcticAppendSpot": ArtifactReason(
        reason="Heal-loop fire-and-forget spot/on-demand re-dispatch "
        "(config-I2702) of the missing post-market-arctic-append "
        "workload via alpha-engine-data-spot-dispatcher, "
        "force_on_demand=true. Operational only, no rendered artifact; "
        "the append output is substrate (see PostMarketArcticAppend)."
    ),
    # config-I2702 deliverable #3(b): re-probe after the heal-loop-
    # dispatched workloads complete. Overwrites $.precondition_probe with
    # the fresh result — never trusts the earlier probe.
    "HealReProbe": ArtifactReason(
        reason="Heal-loop re-probe (config-I2702) after the re-dispatched "
        "data-spot workload(s) complete — invokes "
        "alpha-engine-eod-precondition-probe again and overwrites "
        "$.precondition_probe with the fresh result for "
        "HealCheckConverged / the next HealLoopGate deadline check. No "
        "persisted artifact — a re-probe, not a producer."
    ),
    # config-I2702 closes-when "convergence notification": informational
    # (non-paging) — the self-heal succeeded without any operator action.
    "HealConvergedNotify": ArtifactReason(
        reason="Informational (non-paging) SNS publish (config-I2702) — "
        "the EOD self-heal loop converged (SPY close verified present) "
        "and a reconcile-only replay execution was dispatched "
        "automatically; no operator action needed. No persisted artifact "
        "(the email IS the surface); see EODReconcile for the briefing "
        "the dispatched replay produces."
    ),
    # config-I2702 deliverable #3(d): the loud page. Fires only when the
    # heal loop exhausted its attempt budget, passed the probe-computed
    # deadline, or a replay execution still can't converge.
    "HealNonConvergent": ArtifactReason(
        reason="Loud fail-open SNS alert (config-I2702) fired when the "
        "EOD self-heal loop did NOT converge — exhausted its 2-attempt "
        "budget and/or passed its deadline without SPY close landing in "
        "ArcticDB. Requires a manual operator-replay (I2700 shape: "
        "skip_post_market_data=true, skip_capture_snapshot=true). No "
        "persisted artifact (the email IS the surface)."
    ),
    # config-I2702: loud page for the narrower case where the precondition
    # itself converged but the auto-replay StartExecution call failed.
    "HealReplayDispatchFailed": ArtifactReason(
        reason="Loud SNS alert (config-I2702) — the heal-loop precondition "
        "converged (SPY close verified present) but the auto-replay "
        "StartExecution call itself failed; requires a manual "
        "operator-replay (I2700 shape). No persisted artifact (the email "
        "IS the surface)."
    ),
    "DailySubstrateHealthCheck": ArchivePageRef(
        page="fleet-status",
        artifact_label="Daily substrate health check",
    ),
    "PublishSubstrateHealthCheckDegradedAlert": ArtifactReason(
        reason="Best-effort fail-open SNS alert (config#2326) fired when the "
        "EOD daily substrate health check degrades (crashed, timed out, or "
        "finished non-Success) — surfaces the miss without blocking the "
        "cost-guard instance stop; its own Catch still routes to "
        "StopTradingInstance so an SNS-side failure can't delay/block the "
        "cost-guard either. No persisted artifact (the email IS the "
        "surface); see DailySubstrateHealthCheck for the check this alert "
        "reports on."
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


def lookup_registry(state_name: str) -> ArchivePageRef | ArtifactReason | None:
    """Return the registry entry for ``state_name`` (None if absent).

    ``None`` here signals "this state is not in the registry" — distinct
    from :class:`ArtifactReason` ("registered as substrate-only with this
    reason"). The dashboard consumer should treat ``None`` as a CI-time
    test failure (registry drift); it should NEVER render in production.
    """
    return STATE_TO_ARCHIVE_PAGE.get(state_name)


# ── Work-verdict declarations (alpha-engine-config-I8045) ─────────────────
#
# Two declarations, and the whole three-way verdict in
# :mod:`nousergon_lib.pipeline_status.work` rests on them. Both are
# DECLARED rather than inferred, per ``observability-policy.md`` §8.3 —
# a skip that has to be guessed at from a duration is a threshold wearing a
# taxonomy's clothes, and it silently reclassifies every time the graph
# changes shape.


#: The substantive spine of each pipeline: the stages whose entry is what
#: "the pipeline ran" MEANS. An execution that succeeds without entering
#: these did not do the work, however green its status.
#:
#: Lifted here from ``crucible-dashboard/loaders/pipeline_status_loader.py``
#: (``RELIABILITY_STAGE_ORDER``) on second adoption — the console adapters,
#: the grading tiles, the watchdog Lambdas and the SF-watch scripts all need
#: the same list, and a spine that lives in one consumer is a spine the
#: other consumers get wrong. ``shared-code-policy.md``.
#:
#: Ordered deepest-last: :func:`cycles._depth_of` reads position as progress.
#:
#: Kept in sync with the live SF JSONs by a producer-side contract test in
#: ``nousergon-data`` — a stage rename that is not mirrored here blinds every
#: reader matching on the old name, and that blindness reports as the benign
#: "that stage did not run" (the I6857 defect).
PIPELINE_STAGE_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "ne-weekly-freshness-pipeline": (
        "MorningEnrich",
        "DataPhase1",
        "RAGIngestion",
        "Scanner",
        "SignalsEnvelope",
        "PredictorTraining",
        "DataPhase2",
        "Backtester",
        "ParityParallel",
        "PitParityCompare",
        "ModelZooSelect",
        "ModelZooTrainMap",
        "EvaluatorDiagnostics",
        "EvaluatorOptimize",
        "ReportCard",
        "Director",
    ),
    "ne-preopen-trading-pipeline": (
        "StartExecutorEC2",
        "CodeFreshnessGate",
        "LaunchMorningEnrichSpot",
        "LaunchMorningArcticAppendSpot",
        "PredictorInference",
        "CheckPredictorCoverage",
        "RunMorningPlanner",
        "RunDaemon",
    ),
    "ne-postclose-trading-pipeline": (
        "LaunchPostMarketDataSpot",
        "LaunchPostMarketArcticAppendSpot",
        "CaptureSnapshot",
        "EODReconcile",
        "StopTradingInstance",
    ),
}


#: Terminal states that end an execution SUCCEEDED having deliberately not
#: done the work. Reaching one is CORRECT behaviour and must never page —
#: and must never be counted as a completed cycle either.
#:
#: Verified against the live state-machine definitions on 2026-08-21:
#:
#: - ``WeeklyRunDaySkip`` (``Succeed``) — the weekly cron fires THU-SAT and
#:   this gate self-selects the single correct day, so on a normal week 2 of
#:   the 3 firings land here before any spend. Its own definition comment
#:   calls it a "designed no-op ... this state produces no artifacts on
#:   purpose". Measured: 2.7-5.7s, five states entered, zero spine stages.
#: - ``ShellRunComplete`` (``Succeed``) — the Friday-PM preflight-only "shell
#:   run". Green preflight, no completion marker, no artifacts.
#: - ``NotifyHolidaySkip`` (``Task`` with ``End: true``) — the preopen
#:   market-holiday skip. A Task, not a ``Succeed``: a reader that looks only
#:   for ``Succeed``-typed terminals misses it, which is why this is a
#:   declared name list rather than a walk of the definition by type.
#:
#: ``ne-postclose-trading-pipeline`` has no skip terminal — its market-hours
#: gate routes to ``MarketHoursBlocked``, a ``Fail``.
SKIP_TERMINALS: Final[dict[str, frozenset[str]]] = {
    "ne-weekly-freshness-pipeline": frozenset({"WeeklyRunDaySkip", "ShellRunComplete"}),
    "ne-preopen-trading-pipeline": frozenset({"NotifyHolidaySkip"}),
    "ne-postclose-trading-pipeline": frozenset(),
}


def stage_order_for(state_machine: str) -> tuple[str, ...]:
    """Declared substantive spine for a state machine ARN or bare name.

    Returns ``()`` for an undeclared pipeline — callers must treat that as
    "cannot be judged" and raise, never as "nothing was expected, so it
    passed". :func:`work.classify_work` does exactly that.
    """
    name = state_machine.rsplit(":", 1)[-1] if state_machine else ""
    return PIPELINE_STAGE_ORDER.get(name, ())


def skip_terminals_for(state_machine: str) -> frozenset[str]:
    """Declared no-work terminals for a state machine ARN or bare name."""
    name = state_machine.rsplit(":", 1)[-1] if state_machine else ""
    return SKIP_TERMINALS.get(name, frozenset())
