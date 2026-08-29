"""The arena engine — one cycle: ladders, pointer decision, retirements, artifact.

This is the single implementation all four slots (universe cut, selection
producer, model M, strategy S) consume, so the rules cannot drift into four
versions (`shared-code-policy.md`). It is pure compute: no I/O, no S3, no
logging. Callers supply per-date scores and receive an :class:`ArenaCycle`
to persist.

**The decision rule, from Brian's rulings of 2026-08-29:**

- The pointer goes to the arm leading the incumbent on the **longest window
  the two of them share**, among arms whose lead the **anytime-valid
  sequence** supports. If no arm's lead is supported, the incumbent holds.
- The pointer moves **freely, in both directions, with no cooldown and no
  hysteresis** — "if for a time period version 1 beats version 2, but over
  time version 2 regains the edge, then version 1 should be champion while it
  beats, but should be replaced by version 2 when version 2 regains the
  edge." What makes that safe is that the decision window is CUMULATIVE: a
  single bad week cannot flip a cumulative ranking. A trailing window under
  the same rule would thrash.
- An arm is **retired** when it is at least ``grace_weeks`` old AND at least
  ``cap`` arms beat it pairwise. Arms are never blocked from being created;
  the pool may exceed ``cap`` while a newly added recipe is inside its grace
  window.

**An arm is a recipe, and a refit is not a new arm.** The roster is a set of
recipes — features, hyperparameters, training-window rule and refit cadence,
all fixed at registration — whose fitted weights refresh on the schedule the
recipe itself declares. A retrain is therefore the arm doing its job, and its
score series stays continuous across every refit. A new arm exists only when
a recipe is deliberately added or changed.

**Improper training is a HARD TASK FAILURE, never a miss.** Brian ruled
2026-08-29: "if any of the arms is not trained properly then the predictor
module should fail the task." :func:`run_cycle` raises
:class:`TrainingIntegrityError` rather than scoring, because arms in a slot
share a training substrate — a defect that spoils one arm's fit is evidence
the whole cycle's inputs are compromised, and recording it as a miss lets the
pipeline continue and publish a verdict built on void inputs. That is exactly
what happened the week of 2026-08-29: ``PredictorTraining`` returned SSM
``Status: Success``, all four zoo specs reported ``spec_status: OK``,
``ModelZooSelect`` wrote a complete leaderboard and ``branch_b_degraded`` was
``false`` — on a week when every model was fitted with seven features
hard-zeroed. §7.2's "a record asserting an action that never happened."

``ArmSeries.misses`` keeps its original and much narrower meaning: an arm
that legitimately had nothing to say for a cycle, such as a cut that selected
zero names. "This arm had nothing to say" and "this arm was trained on broken
inputs" must never render identically.

**Two hard preconditions on SERVING, independent of ranking.** An arm may
lead the ladder and still not be allowed to serve. Both are supplied by the
caller as :class:`ServingPrecondition` results because both are slot-specific
facts the engine cannot compute:

1. **The behavioural veto** (slot M). Measured 2026-08-29 to be the only
   guard that actually worked. It must stay in its scale-DEPENDENT form: a
   scale-invariant version would have passed both the collapsed 2026-08-28
   model AND the 2026-08-21 model that produced five live sessions with zero
   high-confidence names (standardized ratios 0.943 and 0.973 — both look
   healthy once you divide the collapse away).
2. **Input completeness.** An arm scored on partial inputs may rank first
   and still be unfit to trade.

A precondition failure on the INCUMBENT forces the pointer to move to the
best eligible arm even without a supported lead — the alternative is serving
an arm that is known to be unfit. If no arm is eligible, the cycle's status
is ``unservable`` and it fails loud; it is never an empty pass (§7.2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .arms import ArmRegister
from .confseq import DEFAULT_ALPHA, DEFAULT_OPT_N, ConfSeqBound, confidence_sequence
from .ladder import ScoreLadder, build_ladder
from .ranking import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    PairwiseRanking,
    rank_pairwise,
)
from .window import ArmSeries, PairedWindow, pair_on_common_window

__all__ = [
    "ArenaConfig",
    "TrainingStatus",
    "TrainingIntegrityError",
    "assert_training_integrity",
    "ServingPrecondition",
    "Comparison",
    "PointerDecision",
    "RetirementVerdict",
    "ArenaCycle",
    "decide_pointer",
    "evaluate_retirements",
    "run_cycle",
    "BENCHMARK_POPULATION",
    "SELECTION_SLOT_KINDS",
    "ARENA_CYCLE_SCHEMA_VERSION",
]

ARENA_CYCLE_SCHEMA_VERSION = 1

#: The default benchmark: the population the arm selected FROM.
BENCHMARK_POPULATION = "population"

#: Slot kinds whose job is to beat the population they drew from, and which
#: therefore may never be graded against SPY. Found 2026-08-17: arms were
#: graded against SPY when SPY trailed the population they were drawn from by
#: 140bp at 21d, which inverted wins and losses outright.
SELECTION_SLOT_KINDS = ("universe_cut", "selection_producer")


class ArenaConfigError(ValueError):
    """A slot configuration that cannot produce a meaningful comparison."""


class TrainingIntegrityError(RuntimeError):
    """At least one arm was not trained properly. The cycle does not run.

    Deliberately a hard raise and not a status field: a status field can be
    read, logged, and ignored by a caller that then writes a complete-looking
    artifact. Raising means the task fails and the pipeline stops, which is
    the ruling (Brian, 2026-08-29).
    """


@dataclass(frozen=True)
class TrainingStatus:
    """Whether one arm's most recent fit is sound enough to be scored on.

    Supplied by the slot — the engine cannot inspect a training run. ``ok``
    is False for a degenerate, partial, or input-compromised fit. It is NOT
    the place to record "this arm produced no output this cycle"; that is a
    miss, and it belongs in :attr:`ArmSeries.misses`.
    """

    arm_id: str
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "ok": self.ok, "reason": self.reason}


def assert_training_integrity(
    statuses: Mapping[str, TrainingStatus],
    required_arms: Sequence[str],
) -> None:
    """Raise unless every required arm reports a sound fit.

    A missing status is as fatal as a failed one: an arm whose training was
    never asserted about is indistinguishable from an arm whose training
    silently failed, and the whole point of the ruling is that the cycle must
    not proceed on inputs nobody vouched for.
    """
    missing = sorted(a for a in required_arms if a not in statuses)
    if missing:
        raise TrainingIntegrityError(
            f"no training status reported for arm(s) {missing}; an unasserted fit is "
            "treated as a failed fit — the cycle does not run on inputs "
            "nobody vouched for (Brian ruling 2026-08-29)"
        )
    failed = sorted(a for a in required_arms if not statuses[a].ok)
    if failed:
        detail = "; ".join("{}: {}".format(a, statuses[a].reason or "no reason given") for a in failed)
        raise TrainingIntegrityError(
            f"arm(s) not trained properly, so the whole cycle fails: {detail}. Arms "
            "in a slot share a training substrate, so a defect that spoils one "
            "arm's fit is evidence the cycle's inputs are compromised. This is "
            "a TASK FAILURE, not a miss and not a degraded run "
            "(Brian ruling 2026-08-29)."
        )


@dataclass(frozen=True)
class ArenaConfig:
    """Per-slot parameters. Every one is a slot fact, none is a fleet constant."""

    slot: str
    slot_kind: str
    benchmark: str = BENCHMARK_POPULATION
    #: Two-sided level of the anytime-valid sequence governing PROMOTION.
    alpha: float = DEFAULT_ALPHA
    #: Declared bound on a per-date score DIFFERENCE, in the score's units.
    #: The sub-Gaussian scale is derived from it, so the interval's validity
    #: is checkable from configuration alone.
    diff_clip: float = 0.05
    variance_mode: str = "declared"
    opt_n: int = DEFAULT_OPT_N
    #: Minimum paired dates before a comparison is attempted at all. This is
    #: NOT a thin-evidence gate — the confidence sequence is the evidence bar
    #: — it only refuses a window from which no statistic can be formed.
    min_paired_dates: int = 1
    #: Brian's cap, applied as a RETIREMENT criterion, never as an admission gate.
    cap: int = 5
    #: An arm younger than this is never retired, whatever its ranking.
    grace_weeks: int = 4
    #: Never retire below this many active arms. Two arms are the bare
    #: minimum for a comparison to exist at all; three leaves slack for one
    #: arm to miss a cycle or fail a serving precondition and still leave a
    #: live comparison. A slot stranded at one arm is the 2026-08-21/28
    #: `no_promotable_challenger` defect this engine exists to prevent.
    min_active_arms: int = 3
    #: §3 — a retired arm keeps being scored for this many cycles past
    #: retirement, so "we retired the wrong one" stays detectable.
    retired_trailing_cycles: int = 8
    #: Evidence bar for a pairwise loss in the RETIREMENT ranking. Point
    #: estimate by default: the four-week grace period IS the evidence bar
    #: for retirement, and the sequence is the evidence bar for SERVING.
    #: Requiring anytime-valid support at four weeks would retire nothing and
    #: defeat the cap.
    retire_evidence: str = EVIDENCE_POINT
    #: Emitted-ladder size cap. Never affects a decision.
    max_ladder_weeks: int | None = None

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ArenaConfigError(f"cap must be >= 1; got {self.cap}")
        if self.min_active_arms < 2:
            raise ArenaConfigError(
                "min_active_arms must be >= 2: a slot with one arm produces "
                "ZERO comparisons, which is exactly the "
                "`no_promotable_challenger` defect of 2026-08-21 and "
                f"2026-08-28; got {self.min_active_arms}"
            )
        if self.min_active_arms > self.cap:
            raise ArenaConfigError(
                f"min_active_arms ({self.min_active_arms}) exceeds cap ({self.cap}); the floor would "
                "permanently block every retirement the cap requires"
            )
        if self.grace_weeks < 1:
            raise ArenaConfigError(f"grace_weeks must be >= 1; got {self.grace_weeks}")
        if self.retire_evidence not in (EVIDENCE_POINT, EVIDENCE_ANYTIME_VALID):
            raise ArenaConfigError(
                "retire_evidence must be 'point' or 'anytime_valid'; got "
                f"{self.retire_evidence!r}"
            )
        if self.slot_kind in SELECTION_SLOT_KINDS and self.benchmark != BENCHMARK_POPULATION:
            raise ArenaConfigError(
                f"slot {self.slot!r} is a selection-stage slot ({self.slot_kind}) and must be graded "
                f"against the POPULATION it selected from, not {self.benchmark!r}. Grading a "
                "selection stage against SPY inverted wins and losses on "
                "2026-08-17, when SPY trailed the drawn-from population by "
                "140bp at 21d."
            )


@dataclass(frozen=True)
class ServingPrecondition:
    """A hard gate on SERVING, evaluated outside the engine and passed in."""

    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}


def _eligible(preconditions: Sequence[ServingPrecondition]) -> bool:
    return all(p.passed for p in preconditions)


@dataclass(frozen=True)
class Comparison:
    """One challenger measured against the incumbent on their common window."""

    challenger: str
    incumbent: str
    window: PairedWindow
    bound: ConfSeqBound | None
    status: str
    reason: str

    @property
    def supported(self) -> bool:
        return self.bound is not None and self.bound.supported

    def to_dict(self) -> dict[str, Any]:
        payload = self.window.to_dict()
        payload.update(
            {
                "challenger": self.challenger,
                "incumbent": self.incumbent,
                "status": self.status,
                "reason": self.reason,
                "confidence_sequence": self.bound.to_dict() if self.bound else None,
                "supported": self.supported,
            }
        )
        return payload


@dataclass(frozen=True)
class PointerDecision:
    """Where the champion pointer sits after this cycle, and why."""

    slot: str
    as_of: str
    incumbent: str | None
    champion: str | None
    moved: bool
    status: str  # decided | held | unmeasurable | unservable | bootstrap
    reason: str
    comparisons: tuple[Comparison, ...]
    ineligible: Mapping[str, tuple[ServingPrecondition, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "as_of": self.as_of,
            "incumbent": self.incumbent,
            "champion": self.champion,
            "moved": self.moved,
            "status": self.status,
            "reason": self.reason,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "ineligible": {
                arm: [p.to_dict() for p in checks]
                for arm, checks in sorted(self.ineligible.items())
            },
        }


@dataclass(frozen=True)
class RetirementVerdict:
    """Whether one arm is retired this cycle, and the reason either way."""

    arm_id: str
    retire: bool
    reason: str
    age_weeks: int
    pairwise_losses: int
    is_champion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "retire": self.retire,
            "reason": self.reason,
            "age_weeks": self.age_weeks,
            "pairwise_losses": self.pairwise_losses,
            "is_champion": self.is_champion,
        }


@dataclass(frozen=True)
class ArenaCycle:
    """The durable artifact for one evaluation cycle of one slot."""

    schema_version: int
    slot: str
    slot_kind: str
    benchmark: str
    as_of: str
    ladders: tuple[ScoreLadder, ...]
    ranking: PairwiseRanking | None
    decision: PointerDecision
    retirements: tuple[RetirementVerdict, ...]
    scored_arms: tuple[str, ...]
    active_arms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "slot_kind": self.slot_kind,
            "benchmark": self.benchmark,
            "as_of": self.as_of,
            "scored_arms": list(self.scored_arms),
            "active_arms": list(self.active_arms),
            "ladders": [ladder.to_dict() for ladder in self.ladders],
            "ranking": self.ranking.to_dict() if self.ranking else None,
            "decision": self.decision.to_dict(),
            "retirements": [verdict.to_dict() for verdict in self.retirements],
        }


def decide_pointer(
    config: ArenaConfig,
    as_of: str,
    incumbent: str | None,
    series_by_arm: Mapping[str, ArmSeries],
    preconditions: Mapping[str, Sequence[ServingPrecondition]] | None = None,
) -> PointerDecision:
    """Move the pointer to the best supported lead, or hold the incumbent.

    Free movement in both directions, no cooldown, no hysteresis margin
    (Brian ruling 2026-08-29). The self-damping property that makes this safe
    is that the comparison window is cumulative, not trailing.
    """
    checks = {arm: tuple(preconditions.get(arm, ())) for arm in series_by_arm} if preconditions else dict.fromkeys(series_by_arm, ())
    ineligible = {arm: c for arm, c in checks.items() if not _eligible(c)}
    eligible_arms = sorted(arm for arm in series_by_arm if arm not in ineligible)

    if not eligible_arms:
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=None,
            moved=False,
            status="unservable",
            reason=(
                "no arm passes its serving preconditions; the slot has nothing "
                "it is permitted to serve"
            ),
            comparisons=(),
            ineligible=ineligible,
        )

    incumbent_eligible = incumbent is not None and incumbent in eligible_arms

    if incumbent is None or incumbent not in series_by_arm:
        # Bootstrap (§9.1): no incumbent, an arm must serve, and the
        # alternative is no production behaviour at all.
        ranking = rank_pairwise(
            {arm: series_by_arm[arm] for arm in eligible_arms},
            created_dates=dict.fromkeys(eligible_arms, as_of),
            as_of=as_of,
            evidence_mode=EVIDENCE_POINT,
            alpha=config.alpha,
            clip=config.diff_clip,
            variance_mode=config.variance_mode,
            opt_n=config.opt_n,
            min_dates=config.min_paired_dates,
        )
        chosen = ranking.ordering[0]
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=chosen,
            moved=chosen != incumbent,
            status="bootstrap",
            reason=(
                "no eligible incumbent; bootstrap-promoted the highest Copeland "
                "arm (champion-challenger-policy.md §9.1)"
            ),
            comparisons=(),
            ineligible=ineligible,
        )

    comparisons: list[Comparison] = []
    for challenger in eligible_arms:
        if challenger == incumbent:
            continue
        window = pair_on_common_window(
            series_by_arm[challenger],
            series_by_arm[incumbent],
            min_dates=config.min_paired_dates,
        )
        if not window.measurable:
            comparisons.append(
                Comparison(
                    challenger=challenger,
                    incumbent=incumbent,
                    window=window,
                    bound=None,
                    status="unmeasurable",
                    reason=window.unmeasurable_reason or "unmeasurable",
                )
            )
            continue
        bound = confidence_sequence(
            window.diffs,
            alpha=config.alpha,
            clip=config.diff_clip,
            variance_mode=config.variance_mode,
            opt_n=config.opt_n,
        )
        comparisons.append(
            Comparison(
                challenger=challenger,
                incumbent=incumbent,
                window=window,
                bound=bound,
                status="measured",
                reason=(
                    "lead supported by the anytime-valid sequence"
                    if bound.supported
                    else "lead not supported by the anytime-valid sequence"
                ),
            )
        )

    # (lower bound, comparison) pairs. Building the tuple here rather than
    # reaching through `c.bound` at the ranking site keeps the bound's
    # presence a fact of the list's construction instead of an invariant a
    # reader has to hold in their head.
    supported: list[tuple[float, Comparison]] = [
        (c.bound.lower, c) for c in comparisons if c.bound is not None and c.bound.supported
    ]

    if not incumbent_eligible:
        # The incumbent is not permitted to serve. The pointer MUST move, and
        # a supported lead is not required — continuing to serve a known-unfit
        # arm is never the safer option.
        candidates = supported or [
            (c.bound.lower if c.bound else float("-inf"), c)
            for c in comparisons
            if c.status == "measured"
        ]
        if candidates:
            chosen = max(candidates, key=lambda item: (item[0], item[1].challenger))[1].challenger
        else:
            chosen = eligible_arms[0]
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=chosen,
            moved=chosen != incumbent,
            status="decided",
            reason=(
                "incumbent {} failed a serving precondition ({}); pointer "
                "forced to the best eligible arm".format(
                    incumbent,
                    "; ".join(
                        f"{p.name}: {p.reason}"
                        for p in ineligible.get(incumbent, ())
                        if not p.passed
                    ),
                )
            ),
            comparisons=tuple(comparisons),
            ineligible=ineligible,
        )

    if not comparisons:
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=incumbent,
            moved=False,
            status="unmeasurable",
            reason=(
                "slot has no eligible challenger to compare against; a slot "
                "with one arm produces zero comparisons "
                "(champion-challenger-policy.md §9.2 requires the champion "
                "still be scored)"
            ),
            comparisons=(),
            ineligible=ineligible,
        )

    if all(c.status == "unmeasurable" for c in comparisons):
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=incumbent,
            moved=False,
            status="unmeasurable",
            reason=(
                "no challenger shares a usable window with the incumbent: "
                + "; ".join(c.reason for c in comparisons)
            ),
            comparisons=tuple(comparisons),
            ineligible=ineligible,
        )

    if not supported:
        return PointerDecision(
            slot=config.slot,
            as_of=as_of,
            incumbent=incumbent,
            champion=incumbent,
            moved=False,
            status="held",
            reason="no challenger's lead is supported by the anytime-valid sequence",
            comparisons=tuple(comparisons),
            ineligible=ineligible,
        )

    # Rank supported challengers by the SUPPORTED lead — the confidence
    # sequence's lower bound. This is what makes leads on windows of different
    # lengths comparable without a cross-window aggregation: a short window
    # produces a wide interval and therefore a small lower bound on its own.
    winner_lower, winner = max(supported, key=lambda item: (item[0], item[1].challenger))
    return PointerDecision(
        slot=config.slot,
        as_of=as_of,
        incumbent=incumbent,
        champion=winner.challenger,
        moved=winner.challenger != incumbent,
        status="decided",
        reason=(
            f"{winner.challenger} leads {incumbent} by {winner.window.mean_diff:.6g} over {winner.window.n_dates} paired date(s) ({winner.window.weeks} week(s)); "
            f"confidence-sequence lower bound {winner_lower:.6g} > 0"
        ),
        comparisons=tuple(comparisons),
        ineligible=ineligible,
    )


def evaluate_retirements(
    config: ArenaConfig,
    as_of: str,
    register: ArmRegister,
    ranking: PairwiseRanking,
    champion: str | None,
) -> tuple[RetirementVerdict, ...]:
    """Apply Brian's cap-with-grace rule, in his literal form.

    An active arm is retired when **both** hold:

    - it is at least ``grace_weeks`` old, and
    - at least ``cap`` other arms beat it pairwise, each on that pair's own
      longest common window.

    Three things can veto a retirement, and each is reported rather than
    silently applied: the arm is the champion; retiring would drop the active
    pool below ``min_active_arms``; or the arm is inside its grace window.

    **The champion cannot be retired by this rule, by construction** — it is
    excluded explicitly here, and it is also excluded structurally: an arm
    that ``cap`` arms beat pairwise cannot simultaneously hold a lead that
    the pointer decision supports. Both are asserted in the test suite.
    """
    active = list(register.active_arms())
    verdicts: list[RetirementVerdict] = []
    remaining = len(active)

    # Deterministic order: worst-standing first, so that when the floor binds
    # it is the WORST arms that survive on the floor, never an arbitrary set.
    ordered = sorted(
        active,
        key=lambda a: ranking.ordering.index(a) if a in ranking.ordering else -1,
        reverse=True,
    )

    for arm in ordered:
        state = register.state(arm)
        age_weeks = state.age_weeks(as_of)
        standing = ranking.standings.get(arm)
        losses = standing.losses if standing else 0

        if arm == champion:
            verdicts.append(
                RetirementVerdict(arm, False, "champion: the serving arm is never retired", age_weeks, losses, True)
            )
            continue
        if standing is None:
            verdicts.append(
                RetirementVerdict(
                    arm,
                    False,
                    "unranked: no pairwise standing this cycle, so no evidence to retire on",
                    age_weeks,
                    0,
                    False,
                )
            )
            continue
        if age_weeks < config.grace_weeks:
            verdicts.append(
                RetirementVerdict(
                    arm,
                    False,
                    f"grace: {age_weeks} week(s) old, grace period is {config.grace_weeks}",
                    age_weeks,
                    losses,
                    False,
                )
            )
            continue
        if losses < config.cap:
            verdicts.append(
                RetirementVerdict(
                    arm,
                    False,
                    f"in top {config.cap}: only {losses} arm(s) beat it pairwise",
                    age_weeks,
                    losses,
                    False,
                )
            )
            continue
        if remaining - 1 < config.min_active_arms:
            verdicts.append(
                RetirementVerdict(
                    arm,
                    False,
                    (
                        f"floor: retiring would leave {remaining - 1} active arm(s), below "
                        f"min_active_arms={config.min_active_arms}; a slot stranded below the floor "
                        "produces zero comparisons"
                    ),
                    age_weeks,
                    losses,
                    False,
                )
            )
            continue

        remaining -= 1
        verdicts.append(
            RetirementVerdict(
                arm,
                True,
                (
                    f"{losses} arm(s) beat it pairwise (cap {config.cap}) and it is {age_weeks} week(s) "
                    f"old (grace {config.grace_weeks})"
                ),
                age_weeks,
                losses,
                False,
            )
        )

    return tuple(sorted(verdicts, key=lambda v: v.arm_id))


def run_cycle(
    config: ArenaConfig,
    as_of: str,
    register: ArmRegister,
    series_by_arm: Mapping[str, ArmSeries],
    incumbent: str | None,
    preconditions: Mapping[str, Sequence[ServingPrecondition]] | None = None,
    training: Mapping[str, TrainingStatus] | None = None,
) -> ArenaCycle:
    """Score every arm, decide the pointer, evaluate retirements, emit the artifact.

    ``series_by_arm`` must cover every arm the register says should be scored
    this cycle — active arms plus retired arms still inside their §3 trailing
    window. A missing series is a defect, not an omission, and raises.

    ``training`` must carry a :class:`TrainingStatus` for every ACTIVE arm in
    any slot whose arms are fitted. Any arm reporting an unsound fit — or no
    status at all — fails the whole cycle with
    :class:`TrainingIntegrityError`. Slots whose arms are not fitted (a
    deterministic universe cut, say) pass ``training=None``.
    """
    expected = set(register.scored_arms(as_of, config.retired_trailing_cycles))
    supplied = set(series_by_arm)
    missing = sorted(expected - supplied)
    if missing:
        raise ValueError(
            f"no series supplied for arm(s) {missing}; every registered arm is scored "
            f"every cycle, and a retired arm keeps being scored for {config.retired_trailing_cycles} cycle(s) "
            "so that 'we retired the wrong one' stays detectable "
            "(champion-challenger-policy.md §3)"
        )
    unregistered = sorted(supplied - set(register.all_arms()))
    if unregistered:
        raise ValueError(
            f"series supplied for unregistered arm(s) {unregistered}; writing shadow output "
            "without a register row is the `thinktank_coverage` defect — the "
            "data rots unnoticed (champion-challenger-policy.md §3)"
        )

    if training is not None:
        assert_training_integrity(training, register.active_arms())

    ladders = tuple(
        build_ladder(series_by_arm[arm], as_of, max_weeks=config.max_ladder_weeks)
        for arm in sorted(series_by_arm)
    )

    active = register.active_arms()
    active_series = {arm: series_by_arm[arm] for arm in active if arm in series_by_arm}
    ranking = None
    if len(active_series) >= 2:
        ranking = rank_pairwise(
            active_series,
            created_dates={arm: register.state(arm).record.created_date for arm in active_series},
            as_of=as_of,
            evidence_mode=config.retire_evidence,
            alpha=config.alpha,
            clip=config.diff_clip,
            variance_mode=config.variance_mode,
            opt_n=config.opt_n,
            min_dates=config.min_paired_dates,
        )

    decision = decide_pointer(
        config=config,
        as_of=as_of,
        incumbent=incumbent,
        series_by_arm=active_series,
        preconditions=preconditions,
    )

    retirements: tuple[RetirementVerdict, ...] = ()
    if ranking is not None:
        retirements = evaluate_retirements(
            config=config,
            as_of=as_of,
            register=register,
            ranking=ranking,
            champion=decision.champion,
        )

    return ArenaCycle(
        schema_version=ARENA_CYCLE_SCHEMA_VERSION,
        slot=config.slot,
        slot_kind=config.slot_kind,
        benchmark=config.benchmark,
        as_of=as_of,
        ladders=ladders,
        ranking=ranking,
        decision=decision,
        retirements=retirements,
        scored_arms=tuple(sorted(series_by_arm)),
        active_arms=active,
    )
