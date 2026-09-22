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
- A challenger is promotable only once its **paired** window with the
  incumbent reaches ``promote_min_weeks`` (Brian ruling 2026-09-01). The
  evidence bar itself is per-slot: ``promote_evidence`` is the anytime-valid
  sequence by default, or ``point`` — the largest positive mean paired
  difference among age-eligible challengers — where a slot has declared that
  delta (Brian ruling 2026-09-12, `alpha-engine-config-I10546`). Under
  ``point`` the sequence is still computed and emitted; it just does not
  decide.
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
    "ArmExclusion",
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

#: The pointer ranks on the mean paired difference. Every slot's behaviour
#: before ``ArenaConfig.promote_statistic`` existed, and still the default.
STATISTIC_MEAN_DIFF = "mean_diff"

#: The pointer ranks on the difference of the two arms' information ratios over
#: the pair's common window. For a slot whose arms carry different widths on
#: purpose — see ``ArenaConfig.promote_statistic``.
STATISTIC_INFORMATION_RATIO = "information_ratio"

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
    #: Brian's ruling, 2026-09-01: a new arm is promotable only after this many
    #: paired weeks (20 paired trading days at the default of 4) against the
    #: incumbent. Scored and laddered from week one; the pointer cannot move
    #: to the arm before this threshold — within eligibility the decision is
    #: the confidence sequence alone. Previously carried on crucible's
    #: `SlotSpec` as a declared second place because this field did not exist
    #: here (`alpha-engine-config-I9763`, `-I10504`).
    promote_min_weeks: int = 4
    #: Evidence bar for SERVING. The anytime-valid sequence is the policy
    #: default (`champion-challenger-policy.md` §5.0): a lead promotes only
    #: when the confidence sequence's lower bound clears zero. ``point``
    #: promotes the age-eligible challenger with the largest positive mean
    #: paired difference instead, and is a PER-SLOT DECLARED DELTA — never a
    #: fleet default — recorded under §5.2(B) with its skipped-SOTA rationale
    #: (Brian ruling 2026-09-12, `alpha-engine-config-I10546`: the
    #: ``universe_cut`` slot promotes the point-estimate leader after two
    #: paired weeks, because a paper account re-deciding weekly makes a
    #: reversible false promotion cheaper than never promoting at all). The
    #: confidence sequence is still computed and emitted on every comparison
    #: under ``point``, so the evidence the sequence would have required stays
    #: on the record even when it did not decide.
    promote_evidence: str = EVIDENCE_ANYTIME_VALID
    #: WHICH statistic the pointer ranks on, orthogonal to the evidence bar
    #: above. ``mean_diff`` (the default, and every slot's behaviour before
    #: this field existed) ranks on the mean paired difference.
    #: ``information_ratio`` ranks on the difference of the two arms' own
    #: information ratios over the pair's common window.
    #:
    #: A slot needs the second exactly when its arms carry DIFFERENT WIDTHS on
    #: purpose. A raw mean then selects for concentration rather than skill —
    #: mean alpha per name declines with depth whenever a ranking carries any
    #: signal — so the narrowest arm wins a comparison it did not earn. IR
    #: prices the concentration (IR ~= IC x sqrt(breadth)).
    #: `alpha-engine-config-I11403`, for crucible-research's `research` slot.
    promote_statistic: str = STATISTIC_MEAN_DIFF
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
        if self.promote_min_weeks < 1:
            raise ArenaConfigError(
                f"promote_min_weeks must be >= 1; got {self.promote_min_weeks}. Zero "
                "would promote an arm on its first cycle, which is the eligibility "
                "age Brian's 2026-09-01 ruling exists to set."
            )
        if self.retire_evidence not in (EVIDENCE_POINT, EVIDENCE_ANYTIME_VALID):
            raise ArenaConfigError(
                "retire_evidence must be 'point' or 'anytime_valid'; got "
                f"{self.retire_evidence!r}"
            )
        if self.promote_evidence not in (EVIDENCE_POINT, EVIDENCE_ANYTIME_VALID):
            raise ArenaConfigError(
                "promote_evidence must be 'point' or 'anytime_valid'; got "
                f"{self.promote_evidence!r}"
            )
        if self.promote_statistic not in (STATISTIC_MEAN_DIFF, STATISTIC_INFORMATION_RATIO):
            raise ArenaConfigError(
                "promote_statistic must be 'mean_diff' or 'information_ratio'; "
                f"got {self.promote_statistic!r}"
            )
        if (
            self.promote_statistic == STATISTIC_INFORMATION_RATIO
            and self.promote_evidence != EVIDENCE_POINT
        ):
            raise ArenaConfigError(
                f"slot {self.slot!r} declares promote_statistic="
                f"{STATISTIC_INFORMATION_RATIO!r} with promote_evidence="
                f"{self.promote_evidence!r}. The anytime-valid sequence is a "
                "bound on the MEAN of bounded per-date differences "
                "(champion-challenger-policy.md §5.0); it says nothing about a "
                "difference of two ratios, and applying it there would report a "
                "coverage guarantee the construction does not have. A slot "
                "ranking on the information ratio must declare "
                f"promote_evidence={EVIDENCE_POINT!r}, whose bar is the point "
                "estimate it actually ranks on. The sequence is still computed "
                "and emitted on mean_diff for every comparison, so the evidence "
                "it would have required stays on the record."
            )
        if self.slot_kind in SELECTION_SLOT_KINDS and self.benchmark != BENCHMARK_POPULATION:
            raise ArenaConfigError(
                f"slot {self.slot!r} is a selection-stage slot ({self.slot_kind}) and must be graded "
                f"against the POPULATION it selected from, not {self.benchmark!r}. Grading a "
                "selection stage against SPY inverted wins and losses on "
                "2026-08-17, when SPY trailed the drawn-from population by "
                "140bp at 21d."
            )

    def to_dict(self) -> dict[str, Any]:
        """The decision-governing knobs, for the cycle record.

        ``slot``, ``slot_kind`` and ``benchmark`` are deliberately omitted:
        they are already top-level fields of :class:`ArenaCycle` and a fact
        emitted twice is a fact that can disagree with itself. Everything
        here is a parameter that changed an outcome — a reader reconstructing
        why the pointer did what it did needs the thresholds that were in
        force, not the thresholds in today's config (`principles.md` §2.1).
        """
        return {
            "alpha": self.alpha,
            "diff_clip": self.diff_clip,
            "variance_mode": self.variance_mode,
            "opt_n": self.opt_n,
            "min_paired_dates": self.min_paired_dates,
            "cap": self.cap,
            "grace_weeks": self.grace_weeks,
            "promote_min_weeks": self.promote_min_weeks,
            "promote_evidence": self.promote_evidence,
            "promote_statistic": self.promote_statistic,
            "min_active_arms": self.min_active_arms,
            "retired_trailing_cycles": self.retired_trailing_cycles,
            "retire_evidence": self.retire_evidence,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, slot: str, slot_kind: str, benchmark: str
    ) -> ArenaConfig:
        """The inverse of :meth:`to_dict`.

        ``slot``, ``slot_kind`` and ``benchmark`` are required KEYWORD
        arguments rather than read off ``data``, because :meth:`to_dict`
        deliberately omits them — they already live on the enclosing
        :class:`ArenaCycle` and a fact emitted twice is a fact that can
        disagree with itself. A caller reconstructing a whole cycle passes
        the cycle's own top-level fields; :meth:`ArenaCycle.from_dict` does
        exactly that.

        ``max_ladder_weeks`` is NOT recoverable — :meth:`to_dict` never
        emits it ("never affects a decision") — so it is left at its
        dataclass default (``None``, no cap) rather than guessed. That is
        the one field on which a reconstructed ``ArenaConfig`` can
        legitimately differ from the config a cycle was actually produced
        under; nothing it governs (ladder emission size) is read by a
        decision.
        """
        return cls(
            slot=slot,
            slot_kind=slot_kind,
            benchmark=benchmark,
            alpha=float(data["alpha"]),
            diff_clip=float(data["diff_clip"]),
            variance_mode=str(data["variance_mode"]),
            opt_n=int(data["opt_n"]),
            min_paired_dates=int(data["min_paired_dates"]),
            cap=int(data["cap"]),
            grace_weeks=int(data["grace_weeks"]),
            promote_min_weeks=int(data["promote_min_weeks"]),
            promote_evidence=str(data["promote_evidence"]),
            # `.get` with the pre-field default: a cycle recorded before
            # `promote_statistic` existed was decided on the mean paired
            # difference, and reconstructing it as anything else would
            # misreport why that pointer moved.
            promote_statistic=str(data.get("promote_statistic", STATISTIC_MEAN_DIFF)),
            min_active_arms=int(data["min_active_arms"]),
            retired_trailing_cycles=int(data["retired_trailing_cycles"]),
            retire_evidence=str(data["retire_evidence"]),
        )


@dataclass(frozen=True)
class ServingPrecondition:
    """A hard gate on SERVING, evaluated outside the engine and passed in."""

    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ServingPrecondition:
        """The inverse of :meth:`to_dict`. Every field is stored directly."""
        return cls(
            name=str(data["name"]),
            passed=bool(data["passed"]),
            reason=str(data.get("reason") or ""),
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Comparison:
        """The inverse of :meth:`to_dict`.

        ``window`` is reconstructed from the SAME dict: ``to_dict`` starts
        from ``self.window.to_dict()`` and layers ``challenger``/
        ``incumbent``/... on top rather than nesting it, so the window's own
        keys (``arm_a``, ``arm_b``, ``n_dates``, ...) are top-level here
        too — :meth:`PairedWindow.from_dict` reads exactly those. See that
        method for what is and is not recoverable about the per-date series.
        ``supported`` is not read back: it is the ``bound.supported``
        property, recomputed from the restored bound.
        """
        bound = data.get("confidence_sequence")
        return cls(
            challenger=str(data["challenger"]),
            incumbent=str(data["incumbent"]),
            window=PairedWindow.from_dict(data),
            bound=ConfSeqBound.from_dict(bound) if bound is not None else None,
            status=str(data["status"]),
            reason=str(data.get("reason") or ""),
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointerDecision:
        """The inverse of :meth:`to_dict`."""
        return cls(
            slot=str(data["slot"]),
            as_of=str(data["as_of"]),
            incumbent=data.get("incumbent"),
            champion=data.get("champion"),
            moved=bool(data["moved"]),
            status=str(data["status"]),
            reason=str(data.get("reason") or ""),
            comparisons=tuple(Comparison.from_dict(c) for c in data.get("comparisons") or ()),
            ineligible={
                str(arm): tuple(ServingPrecondition.from_dict(p) for p in checks)
                for arm, checks in (data.get("ineligible") or {}).items()
            },
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetirementVerdict:
        """The inverse of :meth:`to_dict`. Every field is stored directly."""
        return cls(
            arm_id=str(data["arm_id"]),
            retire=bool(data["retire"]),
            reason=str(data.get("reason") or ""),
            age_weeks=int(data["age_weeks"]),
            pairwise_losses=int(data["pairwise_losses"]),
            is_champion=bool(data["is_champion"]),
        )


@dataclass(frozen=True)
class ArmExclusion:
    """One arm the cycle did NOT score, and why it was absent.

    A cycle is graded AS OF a trading day, so it contains exactly the arms
    that were in the arena on that day. Every other registered arm is
    excluded — and named here, with the two dates that decide it, so a reader
    comparing the cycle's arm count against the register today can tell an
    intentional point-in-time exclusion from a silently dropped arm
    (`principles.md` §2.7, `alpha-engine-config-I11084`).
    """

    arm_id: str
    reason: str
    created_date: str
    filed_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "reason": self.reason,
            "created_date": self.created_date,
            "filed_date": self.filed_date,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArmExclusion:
        """The inverse of :meth:`to_dict`. Every field is stored directly."""
        filed = data.get("filed_date")
        return cls(
            arm_id=str(data["arm_id"]),
            reason=str(data.get("reason") or ""),
            created_date=str(data["created_date"]),
            filed_date=None if filed is None else str(filed),
        )


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
    #: Every arm that was live in the slot ON `as_of`, CONTROLS INCLUDED.
    #: Kept for backward compatibility with existing consumers reading arm
    #: counts off this field — do not repurpose it to exclude controls; use
    #: `promotable_arms` for that instead. POINT-IN-TIME, like the rest of
    #: the cycle: an arm registered after `as_of` is absent from this field
    #: and named in `not_yet_registered_arms` instead
    #: (`alpha-engine-config-I11084`).
    active_arms: tuple[str, ...]
    #: `active_arms` with controls excluded — the pool `min_active_arms`
    #: actually governs (`evaluate_retirements`'s docstring,
    #: alpha-engine-config-I9770). A control is a real point of comparison
    #: but can never take the pointer, so it can never serve as the floor's
    #: slack; `len(promotable_arms)` is the number a consumer should compare
    #: against a slot's `min_active_arms` floor, not `len(active_arms)`.
    #: Computed by the same helper (`_promotable_arms`) that
    #: `evaluate_retirements` uses to check the floor, so the two can never
    #: disagree about which arms count.
    promotable_arms: tuple[str, ...]
    #: The parameters in force for THIS cycle. Emitted so a reader can
    #: reconstruct the decision without the current config, which may since
    #: have changed — `promote_evidence` and `promote_min_weeks` in particular
    #: change what a given set of comparisons decides.
    config: ArenaConfig
    #: Registered arms the cycle deliberately did NOT score because they were
    #: not in the arena on ``as_of`` — filed later, or declaring a later
    #: ``created_date``. Emitted on EVERY cycle, the empty tuple included, so
    #: that "no arm was excluded" is a reading rather than an absence
    #: (`alpha-engine-config-I11084`). Defaulted only so a hand-built cycle in
    #: a test need not state it; :func:`run_cycle` always sets it.
    not_yet_registered_arms: tuple[ArmExclusion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "slot_kind": self.slot_kind,
            "benchmark": self.benchmark,
            "as_of": self.as_of,
            "config": self.config.to_dict(),
            "scored_arms": list(self.scored_arms),
            "active_arms": list(self.active_arms),
            "promotable_arms": list(self.promotable_arms),
            "ladders": [ladder.to_dict() for ladder in self.ladders],
            "ranking": self.ranking.to_dict() if self.ranking else None,
            "decision": self.decision.to_dict(),
            "retirements": [verdict.to_dict() for verdict in self.retirements],
            "not_yet_registered_arms": [
                exclusion.to_dict() for exclusion in self.not_yet_registered_arms
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArenaCycle:
        """The inverse of :meth:`to_dict` — the whole `arena_cycle` artifact.

        This is the entry point a consumer that only ever reads the artifact
        (never recomputes it) actually calls; every sibling ``from_dict`` in
        this package exists to make this one possible. ``ArenaConfig`` is
        reconstructed from the nested ``config`` payload PLUS this cycle's
        own ``slot``/``slot_kind``/``benchmark`` — see
        :meth:`ArenaConfig.from_dict` for why those three are not inside
        ``config`` at all.

        Round-trips exactly through :meth:`to_dict` for every field this
        artifact actually serialises. The one documented exception is
        ``ArenaConfig.max_ladder_weeks``, which ``to_dict`` never emits
        (see that method) and which this cycle's own decision never reads.
        """
        config = ArenaConfig.from_dict(
            data["config"],
            slot=str(data["slot"]),
            slot_kind=str(data["slot_kind"]),
            benchmark=str(data["benchmark"]),
        )
        ranking = data.get("ranking")
        return cls(
            schema_version=int(data["schema_version"]),
            slot=str(data["slot"]),
            slot_kind=str(data["slot_kind"]),
            benchmark=str(data["benchmark"]),
            as_of=str(data["as_of"]),
            ladders=tuple(ScoreLadder.from_dict(d) for d in data.get("ladders") or ()),
            ranking=PairwiseRanking.from_dict(ranking) if ranking is not None else None,
            decision=PointerDecision.from_dict(data["decision"]),
            retirements=tuple(
                RetirementVerdict.from_dict(v) for v in data.get("retirements") or ()
            ),
            scored_arms=tuple(str(a) for a in data.get("scored_arms") or ()),
            active_arms=tuple(str(a) for a in data.get("active_arms") or ()),
            promotable_arms=tuple(str(a) for a in data.get("promotable_arms") or ()),
            config=config,
            not_yet_registered_arms=tuple(
                ArmExclusion.from_dict(d) for d in data.get("not_yet_registered_arms") or ()
            ),
        )


def promotion_statistic(config: ArenaConfig, window: PairedWindow) -> float | None:
    """The quantity ``config`` ranks the pointer on, for one pair's window.

    ONE function, so eligibility and ranking can never be computed from two
    different statistics — the property `_promotable`'s docstring already
    claimed and which a second call site would quietly break.

    ``None`` means "not estimable on this window", which is NOT a loss: an arm
    whose information ratio cannot be formed (one date, or a constant series)
    has not lost the comparison, it has not been in one. Callers must filter
    on None before comparing.
    """
    if config.promote_statistic == STATISTIC_INFORMATION_RATIO:
        return window.ir_diff
    return window.mean_diff


def _leads(config: ArenaConfig, window: PairedWindow) -> bool:
    """Does the challenger lead the incumbent on the statistic that decides?"""
    value = promotion_statistic(config, window)
    return value is not None and value > 0


def _age_eligible(config: ArenaConfig, window: PairedWindow) -> bool:
    """Has this pair been measured together for long enough to promote on?

    The bar is the PAIRED window — the dates on which both arms produced —
    not either arm's own age. An arm registered months ago that has shared
    one week with the incumbent has one week of evidence against it.
    """
    return window.weeks >= config.promote_min_weeks


def _measured_reason(
    config: ArenaConfig, window: PairedWindow, bound: ConfSeqBound
) -> str:
    """Why this measured comparison can or cannot take the pointer.

    A comparison below the age bar keeps status ``measured`` and stays in the
    record: it WAS measured, and a reader must be able to see the lead that
    was not acted on. Dropping it, or restating it as ``unmeasurable``, would
    make a deliberate hold indistinguishable from an absent comparison —
    §7.2's dominant bug class.
    """
    if not _age_eligible(config, window):
        return (
            f"below promote_min_weeks: {window.weeks} paired week(s) < "
            f"{config.promote_min_weeks}; measured but not promotable this cycle"
        )
    if config.promote_evidence == EVIDENCE_POINT:
        if promotion_statistic(config, window) is None:
            return (
                f"{config.promote_statistic} is not estimable on this window "
                f"({window.n_dates} paired date(s)) — measured, but there is no "
                "point estimate to rank on"
            )
        return (
            f"leads the incumbent on point estimate of {config.promote_statistic} "
            "(promote_evidence=point)"
            if _leads(config, window)
            else f"does not lead the incumbent on point estimate of "
            f"{config.promote_statistic} (promote_evidence=point)"
        )
    return (
        "lead supported by the anytime-valid sequence"
        if bound.supported
        else "lead not supported by the anytime-valid sequence"
    )


def _lead_margin(config: ArenaConfig, window: PairedWindow) -> float:
    """The winner's margin, in the units the decision was taken in.

    Reported as the DECIDING statistic rather than always as ``mean_diff``: a
    pointer that moved on an information-ratio lead and whose record states a
    mean-difference margin is a record that explains the wrong decision
    (`principles.md` §2.1 — reconstructable from durable artifacts alone).
    """
    value = promotion_statistic(config, window)
    return window.mean_diff if value is None else value


def _fallback_rank_key(config: ArenaConfig, comparison: Comparison) -> float:
    """The ordering key for the unfit-incumbent fallback branch.

    ``-inf`` for a comparison whose deciding statistic is not estimable, so it
    sorts BELOW every comparison that has one but stays a candidate: this
    branch must yield an arm — the incumbent cannot serve — and dropping the
    unestimable ones could empty it. Using zero instead would rank an
    unmeasured arm above a genuinely losing one.
    """
    if config.promote_evidence == EVIDENCE_POINT:
        value = promotion_statistic(config, comparison.window)
        return float("-inf") if value is None else value
    return comparison.bound.lower if comparison.bound else float("-inf")


def _promotable(
    config: ArenaConfig, comparisons: Sequence[Comparison]
) -> list[tuple[float, Comparison]]:
    """``(rank key, comparison)`` for every challenger allowed to take the pointer.

    Two bars, both hard: the paired-week age (``promote_min_weeks``) and the
    configured evidence mode. The rank key is the quantity the mode decided
    on — the confidence-sequence lower bound under ``anytime_valid``, and
    under ``point`` whatever ``config.promote_statistic`` names, resolved
    through :func:`promotion_statistic` so that ranking and eligibility can
    never be computed from two different statistics.
    """
    eligible = [
        c
        for c in comparisons
        if c.status == "measured" and c.window.measurable and _age_eligible(config, c.window)
    ]
    if config.promote_evidence == EVIDENCE_POINT:
        return [
            (promotion_statistic(config, c.window), c)
            for c in eligible
            if _leads(config, c.window)
        ]
    return [(c.bound.lower, c) for c in eligible if c.bound is not None and c.bound.supported]


def _hold_reason(config: ArenaConfig, comparisons: Sequence[Comparison]) -> str:
    """Why the incumbent held: which bar nothing cleared, and what was below it.

    A hold caused entirely by the age bar reads differently from a hold on
    the evidence — the first clears itself with time and the second may not —
    so the reason names the blocked challengers rather than reporting a bare
    "no challenger qualified".
    """
    measured = [c for c in comparisons if c.status == "measured" and c.window.measurable]
    too_young = [c for c in measured if not _age_eligible(config, c.window)]
    mode = (
        "no age-eligible challenger leads the incumbent on point estimate of "
        f"{config.promote_statistic} (promote_evidence=point)"
        if config.promote_evidence == EVIDENCE_POINT
        else "no age-eligible challenger's lead is supported by the anytime-valid sequence"
    )
    if too_young:
        detail = "; ".join(
            f"{c.challenger}: {c.window.weeks} paired week(s)" for c in sorted(too_young, key=lambda c: c.challenger)
        )
        return (
            f"{mode}. Below promote_min_weeks={config.promote_min_weeks}: {detail}"
        )
    return mode


def decide_pointer(
    config: ArenaConfig,
    as_of: str,
    incumbent: str | None,
    series_by_arm: Mapping[str, ArmSeries],
    preconditions: Mapping[str, Sequence[ServingPrecondition]] | None = None,
) -> PointerDecision:
    """Move the pointer to the best promotable lead, or hold the incumbent.

    Free movement in both directions, no cooldown, no hysteresis margin
    (Brian ruling 2026-08-29). The self-damping property that makes this safe
    is that the comparison window is cumulative, not trailing.

    Two bars stand between a lead and the pointer:

    - **Age.** A challenger whose PAIRED window with the incumbent is shorter
      than ``config.promote_min_weeks`` can never be chosen. Its comparison is
      still measured, still carries its confidence bound, and stays in
      ``comparisons`` with status ``measured`` and a reason naming the
      shortfall — a lead held back is a fact about this cycle, not an absence.
    - **Evidence**, per ``config.promote_evidence``: the anytime-valid
      sequence's lower bound above zero (the policy default), or the largest
      positive mean paired difference (``point``, a per-slot declared delta —
      Brian ruling 2026-09-12, `alpha-engine-config-I10546`). The bound is
      computed and emitted either way.
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
                reason=_measured_reason(config, window, bound),
            )
        )

    # (rank key, comparison) pairs for the challengers this cycle is allowed
    # to promote. Building the tuple here rather than reaching through
    # `c.bound` at the ranking site keeps the bound's presence a fact of the
    # list's construction instead of an invariant a reader has to hold in
    # their head. `_promotable` applies BOTH bars — the paired-week age and
    # the configured evidence mode — so there is exactly one place a
    # challenger can become choosable.
    supported: list[tuple[float, Comparison]] = _promotable(config, comparisons)

    if not incumbent_eligible:
        # The incumbent is not permitted to serve. The pointer MUST move, and
        # a supported lead is not required — continuing to serve a known-unfit
        # arm is never the safer option.
        #
        # The age bar and the evidence bar are PROMOTION bars, so they rank
        # this branch's candidates but cannot empty it: with the incumbent
        # barred from serving, "nothing clears the bar" must still yield an
        # arm, or the slot serves an arm it knows is unfit. The ordered
        # fallback — promotable first, then any measured comparison, then the
        # first eligible arm — is why the age rule is applied here as a
        # PREFERENCE and everywhere else as a veto.
        candidates = supported or [
            (
                _fallback_rank_key(config, c),
                c,
            )
            for c in comparisons
            if c.status == "measured" and c.window.measurable
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
            reason=_hold_reason(config, comparisons),
            comparisons=tuple(comparisons),
            ineligible=ineligible,
        )

    # Rank promotable challengers by the statistic the evidence mode decided
    # on. Under `anytime_valid` that is the confidence sequence's lower bound,
    # which is what makes leads on windows of different lengths comparable
    # without a cross-window aggregation: a short window produces a wide
    # interval and therefore a small lower bound on its own. Under `point` it
    # is the mean paired difference, and `promote_min_weeks` is the only thing
    # standing between a one-week fluke and the pointer — which is why that
    # mode is declared per slot with its minimum, never on its own.
    winner_key, winner = max(supported, key=lambda item: (item[0], item[1].challenger))
    if config.promote_evidence == EVIDENCE_POINT:
        evidence = f"point estimate, mean paired difference {winner_key:.6g} > 0"
    else:
        evidence = f"anytime-valid sequence, lower bound {winner_key:.6g} > 0"
    return PointerDecision(
        slot=config.slot,
        as_of=as_of,
        incumbent=incumbent,
        champion=winner.challenger,
        moved=winner.challenger != incumbent,
        status="decided",
        reason=(
            f"{winner.challenger} leads {incumbent} by "
            f"{_lead_margin(config, winner.window):.6g} "
            f"{config.promote_statistic} over {winner.window.n_dates} paired "
            f"date(s) ({winner.window.weeks} week(s)); "
            f"decided on the {evidence} "
            f"(promote_evidence={config.promote_evidence}, promote_min_weeks={config.promote_min_weeks})"
        ),
        comparisons=tuple(comparisons),
        ineligible=ineligible,
    )


def _promotable_arms(register: ArmRegister, as_of: str | None = None) -> tuple[str, ...]:
    """The active pool with controls excluded — the ``min_active_arms`` floor.

    A control (synthetic benchmark) is a real point of comparison but can
    never take the pointer, so it can never serve as the floor's slack
    (`evaluate_retirements`'s docstring, alpha-engine-config-I9770). This is
    the single computation both `evaluate_retirements` (which checks the
    floor) and `run_cycle` (which emits it as `ArenaCycle.promotable_arms`)
    read, so the "controls excluded" rule lives in exactly one place.

    ``as_of`` makes the pool POINT-IN-TIME: an arm registered after that day
    was not in the arena then, so it is neither part of the floor nor a
    candidate for retirement (`alpha-engine-config-I11084`).
    """
    active = register.active_arms(as_of)
    control_ids = frozenset(a for a in active if register.state(a).record.control)
    return tuple(a for a in active if a not in control_ids)


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

    **Control arms (`alpha-engine-config-I9770`) are excluded from the cap in
    both directions, never as a side effect of §6.2's ranking.** A control is
    a real point of comparison — it stays in ``ranking`` unchanged, exactly
    like any other arm, because §6.2's pairwise aggregation must not be
    slot-aware. Only this function, which owns the cap/grace/floor rule, is
    control-aware:

    - a control never receives a retirement verdict at all (never in
      ``ordered``, never in the returned tuple) — it is not part of the
      real-arm rotation the cap governs;
    - a control's pairwise **win** never counts toward another arm's
      ``pairwise_losses`` — a benchmark beating a real arm says nothing about
      that arm's standing among its actual competitors. ``ranking.standings``
      is control-blind by construction (§6.2's ranking must not be
      slot-aware), so this function subtracts, from each arm's reported
      ``standing.losses``, the count of that arm's losses in
      ``ranking.verdicts`` whose winner is a control. This composes with any
      ``standings`` a caller supplies (including a hand-built
      :class:`~nousergon_lib.arena.ranking.ArmStanding` in a test) rather than
      requiring ``standings`` and ``verdicts`` to agree;
    - ``min_active_arms`` is checked against the **real-arm** pool size
      (``active`` below already excludes controls), not
      ``register.active_arms()`` — a deliberate choice: the floor exists so a
      comparison always has slack (§6.1), and a control cannot serve as that
      slack since it can never lose its exemption or take the pointer.
      ``register.active_arms()`` itself is unchanged and still includes
      controls, since `run_cycle` and the emitted ``ArenaCycle.active_arms``
      correctly report every live arm, controls included.
    """
    control_ids = frozenset(
        a for a in register.active_arms(as_of) if register.state(a).record.control
    )
    active = list(_promotable_arms(register, as_of))
    control_losses: dict[str, int] = dict.fromkeys(active, 0)
    for verdict in ranking.verdicts:
        if verdict.winner in control_ids and verdict.loser in control_losses:
            control_losses[verdict.loser] += 1

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
        losses = (standing.losses if standing else 0) - control_losses.get(arm, 0)

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

    **The cycle is POINT-IN-TIME.** It contains exactly the arms that were in
    the arena on ``as_of``: an arm whose ``registered`` row was filed later,
    or whose recipe declares a later ``created_date``, is EXCLUDED from the
    cycle rather than aged. It is not scored, not ranked, not eligible for the
    pointer and receives no retirement verdict, and it is named in
    :attr:`ArenaCycle.not_yet_registered_arms` with the dates that decide it.

    Neither of the two shortcuts is taken, and both are forbidden
    (`alpha-engine-config-I11084`):
    :func:`~nousergon_lib.arena.window.elapsed_weeks` still RAISES on a
    negative age — that refusal is the only reason this defect was ever found
    — and an absent arm's age is never clamped to zero, which would hand an
    arm that did not exist a standing and a retirement verdict on a day it
    was absent from (the `-I10948` defect re-entering through the writer).

    A series supplied for a not-yet-registered arm is DROPPED for the same
    reason, not scored: a shadow backfilled for a recipe that did not exist
    on ``as_of`` is a look-ahead on that day whatever produced it. The drop is
    reported in the same field, so it is never silent.
    """
    absent = {
        arm: register.state(arm) for arm in register.not_yet_registered(as_of)
    }
    not_yet_registered_arms = tuple(
        ArmExclusion(
            arm_id=arm,
            reason=state.absence_reason(as_of),
            created_date=state.record.created_date,
            filed_date=state.filed_date,
        )
        for arm, state in sorted(absent.items())
    )
    series_by_arm = {
        arm: series for arm, series in series_by_arm.items() if arm not in absent
    }

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
        assert_training_integrity(training, register.active_arms(as_of))

    ladders = tuple(
        build_ladder(series_by_arm[arm], as_of, max_weeks=config.max_ladder_weeks)
        for arm in sorted(series_by_arm)
    )

    active = register.active_arms(as_of)
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
        promotable_arms=_promotable_arms(register, as_of),
        config=config,
        not_yet_registered_arms=not_yet_registered_arms,
    )
