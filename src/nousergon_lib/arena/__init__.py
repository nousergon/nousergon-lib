"""The champion/challenger arena — one scoring engine for every swappable slot.

Normative source: ``nous-ergon-ops/policies/champion-challenger-policy.md``.
This package is the fleet's single implementation of that policy's decision
machinery, so the four slots — universe cut, selection producer, model (M),
strategy (S) — consume one set of rules rather than four drifting copies
(`shared-code-policy.md`).

**What it does, in the order a cycle runs:**

1. :mod:`~nousergon_lib.arena.arms` — the append-only, immutable arm
   register. An arm is a RECIPE — features, hyperparameters, training-window
   rule and refit cadence — and its id encodes its own spec hash, so a
   changed recipe is a NEW arm and cannot inherit a record. A scheduled
   refit is the arm doing its job: it is recorded, and it neither changes
   the id nor resets the series (Brian rulings 2026-08-29).
2. :mod:`~nousergon_lib.arena.ladder` — the per-arm score ladder: 1, 2, 3,
   … N-week scores, every rung recomputed every cycle. The track record.
3. :mod:`~nousergon_lib.arena.window` — longest-common-window pairing. Two
   arms are compared ONLY on dates where both produced, paired per date.
4. :mod:`~nousergon_lib.arena.confseq` — the anytime-valid confidence
   sequence that replaces minimum-week and ``thin_evidence`` gates entirely.
5. :mod:`~nousergon_lib.arena.ranking` — Condorcet-style pairwise-wins
   ranking, which is how arms of very different ages are ranked without ever
   comparing incomparable windows.
6. :mod:`~nousergon_lib.arena.engine` — the pointer decision, the
   cap-with-grace retirement rule, and the durable cycle artifact.

Pure compute: no I/O, no logging, no third-party dependency. Import it from a
Lambda without pulling numpy.

**Improper training fails the task.** :func:`~nousergon_lib.arena.engine.run_cycle`
raises :class:`~nousergon_lib.arena.engine.TrainingIntegrityError` when any
active arm reports an unsound fit, or reports nothing. It is never recorded
as a miss — ``ArmSeries.misses`` means "this arm legitimately had nothing to
say", which must never render the same as "this arm was trained on broken
inputs".

**Three things this package deliberately does not do.** It does not apply a
benchmark (the correct benchmark is a per-slot fact, and the config refuses
SPY for a selection-stage slot). It does not evaluate serving preconditions
(the M-slot behavioural veto and input completeness are slot facts, passed
in as results). And it does not gate admission — the cap of 5 is a
retirement criterion with a grace period, never a bar on creating an arm.
"""

from __future__ import annotations

from .arms import (
    EVENT_REFIT,
    EVENT_REGISTERED,
    EVENT_RETIRED,
    ArmEvent,
    ArmRecord,
    ArmRegister,
    ArmState,
    ImmutableArmError,
    derive_arm_id,
)
from .confseq import ConfSeqBound, confidence_sequence
from .engine import (
    ARENA_CYCLE_SCHEMA_VERSION,
    ArenaConfig,
    ArenaCycle,
    Comparison,
    PointerDecision,
    RetirementVerdict,
    ServingPrecondition,
    TrainingIntegrityError,
    TrainingStatus,
    assert_training_integrity,
    decide_pointer,
    evaluate_retirements,
    run_cycle,
)
from .ladder import LadderRung, ScoreLadder, build_ladder
from .ranking import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    ArmStanding,
    PairVerdict,
    PairwiseRanking,
    rank_pairwise,
)
from .selection import (
    DEFAULT_SCORE_CEILING,
    DEFAULT_SCORE_FLOOR,
    rank_by_alpha,
    rank_to_score,
)
from .window import ArmSeries, PairedWindow, pair_on_common_window

__all__ = [
    "ARENA_CYCLE_SCHEMA_VERSION",
    "EVENT_REFIT",
    "EVENT_REGISTERED",
    "EVENT_RETIRED",
    "TrainingIntegrityError",
    "TrainingStatus",
    "assert_training_integrity",
    "ArenaConfig",
    "ArenaCycle",
    "ArmEvent",
    "ArmRecord",
    "ArmRegister",
    "ArmSeries",
    "ArmStanding",
    "ArmState",
    "Comparison",
    "ConfSeqBound",
    "DEFAULT_SCORE_CEILING",
    "DEFAULT_SCORE_FLOOR",
    "EVIDENCE_ANYTIME_VALID",
    "EVIDENCE_POINT",
    "ImmutableArmError",
    "LadderRung",
    "PairVerdict",
    "PairedWindow",
    "PairwiseRanking",
    "PointerDecision",
    "RetirementVerdict",
    "ScoreLadder",
    "ServingPrecondition",
    "build_ladder",
    "confidence_sequence",
    "decide_pointer",
    "derive_arm_id",
    "evaluate_retirements",
    "pair_on_common_window",
    "rank_by_alpha",
    "rank_pairwise",
    "rank_to_score",
    "run_cycle",
]
