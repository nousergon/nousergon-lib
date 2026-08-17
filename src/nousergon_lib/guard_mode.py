"""Observe-then-enforce staging for a guard on a scheduled production path.

`sf-pipeline-policy.md` §7a (clause `SFP-7a-new-guard-observes-first`): a check
newly added to a scheduled pipeline path, whose verdict can halt a stage or
fail a run, does not enforce on its first production execution. It observes for
a declared number of cycles — predicate live, consequence not — carries its
promotion criterion in its own module, and is loud while observing.

This module is the spelling of that rule, so a new guard gets the staging
without its author hand-rolling it.

**Why a helper and not a convention.** The rule exists because of the
2026-08-17 EOD gap: a correct zero-variance guard merged at 18:59 UTC, ran for
the first time anywhere on the 20:00 UTC production EOD, and cost the day its
ArcticDB append and its reconcile — not because the predicate was wrong, but
because its blast radius had never been observed. The fix in that repo added a
``zero_variance_fatal`` boolean to one function. That works, and it is exactly
the shape that does not survive being reinvented: the next author writes a
differently-named flag, defaults it the other way, or forgets the promotion
criterion, and the rule quietly stops being followed while every individual PR
looks reasonable.

**The promotion criterion is not optional.** A guard parked in observe mode
forever is the same defect one direction over — the predicate runs, the defect
persists, and nothing ever stops. So ``GuardStaging`` cannot be constructed
without one, and ``describe()`` renders it for the tracked item.

Usage::

    from nousergon_lib.guard_mode import GuardStaging, GuardMode

    _ZERO_VARIANCE = GuardStaging(
        name="feature_store_zero_variance",
        mode=GuardMode.OBSERVE,
        promotion_criterion="enforce from 2026-09-01, or after 10 clean daily runs",
        tracked_issue="alpha-engine-config-I7539",
    )

    offenders = find_zero_variance_columns(df, cols)
    if offenders:
        _ZERO_VARIANCE.verdict(
            log,
            f"zero-variance column(s): {offenders}",
            raise_with=RuntimeError,
        )

In OBSERVE the verdict is logged at ERROR and returns; in ENFORCE the same
message is logged and then raised. Nothing else differs, which is the point:
the observation and the enforcement report identically, so the surface a person
reads does not quietly change shape at promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

__all__ = ["GuardMode", "GuardStaging"]


class GuardMode(str, Enum):
    """Whether a guard's verdict has a consequence yet."""

    #: Predicate runs, verdict is logged at ERROR, nothing raises.
    OBSERVE = "observe"
    #: Predicate runs, verdict is logged at ERROR, then raises.
    ENFORCE = "enforce"


@dataclass(frozen=True)
class GuardStaging:
    """One guard's staging state, its promotion criterion, and its tracker.

    Args:
        name: Stable identifier for the guard, used in every log line so the
            observe-mode verdicts are greppable and countable before promotion.
        mode: :class:`GuardMode`.
        promotion_criterion: How this guard leaves OBSERVE — a count of clean
            cycles, a date, or a named condition. Required in BOTH modes: in
            OBSERVE it is the exit condition, and in ENFORCE it is the record of
            why the guard was promoted, which is what a later reader needs when
            deciding whether a regression means demoting it again.
        tracked_issue: The item carrying the `Re-exam:` line. Required for the
            same reason the criterion is: an intention with no tracked home is
            not a plan.
    """

    name: str
    mode: GuardMode
    promotion_criterion: str
    tracked_issue: str

    def __post_init__(self) -> None:
        # Fail at construction, not at first verdict. A guard whose staging is
        # malformed must not reach production and discover it on the one run
        # where the predicate fires.
        for field in ("name", "promotion_criterion", "tracked_issue"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GuardStaging.{field} is required and must be non-empty — "
                    "sf-pipeline-policy.md §7a. A guard with no promotion "
                    "criterion or no tracked item is one that stays in observe "
                    "mode forever, which is the failure this rule exists to "
                    "prevent as much as the one it is named for."
                )
        if not isinstance(self.mode, GuardMode):
            raise ValueError(f"GuardStaging.mode must be a GuardMode, got {self.mode!r}")

    @property
    def enforcing(self) -> bool:
        return self.mode is GuardMode.ENFORCE

    def describe(self) -> str:
        """One line for a log header, an alert body, or a tracked-item comment."""
        return (
            f"guard={self.name} mode={self.mode.value} "
            f"promotion={self.promotion_criterion!r} tracked={self.tracked_issue}"
        )

    def verdict(
        self,
        log: logging.Logger,
        message: str,
        *,
        raise_with: Callable[[str], BaseException] = RuntimeError,
    ) -> None:
        """Report a FAILING verdict, and raise it only when enforcing.

        Called only when the predicate has already failed — this class does not
        evaluate anything, it decides what a failure is allowed to do.

        The message is logged at ERROR in BOTH modes. That is deliberate and is
        the third obligation of §7a: a verdict nobody reads is a suppression,
        not an observation (`principles.md` §2.7). An observe-mode guard that
        logged at INFO, or not at all, would be indistinguishable from a guard
        that never fired, and the observation period would produce no evidence
        for the promotion decision it exists to inform.
        """
        log.error(
            "[%s] %s — %s",
            self.mode.value.upper(),
            message,
            self.describe(),
        )
        if self.enforcing:
            raise raise_with(f"{message} ({self.describe()})")
