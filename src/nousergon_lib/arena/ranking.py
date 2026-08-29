"""Pairwise-wins ranking — how a 4-week arm and a 54-week arm are ranked together.

**The problem.** Brian's cap rule (2026-08-29) needs a "top 5", and the pool
legitimately mixes arms of wildly different ages. Two wrong ways to build
that ranking, both of which reintroduce the defect this engine exists to
remove:

- **Rank on each arm's own longest score.** That compares a 4-week number
  against a 54-week number — the incomparable-window defect, removed from
  the promotion path, walking back in through retirement.
- **Rank the whole pool on one common window.** The pool's common window is
  the NEWEST arm's history, so a single fresh arm truncates every established
  arm's record to a few dates and throws the evidence away.

**The rule built here: Condorcet-style pairwise aggregation.** Every pair is
compared on THEIR OWN longest common window, paired per date
(:func:`nousergon_lib.arena.window.pair_on_common_window`). Each comparison
therefore rests on valid common support, and no cross-window comparison ever
happens. An arm's standing is its count of pairwise **losses**: "not in the
top ``cap``" means at least ``cap`` arms beat it head to head. A young arm is
only ever judged against the slice of each older arm's history it actually
overlaps.

**Why loss-count and not rank position.** Loss-count needs no aggregation
across incomparable windows at all — it is a pure count of window-honest
verdicts. It is also self-consistently a cap: under a strict total order
exactly ``cap`` arms can have fewer than ``cap`` losses, so the rule keeps
``cap`` arms without ever having to sort them on a common scale.

**Condorcet cycles are possible and are resolved conservatively.** A cycle
(A beats B beats C beats A) can leave more than ``cap`` arms with fewer than
``cap`` losses. The pool then temporarily exceeds the cap and nobody extra is
retired. That is the deliberate choice: retirement destroys optionality (§6)
and the counterfactual is unrecoverable, so an ambiguous ranking resolves
toward keeping an arm. The cycle itself is reported
(:attr:`PairwiseRanking.cycles_present`) rather than silently smoothed away.

The presentation ordering (:attr:`PairwiseRanking.ordering`) uses Copeland
score, then average margin, then age, then arm id — deterministic, and used
for REPORTING and for bootstrap selection only. Retirement never consults it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .confseq import ConfSeqBound, confidence_sequence
from .window import ArmSeries, PairedWindow, pair_on_common_window

__all__ = [
    "PairVerdict",
    "ArmStanding",
    "PairwiseRanking",
    "rank_pairwise",
    "EVIDENCE_POINT",
    "EVIDENCE_ANYTIME_VALID",
]

#: A pairwise loss is a point-estimate loss on the pair's common window.
EVIDENCE_POINT = "point"

#: A pairwise loss counts only when the anytime-valid sequence supports it.
EVIDENCE_ANYTIME_VALID = "anytime_valid"

_EVIDENCE_MODES = (EVIDENCE_POINT, EVIDENCE_ANYTIME_VALID)


@dataclass(frozen=True)
class PairVerdict:
    """One head-to-head verdict on the pair's own longest common window."""

    arm_a: str
    arm_b: str
    window: PairedWindow
    bound: ConfSeqBound | None
    winner: str | None
    loser: str | None
    reason: str

    @property
    def measurable(self) -> bool:
        return self.window.measurable

    def to_dict(self) -> dict[str, Any]:
        payload = self.window.to_dict()
        payload["winner"] = self.winner
        payload["loser"] = self.loser
        payload["reason"] = self.reason
        payload["confidence_sequence"] = self.bound.to_dict() if self.bound else None
        return payload


@dataclass(frozen=True)
class ArmStanding:
    """One arm's aggregate standing over its pairwise verdicts."""

    arm_id: str
    wins: int
    losses: int
    ties: int
    unmeasurable: int
    mean_margin: float
    created_date: str

    @property
    def copeland(self) -> int:
        return self.wins - self.losses

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "unmeasurable": self.unmeasurable,
            "copeland": self.copeland,
            "mean_margin": self.mean_margin,
            "created_date": self.created_date,
        }


@dataclass(frozen=True)
class PairwiseRanking:
    """Every pairwise verdict plus the per-arm standings derived from them."""

    as_of: str
    evidence_mode: str
    verdicts: tuple[PairVerdict, ...]
    standings: Mapping[str, ArmStanding]
    ordering: tuple[str, ...]
    cycles_present: bool

    def losses(self, arm_id: str) -> int:
        return self.standings[arm_id].losses

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "evidence_mode": self.evidence_mode,
            "cycles_present": self.cycles_present,
            "ordering": list(self.ordering),
            "standings": [self.standings[a].to_dict() for a in self.ordering],
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def _verdict(
    series_a: ArmSeries,
    series_b: ArmSeries,
    evidence_mode: str,
    alpha: float,
    clip: float | None,
    variance_mode: str,
    opt_n: int,
    min_dates: int,
) -> PairVerdict:
    window = pair_on_common_window(series_a, series_b, min_dates=min_dates)
    if not window.measurable:
        return PairVerdict(
            arm_a=series_a.arm_id,
            arm_b=series_b.arm_id,
            window=window,
            bound=None,
            winner=None,
            loser=None,
            reason=f"unmeasurable: {window.unmeasurable_reason}",
        )

    bound = confidence_sequence(
        window.diffs,
        alpha=alpha,
        clip=clip,
        variance_mode=variance_mode,
        opt_n=opt_n,
    )

    if evidence_mode == EVIDENCE_ANYTIME_VALID:
        if bound.supported:
            winner, loser, reason = series_a.arm_id, series_b.arm_id, "anytime_valid_lead"
        elif bound.supported_negative:
            winner, loser, reason = series_b.arm_id, series_a.arm_id, "anytime_valid_lead"
        else:
            winner = loser = None
            reason = "tie: lead not supported by the confidence sequence"
    else:
        mean_diff = window.mean_diff
        if mean_diff > 0.0:
            winner, loser, reason = series_a.arm_id, series_b.arm_id, "point_estimate_lead"
        elif mean_diff < 0.0:
            winner, loser, reason = series_b.arm_id, series_a.arm_id, "point_estimate_lead"
        else:
            winner = loser = None
            reason = "tie: exactly equal means on the common window"

    return PairVerdict(
        arm_a=series_a.arm_id,
        arm_b=series_b.arm_id,
        window=window,
        bound=bound,
        winner=winner,
        loser=loser,
        reason=reason,
    )


def rank_pairwise(
    series_by_arm: Mapping[str, ArmSeries],
    created_dates: Mapping[str, str],
    as_of: str,
    evidence_mode: str = EVIDENCE_POINT,
    alpha: float = 0.05,
    clip: float | None = None,
    variance_mode: str = "declared",
    opt_n: int = 26,
    min_dates: int = 1,
) -> PairwiseRanking:
    """Compare every pair on its own longest common window and aggregate.

    ``created_dates`` supplies the age tie-break and is required for every
    arm — an arm whose age is unknown cannot be ranked deterministically, and
    a default would silently favour or condemn it.
    """
    if evidence_mode not in _EVIDENCE_MODES:
        raise ValueError(
            f"evidence_mode must be one of {_EVIDENCE_MODES}; got {evidence_mode!r}"
        )
    missing = sorted(set(series_by_arm) - set(created_dates))
    if missing:
        raise ValueError(
            f"no created_date for arm(s) {missing}; every ranked arm must be in the "
            "register (champion-challenger-policy.md §3)"
        )

    arms = sorted(series_by_arm)
    verdicts: list[PairVerdict] = []
    wins = dict.fromkeys(arms, 0)
    losses = dict.fromkeys(arms, 0)
    ties = dict.fromkeys(arms, 0)
    unmeasurable = dict.fromkeys(arms, 0)
    margin_sum = dict.fromkeys(arms, 0.0)
    margin_n = dict.fromkeys(arms, 0)
    beats: dict[str, set] = {a: set() for a in arms}

    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1 :]:
            verdict = _verdict(
                series_by_arm[arm_a],
                series_by_arm[arm_b],
                evidence_mode=evidence_mode,
                alpha=alpha,
                clip=clip,
                variance_mode=variance_mode,
                opt_n=opt_n,
                min_dates=min_dates,
            )
            verdicts.append(verdict)
            if not verdict.measurable:
                unmeasurable[arm_a] += 1
                unmeasurable[arm_b] += 1
                continue
            margin = verdict.window.mean_diff
            margin_sum[arm_a] += margin
            margin_sum[arm_b] -= margin
            margin_n[arm_a] += 1
            margin_n[arm_b] += 1
            won, lost = verdict.winner, verdict.loser
            if won is None or lost is None:
                ties[arm_a] += 1
                ties[arm_b] += 1
            else:
                wins[won] += 1
                losses[lost] += 1
                beats[won].add(lost)

    standings = {
        arm: ArmStanding(
            arm_id=arm,
            wins=wins[arm],
            losses=losses[arm],
            ties=ties[arm],
            unmeasurable=unmeasurable[arm],
            mean_margin=(margin_sum[arm] / margin_n[arm]) if margin_n[arm] else 0.0,
            created_date=created_dates[arm],
        )
        for arm in arms
    }

    ordering = tuple(
        sorted(
            arms,
            key=lambda a: (
                -standings[a].copeland,
                -standings[a].mean_margin,
                standings[a].created_date,
                a,
            ),
        )
    )

    return PairwiseRanking(
        as_of=as_of,
        evidence_mode=evidence_mode,
        verdicts=tuple(verdicts),
        standings=standings,
        ordering=ordering,
        cycles_present=_has_cycle(beats),
    )


def _has_cycle(beats: Mapping[str, Any]) -> bool:
    """True when the beats-relation contains a directed cycle (Condorcet cycle)."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(beats, WHITE)

    def visit(node: str) -> bool:
        colour[node] = GREY
        for nxt in beats[node]:
            if colour[nxt] == GREY:
                return True
            if colour[nxt] == WHITE and visit(nxt):
                return True
        colour[node] = BLACK
        return False

    for node in beats:
        if colour[node] == WHITE and visit(node):
            return True
    return False
