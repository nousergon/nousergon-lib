"""The per-arm score ladder — 1, 2, 3, … N-week scores, recomputed every cycle.

**Brian's ruling, 2026-08-29, verbatim:**

    "we should have a new score each week. an arm should win if the longest
    running score possible outperforms the champion. so each arm should have
    1, 2, 3, 4, 52, 53, 54 etc week scores, and the longest running score for
    each arm is the score that is compared."

The ladder IS the track record: rung ``h`` is the arm's mean score over the
trailing ``h`` calendar weeks ending at ``as_of``, for every ``h`` from 1 up
to the rung that covers the arm's whole history. Every rung is emitted every
cycle, so the shape of an arm's performance across horizons is a durable
artifact rather than a number somebody recomputed on demand.

**The ladder is a REPORT; it is not where the decision is taken.** The
decision uses the longest window two arms actually share
(:mod:`nousergon_lib.arena.window`). "Longest" is a **deterministic,
pre-registered rule** — the single window the pair shares — and therefore
NOT a multiple-comparison problem: nothing selects among the rungs.

**This is the sentence that must not be softened.** Anyone who later
"improves" this by picking whichever rung shows the biggest lead converts a
pre-registered statistic into a search over 52 horizons, which IS mining and
which the DSR/PBO battery of §5.1 is not measuring — those deflate for
strategy trials, not for a horizon search taken inside a single trial.
:func:`best_rung` deliberately does not exist. If you are here to add it,
the answer is no; take it to Brian as a policy amendment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta
from typing import Any

from .window import ArmSeries, span_weeks

__all__ = ["LadderRung", "ScoreLadder", "build_ladder"]


def _parse(value: str) -> _date:
    parts = value.split("-")
    return _date(int(parts[0]), int(parts[1]), int(parts[2]))


@dataclass(frozen=True)
class LadderRung:
    """The arm's mean score over the trailing ``weeks`` weeks ending at ``as_of``."""

    weeks: int
    n_dates: int
    n_misses: int
    start_date: str
    end_date: str
    mean_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "weeks": self.weeks,
            "n_dates": self.n_dates,
            "n_misses": self.n_misses,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mean_score": self.mean_score,
        }


@dataclass(frozen=True)
class ScoreLadder:
    """Every rung for one arm, plus the arm's total span."""

    arm_id: str
    as_of: str
    rungs: tuple[LadderRung, ...]
    total_weeks: int
    total_dates: int
    total_misses: int

    @property
    def longest(self) -> LadderRung | None:
        """The full-history rung — the one Brian's ruling calls "the score"."""
        return self.rungs[-1] if self.rungs else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "as_of": self.as_of,
            "total_weeks": self.total_weeks,
            "total_dates": self.total_dates,
            "total_misses": self.total_misses,
            "rungs": [rung.to_dict() for rung in self.rungs],
        }


def build_ladder(series: ArmSeries, as_of: str, max_weeks: int | None = None) -> ScoreLadder:
    """Build every rung 1..N for ``series`` as of ``as_of``.

    ``max_weeks`` caps the emitted ladder for artifact size only; it never
    changes which window a decision uses. An arm with no scored dates
    produces an empty ladder — that is a legible "no history", distinct from
    a ladder of zeros.
    """
    as_of_date = _parse(as_of)
    scored = {d: v for d, v in series.scores.items() if _parse(d) <= as_of_date}
    if not scored:
        return ScoreLadder(
            arm_id=series.arm_id,
            as_of=as_of,
            rungs=(),
            total_weeks=0,
            total_dates=0,
            total_misses=len([d for d in series.misses if _parse(d) <= as_of_date]),
        )

    earliest = min(scored)
    total_weeks = span_weeks(earliest, as_of)
    horizon = total_weeks if max_weeks is None else min(total_weeks, max_weeks)

    misses = [d for d in series.misses if _parse(d) <= as_of_date]

    rungs: list[LadderRung] = []
    for weeks in range(1, horizon + 1):
        window_start = as_of_date - timedelta(days=weeks * 7 - 1)
        in_window = {d: v for d, v in scored.items() if _parse(d) >= window_start}
        if not in_window:
            # A trailing window with no observations is a genuine gap in the
            # arm's output, not a zero. Recording it as a rung with n_dates=0
            # and mean_score None would break the artifact's numeric contract,
            # so the rung is omitted and the absence is visible as a skipped
            # `weeks` value in the emitted ladder.
            continue
        rung_dates = sorted(in_window)
        rungs.append(
            LadderRung(
                weeks=weeks,
                n_dates=len(rung_dates),
                n_misses=len([d for d in misses if _parse(d) >= window_start]),
                start_date=rung_dates[0],
                end_date=rung_dates[-1],
                mean_score=sum(in_window.values()) / len(in_window),
            )
        )

    return ScoreLadder(
        arm_id=series.arm_id,
        as_of=as_of,
        rungs=tuple(rungs),
        total_weeks=total_weeks,
        total_dates=len(scored),
        total_misses=len(misses),
    )
