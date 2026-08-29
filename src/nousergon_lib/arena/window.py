"""Longest-common-window pairing — the ONLY basis on which two arms are compared.

**Why this exists.** ``champion-challenger-policy.md`` §4 has required
"same cohort dates … the intersection is reported alongside the metric"
since 2026-07-28, and the requirement was being violated in production.
Measured 2026-08-29: an incumbent scored over **2** dates successfully
defended against a challenger rejected for having **4** — two arms judged
on bases 3.6x apart, on windows that barely overlapped. A comparison
between an arm's good month and another's bad quarter is not a comparison,
and nothing in the fleet was enforcing that.

This module makes the rule mechanical: two arms are compared ONLY over the
set of dates on which **both** produced output, paired per date. Every
statistic downstream (:mod:`nousergon_lib.arena.confseq`,
:mod:`nousergon_lib.arena.ranking`) consumes a :class:`PairedWindow` and
therefore cannot see un-paired data even by accident.

**Failure is loud.** An empty or too-short intersection produces a
:class:`PairedWindow` with ``n_dates`` below the caller's floor and an
``unmeasurable_reason`` string. It never renders as a pass, a tie, or a
zero — §7.2's dominant bug class is "a well-formed artifact containing
nothing", and an empty comparison is exactly that shape.

**A miss is data.** :class:`ArmSeries` carries ``misses`` — dates on which
the arm was expected to produce output and did not — separately from
``scores``. §3 requires that silent absence and a genuine zero never render
identically, so absence gets its own field rather than being represented as
a zero score or by omission from an expectation set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "ArmSeries",
    "elapsed_weeks",
    "PairedWindow",
    "pair_on_common_window",
    "span_weeks",
]

_DAYS_PER_WEEK = 7


def _as_date(value: str):
    from datetime import date

    parts = value.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"arena window dates must be ISO YYYY-MM-DD; got {value!r}"
        )
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def span_weeks(start_date: str, end_date: str) -> int:
    """Inclusive calendar span of ``[start_date, end_date]`` in whole weeks.

    A single date is one week (a one-observation window is not a zero-week
    window — rendering it as zero would make an arm's first cycle
    indistinguishable from an arm with no history at all).
    """
    delta = (_as_date(end_date) - _as_date(start_date)).days
    if delta < 0:
        raise ValueError(
            f"end_date {end_date} precedes start_date {start_date}"
        )
    return delta // _DAYS_PER_WEEK + 1


def elapsed_weeks(start_date: str, end_date: str) -> int:
    """FULL weeks elapsed between two dates. 27 days is 3 weeks, not 4.

    Distinct from :func:`span_weeks`, which is the INCLUSIVE span of a window
    and is what a ladder rung reports. Ages and grace periods use this one.
    """
    delta = (_as_date(end_date) - _as_date(start_date)).days
    if delta < 0:
        raise ValueError(
            f"end_date {end_date} precedes start_date {start_date}"
        )
    return delta // _DAYS_PER_WEEK


@dataclass(frozen=True)
class ArmSeries:
    """One arm's per-date scores, already expressed against the slot benchmark.

    ``scores`` maps ISO date -> the arm's score for that date. The score is
    benchmark-relative by construction: the engine never applies a benchmark
    itself, because the correct benchmark is a per-slot fact (see
    :class:`nousergon_lib.arena.engine.ArenaConfig` and its refusal to let a
    selection-stage slot grade against SPY).

    ``misses`` holds dates on which this arm was expected to produce and did
    not. They are excluded from every window — a miss is not a zero — but
    they are carried so the artifact can report them (§3).
    """

    arm_id: str
    scores: Mapping[str, float]
    misses: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("ArmSeries.arm_id must be non-empty")
        overlap = set(self.scores) & set(self.misses)
        if overlap:
            raise ValueError(
                f"arm {self.arm_id}: dates recorded as BOTH scored and missed: {sorted(overlap)}"
            )
        for key, value in self.scores.items():
            _as_date(key)
            if value != value:  # NaN
                raise ValueError(
                    f"arm {self.arm_id}: NaN score on {key}; a missing score is a MISS, "
                    "not a NaN (champion-challenger-policy.md §3)"
                )

    @property
    def dates(self) -> frozenset[str]:
        return frozenset(self.scores)

    @property
    def n_dates(self) -> int:
        return len(self.scores)

    @property
    def first_date(self) -> str | None:
        return min(self.scores) if self.scores else None

    @property
    def last_date(self) -> str | None:
        return max(self.scores) if self.scores else None


@dataclass(frozen=True)
class PairedWindow:
    """The per-date paired difference ``arm_a - arm_b`` over their common dates.

    ``unmeasurable_reason`` is ``None`` exactly when the window is usable.
    Callers MUST check :attr:`measurable`; there is deliberately no
    "empty means tie" path.
    """

    arm_a: str
    arm_b: str
    dates: tuple[str, ...]
    diffs: tuple[float, ...]
    scores_a: tuple[float, ...]
    scores_b: tuple[float, ...]
    unmeasurable_reason: str | None = None

    @property
    def measurable(self) -> bool:
        return self.unmeasurable_reason is None

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def start_date(self) -> str | None:
        return self.dates[0] if self.dates else None

    @property
    def end_date(self) -> str | None:
        return self.dates[-1] if self.dates else None

    @property
    def weeks(self) -> int:
        if not self.dates:
            return 0
        return span_weeks(self.dates[0], self.dates[-1])

    @property
    def mean_diff(self) -> float:
        if not self.diffs:
            raise ValueError(
                f"mean_diff on an unmeasurable window ({self.arm_a} vs {self.arm_b}): {self.unmeasurable_reason}"
            )
        return sum(self.diffs) / len(self.diffs)

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n_dates": self.n_dates,
            "weeks": self.weeks,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mean_diff": self.mean_diff if self.measurable else None,
            "unmeasurable_reason": self.unmeasurable_reason,
        }


def pair_on_common_window(
    series_a: ArmSeries,
    series_b: ArmSeries,
    min_dates: int = 1,
) -> PairedWindow:
    """Pair two arms per date over the LONGEST window on which both produced.

    This is the decision statistic's only input. The window is the full
    intersection of both arms' scored dates — "longest" is not a search over
    candidate windows, it is the single deterministic window the two arms
    share, which is why the horizon ladder is not a multiple-comparison
    problem (see :mod:`nousergon_lib.arena.ladder`).

    Returns an unmeasurable :class:`PairedWindow` — never raises — when the
    intersection is smaller than ``min_dates``, carrying the reason so the
    caller can record ``unmeasurable`` per §7.2.
    """
    if min_dates < 1:
        raise ValueError(f"min_dates must be >= 1; got {min_dates}")
    if series_a.arm_id == series_b.arm_id:
        raise ValueError(
            f"cannot pair arm {series_a.arm_id} against itself — an arm compared to itself is "
            "the promotion-gate defect of 2026-08-28 "
            "(champion-challenger-policy.md §4)"
        )

    common = sorted(series_a.dates & series_b.dates)
    if len(common) < min_dates:
        return PairedWindow(
            arm_a=series_a.arm_id,
            arm_b=series_b.arm_id,
            dates=(),
            diffs=(),
            scores_a=(),
            scores_b=(),
            unmeasurable_reason=(
                f"common_window_too_short: {len(common)} paired date(s), need {min_dates} "
                f"({series_a.arm_id}: {series_a.n_dates} dates, {series_b.arm_id}: {series_b.n_dates} dates)"
            ),
        )

    scores_a = tuple(float(series_a.scores[d]) for d in common)
    scores_b = tuple(float(series_b.scores[d]) for d in common)
    return PairedWindow(
        arm_a=series_a.arm_id,
        arm_b=series_b.arm_id,
        dates=tuple(common),
        diffs=tuple(a - b for a, b in zip(scores_a, scores_b)),
        scores_a=scores_a,
        scores_b=scores_b,
        unmeasurable_reason=None,
    )
