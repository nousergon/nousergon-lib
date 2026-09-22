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
from typing import Any

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


def _normalize_lineage(
    arm_id: str, lineage: Mapping[Any, Any] | None
) -> dict[str, tuple[str, ...]]:
    """Sort and de-duplicate each dimension's values; refuse an empty claim.

    Deterministic ordering is what makes two runs of the same cycle produce
    byte-identical artifacts. A dimension mapped to no values is refused
    rather than emitted: "this arm declared `feature_version` and it took no
    value" is a well-formed record of nothing, which is the §7.2 shape — a
    slot with no provenance to report omits the dimension.
    """
    if not lineage:
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for dimension, values in lineage.items():
        if not isinstance(dimension, str) or not dimension:
            raise ValueError(
                f"arm {arm_id}: lineage dimension names must be non-empty strings; "
                f"got {dimension!r}"
            )
        if isinstance(values, str):
            raise ValueError(
                f"arm {arm_id}: lineage[{dimension!r}] must be a sequence of values, not a "
                "bare string — a single version is a one-element sequence, and accepting "
                "the string would silently record it as its own characters"
            )
        collected = sorted({str(v) for v in values})
        if not collected or any(not v for v in collected):
            raise ValueError(
                f"arm {arm_id}: lineage[{dimension!r}] must carry at least one non-empty "
                "value; a dimension with nothing to report is omitted, never declared empty"
            )
        normalized[dimension] = tuple(collected)
    return normalized


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

    ``lineage`` is the **slot's own** provenance for this series, and the
    engine never reads it. It maps a dimension name the slot chooses — the
    M slot uses ``feature_version`` — to the DISTINCT values that dimension
    took across the dates in ``scores``. One value means the whole series
    rests on one upstream version; two or more means it spans a change, and
    that is the fact a verdict surface has to carry (see
    :class:`nousergon_lib.arena.ladder.ScoreLadder`, which emits it).

    **Why it is an opaque map and not a ``feature_version`` field.** This
    engine scores four slot kinds and only one of them reads a feature layer
    at all. A ``feature_version`` on a slot-agnostic contract would make the
    engine know about a specific consumer, which `principles.md` §2.8
    forbids: address the capability class — "a slot has provenance" — not
    the consumer. Nothing in this package inspects a key or a value.

    It records WHICH values, not which date took which. The per-date
    attribution already exists, once, in the produce run's manifest
    ``inputs[]``; duplicating it here would create a second copy that can
    disagree with the first. What the verdict surface could not answer
    before, and now can, is "one version or several".
    """

    arm_id: str
    scores: Mapping[str, float]
    misses: frozenset[str] = field(default_factory=frozenset)
    lineage: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("ArmSeries.arm_id must be non-empty")
        object.__setattr__(self, "lineage", _normalize_lineage(self.arm_id, self.lineage))
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
    #: Each arm's own information ratio ON THIS WINDOW — see
    #: :func:`_information_ratio`. STORED rather than computed on demand
    #: because :meth:`to_dict` emits only aggregates and :meth:`from_dict`
    #: therefore has no per-date series to recompute them from: a reader
    #: reconstructing a cycle must get back the ratio that DECIDED it, not a
    #: null standing in for a number the artifact actually carried.
    #:
    #: Left as None by a caller, they are derived from ``scores_a``/``scores_b``
    #: in ``__post_init__``, so every construction from real per-date data is
    #: correct by construction and no call site can forget to pass them.
    information_ratio_a: float | None = None
    information_ratio_b: float | None = None

    def __post_init__(self) -> None:
        # Only fills what the caller left unset. `from_dict` passes the stored
        # values with empty score tuples, and must keep them.
        if self.information_ratio_a is None and self.scores_a:
            object.__setattr__(self, "information_ratio_a", _information_ratio(self.scores_a))
        if self.information_ratio_b is None and self.scores_b:
            object.__setattr__(self, "information_ratio_b", _information_ratio(self.scores_b))

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

    @property
    def ir_diff(self) -> float | None:
        """``information_ratio_a - information_ratio_b``, or None when either
        leg is unestimable.

        WHY A SLOT WOULD DECIDE ON THIS. When arms in a slot are allowed
        DIFFERENT widths on purpose — the "what does depth cost?" experiment —
        a raw mean selects for concentration rather than skill, because mean
        alpha per name declines with depth whenever a ranking carries any
        signal. IR ~= IC x sqrt(breadth) prices that: a narrow arm must beat a
        wider one by enough to pay for the dispersion its concentration bought.

        None rather than a signed default: an arm whose IR cannot be estimated
        has not lost the comparison, it has not been in one, and a slot ranking
        on this must be able to tell the two apart.
        """
        a, b = self.information_ratio_a, self.information_ratio_b
        return None if a is None or b is None else a - b

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n_dates": self.n_dates,
            "weeks": self.weeks,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mean_diff": self.mean_diff if self.measurable else None,
            # Emitted on EVERY window, whatever the slot decided on, so a
            # reader can see the statistic that did not decide alongside the
            # one that did.
            "information_ratio_a": self.information_ratio_a if self.measurable else None,
            "information_ratio_b": self.information_ratio_b if self.measurable else None,
            "ir_diff": self.ir_diff if self.measurable else None,
            "unmeasurable_reason": self.unmeasurable_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PairedWindow:
        """The inverse of :meth:`to_dict` — for a READER, not a re-scorer.

        ``to_dict`` never serialises the per-date ``dates``/``diffs``/
        ``scores_a``/``scores_b`` — only the aggregates a decision reader
        needs (``n_dates``, ``weeks``, ``start_date``, ``end_date``,
        ``mean_diff``). Every property this class exposes (``n_dates``,
        ``weeks``, ``start_date``, ``end_date``, ``mean_diff``,
        ``measurable``) is derived from those four sequences, so
        reconstructing a window that answers them identically requires
        *some* sequence to sit behind each property — not the original one,
        which was never retained.

        The reconstruction is synthetic and deliberately documented as such:
        ``dates`` is filled with ``start_date`` repeated up to ``end_date``
        so ``len(dates) == n_dates`` and ``dates[0]``/``dates[-1]`` match
        (``weeks`` is computed from those two endpoints alone — see
        :func:`span_weeks` — so the filler in between never affects it).
        ``diffs`` is the single-element tuple ``(mean_diff,)`` rather than
        ``n_dates`` values that would have to average to it: a caller
        cannot recover the original per-date differences from an artifact
        that never stored them, and re-deriving a *plausible* per-date series
        would be a fabrication. A one-element series makes ``mean_diff``
        exact (``sum((x,)) / 1 == x``) without pretending to be the real
        one. ``scores_a``/``scores_b`` are never read by anything outside
        this class's own construction and are left empty.

        Callers that need the real per-date series must recompute it from
        the underlying score series (`nousergon_lib.arena.window
        .pair_on_common_window`), never from this artifact.
        """
        unmeasurable_reason = data.get("unmeasurable_reason")
        if unmeasurable_reason is not None:
            return cls(
                arm_a=str(data["arm_a"]),
                arm_b=str(data["arm_b"]),
                dates=(),
                diffs=(),
                scores_a=(),
                scores_b=(),
                unmeasurable_reason=str(unmeasurable_reason),
            )
        # The stored ratios, restored verbatim. They cannot be recomputed here
        # — `to_dict` emits no per-date scores — and a reader reconstructing a
        # cycle must get back the number that DECIDED the pointer.
        ir_a = data.get("information_ratio_a")
        ir_b = data.get("information_ratio_b")
        n_dates = int(data["n_dates"])
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        if n_dates <= 0 or start_date is None or end_date is None:
            dates: tuple[str, ...] = ()
        elif n_dates == 1:
            dates = (str(start_date),)
        else:
            dates = (str(start_date),) + (str(start_date),) * (n_dates - 2) + (str(end_date),)
        mean_diff = data.get("mean_diff")
        diffs = (float(mean_diff),) if mean_diff is not None else ()
        return cls(
            arm_a=str(data["arm_a"]),
            arm_b=str(data["arm_b"]),
            dates=dates,
            diffs=diffs,
            scores_a=(),
            scores_b=(),
            unmeasurable_reason=None,
            information_ratio_a=None if ir_a is None else float(ir_a),
            information_ratio_b=None if ir_b is None else float(ir_b),
        )



def _information_ratio(scores: Sequence[float]) -> float | None:
    """Mean over SAMPLE standard deviation, or None where neither is defined.

    Deliberately NOT annualised. Annualising assumes independent periods, and
    a slot's cohort dates routinely overlap — scaling by sqrt(periods) there
    would report a ratio inflated by the overlap rather than by skill.
    """
    n = len(scores)
    if n < 2:
        return None
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    if var <= 0:
        return None
    return mean / (var ** 0.5)


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
