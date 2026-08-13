"""seam_spread — one number per pipeline seam: output-population performance
minus input-population performance, over the same window.

This is the grading primitive behind Brian's directive (alpha-engine-config#7214):

    "performance should typically be population of inputs vs population of
     outputs. for predictor this should be the performance of the portfolio
     over the time window against the performance of the scanner attractiveness
     top 20 over the same time window."

A *seam* is a place where the system narrows a population: the scanner narrows
~900 names to a top-20; the predictor narrows the top-20 to a portfolio; the
executor narrows planned orders to realized fills. The seam is doing work if
and only if the selected sub-population out-performs the population it was
selected from, measured **the same way, over the same window**.

Why this module exists in the shared library
--------------------------------------------
The date-clustered mean estimator it implements is already hand-written three
times inside ``crucible-backtester`` (``analysis/attractiveness_eval.py::
_per_date_ics`` / ``_clustered_ic_block``, ``analysis/end_to_end.py::
_cio_layer_attribution``'s ``{layer}_date_ic``, and ``_trajectory_forward_ic``),
and the grader in ``crucible-evaluator`` needs the identical estimator to
*recompute* what those producers publish. Under the shared-code policy that is
well past the second-adoption trigger. Everything here is pure (no IO, no S3,
no pandas, stdlib only) so it is unit-testable in isolation and can run inside
a Lambda, a spot job, or a reader's own script.

The conventions, stated so they can be argued with
--------------------------------------------------
Every one of these is a *choice*, and each is enforced mechanically rather
than documented and hoped for.

1. **The cohort is frozen at the cohort date.** A population is exactly the
   set of names named by the artifact published on ``cohort_date``. Nothing is
   added later. Taking today's roster and looking back is the survivorship
   trap this rule exists to close.

2. **The output population must be a subset of the input population on the
   same date** (``require_subset``, default on). "Inputs vs outputs" is only a
   meaningful subtraction when the outputs were selected *from* the inputs. A
   name in the output that never appeared in the input is a contract
   violation and raises — it means the two artifacts were joined across
   different dates, different universes, or different key normalizations.

3. **On a SELECTION seam, the same name on the same date carries the same
   outcome on both sides** (``require_outcome_agreement``, default on, to
   within ``OUTCOME_TOLERANCE``). This is what makes "same return convention"
   checkable instead of assertable: if the input side measured a 21-day log
   alpha and the output side measured a 21-day simple return, the shared names
   disagree and this raises.

   Two of the four seams in alpha-engine-config#7214 are **not** selection
   seams — they are *measurement* seams, where both sides describe the same
   objects and the difference between them IS the number. The executor's
   planned order and its realized fill are the same order priced twice; the
   backtester's simulation and the realized trading history are the same
   trades run twice. For those, ``require_outcome_agreement=False`` — the
   subset guard still applies (a simulation that invents a trade that never
   happened is still a contract violation), but disagreement is the signal
   rather than the defect.

4. **Equal weight within a date, equal weight across dates** (date-clustered,
   "weeks-as-N"). Pooling names across dates lets one large cross-section
   dominate and pseudo-replicates a single market move into hundreds of
   observations; the pooled estimate is reported separately and is explicitly
   an inflated reference, never the headline. This mirrors the estimator
   already used for the attractiveness IC.

5. **Unresolved outcomes are dropped and COUNTED, never dropped silently.**
   A name with no realized outcome (delisted, halted, acquired, or simply not
   yet matured to the horizon) cannot contribute a return. Dropping it is the
   survivorship trap in its second form, so both sides publish
   ``n_unresolved`` and ``unresolved_rate``, and ``max_unresolved_rate``
   raises when either side has lost more of its population than the caller is
   willing to accept. **The asymmetry is the thing to watch**: a top-20 whose
   names all survive, measured against a ~900-name universe that quietly lost
   its delistings, is not a like-for-like comparison, and
   ``unresolved_rate_gap`` puts that number on the surface.

6. **Only dates present on BOTH sides are used.** A date with an input
   population and no output population (or vice versa) is excluded and named
   in ``dates_dropped`` — a renormalization the reader can see, rather than
   one the composite performs invisibly (alpha-engine-config#7202).

7. **The spread is a difference of means, not a mean of differences —
   except per date, where they coincide.** Each date contributes
   ``mean(output) - mean(input)``; the headline is the equal-weighted mean of
   those per-date spreads. This is identical to differencing the two
   date-clustered means whenever both sides have the same date set, which
   guard 6 enforces.

Units are the caller's: whatever ``Observation.outcome`` is denominated in
(decimal log alpha, simple return, basis points of shortfall), the spread is
in the same unit. The unit is carried on the published record as
``outcome_unit`` so a reader is never guessing.

Verifiability
-------------
``SeamSpread.to_dict()`` publishes every per-date row it averaged.
``recompute_spread()`` takes that published dict and reproduces the headline
number importing nothing from any producer — the reader-side recompute shape
alpha-engine-config#7214 requires. A published record whose ``spread`` does not
match its own per-date rows is self-refuting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "OUTCOME_TOLERANCE",
    "DateMean",
    "DateSpread",
    "Observation",
    "PopulationMean",
    "SeamSpread",
    "date_clustered_mean",
    "recompute_spread",
    "seam_spread",
]

#: Two sides of a seam may carry the same (date, key) outcome through
#: different float paths (a JSON round-trip, a SQL cast). Anything larger than
#: this is a genuine convention disagreement, not representation noise.
OUTCOME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Observation:
    """One member of one population on one date, with its realized outcome.

    ``outcome`` is ``None`` when the outcome is *unresolved* — the name exists
    in the population but no realized number can be attached to it (delisted
    inside the window, no price at the horizon, horizon not yet matured).
    ``None`` is a first-class state here precisely so that dropping it is
    counted rather than silent.
    """

    cohort_date: str
    key: str
    outcome: float | None = None

    def __post_init__(self) -> None:
        if not self.cohort_date:
            raise ValueError("Observation.cohort_date must be a non-empty string")
        if not self.key:
            raise ValueError("Observation.key must be a non-empty string")
        if self.outcome is not None:
            value = float(self.outcome)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(
                    f"Observation.outcome must be finite or None (got {self.outcome!r} for "
                    f"{self.cohort_date}/{self.key}); a NaN outcome is an unresolved outcome and must "
                    "be passed as None so it is counted"
                )


@dataclass(frozen=True)
class DateMean:
    """The equal-weighted mean outcome of one population on one date."""

    cohort_date: str
    mean: float
    n_resolved: int
    n_unresolved: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_date": self.cohort_date,
            "mean": self.mean,
            "n_resolved": self.n_resolved,
            "n_unresolved": self.n_unresolved,
        }


@dataclass(frozen=True)
class DateSpread:
    """One date's contribution to the seam spread."""

    cohort_date: str
    input_mean: float
    output_mean: float
    spread: float
    n_input_resolved: int
    n_output_resolved: int
    n_input_unresolved: int
    n_output_unresolved: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_date": self.cohort_date,
            "input_mean": self.input_mean,
            "output_mean": self.output_mean,
            "spread": self.spread,
            "n_input_resolved": self.n_input_resolved,
            "n_output_resolved": self.n_output_resolved,
            "n_input_unresolved": self.n_input_unresolved,
            "n_output_unresolved": self.n_output_unresolved,
        }


@dataclass(frozen=True)
class PopulationMean:
    """Date-clustered mean of one population, with its coverage accounting."""

    mean: float | None
    pooled_mean: float | None
    n_dates: int
    n_resolved: int
    n_unresolved: int
    per_date: tuple[DateMean, ...]

    @property
    def unresolved_rate(self) -> float | None:
        """Fraction of the population that carried no realized outcome.

        ``None`` when the population is empty — "could not measure" is not
        "measured zero" (alpha-engine-config#7105).
        """
        total = self.n_resolved + self.n_unresolved
        if total == 0:
            return None
        return self.n_unresolved / total

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "pooled_mean": self.pooled_mean,
            "n_dates": self.n_dates,
            "n_resolved": self.n_resolved,
            "n_unresolved": self.n_unresolved,
            "unresolved_rate": self.unresolved_rate,
            "per_date": [d.to_dict() for d in self.per_date],
        }


@dataclass(frozen=True)
class SeamSpread:
    """The one number for a seam, plus everything needed to recompute it."""

    seam: str
    seam_kind: str
    outcome_unit: str
    window_start: str
    window_end: str
    spread: float | None
    input_population: PopulationMean
    output_population: PopulationMean
    per_date: tuple[DateSpread, ...]
    dates_dropped: tuple[str, ...]
    status: str
    status_reason: str | None = None

    @property
    def n_dates(self) -> int:
        return len(self.per_date)

    @property
    def unresolved_rate_gap(self) -> float | None:
        """Output unresolved rate minus input unresolved rate.

        The survivorship tell. A materially negative value means the input
        population lost names the output population kept, which flatters the
        seam by removing the input's worst outcomes; a materially positive one
        means the reverse. ``None`` when either side is empty.
        """
        out = self.output_population.unresolved_rate
        inp = self.input_population.unresolved_rate
        if out is None or inp is None:
            return None
        return out - inp

    def to_dict(self) -> dict[str, object]:
        """The publishable record. Carries every per-date row it averaged."""
        return {
            "schema_version": 1,
            "seam": self.seam,
            "seam_kind": self.seam_kind,
            "outcome_unit": self.outcome_unit,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "estimator": "date_clustered_mean_spread",
            "status": self.status,
            "status_reason": self.status_reason,
            "spread": self.spread,
            "n_dates": self.n_dates,
            "unresolved_rate_gap": self.unresolved_rate_gap,
            "input_population": self.input_population.to_dict(),
            "output_population": self.output_population.to_dict(),
            "per_date": [d.to_dict() for d in self.per_date],
            "dates_dropped": list(self.dates_dropped),
        }


def _group(observations: Iterable[Observation]) -> dict[str, dict[str, float | None]]:
    """(cohort_date -> key -> outcome), rejecting duplicate keys per date.

    A duplicate ``(cohort_date, key)`` is a producer defect: the same name
    counted twice in one cross-section silently double-weights it. It raises
    rather than de-duplicating, because either copy could be the right one and
    picking one is a guess.
    """
    grouped: dict[str, dict[str, float | None]] = {}
    for obs in observations:
        bucket = grouped.setdefault(obs.cohort_date, {})
        if obs.key in bucket:
            raise ValueError(
                f"duplicate observation for {obs.cohort_date}/{obs.key}: a name may appear at most "
                "once per cohort date, or it is silently double-weighted"
            )
        bucket[obs.key] = None if obs.outcome is None else float(obs.outcome)
    return grouped


def date_clustered_mean(
    observations: Iterable[Observation],
    *,
    min_names_per_date: int = 1,
) -> PopulationMean:
    """Equal-weight within a date, then equal-weight across dates.

    A date with fewer than ``min_names_per_date`` *resolved* outcomes does not
    produce a cross-sectional mean worth averaging, so it contributes no date
    row — but its members are still counted in ``n_resolved`` /
    ``n_unresolved``, so the coverage accounting stays honest about what was
    seen versus what was used.

    Also reports ``pooled_mean`` (every resolved name weighted equally,
    ignoring dates). It is a diagnostic only: a large gap between ``mean`` and
    ``pooled_mean`` means the date cross-sections differ wildly in size, and
    the pooled number is the one that lies.
    """
    if min_names_per_date < 1:
        raise ValueError("min_names_per_date must be >= 1")

    grouped = _group(observations)
    per_date: list[DateMean] = []
    n_resolved = 0
    n_unresolved = 0
    pooled_sum = 0.0

    for cohort_date in sorted(grouped):
        outcomes = grouped[cohort_date]
        resolved = [v for v in outcomes.values() if v is not None]
        unresolved = len(outcomes) - len(resolved)
        n_resolved += len(resolved)
        n_unresolved += unresolved
        pooled_sum += sum(resolved)
        if len(resolved) < min_names_per_date:
            continue
        per_date.append(
            DateMean(
                cohort_date=cohort_date,
                mean=sum(resolved) / len(resolved),
                n_resolved=len(resolved),
                n_unresolved=unresolved,
            )
        )

    mean = sum(d.mean for d in per_date) / len(per_date) if per_date else None
    pooled = pooled_sum / n_resolved if n_resolved else None
    return PopulationMean(
        mean=mean,
        pooled_mean=pooled,
        n_dates=len(per_date),
        n_resolved=n_resolved,
        n_unresolved=n_unresolved,
        per_date=tuple(per_date),
    )


def _assert_same_convention(
    seam: str,
    cohort_date: str,
    inputs: Mapping[str, float | None],
    outputs: Mapping[str, float | None],
    require_subset: bool,
    require_outcome_agreement: bool,
) -> None:
    for key, out_value in outputs.items():
        if key not in inputs:
            if require_subset:
                raise ValueError(
                    f"seam {seam}: {key} appears in the output population on {cohort_date} but "
                    "not in the input population. 'inputs vs outputs' is only a "
                    "meaningful subtraction when the outputs were selected FROM "
                    "the inputs; this usually means the two artifacts were "
                    "joined across different dates, universes, or key "
                    "normalizations."
                )
            continue
        if not require_outcome_agreement:
            continue
        in_value = inputs[key]
        if in_value is None or out_value is None:
            if (in_value is None) != (out_value is None):
                raise ValueError(
                    f"seam {seam}: {key} on {cohort_date} is resolved on one side of the seam "
                    "and unresolved on the other. The same name must carry the "
                    "same realized outcome on both sides or the populations are "
                    "not being measured the same way."
                )
            continue
        if abs(in_value - out_value) > OUTCOME_TOLERANCE:
            raise ValueError(
                f"seam {seam}: {key} on {cohort_date} carries outcome {in_value!r} in the input "
                f"population and {out_value!r} in the output population. Both sides must "
                "use the SAME return convention over the SAME window; this "
                "difference is larger than float round-trip noise "
                f"({OUTCOME_TOLERANCE})."
            )


def seam_spread(
    *,
    seam: str,
    outcome_unit: str,
    window_start: str,
    window_end: str,
    input_population: Sequence[Observation],
    output_population: Sequence[Observation],
    min_names_per_date: int = 1,
    min_dates: int = 1,
    max_unresolved_rate: float | None = None,
    require_subset: bool = True,
    require_outcome_agreement: bool = True,
) -> SeamSpread:
    """Compute one seam's number: output performance minus input performance.

    Args:
        seam: seam identifier, e.g. ``"scanner"`` — carried onto the record.
        outcome_unit: what ``Observation.outcome`` is denominated in, e.g.
            ``"log_alpha_21d"`` or ``"bps"``. Published so a reader is never
            guessing at the unit of the headline number.
        window_start / window_end: the window both populations were measured
            over, ``YYYY-MM-DD``. Carried, not derived — the caller knows
            which window it asked its data source for, and a window inferred
            from whatever rows came back is a window that silently moves.
        input_population / output_population: the two cohorts.
        min_names_per_date: minimum resolved names for a date to contribute.
        min_dates: minimum contributing dates before a spread is reported at
            all. Below it the result is ``status="insufficient_data"`` with
            ``spread=None`` — an honest absence, never a fabricated zero.
        max_unresolved_rate: raise if either side lost more than this fraction
            of its population to unresolved outcomes. ``None`` disables the
            guard but never disables the *reporting* of the rate.
        require_subset: enforce that every output name appears in the input
            population on the same date.
        require_outcome_agreement: enforce that a name shared by both sides
            carries the same outcome on both. ``True`` for a SELECTION seam
            (scanner, predictor), where the outcome is a property of the name
            and any disagreement means the two sides used different
            conventions. ``False`` for a MEASUREMENT seam (executor,
            backtester), where both sides describe the same objects and the
            disagreement is the number being reported.

    Returns:
        A :class:`SeamSpread`. ``status`` is ``"ok"`` or
        ``"insufficient_data"``; it is never ``"ok"`` with a ``None`` spread.

    Raises:
        ValueError: on a convention violation — a non-subset output, a name
            whose outcome disagrees between the two sides, a duplicate name in
            one cross-section, or an unresolved rate above
            ``max_unresolved_rate``. These are contract violations, not data
            states, and a grade computed over them would be wrong rather than
            missing.
    """
    if not seam:
        raise ValueError("seam must be a non-empty identifier")
    if not outcome_unit:
        raise ValueError(
            "outcome_unit must be stated: a spread whose unit is unpublished "
            "cannot be compared to anything, including its own history"
        )
    if min_dates < 1:
        raise ValueError("min_dates must be >= 1")
    if max_unresolved_rate is not None and not 0.0 <= max_unresolved_rate <= 1.0:
        raise ValueError("max_unresolved_rate must be in [0, 1] or None")

    grouped_in = _group(input_population)
    grouped_out = _group(output_population)

    shared = sorted(set(grouped_in) & set(grouped_out))
    dropped = tuple(sorted(set(grouped_in) ^ set(grouped_out)))

    for cohort_date in shared:
        _assert_same_convention(
            seam,
            cohort_date,
            grouped_in[cohort_date],
            grouped_out[cohort_date],
            require_subset,
            require_outcome_agreement,
        )
    # A non-subset output on a date the input side never published is still a
    # convention violation worth raising, even though the date is dropped.
    if require_subset:
        for cohort_date in sorted(set(grouped_out) - set(grouped_in)):
            if grouped_out[cohort_date]:
                raise ValueError(
                    f"seam {seam}: the output population has {len(grouped_out[cohort_date])} name(s) on {cohort_date} and "
                    "the input population has no rows for that date at all. The "
                    "seam cannot be graded on a date whose input population was "
                    "never published."
                )

    shared_in = [
        Observation(d, k, v) for d in shared for k, v in sorted(grouped_in[d].items())
    ]
    shared_out = [
        Observation(d, k, v) for d in shared for k, v in sorted(grouped_out[d].items())
    ]
    in_mean = date_clustered_mean(shared_in, min_names_per_date=min_names_per_date)
    out_mean = date_clustered_mean(shared_out, min_names_per_date=min_names_per_date)

    if max_unresolved_rate is not None:
        for label, pop in (("input", in_mean), ("output", out_mean)):
            rate = pop.unresolved_rate
            if rate is not None and rate > max_unresolved_rate:
                raise ValueError(
                    f"seam {seam}: the {label} population lost {rate:.1%} of its names to "
                    f"unresolved outcomes, above the {max_unresolved_rate:.1%} ceiling. Names that "
                    "leave the cross-section are the survivorship trap; a spread "
                    "computed over the remainder would flatter whichever side "
                    "kept its losers."
                )

    in_by_date = {d.cohort_date: d for d in in_mean.per_date}
    out_by_date = {d.cohort_date: d for d in out_mean.per_date}
    per_date: list[DateSpread] = []
    for cohort_date in sorted(set(in_by_date) & set(out_by_date)):
        lhs = in_by_date[cohort_date]
        rhs = out_by_date[cohort_date]
        per_date.append(
            DateSpread(
                cohort_date=cohort_date,
                input_mean=lhs.mean,
                output_mean=rhs.mean,
                spread=rhs.mean - lhs.mean,
                n_input_resolved=lhs.n_resolved,
                n_output_resolved=rhs.n_resolved,
                n_input_unresolved=lhs.n_unresolved,
                n_output_unresolved=rhs.n_unresolved,
            )
        )

    if len(per_date) < min_dates:
        return SeamSpread(
            seam=seam,
            seam_kind="selection" if require_outcome_agreement else "measurement",
            outcome_unit=outcome_unit,
            window_start=window_start,
            window_end=window_end,
            spread=None,
            input_population=in_mean,
            output_population=out_mean,
            per_date=tuple(per_date),
            dates_dropped=dropped,
            status="insufficient_data",
            status_reason=(
                f"{len(per_date)} usable cohort date(s), need {min_dates} (a date is usable when BOTH "
                f"populations have at least {min_names_per_date} resolved outcome(s) on it)"
            ),
        )

    spread = sum(d.spread for d in per_date) / len(per_date)
    return SeamSpread(
        seam=seam,
        seam_kind="selection" if require_outcome_agreement else "measurement",
        outcome_unit=outcome_unit,
        window_start=window_start,
        window_end=window_end,
        spread=spread,
        input_population=in_mean,
        output_population=out_mean,
        per_date=tuple(per_date),
        dates_dropped=dropped,
        status="ok",
        status_reason=None,
    )


def recompute_spread(published: Mapping[str, object]) -> float | None:
    """Reader-side recompute of a published seam record.

    Takes the dict emitted by :meth:`SeamSpread.to_dict` (or its JSON
    round-trip) and reproduces the headline ``spread`` from the ``per_date``
    rows alone. Imports nothing from any producer and reads no artifact other
    than the one handed in — so a reader can hold a published number to
    account without trusting, or even having, the code that wrote it.

    Returns ``None`` when the record carries no usable date rows, matching the
    ``insufficient_data`` status. Raises ``ValueError`` on a malformed record
    — a record that cannot be checked must not read as a record that passed.
    """
    rows = published.get("per_date")
    if rows is None:
        raise ValueError("published seam record has no 'per_date' rows to recompute from")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("published 'per_date' must be a list of rows")
    if not rows:
        return None

    spreads: list[float] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"published per_date[{i}] is not an object")
        for field in ("input_mean", "output_mean"):
            if field not in row:
                raise ValueError(
                    f"published per_date[{i}] is missing {field!r}; the record does "
                    "not carry enough to be recomputed"
                )
        spreads.append(float(row["output_mean"]) - float(row["input_mean"]))
    return sum(spreads) / len(spreads)
