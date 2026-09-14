"""The phase ladder — one durable row per phase.

Until this existed in crucible, "which phase is this on, and is its gate met"
was published only as prose in issue comments, written by hand. On one issue
the phase-1 reading moved 18/24 -> 19/23 -> 21/23 across three comments, one of
them explicitly correcting another, and phase 1 was CLOSED while phase 0's gate
had never been read at all. Three defects, one cause: the number that says
whether a phase is done had no surface of its own.

The ladder is that surface. It is a PROJECTION over gate readings already in
the store plus the registered clause lists — it runs nothing, and it invents no
state a gate did not measure.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from nousergon_lib.gates.keys import gate_prefix
from nousergon_lib.gates.result import GateResult, gate_state_for
from nousergon_lib.gates.store import GateStore, list_store_keys

__all__ = [
    "LADDER_CONSOLE_STATE",
    "LADDER_SCHEMA_PATH",
    "LADDER_SCHEMA_VERSION",
    "LADDER_STATES",
    "Ladder",
    "Phase",
    "PhaseRow",
    "build_ladder",
    "ladder_payload",
    "ladder_schema",
    "last_read",
    "validate_ladder_document",
]

LADDER_SCHEMA_VERSION = "phase_ladder.v1"

#: The closed set of ladder states.
#:
#: * ``MET`` / ``UNMET``   — the gate was read and answered.
#: * ``UNMEASURED``        — no clause list is registered, or the gate has no
#:                           clauses. Nothing has ever measured this phase.
#: * ``UNMEASURABLE``      — the reading itself failed. Red, never folded into
#:                           ``UNMET``: "it says no" and "I could not ask" are
#:                           different facts, and the second is about us.
#: * ``OUT_OF_ORDER``      — a later phase is being graded while an earlier
#:                           phase's gate is not met.
LADDER_STATES: tuple[str, ...] = ("MET", "UNMET", "UNMEASURED", "UNMEASURABLE", "OUT_OF_ORDER")

#: How a ladder state renders on the fleet console, in `observability-policy`
#: §8.3's vocabulary. ``UNMEASURED`` maps to ``UNREPORTED`` and therefore counts
#: against the transparency gap whose objective is zero — a phase nobody can
#: read is unobserved, not healthy. ``UNMEASURABLE`` and ``OUT_OF_ORDER`` both
#: map to ``FAILED``: one is an access fault, the other an invariant breach, but
#: neither is a slow phase or a plain shortfall.
LADDER_CONSOLE_STATE: dict[str, str] = {
    "MET": "HEALTHY",
    "UNMET": "DEGRADED",
    "UNMEASURED": "UNREPORTED",
    "UNMEASURABLE": "FAILED",
    "OUT_OF_ORDER": "FAILED",
}


def _check_ladder_console_coverage(states: Iterable[str], console_map: dict[str, str]) -> None:
    """Refuse a ladder state with no declared console rendering.

    A plain function, not a module-level ``assert``: ``assert`` is compiled out
    under ``python -O`` — the one guard construct guaranteed absent in an
    optimized interpreter, and this is the guard that stops a ladder state
    reaching a console surface with no declared rendering.
    """
    gap = set(states) - set(console_map)
    if gap:
        raise ValueError(
            f"LADDER_CONSOLE_STATE is missing {sorted(gap)} — every ladder state must declare how "
            "it renders before it can reach a surface."
        )


_check_ladder_console_coverage(LADDER_STATES, LADDER_CONSOLE_STATE)


@dataclass(frozen=True)
class Phase:
    """One rung of a ladder, and the gate that lets it exit."""

    id: str
    number: int
    title: str
    #: The tracker ref, e.g. ``alpha-engine-config-I10748``. Supplied rather
    #: than derived: the engine has two adopters and must not hard-code one
    #: tracker repository.
    tracker: str
    tracker_url: str
    #: The registered gate name, or ``None`` when no gate has been written for
    #: this phase yet. ``None`` is never a pass: it renders ``UNMEASURED``.
    gate: str | None = None

    @classmethod
    def on_alpha_engine_config(cls, *, id: str, number: int, title: str, issue: int, gate: str | None = None) -> Phase:
        """A rung tracked on ``nousergon/alpha-engine-config``.

        The alpha-engine ecosystem files to that tracker whatever repo the code
        lands in, so the ref is ``alpha-engine-config-I<N>`` — deliberately the
        SAME identifier a console `git-host` adapter mints for the issue, so the
        two claims merge into one row instead of rendering the phase twice.
        """
        return cls(
            id=id,
            number=number,
            title=title,
            tracker=f"alpha-engine-config-I{issue}",
            tracker_url=f"https://github.com/nousergon/alpha-engine-config/issues/{issue}",
            gate=gate,
        )


def last_read(store: GateStore, gate: str) -> tuple[str | None, bool]:
    """The most recent day a reading of ``gate`` was filed for, and whether the
    listing itself could not be read.

    ``(None, False)`` when the gate has never been read — the field a row
    publishes as "when was this last measured", so a row that cannot answer it
    says so rather than borrowing the ladder's own generation time. A freshly
    written ladder full of never-measured phases would otherwise look entirely
    fresh.

    ``(None, True)`` is a THIRD, different answer: the listing failed. A bare
    list call here would raise straight out of :func:`build_ladder`, defeating
    the per-clause containment everywhere else.
    """
    listed = list_store_keys(store, gate_prefix(gate))
    if listed.problem is not None:
        return None, listed.access_problem
    days: list[str] = []
    for key in listed.keys or []:
        parts = key.split("/")
        if key.endswith("/gate.json") and len(parts) == 4:
            days.append(parts[2])
    return (max(days) if days else None), False


@dataclass(frozen=True)
class PhaseRow:
    """One phase, as the console reads it."""

    phase: Phase
    state: str
    gate_state: str
    detail: str
    clauses_met: int | None
    clauses_total: int | None
    #: How many of this phase's clauses read UNMEASURABLE. ``None`` exactly
    #: where ``clauses_total`` is ``None`` (no clause list registered at all);
    #: otherwise always a count, ``0`` included, so its absence on the wire is
    #: never ambiguous with "not counted".
    clauses_unmeasurable: int | None
    met_ratio: float | None
    read_on: str | None
    blocked_by: str | None

    def to_dict(self, generated_utc: str) -> dict[str, Any]:
        return {
            "decision_id": self.phase.tracker,
            "phase": self.phase.id,
            "number": self.phase.number,
            "title": self.phase.title,
            "tracker": self.phase.tracker,
            "tracker_url": self.phase.tracker_url,
            "gate": self.phase.gate,
            "state": self.state,
            "console_state": LADDER_CONSOLE_STATE[self.state],
            "gate_state": self.gate_state,
            "detail": self.detail,
            "clauses_met": self.clauses_met,
            "clauses_total": self.clauses_total,
            "clauses_unmeasurable": self.clauses_unmeasurable,
            # `null`, never 0.0, when nothing was measured. Zero is a
            # measurement; absence is not, and rendering one as the other is
            # the whole defect (principle 7).
            "met_ratio": self.met_ratio,
            "read_on": self.read_on,
            "blocked_by": self.blocked_by,
            "generated_utc": generated_utc,
        }


@dataclass
class Ladder:
    """Every phase, in order, with the ladder-level counts."""

    trading_day: dt.date
    generated_utc: str
    rows: list[PhaseRow] = field(default_factory=list)
    #: Which ladder this is when a store holds more than one.
    name: str | None = None

    @property
    def current_phase(self) -> str:
        """The lowest phase whose gate is not met — where the ladder actually is.

        Not "the highest phase with work in it". A phase whose gate has never
        been read is not behind us, and this is the field that says so.
        """
        for row in self.rows:
            if row.gate_state != "MET":
                return row.phase.id
        return "complete"

    @property
    def out_of_order(self) -> list[str]:
        return [r.phase.id for r in self.rows if r.state == "OUT_OF_ORDER"]

    @property
    def unmeasured(self) -> int:
        return sum(1 for r in self.rows if r.gate_state == "UNMEASURED")

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": LADDER_SCHEMA_VERSION,
            "trading_day": self.trading_day.isoformat(),
            "generated_utc": self.generated_utc,
            "current_phase": self.current_phase,
            "phases_total": len(self.rows),
            "phases_met": sum(1 for r in self.rows if r.gate_state == "MET"),
            "unmeasured": self.unmeasured,
            "out_of_order": self.out_of_order,
            "phases": [r.to_dict(self.generated_utc) for r in self.rows],
        }
        if self.name:
            document["ladder"] = self.name
        return document

    def render(self) -> str:
        lines = [
            "{}ladder @ {}: at {}, {}/{} met, {} unmeasured".format(
                f"{self.name} " if self.name else "",
                self.trading_day.isoformat(),
                self.current_phase,
                sum(1 for r in self.rows if r.gate_state == "MET"),
                len(self.rows),
                self.unmeasured,
            )
        ]
        for row in self.rows:
            read = row.read_on or "never"
            lines.append(
                f"  [{row.phase.number}] {row.phase.id} {row.state}: {row.detail} (tracker {row.phase.tracker}, last read {read})"
            )
        return "\n".join(lines)


def build_ladder(
    store: GateStore,
    *,
    phases: Sequence[Phase],
    trading_day: dt.date,
    evaluate: Callable[[str], GateResult],
    now: dt.datetime | None = None,
    readings: dict[str, GateResult] | None = None,
    name: str | None = None,
) -> Ladder:
    """Read every phase's gate out of ``store`` and assemble the ladder.

    Reads only. Each registered gate is evaluated against artifacts already
    filed — the same measurement the gate command publishes — so the ladder can
    never disagree with the per-gate reading beside it.

    ``readings`` lets a caller that already evaluated a gate this run hand that
    reading in rather than have it re-evaluated here; any gate not present is
    evaluated through ``evaluate``.
    """
    moment = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    supplied = readings or {}

    graded: list[tuple[Phase, str, str, int | None, int | None, int | None, float | None, str | None]] = []
    for phase in phases:
        if phase.gate is None:
            graded.append(
                (
                    phase,
                    "UNMEASURED",
                    f"no clause list is registered for {phase.id}; nothing has ever measured whether it may exit",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        reading = supplied.get(phase.gate) or evaluate(phase.gate)
        met = sum(1 for c in reading.clauses if c.met)
        total = len(reading.clauses)
        unmeasurable_count = reading.unmeasurable
        read_on, read_on_access_problem = last_read(store, phase.gate)
        if total == 0:
            # A gate with no clauses measured nothing. `met` would be vacuously
            # true and `met_ratio` is already None here — both are the shape
            # that lets a phase close unmeasured, so the ladder refuses to call
            # it a reading.
            graded.append(
                (
                    phase,
                    "UNMEASURED",
                    f"gate {phase.gate} has no clauses",
                    0,
                    0,
                    0,
                    reading.met_ratio,
                    read_on,
                )
            )
            continue
        unmet = [c.name for c in reading.clauses if not c.met and not c.unmeasurable]
        unmeasurable_names = [c.name for c in reading.clauses if c.unmeasurable]
        detail = f"{met}/{total} clauses met"
        if unmeasurable_names:
            detail = "{}; unmeasurable: {}".format(detail, ", ".join(sorted(unmeasurable_names)[:12]))
        if unmet:
            detail = "{}; holding: {}".format(detail, ", ".join(sorted(unmet)[:12]))
        # The coverage line travels with the row. Without it a phase whose gate
        # grades a SUBSET of its deliverables renders MET with nothing saying so
        # — partial coverage reported as complete.
        if reading.coverage:
            detail = f"{detail}; {reading.coverage}"
        if read_on_access_problem:
            detail = f"{detail}; could not determine when gate {phase.gate} was last read (store access failure)"
        # `UNMEASURABLE` outranks `MET`/`UNMET`: a phase with one unreadable
        # clause, or whose own read-history listing failed, rendering as plain
        # `UNMET` with a specific ratio is indistinguishable from "we checked
        # and it fell short".
        state = "UNMEASURABLE" if read_on_access_problem else gate_state_for(reading)
        graded.append((phase, state, detail, met, total, unmeasurable_count, reading.met_ratio, read_on))

    # Out-of-order: a phase later than the lowest not-met phase that has
    # nevertheless been graded. Derived from readings alone, so nothing here
    # depends on a hand-maintained claim about which phase is "open".
    first_unmet = next((i for i, s in enumerate(graded) if s[1] != "MET"), len(graded))
    rows: list[PhaseRow] = []
    for index, (phase, gate_state, detail, met, total, unmeasurable_count, ratio, read_on) in enumerate(graded):
        state = gate_state
        blocked_by = None
        if index > first_unmet and read_on is not None:
            blocker = graded[first_unmet][0]
            state = "OUT_OF_ORDER"
            blocked_by = blocker.id
            detail = (
                f"{detail}; graded while {blocker.id} ({blocker.tracker}) is {graded[first_unmet][1]} — a later phase may not be exited ahead "
                "of an earlier one"
            )
        rows.append(
            PhaseRow(
                phase=phase,
                state=state,
                gate_state=gate_state,
                detail=detail,
                clauses_met=met,
                clauses_total=total,
                clauses_unmeasurable=unmeasurable_count,
                met_ratio=ratio,
                read_on=read_on,
                blocked_by=blocked_by,
            )
        )

    return Ladder(
        trading_day=trading_day,
        generated_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
        name=name,
    )


LADDER_SCHEMA_PATH = Path(__file__).parent / "schemas" / "phase_ladder.v1.json"


@lru_cache(maxsize=1)
def ladder_schema() -> dict[str, Any]:
    """The ``phase_ladder.v1`` JSON Schema, loaded once.

    A missing schema is a broken build, not a degraded read: it ships inside the
    package.
    """
    if not LADDER_SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"phase ladder schema missing at {LADDER_SCHEMA_PATH}. It ships inside the package; a missing "
            "schema means a broken build."
        )
    return json.loads(LADDER_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _ladder_validator() -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
        raise ImportError(
            "nousergon_lib.gates validates the ladder against its shipped JSON Schema "
            "and needs jsonschema: install nousergon-lib[gates]. Publishing an "
            "unvalidated ladder is not an available degradation — a malformed document "
            "reaches a console that renders it as state."
        ) from exc
    schema = ladder_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_ladder_document(document: dict[str, Any]) -> None:
    """Refuse a ladder document that does not conform to ``phase_ladder.v1``.

    Producer-side validation: a malformed ladder is refused before it reaches
    the store, not discovered by whatever reads it next.
    """
    errors = sorted(_ladder_validator().iter_errors(document), key=lambda e: list(e.path))
    if errors:
        detail = "\n".join("  - {}: {}".format("/".join(str(p) for p in e.path) or "<root>", e.message) for e in errors)
        raise ValueError(f"ladder document does not conform to {LADDER_SCHEMA_VERSION}:\n{detail}")


def ladder_payload(ladder: Ladder) -> bytes:
    """The ladder artifact's bytes, as every publisher writes them.

    Validated before being returned, so there is exactly one place the bytes on
    the wire are produced and checked.
    """
    document = ladder.to_dict()
    validate_ladder_document(document)
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
