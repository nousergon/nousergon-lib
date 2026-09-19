"""The immutable arm register — an append-only event log, never a mutable table.

**Brian's ruling, 2026-08-29, verbatim:**

    "going forward we should not 'change' a vintage, if we make a change the
    updated version becomes a challenger arm."

**An arm is a RECIPE, not a set of frozen weights.** Brian, same day, when
the first reading of the ruling was taken too far:

    "i'm not following how the retrain goes away. Won't we need to retrain
    predictor models on a cadence regardless? What I'm thinking is say the
    pool is 5 arms. we promote the arm that has the longest common cumulative
    record each week as the champion, the others stay in the rotation as
    challengers."

The immutable thing is the **recipe INCLUDING its retrain cadence** —
features, hyperparameters, training-window rule and refit schedule are all
fixed at registration; the fitted weights refresh on that schedule. **A
refit is the arm doing its job, not a change to the arm**, so the arm's
score series stays CONTINUOUS and VALID across every refit. This is the
standard construction: a strategy includes its own refit rule, and its track
record is the record of that whole rule operating over time.

A NEW arm therefore exists only when a recipe is deliberately added or
changed. Then it starts a fresh series and must win like anything else.
Editing a recipe in place destroys the only thing a champion/challenger loop
has — a track record that means what it says.

Refits are still recorded (:data:`EVENT_REFIT`), because "which fit produced
this score" must be reconstructible from durable artifacts alone
(`principles.md` §2.1). Recording one changes no id and resets no series.

**How that is enforced, not merely asked for.** The arm id CONTAINS the hash
of its own spec (:func:`derive_arm_id`), so a changed spec cannot reuse an
id; and :class:`ArmRegister` is an append-only log of
:class:`ArmEvent` records whose state is a fold, so there is no field to
overwrite. Retirement appends a ``retired`` event rather than setting a flag
on the record. ``retired_date`` stays queryable (§6) as a derived property
of the fold — derived, but deterministic and never an inference from prose.

**Shape mirrored from a working precedent.** Predictor registry bundles at
``predictor/registry/{version_id}/`` are already immutable and ETag-verified;
this is the same shape lifted to the register itself so all four slots share
one implementation (`shared-code-policy.md`).

**What this module deliberately does NOT do: admission control.** Brian ruled
2026-08-29 that the cap is a RETIREMENT criterion with a grace period, not an
admission gate:

    "we can temporarily have more than 5 arms in a process if the
    underperforming arms are less than 4 weeks old. at 4 weeks, assuming they
    are still not in the top 5 they are deprecated, removed from the process"

So registration is never blocked and the pool may exceed the cap while a
newly added recipe is inside its grace window. Because the roster is a set of
recipes rather than a weekly vintage cohort, the cap only bites when a sixth
recipe is deliberately added. The rule is applied by
:func:`nousergon_lib.arena.engine.evaluate_retirements`.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .window import elapsed_weeks as _elapsed_weeks

__all__ = [
    "ArmRecord",
    "ArmEvent",
    "ArmRegister",
    "ArmState",
    "ImmutableArmError",
    "derive_arm_id",
    "spec_hash",
    "EVENT_REGISTERED",
    "EVENT_RETIRED",
    "EVENT_REFIT",
]

EVENT_REGISTERED = "registered"
EVENT_RETIRED = "retired"

#: A scheduled refit of an existing recipe. Provenance only: it changes no
#: arm id, starts no new series, and never resets a ladder. A refit is the
#: arm executing its own recipe, not a new arm (Brian ruling 2026-08-29).
EVENT_REFIT = "refit"

_EVENT_KINDS = (EVENT_REGISTERED, EVENT_RETIRED, EVENT_REFIT)

_SPEC_HASH_CHARS = 12


class ImmutableArmError(Exception):
    """Raised on any attempt to change what an existing arm id means.

    Never caught-and-continued anywhere in this package: a register that has
    silently accepted a mutation is worse than one that refused to load,
    because every score attributed to that id afterwards is a lie about its
    own provenance (§7.5).
    """


def _canonical(spec: Mapping[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Stable hash of an arm spec. Key order and whitespace are irrelevant."""
    digest = hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()
    return digest[:_SPEC_HASH_CHARS]


def derive_arm_id(slot: str, name: str, spec: Mapping[str, Any]) -> str:
    """``{slot}:{name}:{spec_hash}`` — a changed spec cannot reuse an id.

    This is the mechanical half of ruling 3. A caller who "tweaks a vintage"
    and re-derives the id gets a different id and therefore a new arm, with
    no way to inherit the old arm's record.
    """
    if not slot or ":" in slot:
        raise ValueError(f"slot must be non-empty and contain no ':'; got {slot!r}")
    if not name or ":" in name:
        raise ValueError(f"name must be non-empty and contain no ':'; got {name!r}")
    return f"{slot}:{name}:{spec_hash(spec)}"


@dataclass(frozen=True)
class ArmRecord:
    """An arm's identity. Written once; never rewritten."""

    arm_id: str
    slot: str
    name: str
    spec_hash: str
    created_date: str
    supersedes: str | None = None
    bootstrap: bool = False
    notes: str = ""
    #: A control (synthetic benchmark) arm. Scored and ranked exactly like
    #: any other arm — it is a real point of comparison — but excluded from
    #: the §6 retirement cap in both directions: a control's pairwise win
    #: never counts toward another arm's `pairwise_losses`, and a control is
    #: never itself retired (`nousergon_lib.arena.engine.evaluate_retirements`).
    #: Additive: absent on every prior record, which folds to False
    #: (`alpha-engine-config-I9770`).
    control: bool = False

    def __post_init__(self) -> None:
        expected_prefix = f"{self.slot}:{self.name}:"
        if not self.arm_id.startswith(expected_prefix):
            raise ImmutableArmError(
                f"arm_id {self.arm_id!r} does not encode slot/name {expected_prefix!r} — an id whose "
                "components disagree with its fields cannot be trusted to "
                "identify a vintage"
            )
        if not self.arm_id.endswith(self.spec_hash):
            raise ImmutableArmError(
                f"arm_id {self.arm_id!r} does not carry spec_hash {self.spec_hash!r}: a changed spec "
                "MUST produce a new arm id (Brian ruling 2026-08-29)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "slot": self.slot,
            "name": self.name,
            "spec_hash": self.spec_hash,
            "created_date": self.created_date,
            "supersedes": self.supersedes,
            "bootstrap": self.bootstrap,
            "notes": self.notes,
            "control": self.control,
        }


@dataclass(frozen=True)
class ArmEvent:
    """One append-only lifecycle event. ``record`` is set for ``registered``.

    ``date`` is the day the EVENT happened — the day the row was appended for
    a ``registered``, the day the arm was retired for a ``retired``, the day
    the refit ran for a ``refit``. It is never the recipe's declared
    ``created_date``, which lives on :class:`ArmRecord` and answers a
    different question (`alpha-engine-config-I10948`); see
    :meth:`ArmRegister.register` for what went wrong when the two were the
    same value.
    """

    kind: str
    arm_id: str
    date: str
    reason: str = ""
    record: ArmRecord | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(
                f"unknown arm event kind {self.kind!r}; known: {_EVENT_KINDS}"
            )
        if self.kind == EVENT_REGISTERED and self.record is None:
            raise ValueError("a 'registered' event must carry its ArmRecord")
        if self.record is not None and self.record.arm_id != self.arm_id:
            raise ImmutableArmError(
                f"event arm_id {self.arm_id!r} disagrees with its record's {self.record.arm_id!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "arm_id": self.arm_id,
            "date": self.date,
            "reason": self.reason,
        }
        if self.record is not None:
            payload["record"] = self.record.to_dict()
        return payload


@dataclass(frozen=True)
class ArmState:
    """The fold of the log for one arm."""

    record: ArmRecord
    retired_date: str | None = None
    retired_reason: str = ""
    #: The day the arm's ``registered`` event was FILED — the event date, not
    #: the recipe's declared ``created_date`` (:meth:`ArmRegister.register`).
    #: It is the only thing that answers "was this row in the register on day
    #: D", which is the axis :meth:`exists_at` and every point-in-time read
    #: below are built on (`alpha-engine-config-I10948`, `-I11084`). Optional
    #: only so a hand-built :class:`ArmState` in a test need not state it; the
    #: fold always sets it.
    filed_date: str | None = None

    @property
    def arm_id(self) -> str:
        return self.record.arm_id

    @property
    def active(self) -> bool:
        return self.retired_date is None

    def exists_at(self, as_of: str) -> bool:
        """Was this arm IN THE ARENA on ``as_of``?

        Two dates have to be at or before ``as_of`` for the answer to be yes,
        and they answer different questions
        (:meth:`ArmRegister.register`'s docstring):

        * ``filed_date`` — the row was in the register that day. An arm filed
          later was not there to be scored, ranked or retired.
        * ``record.created_date`` — the arm's out-of-sample clock had started.
          An arm whose recipe declares a later start has NO AGE on this day,
          and :func:`nousergon_lib.arena.window.elapsed_weeks` correctly
          refuses to invent one.

        Either being in the future of ``as_of`` means the same thing: the arm
        was absent, so it has no standing and no retirement question
        (`alpha-engine-config-I11084`).
        """
        if self.record.created_date > as_of:
            return False
        return self.filed_date is None or self.filed_date <= as_of

    def absence_reason(self, as_of: str) -> str:
        """Why :meth:`exists_at` said no — stated, never implied."""
        parts: list[str] = []
        if self.record.created_date > as_of:
            parts.append(f"created {self.record.created_date}")
        if self.filed_date is not None and self.filed_date > as_of:
            parts.append(f"filed {self.filed_date}")
        if not parts:
            return f"present on {as_of}"
        return f"not yet registered on {as_of} ({', '.join(parts)})"

    def age_weeks(self, as_of: str) -> int:
        """FULL weeks elapsed since ``created_date``.

        Elapsed, not inclusive: an arm created 27 days ago is 3 weeks old, not
        4. A grace period is a promise of a real amount of time, and rounding
        it up would retire an arm a day before the promise expires.
        """
        return _elapsed_weeks(self.record.created_date, as_of)

    def in_trailing_scoring_window(self, as_of: str, trailing_cycles: int, cadence_days: int = 7) -> bool:
        """§3: a retired arm is still scored for a trailing window.

        "We retired the wrong one" must be detectable rather than a matter of
        opinion, so retirement stops an arm SERVING and stops it counting
        toward the cap — it does not stop it being measured.
        """
        if self.retired_date is None:
            return True
        return _elapsed_weeks(self.retired_date, as_of) * 7 <= trailing_cycles * cadence_days


class ArmRegister:
    """Append-only register. Every mutator returns a NEW register."""

    def __init__(self, events: Sequence[ArmEvent] | None = None) -> None:
        self._events: tuple[ArmEvent, ...] = tuple(events or ())
        self._states: dict[str, ArmState] = self._fold(self._events)

    # -- construction ---------------------------------------------------

    @staticmethod
    def _fold(events: Sequence[ArmEvent]) -> dict[str, ArmState]:
        states: dict[str, ArmState] = {}
        for event in events:
            if event.kind == EVENT_REGISTERED:
                if event.record is None:  # pragma: no cover -- guarded in ArmEvent
                    raise ImmutableArmError(
                        f"registered event for {event.arm_id} carries no record"
                    )
                existing = states.get(event.arm_id)
                if existing is not None:
                    if existing.record != event.record:
                        raise ImmutableArmError(
                            f"arm {event.arm_id} re-registered with a DIFFERENT record; an "
                            "arm id is permanently bound to one vintage "
                            "(Brian ruling 2026-08-29)"
                        )
                    raise ImmutableArmError(
                        f"arm {event.arm_id} registered twice; the log is append-only but "
                        "not idempotent — a duplicate registration hides which "
                        "created_date the ladder should start from"
                    )
                states[event.arm_id] = ArmState(record=event.record, filed_date=event.date)
            elif event.kind == EVENT_REFIT:
                if event.arm_id not in states:
                    raise ImmutableArmError(
                        f"refit event for unregistered arm {event.arm_id}"
                    )
                # Deliberately no state change: a refit is the recipe doing
                # its job. The event is kept for provenance only.
                continue
            else:  # EVENT_RETIRED
                state = states.get(event.arm_id)
                if state is None:
                    raise ImmutableArmError(
                        f"retirement event for unregistered arm {event.arm_id}"
                    )
                if state.retired_date is not None:
                    raise ImmutableArmError(
                        f"arm {event.arm_id} retired twice ({state.retired_date} then {event.date}); a second "
                        "retirement would silently rewrite retired_date and "
                        "move the §3 trailing scoring window"
                    )
                states[event.arm_id] = ArmState(
                    record=state.record,
                    retired_date=event.date,
                    retired_reason=event.reason,
                    filed_date=state.filed_date,
                )
        return states

    @classmethod
    def from_dicts(cls, payload: Iterable[Mapping[str, Any]]) -> ArmRegister:
        events: list[ArmEvent] = []
        for item in payload:
            record_payload = item.get("record")
            record = ArmRecord(**dict(record_payload)) if record_payload else None
            events.append(
                ArmEvent(
                    kind=item["kind"],
                    arm_id=item["arm_id"],
                    date=item["date"],
                    reason=item.get("reason", ""),
                    record=record,
                )
            )
        return cls(events)

    # -- reads ----------------------------------------------------------

    @property
    def events(self) -> tuple[ArmEvent, ...]:
        return self._events

    def state(self, arm_id: str) -> ArmState:
        try:
            return self._states[arm_id]
        except KeyError as exc:
            raise KeyError(
                f"arm {arm_id!r} is not registered. An arm that writes shadow "
                "artifacts without a register row is never scored and its "
                "data rots unnoticed (champion-challenger-policy.md §3)"
            ) from exc

    def __contains__(self, arm_id: object) -> bool:
        return arm_id in self._states

    def active_arms(self, as_of: str | None = None) -> tuple[str, ...]:
        """Live arms. With ``as_of``, the arms that were live ON that day.

        ``as_of=None`` is "as the register stands now" and is what every
        present-tense caller wants. Passing a past trading day makes the read
        POINT-IN-TIME: an arm whose row was filed after that day, or whose
        recipe declares a later ``created_date``, was not in the arena then
        and is excluded rather than aged (`alpha-engine-config-I11084`).
        """
        return tuple(
            sorted(
                a
                for a, s in self._states.items()
                if s.active and (as_of is None or s.exists_at(as_of))
            )
        )

    def all_arms(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def not_yet_registered(self, as_of: str) -> tuple[str, ...]:
        """Every arm in the log that did NOT yet exist on ``as_of``.

        The complement of the point-in-time reads above, returned so a
        consumer can NAME the arms it excluded. A cycle that scores fewer
        arms than the register holds, with no statement of why, is
        indistinguishable from one that silently dropped them
        (`principles.md` §2.7).
        """
        return tuple(
            sorted(arm for arm, state in self._states.items() if not state.exists_at(as_of))
        )

    def scored_arms(self, as_of: str, trailing_cycles: int) -> tuple[str, ...]:
        """Arms to score on ``as_of``: active plus retired-but-trailing (§3).

        POINT-IN-TIME on both ends. An arm retired more than
        ``trailing_cycles`` ago has left the window; an arm that had not yet
        been registered on ``as_of`` had not entered it. Both are absences
        from the day being graded, and neither is a series the caller owes
        (`alpha-engine-config-I11084`).
        """
        return tuple(
            sorted(
                arm
                for arm, state in self._states.items()
                if state.exists_at(as_of)
                and state.in_trailing_scoring_window(as_of, trailing_cycles)
            )
        )

    # -- appends --------------------------------------------------------

    def register(
        self,
        slot: str,
        name: str,
        spec: Mapping[str, Any],
        created_date: str,
        supersedes: str | None = None,
        bootstrap: bool = False,
        notes: str = "",
        control: bool = False,
        *,
        filed_on: str | None = None,
    ) -> tuple[ArmRegister, ArmRecord]:
        """Append a new vintage. Returns the new register and the record.

        ``created_date`` and ``filed_on`` answer two different questions and
        are never each other's default:

        * ``created_date`` is what the RECIPE declares — the start of the
          arm's out-of-sample clock, which :meth:`ArmState.age_weeks` and
          every grace/promotion rung count from. It may be months before the
          row exists.
        * ``filed_on`` is the trading day the APPEND happens — when this row
          entered the log. It is the only thing that can answer "was this arm
          in the register on day D", and a grader that asks that question of
          ``created_date`` demands an arm on days before it was filed.

        Copying one into the other is `alpha-engine-config-I10948`: every
        `registered` event in the live U and M registers carried
        ``date == record.created_date``, so seven arms appended on one evening
        read as having been registered weeks earlier, and the clause grading
        "every registered arm was scored" demanded them on days whose cycles
        ran before the rows existed.

        ``filed_on=None`` is a DEPRECATED compatibility path for callers
        pinned to a release older than this one: it reproduces the defect
        above (the event takes ``created_date``) and warns. It is not a
        graceful degrade for a caller that could state the day — every caller
        in this package's own tree passes it — and it is removed once the v1
        `crucible-research` / `crucible-predictor` call sites are retired.
        """
        if filed_on is None:
            warnings.warn(
                "ArmRegister.register(filed_on=...) is not set: the appended `registered` "
                "event will take its date from `created_date`, which is the recipe's "
                "declared date and NOT the day the row was filed. Pass the trading day of "
                "the run doing the append. See alpha-engine-config-I10948.",
                DeprecationWarning,
                stacklevel=2,
            )
            filed_on = created_date
        elif not isinstance(filed_on, str) or not filed_on.strip():
            raise ValueError(
                f"filed_on must be a non-empty ISO-8601 trading day; got {filed_on!r}"
            )
        arm_id = derive_arm_id(slot, name, spec)
        record = ArmRecord(
            arm_id=arm_id,
            slot=slot,
            name=name,
            spec_hash=spec_hash(spec),
            created_date=created_date,
            supersedes=supersedes,
            bootstrap=bootstrap,
            notes=notes,
            control=control,
        )
        if supersedes is not None and supersedes not in self._states:
            raise ImmutableArmError(
                f"arm {arm_id} declares supersedes={supersedes!r}, which is not registered; a "
                "lineage pointer to nothing is worse than none"
            )
        # `filed_on`, NEVER `created_date` — see this method's docstring.
        event = ArmEvent(kind=EVENT_REGISTERED, arm_id=arm_id, date=filed_on, record=record)
        return ArmRegister(self._events + (event,)), record

    def refit(self, arm_id: str, refit_date: str, reason: str = "scheduled refit") -> ArmRegister:
        """Record a scheduled refit of an existing recipe.

        Appends provenance and nothing else. The arm id, its spec hash and
        its score series are all unchanged — a refit is the arm executing the
        cadence its own recipe declares, so the series stays continuous
        across it (Brian ruling 2026-08-29).
        """
        state = self.state(arm_id)
        if state.retired_date is not None:
            raise ImmutableArmError(
                f"arm {arm_id} was retired on {state.retired_date}; a retired arm is not refit, it is "
                "only scored for its §3 trailing window"
            )
        event = ArmEvent(kind=EVENT_REFIT, arm_id=arm_id, date=refit_date, reason=reason)
        return ArmRegister(self._events + (event,))

    def refits(self, arm_id: str) -> tuple[str, ...]:
        """Dates on which ``arm_id`` was refit, oldest first."""
        return tuple(
            e.date for e in self._events if e.kind == EVENT_REFIT and e.arm_id == arm_id
        )

    def retire(self, arm_id: str, retired_date: str, reason: str) -> ArmRegister:
        """Append a retirement event. The arm's record itself is untouched."""
        if not reason:
            raise ValueError(
                "retirement requires a reason; an unexplained retirement cannot "
                "be reviewed later (principles.md §2.1)"
            )
        self.state(arm_id)  # raises with guidance if unregistered
        event = ArmEvent(kind=EVENT_RETIRED, arm_id=arm_id, date=retired_date, reason=reason)
        return ArmRegister(self._events + (event,))

    def to_dicts(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]
