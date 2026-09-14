"""Where a gate reading and the ladder are filed. One shape, one place."""

from __future__ import annotations

__all__ = ["LADDER_KEY", "gate_key", "gate_prefix", "ladder_key"]

#: Where the ladder is filed, relative to the gate root. ONE well-known object,
#: rewritten by every reader, because a console renders current state and never
#: owns history (`console-policy` §1). The ladder's history is the dated gate
#: readings under :func:`gate_prefix`, which are never overwritten.
LADDER_KEY = "gates/ladder.json"


def gate_key(gate: str, trading_day: str) -> str:
    """Where one dated reading of ``gate`` is filed."""
    if not gate:
        raise ValueError(
            "gate must be non-empty — a blank gate files every gate's reading at one "
            "empty-segment key, so the last reader silently overwrites the rest."
        )
    if not trading_day:
        raise ValueError(
            "trading_day must be non-empty — the dated readings are the gate's only "
            "history, and an undated one is a second pointer, not a record."
        )
    return f"gates/{gate}/{trading_day}/gate.json"


def gate_prefix(gate: str) -> str:
    """The prefix under which every dated reading of ``gate`` lives.

    ``gate_key(gate, day)`` starts with this for any ``day`` — the last-read
    listing lists this prefix rather than restating the key shape.
    """
    if not gate:
        raise ValueError(
            "gate must be non-empty — a blank gate would list every gate's readings "
            "under one empty-segment prefix, and an empty listing reads as 'no data' "
            "rather than as the caller's own bug."
        )
    return f"gates/{gate}/"


def ladder_key() -> str:
    """:data:`LADDER_KEY`, as a call so a caller never inlines the literal."""
    return LADDER_KEY
