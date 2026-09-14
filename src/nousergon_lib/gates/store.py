"""The store surface a gate reads, and the two guarded reads it takes.

A gate READS; it never runs (`crucible_v2_rebuild_plan_260901.md` §4.1 rule 1,
carried into `data_collection_plan_260914.md` §4.1 rule 1). Everything it needs
is therefore behind two operations — fetch one object, list a prefix — and this
module is the whole of the store contract the engine depends on.

**Why a Protocol rather than an import of `crucible.store.Store`.** The engine
now has two adopters (`crucible`, `nousergon-data`) whose stores are different
classes with different backends. Binding the engine to one of them would make
the second adopter depend on the first repository, which is the coupling the
lift exists to remove (`shared-code-policy`, second adoption). Any object with
``get_bytes`` and ``list_keys`` is a store here; `crucible.store.Store` and a
`boto3`-backed reader in `nousergon-data` both satisfy it without either
knowing about the other.

**Absence is an answer; an access failure is not.** Both readers below return a
record carrying three distinguishable outcomes — read, absent, and *could not
read* — because collapsing the last two is precisely how a gate publishes green
over a bucket it was denied. `absent` feeds a clause's UNMET; `problem` with
`access_problem` feeds UNMEASURABLE.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DocumentRead",
    "GateStore",
    "KeysRead",
    "list_store_keys",
    "read_store_document",
]


@runtime_checkable
class GateStore(Protocol):
    """The two reads a gate takes. Deliberately read-only.

    A gate that could write through the same handle it grades with could
    satisfy its own clause, which is the one property `read`'s whole design
    forbids. Publishing the reading is a separate, explicit write by the CLI,
    never by the engine.
    """

    def get_bytes(self, key: str) -> bytes:
        """The object's bytes, or raise (``FileNotFoundError`` for absence)."""
        ...

    def list_keys(self, prefix: str = "") -> Iterable[str]:
        """Every key under ``prefix``."""
        ...


#: Exception types that mean "this key is not there", as opposed to "we could
#: not find out". Closed, and small on purpose: anything not named here is an
#: access failure, which is the safe direction — a new error type we have not
#: classified renders UNMEASURABLE (loud) rather than UNMET (a finding filed
#: against an innocent producer).
_ABSENCE_ERRORS: tuple[type, ...] = (FileNotFoundError, KeyError)


def _is_absence(exc: BaseException) -> bool:
    if isinstance(exc, _ABSENCE_ERRORS):
        return True
    # botocore raises ClientError for everything; the code is the only
    # discriminator, and it is reachable without importing botocore here.
    code = getattr(getattr(exc, "response", None), "get", lambda *_: None)("Error") or {}
    return str(code.get("Code") or "") in {"NoSuchKey", "404", "NotFound"}


@dataclass(frozen=True)
class DocumentRead:
    """One JSON object out of the store, or the reason it is not here.

    Exactly one of the three fields is meaningful, and which one is the whole
    point: ``document`` (read it), ``absent`` (it is not there — a clause's
    ANSWER), ``problem`` (we could not find out — never an answer).
    """

    source: str
    document: dict[str, Any] | None = None
    absent: bool = False
    problem: str | None = None
    #: True when ``problem`` is an ACCESS failure (denial, throttle, transport)
    #: rather than a malformed payload. Both are UNMEASURABLE; the flag is what
    #: lets a clause detail say which side of the boundary the fault is on.
    access_problem: bool = False


def read_store_document(store: GateStore, key: str) -> DocumentRead:
    """``key`` as a JSON object, with absence and access failure kept apart."""
    try:
        raw = store.get_bytes(key)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed reading, see below
        # A deliberate catch, not a swallow (`AGENTS.md` fail-loud rule): the
        # failure mode caught is "one object could not be fetched"; the primary
        # deliverable — a gate reading carrying every other clause — survives;
        # and the recording surface is this record's own `problem`, which the
        # clause renders UNMEASURABLE and the ladder counts.
        if _is_absence(exc):
            return DocumentRead(source=key, absent=True)
        return DocumentRead(
            source=key,
            problem=f"{type(exc).__name__}: {exc}",
            access_problem=True,
        )
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a malformed payload is a reading, not a crash
        return DocumentRead(
            source=key,
            problem=f"not valid JSON ({type(exc).__name__}: {exc})",
        )
    if not isinstance(parsed, dict):
        return DocumentRead(
            source=key,
            problem=f"expected a JSON object, got {type(parsed).__name__}",
        )
    return DocumentRead(source=key, document=parsed)


@dataclass(frozen=True)
class KeysRead:
    """The keys under a prefix, or the reason the listing did not happen.

    ``keys == []`` and ``problem is not None`` are different facts. An empty
    listing says the producer has written nothing; a failed listing says we do
    not know, and a gate that grades the second as the first reports a finding
    against a producer whose output it never looked at.
    """

    prefix: str
    keys: list[str] | None = None
    problem: str | None = None
    access_problem: bool = False


def list_store_keys(store: GateStore, prefix: str) -> KeysRead:
    """Every key under ``prefix``, with a failed listing kept distinct."""
    try:
        return KeysRead(prefix=prefix, keys=sorted(store.list_keys(prefix)))
    except Exception as exc:  # noqa: BLE001 - see read_store_document
        return KeysRead(
            prefix=prefix,
            problem=f"{type(exc).__name__}: {exc}",
            access_problem=True,
        )
