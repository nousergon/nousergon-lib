"""nousergon_lib.quant.horizons — the single source of truth for evaluation horizons.

WHY THIS EXISTS
---------------
The evaluation/label horizon (how many trading days forward an outcome is
scored over) was, until this module, encoded in scattered **wide-suffixed
column names** across every producer and consumer in the fleet:
``beat_spy_10d``, ``spy_21d_return``, ``return_5d``, ``log_alpha_21d``, ….

That encoding IS a bug class (config#1456 root cause, EPIC config#1483):

  * Changing the horizon requires a fleet-wide rename across N repos.
  * An INCOMPLETE rename **silently starves** consumers — a reader of a
    now-dead column gets ``None``/stale rows with no error (exactly what bit
    the fleet 2026-05-09 → June: the 10d/30d consumers ran dark for months).

This is the second horizon-driven fleet cutover, so per the standing "lift the
invariant to a chokepoint after the second recurrence" rule the horizon becomes
a **parameter**, not a schema fact. This module is that chokepoint: the one
place that (a) pins the canonical label horizon, (b) enumerates the diagnostic
horizons, and (c) maps a horizon → its wide-column names. Consumers import from
here instead of hardcoding ``_5d``/``_21d`` literals.

CRITICAL NUANCE — the canonical label is NOT an interchangeable scalar
----------------------------------------------------------------------
``PRIMARY_HORIZON`` (21 trading days, log-domain, market-relative,
sector-neutral alpha) is *THE* canonical label the whole system targets — not
one value in an interchangeable set. Diagnostic horizons (e.g. 5d) are
observability only and MUST NOT be treated as alpha-equivalent to the primary.
:class:`HorizonPolicy` keeps the primary first-class + **fail-loud on absence**
(:meth:`HorizonPolicy.require_primary_present`), while diagnostic horizons are
graceful-empty (a consumer filtering for an unproduced diagnostic horizon gets
no rows, not an error). This prevents someone setting ``HORIZON=7`` and silently
producing a non-canonical "alpha".

This module is pure stdlib (no numpy/pandas) — importable without any extra.

Absorbs the embryonic ``_SHORT_OUTCOME`` / ``_LONG_OUTCOME`` /
``_RESOLVED_OUTCOME`` / ``_HORIZON_BLEND`` / ``_SKILL_TARGET`` constants that
lived in ``crucible-backtester/optimizer/weight_optimizer.py`` — that module was
the sole place the parameterization existed, which is itself evidence this is
the right chokepoint (EPIC config#1483 Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

__all__ = [
    "PRIMARY_HORIZON",
    "DIAGNOSTIC_HORIZONS",
    "LabelDefinition",
    "OutcomeColumns",
    "HorizonPolicy",
    "PrimaryHorizonMissing",
    "DEFAULT_POLICY",
    "outcome_columns",
    "canonical_label",
    "is_primary",
    "all_horizons",
    "skill_target_column",
]

# ── Canonical defaults (the ratified HorizonPolicy, config#1483) ──────────────
# 21 trading days, log-domain, market-relative, sector-neutral alpha. Changing
# this is a system-wide decision, never a local override.
PRIMARY_HORIZON: int = 21
# Secondary/diagnostic horizons — observability only, NOT alpha-equivalent to
# the primary. A tuple (immutable) so it can't be mutated in place by a caller.
DIAGNOSTIC_HORIZONS: tuple[int, ...] = (5,)


class PrimaryHorizonMissing(RuntimeError):
    """The canonical primary horizon is absent from a produced/resolved set.

    Raised (never silently degraded) so a producer or consumer that fails to
    emit/resolve the canonical label surfaces at the earliest callsite — the
    no-silent-fails discipline. Diagnostic horizons do NOT raise on absence
    (they are graceful-empty by design); only the primary is fail-loud.
    """


@dataclass(frozen=True)
class LabelDefinition:
    """Pins WHAT the canonical label is, so a bare horizon integer can never
    masquerade as the canonical alpha. Matches the ratified ``label_definition``
    config block (config#1483).
    """

    domain: str = "log"          # log-return domain (not simple/arithmetic)
    relative_to: str = "spy"     # market-relative benchmark
    neutralization: str = "sector"  # sector-neutralized cross-section

    def as_dict(self) -> dict[str, str]:
        return {
            "domain": self.domain,
            "relative_to": self.relative_to,
            "neutralization": self.neutralization,
        }


@dataclass(frozen=True)
class OutcomeColumns:
    """The wide-suffixed ``score_performance`` column names for one horizon.

    This is the naming chokepoint: the mapping horizon → column name lives here
    and nowhere else, so a rename is a one-line change and consumers stop
    hardcoding ``beat_spy_21d`` string literals. Field names mirror the
    long-format ``outcome_record`` contract (``nousergon_lib.contracts``) so the
    Phase-2 dual-write is an unambiguous per-field copy.

    Ground truth verified against the live ``score_performance`` table
    (research.db, 2026-06-27): ``price_{h}d``, ``return_{h}d``,
    ``spy_{h}d_return``, ``beat_spy_{h}d``, ``eval_date_{h}d``,
    ``log_alpha_{h}d`` (``log_alpha`` populated for the primary horizon only).
    """

    horizon_days: int
    price: str
    stock_return: str
    spy_return: str
    beat_spy: str
    eval_date: str
    log_alpha: str


def outcome_columns(horizon_days: int) -> OutcomeColumns:
    """Return the wide-column names for ``horizon_days`` (policy-independent).

    The single source of truth for the ``score_performance`` naming convention.
    Raises ``ValueError`` on a non-positive horizon (a horizon < 1 trading day
    is nonsensical and almost always a bug in the caller).
    """
    h = int(horizon_days)
    if h < 1:
        raise ValueError(f"horizon_days must be >= 1 trading day, got {horizon_days!r}")
    return OutcomeColumns(
        horizon_days=h,
        price=f"price_{h}d",
        stock_return=f"return_{h}d",
        spy_return=f"spy_{h}d_return",
        beat_spy=f"beat_spy_{h}d",
        eval_date=f"eval_date_{h}d",
        log_alpha=f"log_alpha_{h}d",
    )


@dataclass(frozen=True)
class HorizonPolicy:
    """Config-driven horizon policy (the ratified ``HorizonPolicy``, config#1483).

    The default instance (:data:`DEFAULT_POLICY`) carries the ratified fleet
    defaults; an experiment/config may build an override via
    :meth:`from_mapping`. Keeps the primary label first-class + fail-loud and the
    diagnostic set graceful-empty.
    """

    primary_horizon: int = PRIMARY_HORIZON
    diagnostic_horizons: tuple[int, ...] = DIAGNOSTIC_HORIZONS
    label: LabelDefinition = field(default_factory=LabelDefinition)

    def __post_init__(self) -> None:
        if int(self.primary_horizon) < 1:
            raise ValueError(
                f"primary_horizon must be >= 1 trading day, got {self.primary_horizon!r}"
            )
        # Normalize diagnostics to a sorted, de-duplicated tuple of ints and
        # enforce the two invariants that keep the canonical label safe:
        # (1) every diagnostic is a valid horizon; (2) the primary is NOT also
        # listed as a diagnostic (it would be double-counted / mis-typed).
        diag = tuple(sorted({int(h) for h in self.diagnostic_horizons}))
        for h in diag:
            if h < 1:
                raise ValueError(f"diagnostic horizon must be >= 1, got {h!r}")
        if int(self.primary_horizon) in diag:
            raise ValueError(
                f"primary_horizon {self.primary_horizon} must not also appear in "
                f"diagnostic_horizons {diag} — the canonical label is not a diagnostic"
            )
        # frozen dataclass: bypass the setattr guard to store normalized values.
        object.__setattr__(self, "primary_horizon", int(self.primary_horizon))
        object.__setattr__(self, "diagnostic_horizons", diag)

    @property
    def all_horizons(self) -> tuple[int, ...]:
        """Primary first, then diagnostics (ascending)."""
        return (self.primary_horizon, *self.diagnostic_horizons)

    def is_primary(self, horizon_days: int) -> bool:
        return int(horizon_days) == self.primary_horizon

    def outcome_columns(self, horizon_days: int) -> OutcomeColumns:
        return outcome_columns(horizon_days)

    def skill_target_column(self, horizon_days: int) -> str:
        """The continuous fit-target column for a horizon.

        Primary horizon → ``log_alpha_{h}d`` (the canonical market-relative
        alpha); diagnostic horizons → ``return_{h}d`` (raw stock return, since a
        non-canonical horizon has no canonical alpha). Reproduces the backtester's
        embryonic ``_SKILL_TARGET`` map exactly:
        ``{beat_spy_5d: return_5d, beat_spy_21d: log_alpha_21d}``.
        """
        cols = outcome_columns(horizon_days)
        return cols.log_alpha if self.is_primary(horizon_days) else cols.stock_return

    def resolved_gate_column(self) -> str:
        """The column whose non-null-ness marks a resolved (canonical) outcome —
        the starvation gate. Absorbs the backtester's ``_RESOLVED_OUTCOME``
        (= the primary horizon's ``beat_spy`` column)."""
        return outcome_columns(self.primary_horizon).beat_spy

    def require_primary_present(self, resolved_horizons: Any) -> None:
        """Fail-loud if the canonical primary horizon is absent from a produced/
        resolved horizon set. Diagnostic absence is tolerated (graceful-empty);
        only the primary raises :class:`PrimaryHorizonMissing`.
        """
        present = {int(h) for h in resolved_horizons}
        if self.primary_horizon not in present:
            raise PrimaryHorizonMissing(
                f"canonical primary horizon {self.primary_horizon}d absent from "
                f"resolved horizons {sorted(present)} — the canonical label is "
                f"missing (this is a producer starvation bug, not a diagnostic gap)"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HorizonPolicy":
        """Build a policy from a config mapping (the ratified YAML shape)::

            primary_horizon: 21
            diagnostic_horizons: [5]
            label_definition:
              domain: log
              relative_to: spy
              neutralization: sector

        Unknown top-level keys raise (a typo'd key in a live config that
        silently no-ops is the exact failure mode this EPIC exists to kill).
        """
        known = {"primary_horizon", "diagnostic_horizons", "label_definition"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown HorizonPolicy config key(s): {sorted(unknown)}; "
                f"known keys: {sorted(known)}"
            )
        label_data = data.get("label_definition")
        if label_data is None:
            label = LabelDefinition()
        else:
            label_unknown = set(label_data) - {"domain", "relative_to", "neutralization"}
            if label_unknown:
                raise ValueError(
                    f"unknown label_definition key(s): {sorted(label_unknown)}"
                )
            label = LabelDefinition(**label_data)
        return cls(
            primary_horizon=int(data.get("primary_horizon", PRIMARY_HORIZON)),
            diagnostic_horizons=tuple(data.get("diagnostic_horizons", DIAGNOSTIC_HORIZONS)),
            label=label,
        )

    def as_dict(self) -> dict[str, Any]:
        """Round-trippable with :meth:`from_mapping`."""
        return {
            "primary_horizon": self.primary_horizon,
            "diagnostic_horizons": list(self.diagnostic_horizons),
            "label_definition": self.label.as_dict(),
        }

    def with_overrides(self, **kwargs: Any) -> "HorizonPolicy":
        """Return a copy with fields replaced (re-runs validation)."""
        return replace(self, **kwargs)


# The ratified fleet-default policy — the 90% import for consumers that just want
# "the horizons" without threading a config object.
DEFAULT_POLICY = HorizonPolicy()


# ── Module-level convenience bound to DEFAULT_POLICY ──────────────────────────
def canonical_label() -> LabelDefinition:
    """The canonical label definition (log-domain, SPY-relative, sector-neutral)."""
    return DEFAULT_POLICY.label


def is_primary(horizon_days: int) -> bool:
    """True iff ``horizon_days`` is the canonical primary horizon (default policy)."""
    return DEFAULT_POLICY.is_primary(horizon_days)


def all_horizons() -> tuple[int, ...]:
    """Primary + diagnostic horizons under the default policy (primary first)."""
    return DEFAULT_POLICY.all_horizons


def skill_target_column(horizon_days: int) -> str:
    """Continuous fit-target column for ``horizon_days`` under the default policy."""
    return DEFAULT_POLICY.skill_target_column(horizon_days)
