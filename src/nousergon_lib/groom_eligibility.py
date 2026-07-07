"""Groom eligibility + demand-driven slot-dispatch decision (config#1933).

Single source of truth for the label semantics BOTH groom consumers apply:

- ``alpha-engine-config/scripts/groom_driver.py`` — on-box enumeration (its
  inline constants are contract-tested against this module; the spot box does
  not install nousergon-lib at runtime).
- ``nousergon-data`` ``scheduled-groom-dispatcher`` Lambda — pre-boot
  enumerate-then-decide: at each daily slot, count actionable issues per
  complexity tier and decide launch/skip/bundle BEFORE any spot spend.

The 2026-07-07 nousergon-data-PR683 incident (dispatcher silently downgraded
``gated-reverify`` to ``mid-only`` because its private filter set drifted from
the driver's) is the bug class this module closes: one constant set, two
consumers, contract tests on both sides.

Everything here is PURE — no I/O, no GitHub calls. Consumers fetch issues
their own way (gh CLI on-box, urllib in the Lambda) and pass plain label
lists / counts in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

# ── Label semantics (mirrors groom_driver.py — contract-tested both sides) ──

LOW_LABEL = "complexity:low"
MID_LABEL = "complexity:mid"
HIGH_LABEL = "complexity:high"
ULTRA_LABEL = "complexity:ultra"

#: Issues carrying any of these never enter ANY groom queue.
BASE_EXCLUDE_LABELS = frozenset({"groom-digest", "in-progress", "do-not-groom", ULTRA_LABEL})

#: config#1805 gate exclusion: HARD — no automated re-entry path exists.
GATE_HARD_EXCLUDE_LABELS = frozenset({"gate:operator", "gate:decision", "gate:device"})
#: SOFT — excluded unless the issue also carries ``gate-due``.
GATE_SOFT_EXCLUDE_LABELS = frozenset({"gate:date", "gate:data", "gate:live-run", "gate:dependency"})
GATE_DUE_LABEL = "gate-due"

#: Tier order, cheapest first. Unlabeled issues default to "mid".
TIERS = ("low", "mid", "high")

#: Tier → model that works it. A bundled run uses the model of the HIGHEST
#: tier actually present in its queue; high-tier issues never run below Opus.
TIER_MODELS = {
    "low": "claude-haiku-4-5",
    "mid": "claude-sonnet-5",
    "high": "claude-opus-4-8",
}

#: Every issue_filter value the driver accepts. Single-tier forms keep the
#: legacy names; bundled forms are "+"-joined highest-first (config#1933).
SINGLE_TIER_FILTERS = {"low": "low-only", "mid": "mid-only", "high": "high-only"}
BUNDLED_FILTERS = frozenset({"mid+low", "high+mid", "high+low", "high+mid+low"})
VALID_ISSUE_FILTERS = frozenset(
    {"default", "gated-reverify", *SINGLE_TIER_FILTERS.values(), *BUNDLED_FILTERS}
)

#: config#1933 dispatch parameters.
DEFAULT_FLOOR = 8            # min queue size worth a spot boot (= driver floor)
DEFAULT_MAX_WAIT_HOURS = 72  # anti-starvation escape valve (ARCH §66)


def filter_tiers(issue_filter: str) -> tuple[str, ...]:
    """Tiers a driver run with ``issue_filter`` works, cheapest first.

    ``default`` is the historical alias for ``mid-only``; ``gated-reverify``
    works no complexity queue (returns ()).
    """
    if issue_filter == "default":
        return ("mid",)
    if issue_filter == "gated-reverify":
        return ()
    for tier, name in SINGLE_TIER_FILTERS.items():
        if issue_filter == name:
            return (tier,)
    if issue_filter in BUNDLED_FILTERS:
        return tuple(sorted(issue_filter.split("+"), key=TIERS.index))
    raise ValueError(f"unknown issue_filter: {issue_filter!r}")


def filter_for_tiers(tiers: Iterable[str]) -> str:
    """The canonical issue_filter string for a set of tiers (highest-first)."""
    ordered = sorted(set(tiers), key=TIERS.index, reverse=True)
    if not ordered:
        raise ValueError("no tiers")
    if len(ordered) == 1:
        return SINGLE_TIER_FILTERS[ordered[0]]
    name = "+".join(ordered)
    if name not in BUNDLED_FILTERS:
        raise ValueError(f"unsupported tier bundle: {name}")
    return name


def tier_of(labels: Iterable[str]) -> Optional[str]:
    """Complexity tier for a label set, or None if excluded from grooming.

    Unlabeled ⇒ "mid" (the standing default). ``complexity:ultra`` and the
    base excludes ⇒ None.
    """
    label_set = set(labels)
    if label_set & BASE_EXCLUDE_LABELS:
        return None
    if HIGH_LABEL in label_set:
        return "high"
    if LOW_LABEL in label_set:
        return "low"
    return "mid"


def is_gate_excluded(labels: Iterable[str]) -> bool:
    """config#1805 gate exclusion: hard gates always; soft unless gate-due."""
    label_set = set(labels)
    if label_set & GATE_HARD_EXCLUDE_LABELS:
        return True
    return bool(label_set & GATE_SOFT_EXCLUDE_LABELS) and GATE_DUE_LABEL not in label_set


def is_actionable(labels: Iterable[str]) -> Optional[str]:
    """The tier this issue is actionable in, or None (excluded/gated)."""
    tier = tier_of(labels)
    if tier is None or is_gate_excluded(labels):
        return None
    return tier


@dataclass(frozen=True)
class SlotDecision:
    """Outcome of a slot's enumerate-then-decide (config#1933)."""

    launch: bool
    tiers: tuple[str, ...]      # tiers in the queue, cheapest first ((), if skip)
    issue_filter: str           # driver filter to export ("" if skip)
    model: str                  # model for the run ("" if skip)
    reason: str                 # human-readable, rendered in the decision record

    def as_record(self) -> dict:
        return {
            "launch": self.launch, "tiers": list(self.tiers),
            "issue_filter": self.issue_filter, "model": self.model,
            "reason": self.reason,
        }


def decide_slot(
    slot_tier: str,
    counts: Mapping[str, int],
    oldest_wait_hours: Mapping[str, float] | None = None,
    has_actionable_p0: bool = False,
    *,
    floor: int = DEFAULT_FLOOR,
    max_wait_hours: float = DEFAULT_MAX_WAIT_HOURS,
) -> SlotDecision:
    """Decide what (if anything) this slot launches.

    Rules (config#1933, Brian-ratified 2026-07-07):
    - The slot considers its OWN tier plus every LOWER tier whose own count
      is below ``floor`` (those were/will be skipped at their own slot —
      they bundle upward; higher tiers never bundle down).
    - Launch iff the combined queue >= ``floor``, OR the escape valve fires:
      an actionable P0 exists, or any considered tier's oldest actionable
      issue has waited >= ``max_wait_hours`` (anti-starvation, ARCH §66).
    - The run's model = highest tier actually PRESENT in the queue — a
      bundle of only low+mid issues runs on Sonnet even at the Opus slot;
      high-tier issues never run below Opus (COMPLEXITY GUARDRAIL).
    """
    if slot_tier not in TIERS:
        raise ValueError(f"unknown slot tier: {slot_tier!r}")
    oldest_wait_hours = oldest_wait_hours or {}
    slot_idx = TIERS.index(slot_tier)
    considered = [slot_tier] + [t for t in TIERS[:slot_idx] if counts.get(t, 0) < floor]
    present = sorted(
        (t for t in considered if counts.get(t, 0) > 0), key=TIERS.index,
    )
    total = sum(counts.get(t, 0) for t in present)
    if total == 0:
        return SlotDecision(False, (), "", "", f"queue empty at {slot_tier} slot")

    overdue = [t for t in present if oldest_wait_hours.get(t, 0.0) >= max_wait_hours]
    if total >= floor:
        reason = f"{total} actionable across {'+'.join(present)} >= floor {floor}"
    elif has_actionable_p0:
        reason = f"escape valve: actionable P0 present (queue {total} < floor {floor})"
    elif overdue:
        reason = (f"escape valve: {'+'.join(overdue)} oldest waited >= "
                  f"{max_wait_hours:g}h (queue {total} < floor {floor})")
    else:
        return SlotDecision(
            False, tuple(present), "", "",
            f"queue {total} < floor {floor}, no P0, none waited {max_wait_hours:g}h — deferred upward",
        )
    model_tier = present[-1]  # highest present
    return SlotDecision(
        True, tuple(present), filter_for_tiers(present), TIER_MODELS[model_tier], reason,
    )
