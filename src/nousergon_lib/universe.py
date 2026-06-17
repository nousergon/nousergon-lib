"""Universe membership filtering — canonical predicate for the
S&P 500+400 constituent check shared by research's incumbent-exit
evaluator (alpha-engine-research#41) and executor's buy-candidate
filter (alpha-engine#77).

Design context (2026-04-20): the TSM/ASML incident shipped as a
two-layer fix with two independent implementations of the same
predicate — "is ticker in current universe?". Layer 1 drops non-S&P
incumbents at the population-exit stage; Layer 2 drops non-universe
buy_candidates at the executor signal reader. The two layers sourced
the universe from different inputs (Research from in-memory
``scanner_universe`` derived from ``constituents.json``; Executor
from ArcticDB library symbols). Independent implementations risk
silent divergence on universe drift.

This module exposes the membership-test primitive only. It stays
IO-agnostic: callers supply the universe set however their context
sources it (``constituents.json``, ArcticDB ``list_symbols()``,
fixture for tests). The lib makes the SAME predicate available to
both layers; behavioral context (exit-event emission, log shape,
caller-side metadata) stays at the call site. Divergence becomes
impossible by construction: both consumers go through one code path.
"""

from __future__ import annotations

from typing import AbstractSet, Callable, Iterable, TypeVar

T = TypeVar("T")


def _default_key(item: object) -> str | None:
    """Default ticker extractor — handles plain ``str`` and ``{"ticker": str}`` dicts.

    Returns ``None`` for items that don't look like either shape; the
    caller's partition logic treats ``None`` as "not in universe" and
    routes the item to ``dropped``.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        ticker = item.get("ticker")
        return ticker if isinstance(ticker, str) else None
    return None


def filter_to_universe(
    items: Iterable[T],
    universe: AbstractSet[str],
    *,
    key: Callable[[T], str | None] | None = None,
) -> tuple[list[T], list[T]]:
    """Partition ``items`` into ``(kept, dropped)`` by universe membership.

    ``items`` may be an iterable of ticker strings, dicts with a
    ``ticker`` field, or any other type when ``key`` is supplied.
    The extracted ticker is checked against ``universe``; items whose
    key is ``None``, empty, or returns a value not in ``universe`` go
    to ``dropped``.

    Returns the partition without emitting logs or exit events —
    callers apply their own context-specific behavior to the dropped
    list (Research emits ``UNIVERSE_DROP`` exit events; Executor logs
    a warning).

    The universe is a read-only ``AbstractSet[str]`` so either
    ``set`` or ``frozenset`` is accepted. Membership is O(1) per item.
    """
    extract = key or _default_key
    kept: list[T] = []
    dropped: list[T] = []
    for item in items:
        ticker = extract(item)
        if ticker and ticker in universe:
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped


def in_universe(ticker: str | None, universe: AbstractSet[str]) -> bool:
    """Predicate — ``True`` iff ``ticker`` is a non-empty string in ``universe``."""
    return isinstance(ticker, str) and bool(ticker) and ticker in universe
