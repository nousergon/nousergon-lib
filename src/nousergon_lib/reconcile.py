"""
Quantitative parity reconciliation for ``dict[str, DataFrame]`` price stores.

The SOTA observation substrate for contract-safe data-tier migrations: when a
consumer is being moved from one price source to another (e.g. the
``predictor/price_cache_slim/`` parquet tier -> the ArcticDB universe lib,
Wave 4 of the predictor/ S3 namespace rationalization), the cutover decision
must be **data-driven**, not eyeballed. This module turns "do the two sources
agree?" into a single auditable :class:`ParityReport` with:

- ticker-set symmetric difference (coverage),
- per-ticker row-count delta (shape),
- max absolute value delta over the *overlapping* dates/columns (fidelity),
- a binary ``passed`` keyed on an explicit epsilon.

:meth:`ParityReport.as_metrics` is JSON-able so the same object that decides a
gate can be emitted to the metrics surface and observed over a window before
the producer is retired. One implementation, reused by every consumer
migration PR (data macro-breadth, backtester exit-timing) and the final
deletion gate — no per-repo re-implementation of "are these prices equal".

Pure pandas; importable without the ``[arcticdb]`` extra (pandas is imported
lazily inside the function, mirroring the arcticdb module's contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, cast

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass(frozen=True)
class ParityReport:
    """Outcome of comparing two ticker -> DataFrame price stores.

    ``passed`` is the gate signal. ``as_metrics()`` is what you log over the
    observation window. ``summary()`` is the one-line operator string.
    """

    only_in_a: frozenset
    only_in_b: frozenset
    common: frozenset
    rowcount_deltas: Mapping[str, int]  # ticker -> len(a)-len(b), nonzero only
    max_abs_value_delta: float
    worst_cell: Optional[Tuple[str, str, str]]  # (ticker, column, iso-date)
    n_cells_over_epsilon: int
    n_cells_compared: int
    epsilon: float
    value_cols: Tuple[str, ...]
    require_ticker_match: bool
    require_rowcount_match: bool

    @property
    def ticker_sets_match(self) -> bool:
        return not self.only_in_a and not self.only_in_b

    @property
    def rowcounts_match(self) -> bool:
        return not self.rowcount_deltas

    @property
    def passed(self) -> bool:
        ok = self.n_cells_over_epsilon == 0
        if self.require_ticker_match:
            ok = ok and self.ticker_sets_match
        if self.require_rowcount_match:
            ok = ok and self.rowcounts_match
        return ok

    def summary(self) -> str:
        return (
            "parity {verdict}: common={c} only_a={oa} only_b={ob} "
            "rowcount_mismatch_tickers={rc} max_abs_delta={mad:.3e} "
            "cells_over_eps={oe}/{tot} (eps={eps:.1e})"
        ).format(
            verdict="PASS" if self.passed else "FAIL",
            c=len(self.common),
            oa=len(self.only_in_a),
            ob=len(self.only_in_b),
            rc=len(self.rowcount_deltas),
            mad=self.max_abs_value_delta,
            oe=self.n_cells_over_epsilon,
            tot=self.n_cells_compared,
            eps=self.epsilon,
        )

    def as_metrics(self) -> dict:
        """JSON-able metric dict for the observation gate / metrics surface."""
        return {
            "passed": self.passed,
            "ticker_sets_match": self.ticker_sets_match,
            "rowcounts_match": self.rowcounts_match,
            "n_common": len(self.common),
            "n_only_in_a": len(self.only_in_a),
            "n_only_in_b": len(self.only_in_b),
            "only_in_a": sorted(self.only_in_a),
            "only_in_b": sorted(self.only_in_b),
            "n_rowcount_mismatch_tickers": len(self.rowcount_deltas),
            "rowcount_deltas": dict(sorted(self.rowcount_deltas.items())),
            "max_abs_value_delta": self.max_abs_value_delta,
            "worst_cell": list(self.worst_cell) if self.worst_cell else None,
            "n_cells_over_epsilon": self.n_cells_over_epsilon,
            "n_cells_compared": self.n_cells_compared,
            "epsilon": self.epsilon,
            "value_cols": list(self.value_cols),
        }


def reconcile_frame_dicts(
    a: Mapping[str, "pd.DataFrame"],
    b: Mapping[str, "pd.DataFrame"],
    *,
    value_cols: Sequence[str] = ("Close",),
    epsilon: float = 1e-6,
    require_ticker_match: bool = True,
    require_rowcount_match: bool = False,
) -> ParityReport:
    """Compare two ticker -> DataFrame price stores into a :class:`ParityReport`.

    Value fidelity is measured on the **intersection of dates** per common
    ticker (an inner join on the DatetimeIndex), over the ``value_cols``
    present in *both* frames. Row-count deltas are reported separately and,
    by default, do **not** fail the gate: a slim-cache tail slice and an
    ArcticDB ``date_range`` read legitimately differ by a few boundary rows
    while being bit-identical on the overlap — the migration question is
    "do they agree where they overlap", not "are the artifacts shaped
    identically". Set ``require_rowcount_match=True`` for stricter contexts.

    Args:
        a, b: ticker -> pandas DataFrame (DatetimeIndex). Conventionally
            ``a`` = incumbent source, ``b`` = candidate source.
        value_cols: columns compared for numeric equality.
        epsilon: absolute tolerance; a cell counts as a mismatch when
            ``abs(a - b) > epsilon``.
        require_ticker_match: include ticker-set symmetry in ``passed``.
        require_rowcount_match: include row-count equality in ``passed``.
    """
    import pandas as pd  # lazy: keeps module importable without the extra

    keys_a = set(a)
    keys_b = set(b)
    common = keys_a & keys_b

    rowcount_deltas: dict = {}
    max_abs = 0.0
    worst_cell: Optional[Tuple[str, str, str]] = None
    n_over = 0
    n_compared = 0
    cols = tuple(value_cols)

    for ticker in sorted(common):
        fa = a[ticker]
        fb = b[ticker]

        delta = len(fa) - len(fb)
        if delta != 0:
            rowcount_deltas[ticker] = delta

        idx = fa.index.intersection(fb.index)
        if len(idx) == 0:
            continue
        for col in cols:
            if col not in fa.columns or col not in fb.columns:
                continue
            # fa.loc[idx, col] / fb.loc[idx, col] always select a single
            # column across a row index here (col is one column label from
            # `cols`), so to_numeric always receives/returns a Series; the
            # cast narrows pyright away from to_numeric's broader
            # scalar/array-input overload union (its signature has no
            # per-input-type overloads to infer from).
            sa = cast("pd.Series", pd.to_numeric(fa.loc[idx, col], errors="coerce"))
            sb = cast("pd.Series", pd.to_numeric(fb.loc[idx, col], errors="coerce"))
            diff = (sa - sb).abs()
            # NaN on either side -> treat as comparable only where both
            # present; an asymmetric NaN is a real mismatch.
            both_nan = sa.isna() & sb.isna()
            one_nan = (sa.isna() ^ sb.isna())
            diff = diff.where(~both_nan)
            n_compared += int((~both_nan).sum())

            over = (diff > epsilon) | one_nan
            n_over += int(over.sum())

            valid = diff.dropna()
            if not valid.empty:
                cell_max = float(valid.max())
                if cell_max > max_abs:
                    max_abs = cell_max
                    worst_dt = valid.idxmax()
                    worst_cell = (
                        ticker,
                        col,
                        getattr(worst_dt, "isoformat", lambda: str(worst_dt))(),
                    )

    return ParityReport(
        only_in_a=frozenset(keys_a - keys_b),
        only_in_b=frozenset(keys_b - keys_a),
        common=frozenset(common),
        rowcount_deltas=rowcount_deltas,
        max_abs_value_delta=max_abs,
        worst_cell=worst_cell,
        n_cells_over_epsilon=n_over,
        n_cells_compared=n_compared,
        epsilon=epsilon,
        value_cols=cols,
        require_ticker_match=require_ticker_match,
        require_rowcount_match=require_rowcount_match,
    )
