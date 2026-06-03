"""Quantitative portfolio analytics — pure, front-end- and data-source-agnostic.

The shared institutional-analytics engine consumed across the fleet (predictor,
backtester, robodashboard). Every module is dependency-light and unit-testable in
isolation, and *describes/measures* a portfolio (performance, risk, attribution)
with **no advisory logic** — it sits on the "analytics, not advice" side of the
line.

Modules (import the submodule you need — the package keeps no eager imports so the
stdlib-only modules stay importable without numpy):

  - ``factor_risk``    — Σ=B·F·Bᵀ+D ex-ante risk + tracking error (**needs numpy**;
                          install ``alpha-engine-lib[quant]``)
  - ``risk_measures``  — parametric + historical VaR / CVaR (stdlib)
  - ``riskstats``      — volatility, Sharpe, Sortino, max drawdown (stdlib)
  - ``returns``        — XIRR (money-weighted) + time-weighted return (stdlib)
  - ``attribution``    — Brinson-Fachler decomposition + Cariño linking (stdlib)

Example::

    from alpha_engine_lib.quant.risk_measures import historical_cvar
    from alpha_engine_lib.quant.factor_risk import estimate_factor_model, portfolio_risk
"""

from __future__ import annotations
