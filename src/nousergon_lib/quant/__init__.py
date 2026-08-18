"""Quantitative portfolio analytics — pure, front-end- and data-source-agnostic.

The shared institutional-analytics engine consumed across the fleet (predictor,
backtester, robodashboard). Every module is dependency-light and unit-testable in
isolation, and *describes/measures* a portfolio (performance, risk, attribution)
with **no advisory logic** — it sits on the "analytics, not advice" side of the
line.

Modules (import the submodule you need — the package keeps no eager imports so the
stdlib-only modules stay importable without numpy):

  - ``factor_risk``    — Σ=B·F·Bᵀ+D ex-ante risk + tracking error (**needs numpy**;
                          install ``nousergon-lib[quant]``)
  - ``risk_measures``  — parametric + historical VaR / CVaR (stdlib)
  - ``riskstats``      — volatility, Sharpe, Sortino, downside deviation,
                          max drawdown (stdlib). The fleet's ONLY implementation
                          of these — see its module docstring (config-I7597).
  - ``returns``        — XIRR (money-weighted) + time-weighted return (stdlib)
  - ``attribution``    — Brinson-Fachler decomposition + Cariño linking (stdlib)
  - ``transaction_cost`` — √-impact (Almgren-Chriss) cost + tradeability score (stdlib)
  - ``horizons``       — evaluation-horizon chokepoint: canonical primary vs
                          diagnostic horizons + wide-column naming (stdlib)
  - ``selftest``       — shared known-answer self-test runner: the PASS/FAIL/
                          UNKNOWN taxonomy, ``Case`` record, SIGALRM budget and
                          provenance helpers every stage's own numeric battery
                          reuses (stdlib; alpha-engine-config-I7238)

Example::

    from nousergon_lib.quant.risk_measures import historical_cvar
    from nousergon_lib.quant.factor_risk import estimate_factor_model, portfolio_risk
"""

from __future__ import annotations
