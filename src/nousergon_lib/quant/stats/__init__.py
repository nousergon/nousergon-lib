"""Statistical evaluation utilities for signal/strategy quality assessment.

Pure-compute metrics consumed across the fleet (backtester, robodashboard) for
judging signal quality, strategy skill, and selection bias — no I/O. Import the
submodule you need (the package keeps no eager imports). Most need numpy+pandas;
``information_coefficient`` additionally uses scipy when present (with a numpy
fallback). Install ``nousergon-lib[quant-stats]``.

Modules:
  - ``dsr``                     — Probabilistic + Deflated Sharpe (López de Prado)
  - ``calibration``             — Expected Calibration Error (probability vs outcome)
  - ``information_coefficient`` — Spearman rank IC of conviction vs forward return
  - ``expectancy``              — hit-rate × win/loss decomposition
  - ``multiple_testing``        — Benjamini-Hochberg FDR correction
  - ``intervals``               — bootstrap CI, Newey-West SE, Wilson score interval
  - ``risk_matched_benchmark``  — EW-high-vol + beta-matched-SPY baselines + IR
  - ``regime_sortino``          — regime-stratified cross-sectional pick-alpha Sortino

Example::

    from nousergon_lib.quant.stats.dsr import compute_dsr
    from nousergon_lib.quant.stats.multiple_testing import benjamini_hochberg
    from nousergon_lib.quant.stats.intervals import bootstrap_ci, wilson_score_interval
"""

from __future__ import annotations
