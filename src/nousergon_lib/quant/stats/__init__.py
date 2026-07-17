"""Statistical evaluation utilities for signal/strategy quality assessment.

Pure-compute metrics consumed across the fleet (backtester, robodashboard) for
judging signal quality, strategy skill, and selection bias — no I/O. Import the
submodule you need (the package keeps no eager imports). Most need numpy+pandas;
``information_coefficient`` additionally uses scipy when present (with a numpy
fallback). Install ``nousergon-lib[quant-stats]``.

Modules:
  - ``dsr``                     — Probabilistic + Deflated Sharpe (López de Prado)
  - ``pbo``                     — CSCV Probability of Backtest Overfitting (BBLZ 2014)
  - ``calibration``             — Expected Calibration Error (probability vs outcome)
  - ``information_coefficient`` — Spearman rank IC of conviction vs forward return
  - ``expectancy``              — hit-rate × win/loss decomposition
  - ``multiple_testing``        — Benjamini-Hochberg FDR correction
  - ``intervals``               — bootstrap CI, Newey-West SE, Wilson score interval
  - ``risk_matched_benchmark``  — EW-high-vol + beta-matched-SPY baselines + IR
  - ``regime_sortino``          — regime-stratified cross-sectional pick-alpha Sortino
  - ``trial_accumulator``       — cumulative multiple-testing trial count (config#2454);
                                   the shared S3-backed counter ``dsr``'s ``n_trials``
                                   reads from, incremented by the backtester's 4 sweep
                                   producers

Example::

    from nousergon_lib.quant.stats.dsr import compute_dsr
    from nousergon_lib.quant.stats.pbo import cscv_pbo
    from nousergon_lib.quant.stats.multiple_testing import benjamini_hochberg
    from nousergon_lib.quant.stats.intervals import bootstrap_ci, wilson_score_interval
    from nousergon_lib.quant.stats.trial_accumulator import (
        increment_trial_count, read_cumulative_trial_count,
    )
"""

from __future__ import annotations
