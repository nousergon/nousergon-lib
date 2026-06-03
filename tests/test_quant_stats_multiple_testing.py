"""Tests for alpha_engine_lib.quant.stats.multiple_testing — Benjamini-Hochberg FDR."""

from __future__ import annotations

from alpha_engine_lib.quant.stats.multiple_testing import benjamini_hochberg


def test_empty_returns_empty():
    assert benjamini_hochberg([]) == []


def test_all_significant_when_all_tiny():
    p = [0.001, 0.002, 0.003]
    assert benjamini_hochberg(p, alpha=0.05) == [True, True, True]


def test_none_significant_when_all_large():
    p = [0.6, 0.7, 0.8, 0.9]
    assert benjamini_hochberg(p, alpha=0.05) == [False, False, False, False]


def test_preserves_input_order():
    # Smallest p-value is at index 2 — its True must land at index 2.
    p = [0.9, 0.8, 0.0001, 0.7]
    out = benjamini_hochberg(p, alpha=0.05)
    assert out[2] is True
    assert out == [False, False, True, False]


def test_step_up_includes_lower_ranks():
    # Classic BH step-up: a higher-ranked p-value clearing its threshold marks
    # ALL lower-ranked (smaller) p-values significant, even one between thresholds.
    # n=4, alpha=0.05 → thresholds at rank k: 0.0125, 0.025, 0.0375, 0.05.
    # p=[0.001, 0.013, 0.001, 0.04]: rank-4 p=0.04 ≤ 0.05 ⇒ all four significant.
    p = [0.001, 0.013, 0.001, 0.04]
    assert benjamini_hochberg(p, alpha=0.05) == [True, True, True, True]


def test_alpha_scales_rejection():
    p = [0.02, 0.04]
    assert benjamini_hochberg(p, alpha=0.01) == [False, False]
    assert benjamini_hochberg(p, alpha=0.10) == [True, True]
