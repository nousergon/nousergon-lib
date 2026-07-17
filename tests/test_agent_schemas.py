"""Tests for the lifted agent_schemas submodule.

Coverage strategy: each schema gets a "happy path" test confirming it
accepts a valid LLM-style payload, plus a regression test for any
non-trivial validator (clamp, JSON-string-as-list parser). The
agent_id → schema dispatch map gets full coverage so replay tooling
can rely on it.

Schemas are intentionally permissive (extra="allow") to tolerate
forward-compatible drift from the LLM, so most fields don't have
hard validation; the ones that DO (sector_modifiers clamp,
RubricDimensionScore.score range, CIORawOutput.decisions min_length)
each get an explicit test.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

# ── Quant analyst ────────────────────────────────────────────────────────


class TestQuantAnalystOutput:
    def test_accepts_typical_payload(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput

        out = QuantAnalystOutput(
            ranked_picks=[
                {"ticker": "NVDA", "quant_score": 88, "rationale": "AI tailwind"},
                {"ticker": "AAPL", "quant_score": 75, "rationale": "FCF strong"},
            ]
        )
        assert len(out.ranked_picks) == 2
        assert out.ranked_picks[0].ticker == "NVDA"

    def test_quant_score_range_enforced(self):
        from nousergon_lib.agent_schemas import QuantPick

        with pytest.raises(ValidationError):
            QuantPick(ticker="X", quant_score=150)

    def test_extra_fields_allowed(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput

        # Forward-compat: LLM may emit additional keys.
        out = QuantAnalystOutput(
            ranked_picks=[],
            future_field="ok",  # type: ignore[call-arg]
        )
        assert out.ranked_picks == []


# ── Qual analyst ─────────────────────────────────────────────────────────


class TestQualAnalystOutput:
    def test_accepts_assessments_plus_additional_candidate(self):
        from nousergon_lib.agent_schemas import QualAnalystOutput

        out = QualAnalystOutput(
            assessments=[
                {"ticker": "PFE", "qual_score": 70, "bull_case": "pipeline"},
            ],
            additional_candidate={"ticker": "MRK", "bull_case": "oncology"},
        )
        assert out.additional_candidate is not None
        assert out.additional_candidate.ticker == "MRK"


# ── Stance taxonomy (v0.9.0) ─────────────────────────────────────────────


class TestStanceLiteral:
    """Schema-level pin on the 4-stance closed vocabulary.

    Origin: 2026-05-11 stance taxonomy arc. Stance is DERIVED
    downstream of agents (heuristic classifier in alpha-engine-predictor
    reads per-ticker features + FMP catalyst calendar, emits the label
    on predictions.json) — not declared by sector-team agents. The lib
    owns the shared vocabulary so the predictor (emitter) and executor
    (consumer) can pin the same Literal type.

    See StanceLiteral docstring in agent_schemas.py + the gitignored
    plan doc alpha-engine-docs/private/stance-taxonomy-arc-260511.md.
    """

    def test_stance_literal_has_exactly_four_values(self):
        """Closed set of 4 is the load-bearing design decision —
        adding a 5th stance would mean classifier rules need a new
        branch AND executor gates need a new code path. Lock the
        cardinality so a future PR can't quietly slip in a new option
        without surfacing the cross-repo impact."""
        import typing

        from nousergon_lib.agent_schemas import StanceLiteral
        args = typing.get_args(StanceLiteral)
        assert args == ("momentum", "value", "quality", "catalyst"), (
            f"StanceLiteral cardinality drift: got {args}. Adding/removing "
            "a stance requires coordinated changes in the predictor "
            "classifier rules, the executor gates, and the backtester per-"
            "stance attribution. Don't bypass that coordination."
        )

    def test_stance_literal_values_are_lowercase_singular(self):
        """Consumers (predictor classifier, executor gates, backtester
        attribution) match on exact strings. Pin the case + form so a
        future ``Momentum``/``MOMENTUM`` or ``momenta`` drift doesn't
        silently break consumers via missed string match."""
        import typing

        from nousergon_lib.agent_schemas import StanceLiteral
        for v in typing.get_args(StanceLiteral):
            assert v == v.lower(), f"stance vocabulary not lowercase: {v!r}"
            assert not v.endswith("s"), (
                f"stance vocabulary should be singular: {v!r}"
            )

    def test_stance_names_tuple_matches_literal(self):
        """STANCE_NAMES is the canonical iteration order. Must match
        StanceLiteral exactly so the discrete label, continuous
        loadings, and iteration order all align."""
        import typing

        from nousergon_lib.agent_schemas import STANCE_NAMES, StanceLiteral
        assert STANCE_NAMES == typing.get_args(StanceLiteral)


class TestStanceLoadings:
    """Continuous stance loadings — institutional factor-model pattern.

    Each pick gets a 4-element softmax over stance scores instead of a
    forced single-label assignment. Simple consumers fall back to
    ``.argmax()`` for a single StanceLiteral label; nuanced consumers
    (backtester per-loading attribution, future weighted-gate executor)
    read all four loadings.
    """

    def test_typical_payload_round_trips(self):
        import json

        from nousergon_lib.agent_schemas import StanceLoadings

        s = StanceLoadings(momentum=0.65, value=0.10, quality=0.20, catalyst=0.05)
        blob = s.model_dump_json()
        s2 = StanceLoadings.model_validate(json.loads(blob))
        assert s2.momentum == pytest.approx(0.65)
        assert s2.value == pytest.approx(0.10)
        assert s2.quality == pytest.approx(0.20)
        assert s2.catalyst == pytest.approx(0.05)

    def test_loadings_must_sum_to_one(self):
        from nousergon_lib.agent_schemas import StanceLoadings

        # Sum = 0.5 → reject
        with pytest.raises(ValidationError, match="sum to 1"):
            StanceLoadings(momentum=0.5, value=0.0, quality=0.0, catalyst=0.0)
        # Sum = 1.5 → reject
        with pytest.raises(ValidationError, match="sum to 1"):
            StanceLoadings(momentum=0.5, value=0.5, quality=0.5, catalyst=0.0)

    def test_loadings_tolerate_float_roundoff(self):
        """Producer rounds to 6 decimals + softmax involves exp/sum
        which doesn't yield exact 1.0. Allow ±1e-3 tolerance so the
        producer's rounding doesn't trigger spurious validation
        failures."""
        from nousergon_lib.agent_schemas import StanceLoadings

        # Slightly off from 1.0 (4e-4 short) — accept
        s = StanceLoadings(momentum=0.2500, value=0.2500, quality=0.2499, catalyst=0.2497)
        assert abs((s.momentum + s.value + s.quality + s.catalyst) - 1.0) < 1e-3

    def test_negative_loadings_rejected(self):
        """Each loading is a probability — must be ≥ 0."""
        from nousergon_lib.agent_schemas import StanceLoadings

        with pytest.raises(ValidationError):
            StanceLoadings(momentum=-0.1, value=0.5, quality=0.4, catalyst=0.2)

    def test_argmax_returns_dominant_stance(self):
        """The .argmax() convenience method returns the StanceLiteral
        label of the highest-loaded stance. Simple consumers
        (executor v1) use this; nuanced consumers read all four
        loadings directly."""
        from nousergon_lib.agent_schemas import StanceLoadings

        s = StanceLoadings(momentum=0.65, value=0.10, quality=0.20, catalyst=0.05)
        assert s.argmax() == "momentum"

        s = StanceLoadings(momentum=0.10, value=0.65, quality=0.20, catalyst=0.05)
        assert s.argmax() == "value"

        s = StanceLoadings(momentum=0.05, value=0.10, quality=0.20, catalyst=0.65)
        assert s.argmax() == "catalyst"

    def test_argmax_tie_broken_by_canonical_order(self):
        """Ties broken by STANCE_NAMES order (momentum > value >
        quality > catalyst). Deterministic + matches the lib's
        canonical iteration."""
        from nousergon_lib.agent_schemas import StanceLoadings

        s = StanceLoadings(momentum=0.25, value=0.25, quality=0.25, catalyst=0.25)
        assert s.argmax() == "momentum"  # first in canonical order

    def test_extra_fields_rejected(self):
        """extra='forbid' guards the schema — typo in field name (e.g.,
        ``momentmu=0.5``) fails validation rather than silently being
        stored on an attribute the consumer never reads."""
        from nousergon_lib.agent_schemas import StanceLoadings

        with pytest.raises(ValidationError):
            StanceLoadings.model_validate({
                "momentum": 0.65, "value": 0.10, "quality": 0.20,
                "catalyst": 0.05, "momentmu": 0.0,  # typo
            })


# ── Peer review ──────────────────────────────────────────────────────────


class TestJointSelectionOutput:
    """The two-pass flow's Pass 1 schema. Ticker-list + team-rationale
    only — per-ticker rationale moves to Pass 2 (one bounded
    JointFinalizationDecision call per selected ticker)."""

    def test_accepts_typical_payload(self):
        from nousergon_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput(
            selected_tickers=["NVDA", "PLTR", "RKLB"],
            team_rationale="Asymmetric high-R/R slate, AI-infrastructure tilt.",
        )
        assert len(out.selected_tickers) == 3
        assert out.selected_tickers[0] == "NVDA"

    def test_empty_selection_is_valid(self):
        """Edge case: agent emits an empty selection (no candidates clear
        the gate). Schema must accept; downstream gate decides whether
        empty is a hard-fail or graceful no-op."""
        from nousergon_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput()
        assert out.selected_tickers == []
        assert out.team_rationale == ""

    def test_extra_fields_allowed(self):
        """``extra='allow'`` lets the LLM emit forward-compatible fields
        (e.g. ``confidence``) without breaking validation."""
        from nousergon_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput(
            selected_tickers=["NVDA"],
            team_rationale="",
            confidence=0.85,
        )
        assert out.selected_tickers == ["NVDA"]


class TestJointFinalizationOutput:
    def test_accepts_typical_payload(self):
        from nousergon_lib.agent_schemas import JointFinalizationOutput

        out = JointFinalizationOutput(
            selected_decisions=[
                {"ticker": "JPM", "rationale": "Strong NIM"},
            ],
            team_rationale="Sector concentration controlled.",
        )
        assert len(out.selected_decisions) == 1

    def test_string_as_list_parser_recovers(self, caplog):
        """Regression test for the 2026-05-03 SF Sonnet failure mode
        where ``selected_decisions`` was returned as a JSON-encoded
        string. The validator parse-and-continues + emits a WARNING."""
        from nousergon_lib.agent_schemas import JointFinalizationOutput

        encoded = '[{"ticker": "JPM", "rationale": "x"}]'
        with caplog.at_level(logging.WARNING):
            out = JointFinalizationOutput(selected_decisions=encoded)
        assert len(out.selected_decisions) == 1
        assert out.selected_decisions[0].ticker == "JPM"
        assert any("JSON-string" in m for m in caplog.messages)

    def test_invalid_json_string_falls_through_to_pydantic_error(self):
        from nousergon_lib.agent_schemas import JointFinalizationOutput

        with pytest.raises(ValidationError):
            JointFinalizationOutput(selected_decisions="this isn't json")


class TestQuantAcceptanceVerdict:
    def test_accepts_minimal_payload(self):
        from nousergon_lib.agent_schemas import QuantAcceptanceVerdict

        out = QuantAcceptanceVerdict(accept=True, reason="strong tech score")
        assert out.accept is True


# ── Macro economist + critic ─────────────────────────────────────────────


class TestMacroEconomistRawOutput:
    def test_accepts_typical_payload(self):
        from nousergon_lib.agent_schemas import MacroEconomistRawOutput

        out = MacroEconomistRawOutput(
            report_md="Full regime narrative",
            market_regime="bull",
            sector_modifiers={"technology": 1.15, "financials": 1.0},
        )
        assert out.market_regime == "bull"
        assert out.sector_modifiers["technology"] == 1.15

    def test_sector_modifier_clamp_rejects_out_of_band(self):
        from nousergon_lib.agent_schemas import MacroEconomistRawOutput

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(sector_modifiers={"technology": 1.5})  # >1.30

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(sector_modifiers={"technology": 0.5})  # <0.70

    def test_regime_literal_enforced(self):
        from nousergon_lib.agent_schemas import MacroEconomistRawOutput

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(market_regime="exuberant")

    def test_regime_literal_is_3class_caution_rejected(self):
        # v0.42.0 retired "caution" from the macro market_regime taxonomy
        # per caution-regime-retirement-260528.md. The 3-class Ang-Bekaert
        # taxonomy (bull/neutral/bear) is the institutional baseline; the
        # rule-based caution override at the macro-agent layer was double-
        # counted by the continuous regime_intensity_z META_FEATURE.
        # Portfolio-protective hysteresis (risk_on/caution/risk_off) is a
        # separate axis emitted by the predictor drawdown leg.
        from nousergon_lib.agent_schemas import (
            MacroCriticOutput,
            MacroEconomistRawOutput,
        )

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(market_regime="caution")
        with pytest.raises(ValidationError):
            MacroCriticOutput(
                action="revise", critique="elevated stress", suggested_regime="caution",
            )

    def test_regime_literal_accepts_all_3_classes(self):
        from nousergon_lib.agent_schemas import MacroEconomistRawOutput

        for regime in ("bull", "neutral", "bear"):
            out = MacroEconomistRawOutput(market_regime=regime)
            assert out.market_regime == regime


class TestMacroCriticOutput:
    def test_accept_action(self):
        from nousergon_lib.agent_schemas import MacroCriticOutput

        out = MacroCriticOutput(action="accept", critique="looks sound")
        assert out.action == "accept"
        assert out.suggested_regime is None

    def test_revise_with_suggested_regime(self):
        from nousergon_lib.agent_schemas import MacroCriticOutput

        out = MacroCriticOutput(
            action="revise", critique="too bullish", suggested_regime="neutral",
        )
        assert out.suggested_regime == "neutral"


# ── Held-stock thesis update ─────────────────────────────────────────────


class TestHeldThesisUpdateLLMOutput:
    def test_no_score_fields(self):
        from nousergon_lib.agent_schemas import HeldThesisUpdateLLMOutput

        # Schema intentionally has no final_score / qual_score /
        # quant_score — the held-stock LLM update path must NOT
        # overwrite prior_scores. This test pins the contract.
        out = HeldThesisUpdateLLMOutput(
            bull_case="Services growth", conviction=70,
        )
        assert not hasattr(out, "final_score")
        assert out.conviction == 70


# ── CIO ──────────────────────────────────────────────────────────────────


class TestCIORawOutput:
    def test_accepts_decisions_with_advance(self):
        from nousergon_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(
            decisions=[
                {
                    "ticker": "NVDA", "decision": "ADVANCE",
                    "rank": 1, "conviction": 85, "rationale": "RR 2.5",
                },
            ]
        )
        assert len(out.decisions) == 1
        assert out.decisions[0].decision == "ADVANCE"

    def test_min_length_rejects_empty_decisions(self):
        """2026-05-02 PR B regression: Sonnet emitted decisions=[] when
        the prompt's per-candidate cue was stripped. min_length=1
        defends at the schema layer."""
        from nousergon_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[])

    def test_default_factory_also_validates(self):
        """validate_default=True ensures the min_length=1 constraint
        fires when decisions is omitted entirely (default_factory=list
        path), not just when the caller explicitly passes []."""
        from nousergon_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput()

    def test_decision_literal_enforced(self):
        from nousergon_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[
                {"ticker": "X", "decision": "MAYBE"},  # not in literal
            ])

    def test_rule_tags_optional_default_none(self):
        """Legacy artifacts emitted by prompts < v1.3.0 omit rule_tags
        entirely. Schema must default to None so loading historical
        captures keeps working."""
        from nousergon_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "NVDA", "decision": "ADVANCE",
             "rank": 1, "conviction": 85, "rationale": "RR 2.5"},
        ])
        assert out.decisions[0].rule_tags is None

    def test_rule_tags_accepts_single_tag(self):
        from nousergon_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "MCD", "decision": "REJECT",
             "rationale": "Qual<50", "rule_tags": ["qual_veto"]},
        ])
        assert out.decisions[0].rule_tags == ["qual_veto"]

    def test_rule_tags_accepts_multiple_tags(self):
        """Real-world example: REJECT MCD because BOTH qual<50 AND
        Consumer Discretionary is underweight. Multi-tag is the
        common case for REJECTS."""
        from nousergon_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "MCD", "decision": "REJECT",
             "rationale": "Qual<50 + sector underweight",
             "rule_tags": ["qual_veto", "macro_alignment"]},
        ])
        assert out.decisions[0].rule_tags == ["qual_veto", "macro_alignment"]

    def test_rule_tags_rejects_unknown_literal(self):
        """Vocabulary is closed — unknown tags must fail validation
        rather than silently accumulate as freeform strings."""
        from nousergon_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[
                {"ticker": "X", "decision": "REJECT",
                 "rule_tags": ["made_up_tag"]},
            ])

    def test_rule_tag_vocabulary_is_nine_tags(self):
        """Locked vocabulary — adding/removing a tag is a deliberate
        prompt-version + analysis-layer change, not an accident."""
        from typing import get_args

        from nousergon_lib.agent_schemas import CIORuleTagLiteral

        tags = set(get_args(CIORuleTagLiteral))
        assert tags == {
            "qual_veto", "quant_veto", "dual_score_floor",
            "rr_asymmetry", "macro_alignment", "portfolio_fit",
            "catalyst_specificity", "prior_continuity", "other",
        }


# ── LLM-as-judge eval ────────────────────────────────────────────────────


class TestRubricEvalLLMOutput:
    def test_accepts_typical_payload(self):
        from nousergon_lib.agent_schemas import RubricEvalLLMOutput

        out = RubricEvalLLMOutput(
            dimension_scores=[
                {
                    "dimension": "numerical_grounding",
                    "score": 4,
                    "reasoning": "Cited specific multiples.",
                },
            ],
            overall_reasoning="Strong on numerics; rationale could be deeper.",
        )
        assert len(out.dimension_scores) == 1
        assert out.dimension_scores[0].score == 4

    def test_score_range_enforced(self):
        from nousergon_lib.agent_schemas import RubricDimensionScore

        for invalid in (0, 6, -1, 10):
            with pytest.raises(ValidationError):
                RubricDimensionScore(
                    dimension="x", score=invalid, reasoning="r",
                )

    def test_string_as_list_parser_recovers(self, caplog):
        """Same regression class as JointFinalizationOutput — Haiku
        occasionally returns dimension_scores as a JSON-string."""
        from nousergon_lib.agent_schemas import RubricEvalLLMOutput

        encoded = (
            '[{"dimension": "x", "score": 3, "reasoning": "r"}]'
        )
        with caplog.at_level(logging.WARNING):
            out = RubricEvalLLMOutput(
                dimension_scores=encoded, overall_reasoning="ok",
            )
        assert len(out.dimension_scores) == 1
        assert any("JSON-string" in m for m in caplog.messages)


# ── agent_id dispatch ────────────────────────────────────────────────────


class TestSchemaDispatch:
    @pytest.mark.parametrize("agent_id,expected_name", [
        ("sector_quant", "QuantAnalystOutput"),
        ("sector_quant:technology", "QuantAnalystOutput"),
        ("sector_qual:healthcare", "QualAnalystOutput"),
        ("sector_peer_review:financials", "JointFinalizationOutput"),
        ("macro_economist", "MacroEconomistRawOutput"),
        ("ic_cio", "CIORawOutput"),
        ("thesis_update:AAPL", "HeldThesisUpdateLLMOutput"),
    ])
    def test_resolve_known_agent_ids(self, agent_id, expected_name):
        from nousergon_lib.agent_schemas import resolve_schema_for_agent

        cls = resolve_schema_for_agent(agent_id)
        assert cls is not None
        assert cls.__name__ == expected_name

    def test_unknown_agent_returns_none(self):
        from nousergon_lib.agent_schemas import resolve_schema_for_agent

        assert resolve_schema_for_agent("brand_new_agent") is None
        assert resolve_schema_for_agent("") is None
        assert resolve_schema_for_agent(None) is None  # type: ignore[arg-type]

    def test_dispatch_map_covers_six_canonical_families(self):
        from nousergon_lib.agent_schemas import SCHEMA_BY_AGENT_ID_BASE

        # Pin the canonical family list so additions surface in review.
        assert set(SCHEMA_BY_AGENT_ID_BASE.keys()) == {
            "sector_quant",
            "sector_qual",
            "sector_peer_review",
            "macro_economist",
            "ic_cio",
            "thesis_update",
        }
