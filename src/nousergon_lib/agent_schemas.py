"""LLM-output Pydantic schemas — shared contract surface for agents
across alpha-engine-research, replay tooling in alpha-engine-backtester,
and any future consumer that needs to validate agent output against the
canonical shape.

Why these live in the shared lib (not in alpha-engine-research):

  Replay harness invocation isomorphism — the replay path in
  alpha-engine-backtester needs to call ``with_structured_output(Schema)``
  using the EXACT same Pydantic schema the production agent used.
  Without a shared lib, backtester would either need a heavy cross-repo
  dependency on research (pulls langgraph + every research dep) or have
  to vendor a local copy that drifts. This submodule is the canonical
  home — research re-exports from here so existing call sites keep
  working unchanged.

What's here:

  - 14 ``LLMOutput`` and supporting schemas used in
    ``langchain_anthropic.ChatAnthropic.with_structured_output(...)``
    calls across the research pipeline.
  - The literal types they reference (``RegimeLiteral``,
    ``CIORawDecisionLiteral``).

What's NOT here (intentionally):

  - State-machine objects (``SectorTeamOutput``, ``MacroEconomistOutput``,
    ``CIOOutput``, ``InvestmentThesis``) — these are research-internal
    state types, not LLM-output contracts. They live in
    ``alpha-engine-research/graph/state_schemas.py`` and stay there.
  - Domain types coupled to research's tool layer (``ToolCall``,
    ``ExitEvent``, ``PopulationRotationEvent``).

Schema-validation discipline: every class here has
``model_config = ConfigDict(extra="allow")`` because LLM outputs may
include additional fields (forward-compatible drift). Validators that
defend against observed LLM failure modes (e.g. ``selected_decisions``
returned as a JSON-encoded string) move with the class.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Literals ─────────────────────────────────────────────────────────────


RegimeLiteral = Literal["bull", "neutral", "bear"]
"""Macro market regime — output of the macro_economist agent and the
macro critic. Drives sector_modifiers downstream and the executor's
graduated drawdown gate.

3-class Ang-Bekaert taxonomy. The legacy 4th value ``"caution"`` was
retired in v0.42.0 (plan: caution-regime-retirement-260528.md): the
rule-based caution override at the macro-agent layer double-counted
signals already weighted into the continuous ``regime_intensity_z``
META_FEATURE. Portfolio-protective drawdown state is now a separate
axis (``drawdown_tier: Literal["risk_on","caution","risk_off"]``)
emitted by the predictor's drawdown leg; consumers compose the two
axes via most-protective override at decision time."""


CIORawDecisionLiteral = Literal["ADVANCE", "REJECT", "NO_ADVANCE_DEADLOCK"]
"""CIO emits the literal ``NO_ADVANCE_DEADLOCK`` for low-conviction
picks that don't clear the floor; post-processing in the research
layer's ``_parse_cio_response`` may synthesize ``ADVANCE_FORCED`` to
fill below-floor open slots, but that synthesis happens AFTER the LLM
extraction, so the raw schema only enumerates the three values the LLM
is allowed to emit."""


STANCE_NAMES: tuple[str, ...] = ("momentum", "value", "quality", "catalyst")
"""Canonical ordering of stance names. Source of truth for both the
``StanceLiteral`` type and ``StanceLoadings`` field iteration order.
Pinning the tuple lets tests assert equality rather than set-equality,
surfacing unintended reordering. Iteration is also the tie-break order
for ``StanceLoadings.argmax()``."""


StanceLiteral = Literal["momentum", "value", "quality", "catalyst"]
"""Per-pick investment stance — the shared vocabulary across the
stance-taxonomy arc. Routes downstream executor gating:

- ``momentum``  — trend-following; ticker has strong recent price
  action (20d ret > 0, MA50 > MA200, RSI 40-70). Executor applies the
  standard momentum_veto (block if 20d < -X%).
- ``value``     — contrarian; quality business at discounted price
  after sell-off. Executor inverts momentum_veto (requires drawdown to
  qualify) and applies smaller sizing (0.7×) + wider ATR stops (3× vs
  2×). 30d time-bounded — exit if no bounce.
- ``quality``   — defensive; stable earnings, hold-through-cycle.
  Executor relaxes momentum_veto threshold (-15% vs -5%), applies 0.8×
  sizing, disables time decay (longer hold), tighter sector cap.
- ``catalyst``  — event-driven; specific upcoming catalyst (earnings
  beat, FDA approval, M&A) drives thesis. Executor skips momentum_veto
  entirely but requires ``catalyst_date`` (within 30 days) AND applies
  0.6× sizing (event-driven = higher variance) + hard exit on
  catalyst_date+3d if no follow-through.

Origin: 2026-05-11 stance taxonomy arc (private plan at
``alpha-engine-docs/private/stance-taxonomy-arc-260511.md``).

**Stance is DERIVED downstream of agents, not declared by them.**
The sector-team agents (quant + qual + peer review) focus on alpha
generation; a heuristic stance classifier in ``alpha-engine-predictor``
reads per-ticker features (momentum_20d, vol, fundamental ratios,
upcoming earnings) + FMP catalyst calendar and emits the stance label
on ``predictions.json``. The executor consumes ``pred_data["stance"]``
when applying stance-conditional gating. Rationale: factor models at
AQR / BlackRock / Barra derive loadings from data rather than asking
analysts to self-tag; that's the institutional pattern. Adding a 5th
declaration task to the agents would also degrade focus on their
core alpha-generation work.

Closed set of 4 chosen deliberately — small enough for decisive
classification, large enough to cover real strategies. There is no
"mixed" / "other" option on the discrete label because picks ARE
naturally mixed — the ``StanceLoadings`` continuous emission below
captures the mixed exposure faithfully; ``StanceLiteral`` is the
``argmax(loadings)`` convenience label for simple consumers.
"""


class StanceLoadings(BaseModel):
    """Continuous per-stance loadings — institutional factor-model
    pattern. Each field is in ``[0, 1]``; the four fields sum to ``1.0``.

    Most picks have mixed factor exposure (e.g., 0.65 momentum +
    0.20 quality + 0.10 value + 0.05 catalyst). Discrete labels force
    an artificial single-choice when reality is mixed; this model
    captures the mix faithfully.

    Producer: the heuristic stance classifier in alpha-engine-predictor
    (``model/stance_classifier.py``). Smooth functions over per-ticker
    features produce raw scores; ``softmax`` normalizes to a proper
    probability distribution.

    Consumers:
      - **Simple consumers** (executor v1, dashboards): use
        ``StanceLiteral`` (``argmax(loadings)``) instead and route to
        one gate. No need to read this model.
      - **Nuanced consumers** (backtester per-loading attribution,
        future weighted-gate executor v2, future ML stance classifier):
        read this model and weight gates / sizing / attribution by
        each loading.

    The discrete-vs-continuous split lets us ship the simple
    consumer path now (v1 routes by argmax) while leaving the
    institutional-grade continuous data on predictions.json for
    later sophistication (no future schema migration required).
    """

    model_config = ConfigDict(extra="forbid")

    momentum: float = Field(ge=0.0, le=1.0, description="Trending-up factor loading")
    value: float = Field(ge=0.0, le=1.0, description="Oversold-but-defensible factor loading")
    quality: float = Field(ge=0.0, le=1.0, description="Low-vol / defensive factor loading")
    catalyst: float = Field(ge=0.0, le=1.0, description="Event-driven factor loading")

    @model_validator(mode="after")
    def _check_sum_to_one(self):
        """Loadings must form a proper probability distribution. 1e-3
        tolerance accommodates float roundoff in softmax + the producer's
        rounding-to-6-decimals on serialization."""
        total = self.momentum + self.value + self.quality + self.catalyst
        if not (0.999 < total < 1.001):
            raise ValueError(
                f"stance_loadings must sum to 1.0 (±1e-3); got {total:.6f}"
            )
        return self

    def argmax(self) -> StanceLiteral:
        """Convenience: return the dominant stance label (highest
        loading). Ties broken in canonical ``STANCE_NAMES`` order. Used
        by executor v1 + dashboards for single-label routing."""
        pairs = (
            ("momentum", self.momentum),
            ("value", self.value),
            ("quality", self.quality),
            ("catalyst", self.catalyst),
        )
        return max(pairs, key=lambda p: p[1])[0]  # type: ignore[return-value]


CIORuleTagLiteral = Literal[
    "qual_veto",
    "quant_veto",
    "dual_score_floor",
    "rr_asymmetry",
    "macro_alignment",
    "portfolio_fit",
    "catalyst_specificity",
    "prior_continuity",
    "other",
]
"""Per-decision attribution tag identifying which rule(s) drove the
CIO's verdict. Vocabulary mirrors the prompt's EVALUATION CRITERIA
(items 1-5) plus two implicit veto gates and a continuity tag:

- ``qual_veto``         — qual_score < 50 trip
- ``quant_veto``        — quant_score < 50 trip
- ``dual_score_floor``  — both quant + qual < 60 with no compensating R/R
- ``rr_asymmetry``      — R/R-ratio framing as primary justification
- ``macro_alignment``   — sector under/overweight as primary factor
- ``portfolio_fit``     — diversification, concentration, or already-held
- ``catalyst_specificity`` — time-bound, named catalyst as primary factor
- ``prior_continuity``  — prior IC continuity (rolled-over advance)
- ``other``             — escape hatch for non-fitting reasoning

Multiple tags per decision are allowed (a REJECT can be both
``qual_veto`` AND ``macro_alignment``). Optional list[str] | None on
the schema — None means the LLM didn't emit tags (legacy artifacts
from before the v0.7.0 prompt update). Backtester analysis (per-tag
precision over time) reads this field to surface which gates are
systematically over- or under-rejecting."""


# ── Quant analyst (sector_quant) ─────────────────────────────────────────


class QuantPick(BaseModel):
    """One ranked candidate from the quant analyst's ReAct loop."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    quant_score: float = Field(ge=0, le=100)
    rationale: str = ""
    key_metrics: dict = Field(default_factory=dict)


class QuantAnalystOutput(BaseModel):
    """Wrapper for the quant ReAct agent's structured response.

    LangGraph ``create_react_agent(response_format=...)`` runs an extra
    LLM call after the tool-loop terminates to extract this typed shape
    from the conversation."""

    model_config = ConfigDict(extra="allow")

    ranked_picks: list[QuantPick] = Field(default_factory=list)


# ── Qual analyst (sector_qual) ───────────────────────────────────────────


class QualAssessment(BaseModel):
    """One per-ticker qualitative assessment from the qual analyst."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    qual_score: float | None = Field(default=None, ge=0, le=100)
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    conviction: int | None = Field(default=None, ge=0, le=100)


class QualAnalystOutput(BaseModel):
    """Wrapper for the qual ReAct agent's structured response.

    ``additional_candidate`` is the qual-side proposal that the peer-review
    quant gate then accepts or rejects (see :class:`QuantAcceptanceVerdict`).
    """

    model_config = ConfigDict(extra="allow")

    assessments: list[QualAssessment] = Field(default_factory=list)
    additional_candidate: QualAssessment | None = None


# ── Peer review (sector_peer_review) ─────────────────────────────────────


class QuantAcceptanceVerdict(BaseModel):
    """Side-LLM call: peer_review's quant analyst rules on whether to
    accept the qual analyst's added candidate."""

    model_config = ConfigDict(extra="allow")

    accept: bool
    reason: str = ""


class JointFinalizationDecision(BaseModel):
    """One per-ticker decision from peer_review's joint finalization.

    Per-ticker rationale enables LLM-as-judge eval to score the
    synthesis reasoning at decision granularity (rather than one
    rationale string covering all 2-3 picks). Composes with the
    LLM-as-judge workstream (ROADMAP Phase 2 P1).
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    rationale: str = Field(
        default="",
        description=(
            "Why this ticker was selected — name R/R reasoning, score "
            "asymmetry, catalyst. 1-2 sentences."
        ),
    )


class JointSelectionOutput(BaseModel):
    """Side-LLM call: peer_review's two-pass joint finalization, Pass 1.

    Pass 1 emits the ticker-list selection + cross-pick context only,
    deferring per-ticker rationale generation to Pass 2 (which fans out
    to one bounded ``JointFinalizationDecision`` call per selected
    ticker). Two-pass design replaces the prior single-pass
    ``JointFinalizationOutput`` call after 2026-05-03 + 2026-05-06
    truncation incidents where Haiku-emitted rationales blew past
    ``max_tokens_strategic`` mid-emission and the entire selection was
    lost. With the selection separated, Pass 1's output is bounded by
    construction (N tickers × ~10 tokens + a 1-2 sentence team
    rationale = ~200 tokens), eliminating the truncation class for the
    selection step regardless of model verbosity drift.

    The legacy ``JointFinalizationOutput`` schema below stays for
    replay-harness compatibility against historical artifacts.
    """

    model_config = ConfigDict(extra="allow")

    selected_tickers: list[str] = Field(
        default_factory=list,
        description=(
            "Array of selected ticker symbols (e.g. ['NVDA', 'PLTR', "
            "'RKLB']). One entry per pick, structured array (NOT a "
            "JSON-encoded string)."
        ),
    )
    team_rationale: str = Field(
        default="",
        description=(
            "Cross-pick rationale — sector concentration, regime fit "
            "across the slate, asymmetry mix. 1-2 sentences."
        ),
    )


class JointFinalizationOutput(BaseModel):
    """Side-LLM call: peer_review's joint quant+qual finalization, picks
    the team's 2-3 final recommendations from the merged candidate set.

    Per-ticker rationale lives on each ``selected_decisions`` entry;
    ``team_rationale`` carries cross-pick context (sector concentration,
    regime fit across the slate).

    LEGACY single-pass schema. Production peer_review now uses the
    two-pass flow (``JointSelectionOutput`` + per-ticker
    ``JointFinalizationDecision`` calls). This schema stays for replay
    harness invocation against historical artifacts pre-cutover."""

    model_config = ConfigDict(extra="allow")

    selected_decisions: list[JointFinalizationDecision] = Field(
        default_factory=list,
        description=(
            "Array of per-ticker selection decisions. Return one entry "
            "per selected ticker as a structured array (NOT a single "
            "JSON-encoded string). Each entry must be a JSON object "
            "with `ticker` and `rationale` fields."
        ),
    )
    team_rationale: str = Field(
        default="",
        description=(
            "Cross-pick rationale — sector concentration, regime fit "
            "across the slate, asymmetry mix. 1-2 sentences."
        ),
    )

    @field_validator("selected_decisions", mode="before")
    @classmethod
    def _parse_string_as_list(cls, v):
        """Defense for an observed Sonnet failure mode: the model
        occasionally returns ``selected_decisions`` as a JSON-encoded
        string instead of a structured array, even though the tool
        spec declares it as a list. First seen 2026-05-03 in SF
        ``eval-pipeline-validation-2`` where Sonnet returned
        ``'[\\n  {\\n    "ticker": "C..."\\n'`` — valid JSON inside a
        string wrapper.

        We log loudly (so flow-doctor surfaces the drift event in CW
        alarms) and parse-and-continue rather than hard-fail, because
        the downstream cost of a hard-fail is a wasted ~$5 Research
        run; the log entry preserves observability while the parse
        salvages the run. If the string isn't valid JSON list, fall
        through to the normal Pydantic list-type error so the failure
        mode stays loud.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logging.getLogger(__name__).warning(
                        "[joint_finalization_schema] LLM returned "
                        "selected_decisions as JSON-string of length %d "
                        "instead of a structured array; parsed-and-continued "
                        "(see schema-vs-LLM drift class).",
                        len(v),
                    )
                    return parsed
            except json.JSONDecodeError:
                pass
        return v


# ── Macro economist + critic ─────────────────────────────────────────────


class MacroEconomistRawOutput(BaseModel):
    """Wrapping schema for ``run_macro_agent`` output. The agent emits
    free-form prose (``report_md``) interleaved with a JSON block
    carrying structured fields; ``with_structured_output`` extracts both."""

    model_config = ConfigDict(extra="allow")

    report_md: str = ""
    market_regime: RegimeLiteral = "neutral"
    sector_modifiers: dict[str, float] = Field(default_factory=dict)
    sector_ratings: dict[str, dict] = Field(default_factory=dict)
    key_theme: str = ""
    material_changes: list[str] = Field(default_factory=list)

    @field_validator("sector_modifiers")
    @classmethod
    def clamp_modifiers(cls, v: dict[str, float]) -> dict[str, float]:
        """Mirror MacroEconomistOutput's clamp on the [0.70, 1.30] band."""
        for sector, m in v.items():
            if not (0.70 <= float(m) <= 1.30):
                raise ValueError(
                    f"sector_modifiers[{sector!r}]={m} outside [0.70, 1.30]"
                )
        return v


class MacroCriticOutput(BaseModel):
    """Reflection-loop critic output for the macro agent.

    The critic accepts or revises the macro_economist's draft. ``revise``
    triggers another macro_economist call; ``accept`` ends the loop.
    """

    model_config = ConfigDict(extra="allow")

    action: Literal["accept", "revise"]
    critique: str = ""
    suggested_regime: RegimeLiteral | None = None


# ── Held-stock thesis update (sector_team) ───────────────────────────────


class HeldThesisUpdateLLMOutput(BaseModel):
    """LLM-extraction shape for ``_update_thesis_for_held_stock``.

    Intentionally narrative-only — NO score fields. The held-stock LLM
    update path must NOT overwrite prior_scores; the existing strip-nulls
    merge logic exists today specifically because the LLM occasionally
    emits ``final_score: null``. By omitting score fields from the schema
    entirely, the LLM cannot emit them, and the strip-nulls workaround
    becomes unnecessary.

    Field-level ``description`` strings are propagated by
    ``with_structured_output()`` into the tool-input schema the LLM sees,
    so the per-field length/count guidance previously inlined in the
    prompt body lives here now (audit finding F1, PR B 2026-05-02).
    """

    model_config = ConfigDict(extra="allow")

    bull_case: str = Field(default="", description="Bull case narrative — 1-2 sentences, ~200 chars.")
    bear_case: str = Field(default="", description="Bear case narrative — 1-2 sentences, ~200 chars.")
    catalysts: list[str] = Field(default_factory=list, description="Up to 5 catalysts.")
    risks: list[str] = Field(default_factory=list, description="Up to 5 risks.")
    conviction: int | None = Field(default=None, ge=0, le=100, description="Strength of view (0-100). ≥70 high, 40-69 moderate, <40 low.")
    conviction_rationale: str = Field(default="", description="Why this conviction level — ~100 chars.")
    thesis_summary: str = ""
    triggers_response: str = ""


# ── CIO (ic_cio) ─────────────────────────────────────────────────────────


class CIORawDecision(BaseModel):
    """One CIO decision as emitted by the LLM (pre-post-processing).

    Note ``decision`` (LLM-emitted) vs ``thesis_type`` (post-processed,
    used in research's ``CIODecision``): the LLM never emits ``HOLD``
    directly — ``HOLD`` is what the post-processing maps ``REJECT`` to
    for held tickers in the current population. The two shapes are kept
    separate so each can describe its own contract precisely.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    decision: CIORawDecisionLiteral
    rank: int | None = Field(default=None, ge=0, description="1-based rank for ADVANCE picks; null for REJECT / NO_ADVANCE_DEADLOCK.")
    conviction: int | None = Field(default=None, ge=0, le=100, description="Strength of view (0-100).")
    rationale: str = Field(default="", description="Why this decision — name R/R reasoning (sub-scores, rr_ratio, catalyst).")
    rule_tags: list[CIORuleTagLiteral] | None = Field(
        default=None,
        description=(
            "Which gating rule(s) drove this decision. ≥1 tag per decision "
            "in v1.3.0+ prompts; multiple tags allowed (a REJECT can be "
            "both qual_veto AND macro_alignment). None on legacy artifacts "
            "from prompts < v1.3.0."
        ),
    )
    entry_thesis: HeldThesisUpdateLLMOutput | None = Field(default=None, description="Required for ADVANCE; null for REJECT / NO_ADVANCE_DEADLOCK.")


class CIORawOutput(BaseModel):
    """Wrapper for the CIO agent's structured response. The list-of-
    decisions shape mirrors what ``_parse_cio_response`` consumes today
    via balanced-brace JSON extraction.

    ``min_length=1`` is propagated to the LLM via the structured-output
    tool schema description AND validated by the SDK parser. Caught
    2026-05-02: PR B's strip of the CIO prompt's inline JSON example
    let Sonnet emit ``decisions: []`` because the structural cue that
    "one entry per candidate" was lost. The prompt fix (config #21,
    explicit OUTPUT REQUIREMENT block) addresses the LLM-side cue;
    this constraint is the schema-side defense — empty list now
    surfaces as a parsing_error at the call boundary rather than as a
    later "empty decisions" raise inside ``run_cio``.
    """

    # ``validate_default=True`` ensures the ``min_length=1`` constraint
    # fires even when ``decisions`` falls back to ``default_factory=list``.
    # Pydantic v2 skips default validation by default; without this the
    # empty-list rejection only triggers when a caller explicitly passes
    # ``decisions=[]`` — defeating the schema-side defense.
    model_config = ConfigDict(extra="allow", validate_default=True)

    decisions: list[CIORawDecision] = Field(
        default_factory=list,
        min_length=1,
        description="One entry per input candidate. Never empty — every candidate must receive a decision (ADVANCE / REJECT / NO_ADVANCE_DEADLOCK).",
    )


# ── LLM-as-judge eval ────────────────────────────────────────────────────


class RubricDimensionScore(BaseModel):
    """One dimension's score from the eval judge.

    Score is integer 1-5 per the rubric anchors (see
    eval_rubric_*.txt prompts in alpha-engine-config). The ``reasoning``
    string carries the judge's per-dimension justification — used by
    the dashboard's quality-trend page to surface WHY scores dropped,
    not just THAT they dropped.
    """

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(description="Rubric dimension name (e.g. 'numerical_grounding', 'signal_calibration').")
    score: int = Field(ge=1, le=5, description="Integer score 1-5 per the rubric anchors.")
    reasoning: str = Field(description="1-2 sentence justification citing specific artifact content that drove the score.")


class RubricEvalLLMOutput(BaseModel):
    """LLM-extraction shape for the eval judge call.

    The judge LLM (Haiku or Sonnet) produces this against a rubric
    prompt + DecisionArtifact pair. Wrapped in ``RubricEvalArtifact``
    by ``evals.judge.evaluate_artifact`` before persisting to S3.
    """

    model_config = ConfigDict(extra="allow")

    dimension_scores: list[RubricDimensionScore] = Field(
        default_factory=list,
        min_length=1,
        description=(
            "Array of per-dimension score entries. Return one entry "
            "per rubric dimension as a structured array (NOT a single "
            "JSON-encoded string). Each entry must be a JSON object "
            "with `dimension`, `score`, and `reasoning` fields. Order "
            "matches the rubric prompt's dimension list."
        ),
    )
    overall_reasoning: str = Field(
        description="1-2 sentence cross-dimension summary — strongest signal + most concerning gap.",
    )

    @field_validator("dimension_scores", mode="before")
    @classmethod
    def _parse_string_as_list(cls, v):
        """Defense for an observed Haiku failure mode (first surfaced
        2026-05-03 in judge_only smoke against new-format Sat 5/3
        captures): the model occasionally returns ``dimension_scores``
        as a JSON-encoded string instead of a structured array, even
        though the tool spec declares it as a list. Same pattern PR
        #99 fixed for ``JointFinalizationOutput.selected_decisions``.

        We log loudly (so flow-doctor surfaces the drift event in CW
        alarms) and parse-and-continue rather than hard-fail, because
        the downstream cost of a hard-fail is a wasted ~$0.0001 judge
        call and a missing eval datapoint; the log entry preserves
        observability while the parse salvages the run. If the string
        isn't valid JSON list, fall through to the normal Pydantic
        list-type error so the failure mode stays loud.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logging.getLogger(__name__).warning(
                        "[rubric_eval_schema] LLM returned "
                        "dimension_scores as JSON-string of length %d "
                        "instead of a structured array; parsed-and-continued "
                        "(see schema-vs-LLM drift class).",
                        len(v),
                    )
                    return parsed
            except json.JSONDecodeError:
                pass
        return v


# ── agent_id → SchemaClass dispatch ──────────────────────────────────────


SCHEMA_BY_AGENT_ID_BASE: dict[str, type[BaseModel]] = {
    "sector_quant": QuantAnalystOutput,
    "sector_qual": QualAnalystOutput,
    "sector_peer_review": JointFinalizationOutput,
    "macro_economist": MacroEconomistRawOutput,
    "ic_cio": CIORawOutput,
    "thesis_update": HeldThesisUpdateLLMOutput,
}
"""Dispatch map for replay tooling and any other consumer that needs
to resolve an agent_id family to its canonical output schema. The
key is the agent_id base (the part before the first colon — e.g.
``sector_quant`` for ``sector_quant:technology``).

``sector_peer_review`` is mapped to JointFinalizationOutput (the
finalization step's output) rather than QuantAcceptanceVerdict (the
intermediate quant gate); the finalization output is what carries
the team's 2-3 final picks downstream and is the meaningful surface
for replay concordance.

QuantAcceptanceVerdict and MacroCriticOutput are intentionally NOT in
this map — they're side-LLM utility calls inside larger agents, not
the agent's final output."""


def resolve_schema_for_agent(agent_id: str) -> type[BaseModel] | None:
    """Look up the LLM-output schema for an ``agent_id``. Returns None
    when the agent_id family doesn't have a registered schema (matches
    the ``evals.judge.resolve_rubric_for_agent`` pattern).

    The agent_id may be plain (``"macro_economist"``) or namespaced
    (``"sector_quant:technology"``, ``"thesis_update:AAPL"``). The
    colon namespace separator is the existing capture convention.
    """
    base_id = (agent_id or "").split(":", 1)[0]
    return SCHEMA_BY_AGENT_ID_BASE.get(base_id)
