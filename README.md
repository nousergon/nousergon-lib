# alpha-engine-lib

> Part of [**Nous Ergon**](https://nousergon.ai) — Autonomous Multi-Agent Trading System. Repo and S3 names use the underlying project name `alpha-engine`.

[![Part of Nous Ergon](https://img.shields.io/badge/Part_of-Nous_Ergon-1a73e8?style=flat-square)](https://nousergon.ai)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Phase 2 · Reliability](https://img.shields.io/badge/Phase_2-Reliability-e9c46a?style=flat-square)](https://github.com/cipher813/alpha-engine-docs#phase-trajectory)

Shared utility library used by all 6 modules of Nous Ergon. Cross-cutting concerns only — logging, freshness checks, trading-calendar arithmetic, ArcticDB helpers, agent-decision capture, LLM cost tracking. No proprietary trading logic, no model weights, no agent prompts.

The lib's job is to keep the same code from being maintained six times.

---

## Install

```
# requirements.txt
alpha-engine-lib @ git+https://github.com/cipher813/alpha-engine-lib@v0.4.0
```

Tagged releases: `v0.1.0`, `v0.2.0`, `v0.3.0`, `v0.4.0`, etc. Consumers pin to a specific tag. Breaking changes bump the minor version while Alpha Engine is in pre-1.0.

```bash
# With optional extras
pip install "alpha-engine-lib[arcticdb] @ git+https://github.com/cipher813/alpha-engine-lib@v0.4.0"
```

| Extra | Pulls in | When you need it |
|---|---|---|
| `[arcticdb]` | `arcticdb`, `pandas` | Anything that calls `check_arcticdb_fresh` or the ArcticDB read/write helpers |
| `[flow_doctor]` | `flow-doctor` | Logging integration that escalates ERROR-level events to flow-doctor |
| `[rag]` | `psycopg2-binary`, `pgvector`, `numpy` | The `rag` submodule — Neon pgvector RAG retrieval/ingestion |
| `[dev]` | `pytest`, lint tooling | Local development |

## Modules

### `logging` — structured logging + flow-doctor attach

Replaces the near-identical `log_config.py` copies that used to live in alpha-engine-data and alpha-engine-executor. Consumers call `setup_logging` once at process startup:

```python
from alpha_engine_lib.logging import setup_logging

setup_logging("data-collector", flow_doctor_yaml="/path/to/flow-doctor.yaml")
```

- Text mode by default; JSON via `ALPHA_ENGINE_JSON_LOGS=1`
- Flow-doctor attaches as an ERROR-level handler when `FLOW_DOCTOR_ENABLED=1` (requires `[flow_doctor]` extra)

### `preflight` — fail-fast connectivity + freshness checks

Runs at the top of every entrypoint, before any real work starts. Primitives live on `BasePreflight`; each consumer subclasses and overrides `run()`:

```python
from alpha_engine_lib.preflight import BasePreflight

class DataPreflight(BasePreflight):
    def __init__(self, bucket, mode):
        super().__init__(bucket)
        self.mode = mode

    def run(self):
        self.check_env_vars("AWS_REGION")
        if self.mode == "phase1":
            self.check_env_vars("FRED_API_KEY", "POLYGON_API_KEY")
        self.check_s3_bucket()
        if self.mode == "daily":
            self.check_arcticdb_fresh("universe", "SPY", max_stale_days=4)
```

Failed checks raise `RuntimeError` with an explanatory message. Consumers catch nothing — the raise propagates up through `main()` → non-zero exit → Step Function `HandleFailure` → flow-doctor notification. The point is to fail before paying for any LLM calls or downstream work.

### `arcticdb` — read/write helpers + symbol enumeration

Wrappers around the ArcticDB Python client. Standardizes the URI format, library naming, and read paths so each consumer doesn't reinvent the connection logic.

### `dates` — trading-day arithmetic

`now_dual()` returns a `(calendar_date, trading_day)` pair following the rule `trading_day = last_closed_trading_day(now)`. Strictly backward-looking; never ahead. `session_for_timestamp(ts)` resolves any timestamp to its trading session. Used at every artifact-write site to prevent calendar/trading-day drift between modules.

### `trading_calendar` — NYSE holiday detection

Pure-Python NYSE calendar through 2030. No `pandas-market-calendars` dependency.

### `artifact_freshness` — absence-driven S3 artifact monitoring substrate

The lib-side piece of the artifact-freshness-monitor arc closing the silent absence-of-artifact bug class. SF Catch, flow-doctor, and substrate-health-check are all event-driven (failure → alert); this module is the substrate for the absence-driven complement (silence → alert).

```python
from datetime import datetime, date, timezone
from alpha_engine_lib.artifact_freshness import (
    ArtifactSpec, check_freshness, resolve_dedup_key,
)
from alpha_engine_lib.alerts import publish

spec = ArtifactSpec(
    artifact_id="backtest_pit_parity",
    s3_bucket="alpha-engine-research",
    s3_key_template="backtest/{date}/pit_parity.json",
    cadence="saturday_sf",
    sla_minutes_after_cron=180,
    severity="warning",
    owner_repo="alpha-engine-backtester",
    created_at=date(2026, 5, 27),
)
result = check_freshness(s3_client, spec, datetime.now(timezone.utc))
if result.state in ("missing", "stale", "probe_failed"):
    publish(
        f"[{result.state}] {spec.artifact_id}: {result.reason}",
        severity=spec.severity,
        dedup_key=resolve_dedup_key(spec, datetime.now(timezone.utc)),
    )
```

Pure function — `check_freshness(s3_client, spec, now)` returns a `CheckResult` with no side effects beyond the injected `s3_client.head_object`. NYSE-holiday-aware (Memorial Day Monday weekday-SF cron returns `state="fresh"` with a holiday reason). Recovery-substitution-aware (canonical 404 + recovery-key fresh ⇒ `state="fresh"` with `recovery_substituted=True`). Grace-period gate for newly-onboarded specs (default 2 cycles).

The freshness-monitor Lambda (`alpha-engine-data/lambdas/freshness_monitor/`, ships in a follow-up PR) walks the `alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml` SoT, calls this substrate per row, and routes via `alpha_engine_lib.alerts.publish` with the resolved dedup key.

### `decision_capture` — agent decision audit logger

Captures every agent decision as a structured artifact: prompt metadata (id + version), input snapshot, agent output, and cost. Each decision becomes replayable, auditable, and attributable to a specific prompt revision. Backbone of the Phase 2 measurement substrate.

### `cost` — LLM cost tracking

Token-aware cost computation following Anthropic's prompt-caching semantics (cache-write vs cache-read pricing). Used by every LLM call site to attach a `cost_usd` to its output.

### `agent_schemas` — canonical LLM-output Pydantic schemas

Shared contract surface for the 14 LLM-output classes used in `with_structured_output(...)` calls across the research pipeline (sector quant + qual + peer review, macro economist + critic, held-stock thesis update, CIO, eval-judge rubric). Lives here so downstream tooling — replay harness in alpha-engine-backtester, future cheap-model-concordance signals — can validate against the canonical contract without a heavy cross-repo dep on research.

```python
from alpha_engine_lib.agent_schemas import (
    QuantAnalystOutput,
    JointFinalizationOutput,
    CIORawOutput,
    HeldThesisUpdateLLMOutput,
    resolve_schema_for_agent,
)

# Dispatch by captured agent_id (e.g. "sector_quant:technology" → QuantAnalystOutput)
schema = resolve_schema_for_agent(agent_id)
```

`SCHEMA_BY_AGENT_ID_BASE` covers the 6 canonical agent families: `sector_quant`, `sector_qual`, `sector_peer_review`, `macro_economist`, `ic_cio`, `thesis_update`. Validators that defend observed LLM failure modes (sector-modifier clamp, JSON-string-as-list parser, `min_length=1` on CIO decisions) move with their classes.

### `pillars` — canonical 6-pillar attractiveness scoring shapes

Pydantic shapes for the institutional / SOTA refactor of research-module composite scoring — replaces the opaque `quant_score + qual_score` two-bucket model with a canonical 6-pillar decomposition: **Quality / Value / Momentum / Growth / Stewardship / Defensiveness**. Pillar set is the AQR Style Premia / Morningstar Economic Moat / Greenblatt / Piotroski / Fama-French / Asness "QMJ" consensus.

```python
from alpha_engine_lib.pillars import (
    PILLARS,
    MoatAssessment,
    PillarSubscore,
    QualitativePillarAssessment,
)

# Qual Analyst emits via with_structured_output(QualitativePillarAssessment).
# Each of the 6 PillarSubscore fields carries 0-100 + confidence + evidence;
# the Quality pillar additionally carries a structured MoatAssessment
# (Morningstar wide/narrow/none + 6-archetype primary moat type + trend) —
# the qualitative core of Quality, persisted per ticker for time-series
# trend tracking.
```

Each `PillarSubscore` decomposes into optional `quant_component` (from the factor substrate) + `qual_component` (from the agent rubric) for traceability through the composite scoring layer. Catalyst is preserved as an orthogonal `catalyst_horizon_modulation: int ∈ [-20, 20]` (a horizon shift on near-term attractiveness), not a 7th pillar weight.

Origin: 2026-05-20 attractiveness-pillars-260520 plan-doc arc. Phase 1 (this module) ships the schema layer; Phases 2-7 wire it through alpha-engine-research, alpha-engine-data, alpha-engine-backtester, and alpha-engine-dashboard.

### `rag` — semantic retrieval over SEC filings, transcripts, and theses

Neon pgvector backbone shared by `alpha-engine-research` (qual analyst's `query_filings` tool) and `alpha-engine-data` (weekly RAGIngestion step). Re-exports a small surface — `retrieve`, `ingest_document`, `document_exists`, `embed_texts`, `get_connection`, `is_available` — and ships the canonical `schema.sql` as package data.

```python
from alpha_engine_lib.rag import retrieve

results = retrieve(
    query="competitive risks and market position",
    tickers=["AAPL"],
    doc_types=["10-K", "10-Q", "earnings_transcript"],
    top_k=8,
)
```

Requires the `[rag]` extra. Embeddings are Voyage `voyage-3-lite` (512d); the database backend is Neon Postgres with pgvector + HNSW indexes.

### `ssm_dispatcher` — SSM send-command + poll chokepoint

Canonical Python primitive for the `run_ssm` bash helper that previously appeared as a ~54-line mirror across each dispatcher script that drives a spot instance over the SSM transport. The pre-lift shape — base64-wrap the script body, `aws ssm send-command --document-name AWS-RunShellScript`, loop on `get-command-invocation`, stream the `StandardOutputContent` delta, propagate the inner exit — now lives in one place where the polling cadence, error-class handling, and S3 output-key layout match across every consumer.

```bash
python -m alpha_engine_lib.ssm_dispatcher run \
  --instance-id "$INSTANCE_ID" \
  --description "bootstrap" \
  --timeout 3600 \
  --output-bucket "$S3_BUCKET" \
  --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
  --region "$AWS_REGION" \
  --script-stdin <<'BOOTSTRAP'
set -eo pipefail
export HOME=/home/ec2-user AWS_REGION=us-east-1
# ...the script body the SSM target will execute...
BOOTSTRAP
```

Exit 0 on `Success`; exit 1 on any terminal non-Success status, send-command failure, or unrecoverable poll failure; exit 2 on bad CLI input. `InvocationDoesNotExist` during the first 60s after SendCommand counts as a registration race and keeps polling — closes the 2026-05-23 Saturday SF substrate weakness at the chokepoint rather than per-SF-JSON Retry block.

### `ssm_log_capture` — SSM-step log capture + S3 ship-on-exit chokepoint

Pairs with `ssm_dispatcher` on the SSM target side. The dispatcher script tells the target instance to invoke `python -m alpha_engine_lib.ssm_log_capture run --slug X --log /var/log/X.log -- bash <launcher>`; the target wrapper tees the launcher's stdout/stderr to a local log file and to its own stdout (so the SSM `StandardOutputContent` channel still surfaces output to the dispatcher), then on exit ships the full local log to `s3://alpha-engine-research/_ssm_logs/{slug}/{date}/{host}-{time}.log` regardless of the inner exit code. Replaces the inline `trap 'aws s3 cp ...' EXIT` pattern that broke under ASL `States.Array` escape semantics (2026-05-22 Friday-PM dry-pass catch).

### `ec2_spot` — capacity-resilient spot-launch chokepoint

Rotates across `(instance_type × subnet)` combinations on `InsufficientInstanceCapacity` / `InsufficientHostCapacity` / `Unsupported` / `InvalidAvailabilityZone` / `SpotMaxPriceTooLow`; non-capacity errors raise immediately. CLI exit 64 distinguishes capacity exhaustion from generic failure. Replaces the hardcoded single-subnet + single-instance-type launch pattern that mirrored across each dispatcher; landed 2026-05-22 after the third-recurrence-in-a-month spot-launch fragility.

### `locks` — producer-side writer locks via S3 conditional PUT

`universe_writer_lock(writer_id, ttl_seconds=3600)` context manager that uses `PutObject(IfNoneMatch="*")` to claim a single-writer lease on `s3://alpha-engine-research/locks/universe-writer.lock`. The first writer's conditional PUT succeeds; subsequent writers get `LockHeldByAnotherWriterError` carrying the live `LockHolder` body (writer_id + started_at + ttl_epoch + hostname + pid) for operator diagnostics. Soft-TTL self-recovery deletes-and-re-acquires when the on-disk lock's `ttl_epoch` has elapsed; the operator-side S3 lifecycle on `locks/` (expires-after-days=1) is the hard backstop. Release on context exit is best-effort and never masks an inner exception. Closes the producer-side half of the same single-writer-per-resource invariant the SF MutualExclusionGuard (DynamoDB-side) covers at the Step Function entry point; the lib lift is the chokepoint that picks up the third adopter for free (predictor weight-promote, backfill loops, etc.).

### `quant` — portfolio analytics engine (factor risk, VaR/CVaR, attribution, returns)

The shared institutional-analytics engine: pure, front-end- and data-source-agnostic functions that *describe and measure* a portfolio (performance, risk, attribution) with **no advisory logic** — it sits on the "analytics, not advice" side of the line. Lifted from robodashboard's `analytics/` after the 2026-06-03 cross-repo leverage audit, so both the alpha-engine fleet and robodashboard consume one engine instead of parallel reimplementations. Import the submodule you need (the package keeps no eager imports, so the stdlib-only modules import without numpy):

- **`quant.factor_risk`** — statistical factor risk model `Σ = B·F·Bᵀ + D`: `estimate_factor_model` (time-series factor-ETF / Fama-MacBeth loadings), `portfolio_risk` (ex-ante vol + factor/idio split + per-factor variance contribution), `tracking_error`, `benchmark_exposure`, and a numpy-only `ledoit_wolf_cov` (no sklearn). The estimator-agnostic consumption core (`portfolio_risk`/`tracking_error`) consumes any `FactorRiskModel` (B, F, D). **Needs numpy** — `pip install "alpha-engine-lib[quant]"`.
- **`quant.risk_measures`** — parametric (Gaussian, Acklam inverse-normal, no scipy) + historical VaR & CVaR, as positive loss fractions at a horizon (stdlib).
- **`quant.riskstats`** — `volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown` (stdlib).
- **`quant.returns`** — `xirr` (money-weighted, Newton + bisection), `time_weighted_return` (GIPS), `cumulative_return`, `annualize` (stdlib).
- **`quant.attribution`** — single-period Brinson-Fachler decomposition (`brinson_fachler`) + multi-period Cariño linking (`link_periods`) (stdlib).

```python
from alpha_engine_lib.quant.risk_measures import historical_cvar
from alpha_engine_lib.quant.factor_risk import estimate_factor_model, portfolio_risk
```

## How it's used

All six Nous Ergon module repos depend on this lib:

| Module | Repo | What it imports from here |
|---|---|---|
| Data | [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) | `logging`, `preflight`, `arcticdb`, `dates`, `trading_calendar`, `rag` (ingestion), `ec2_spot` + `ssm_log_capture` + `ssm_dispatcher` (spot launchers) |
| Research | [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) | `logging`, `decision_capture`, `cost`, `dates`, `rag` (retrieval), `agent_schemas` (canonical LLM-output contracts) |
| Predictor | [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) | `logging`, `preflight`, `arcticdb`, `dates`, `ec2_spot` + `ssm_log_capture` + `ssm_dispatcher` (spot launcher) |
| Executor | [`alpha-engine`](https://github.com/cipher813/alpha-engine) | `logging`, `preflight`, `arcticdb`, `dates`, `trading_calendar` |
| Backtester | [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) | `logging`, `preflight`, `arcticdb`, `dates`, `agent_schemas` (replay-harness Pydantic validation), `ec2_spot` + `ssm_log_capture` + `ssm_dispatcher` (spot launcher) |
| Dashboard | [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) | `logging`, `arcticdb`, `dates`, hosts the SSM-target `.venv` that `ssm_dispatcher` invokes via `python -m` |

## Development

```bash
git clone https://github.com/cipher813/alpha-engine-lib.git
cd alpha-engine-lib
pip install -e ".[dev,arcticdb,flow_doctor]"
pytest
```

## Scope discipline

This repo is intentionally narrow. Code lands here when **at least two consumers would otherwise maintain their own copy**. New modules land as their own minor release with per-consumer adoption — no lockstep updates.

Code that **does not** belong here:
- Anything tunable (scoring weights, risk thresholds, sizing parameters) → `alpha-engine-config` (private)
- Agent prompts → `alpha-engine-config` (private)
- Module-specific business logic → that module's repo

## License

MIT — see [LICENSE](LICENSE).
