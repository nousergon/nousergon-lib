# alpha-engine-lib — Code Index

> Index of submodules + key files. Companion to [README.md](README.md). System overview lives in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs).
>
> Last reviewed: 2026-05-05

## Module purpose

Shared utility library used by all 6 modules of Nous Ergon — logging, freshness checks, trading-calendar arithmetic, ArcticDB helpers, agent-decision capture, LLM cost tracking, RAG retrieval, canonical agent-output schemas. No proprietary trading logic, no model weights, no agent prompts.

The lib's job is to keep the same code from being maintained six times.

## Where things live

| Concept | File |
|---|---|
| Package version + public surface | [`src/alpha_engine_lib/__init__.py`](src/alpha_engine_lib/__init__.py) |
| Structured logging + flow-doctor attach | [`src/alpha_engine_lib/logging.py`](src/alpha_engine_lib/logging.py) |
| Preflight (`BasePreflight`) + freshness primitives + `check_deploy_drift` | [`src/alpha_engine_lib/preflight.py`](src/alpha_engine_lib/preflight.py) |
| ArcticDB read/write helpers + symbol enumeration | [`src/alpha_engine_lib/arcticdb.py`](src/alpha_engine_lib/arcticdb.py) |
| Trading-day arithmetic — `now_dual()`, `session_for_timestamp()`, `add_trading_days()` | [`src/alpha_engine_lib/dates.py`](src/alpha_engine_lib/dates.py) |
| Pure-Python NYSE calendar (through 2030) | [`src/alpha_engine_lib/trading_calendar.py`](src/alpha_engine_lib/trading_calendar.py) |
| Artifact-freshness substrate — `ArtifactSpec`, `check_freshness`, `resolve_dedup_key` | [`src/alpha_engine_lib/artifact_freshness.py`](src/alpha_engine_lib/artifact_freshness.py) |
| Decision-artifact schema + capture wrapper | [`src/alpha_engine_lib/decision_capture.py`](src/alpha_engine_lib/decision_capture.py) |
| LLM cost tracking — token-aware, cache-hit-aware | [`src/alpha_engine_lib/cost.py`](src/alpha_engine_lib/cost.py) |
| Canonical LLM-output Pydantic schemas (14 classes) + `resolve_schema_for_agent` dispatch | [`src/alpha_engine_lib/agent_schemas.py`](src/alpha_engine_lib/agent_schemas.py) |
| RAG retrieval — `retrieve`, `ingest_document`, `embed_texts` | [`src/alpha_engine_lib/rag/__init__.py`](src/alpha_engine_lib/rag/__init__.py) |
| RAG Postgres + pgvector connection | [`src/alpha_engine_lib/rag/db.py`](src/alpha_engine_lib/rag/db.py) |
| RAG Voyage `voyage-3-lite` (512d) embedding wrapper | [`src/alpha_engine_lib/rag/embeddings.py`](src/alpha_engine_lib/rag/embeddings.py) |
| RAG hybrid retrieval (vector + filters) | [`src/alpha_engine_lib/rag/retrieval.py`](src/alpha_engine_lib/rag/retrieval.py) |
| RAG schema — `chunks` table + HNSW indexes | [`src/alpha_engine_lib/rag/schema.sql`](src/alpha_engine_lib/rag/schema.sql) |

## Versioning + install

| Mechanism | Where |
|---|---|
| Package version | [`src/alpha_engine_lib/__init__.py`](src/alpha_engine_lib/__init__.py) (`__version__`) + [`pyproject.toml`](pyproject.toml) (`version`) — kept in sync |
| Tagged release | `git tag v0.X.Y` at the merge SHA; consumers pin via `git+https://github.com/cipher813/alpha-engine-lib@v0.X.Y` |
| Optional extras | `[arcticdb]` · `[flow_doctor]` · `[rag]` · `[dev]` — see [pyproject.toml](pyproject.toml) |

## Tests

| Coverage | File |
|---|---|
| Logging + flow-doctor wiring | [`tests/test_logging.py`](tests/test_logging.py) |
| Preflight primitives + drift checks | [`tests/test_preflight.py`](tests/test_preflight.py) |
| ArcticDB helpers | [`tests/test_arcticdb.py`](tests/test_arcticdb.py) |
| Trading-day arithmetic | [`tests/test_dates.py`](tests/test_dates.py) |
| NYSE calendar | [`tests/test_trading_calendar.py`](tests/test_trading_calendar.py) |
| Artifact-freshness substrate (spec validation + cycle resolution + 5 check branches) | [`tests/test_artifact_freshness.py`](tests/test_artifact_freshness.py) |
| Decision-capture schema + S3 round-trip (moto) | [`tests/test_decision_capture.py`](tests/test_decision_capture.py) |
| Cost tracking (cache-hit semantics) | [`tests/test_cost.py`](tests/test_cost.py) |
| Agent-output schema validators + dispatch | [`tests/test_agent_schemas.py`](tests/test_agent_schemas.py) |
| RAG retrieval (mocked pgvector) | [`tests/test_rag.py`](tests/test_rag.py) |

```bash
pip install -e ".[dev,arcticdb,flow_doctor,rag]"
pytest
```
