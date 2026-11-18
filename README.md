# SEC EDGAR Natural Language Analyst

Conversational financial analysis workspace for querying SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A) alongside deterministic period-over-period financial metrics (Revenue, Operating Income, Net Income, and R&D expense).

This document covers how to run, test, and operate the app, and lists the few remaining gaps against [`docs/project_reference.md`](docs/project_reference.md). Intentional architectural divergence from the reference spec is fine as long as the trade-off is deliberate and documented.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 22+ and `npm`
- `gcloud` with valid Application Default Credentials (`gcloud auth application-default login`) for a principal that can read BigQuery `sec_edgar`, the SEC corpus bucket, and Vertex AI Search, and call Vertex AI (embeddings + Gemini). There is no offline mode: the app, the tests, and the evals all read live GCP data and call live Gemini.
- Optional for infrastructure operations: Terraform `>= 1.7`

### Install, Run, and Test
```bash
# 1. Create .venv, install Python package + dev dependencies, and install frontend packages
make install

# 2. Start FastAPI (:8000) and Vite React UI (:5173) together for local development
make dev

# 3. Or build the frontend bundle and serve the single-origin app from FastAPI (:8000)
make start

# 4. Run secret scan, backend pytest suite (live GCP + live Gemini, >=80% coverage gate), ESLint, and frontend Vitest suite
make test

# 5. Run the CI lint/format gates (ruff check + format --check, terraform fmt/validate, ESLint),
#    auto-fix them, or run every CI gate at once
make lint
make format
make check
```

> **Packaging & Optional `[ingest]` Extras:** Runtime installs (`pip install .` in `Dockerfile` and `requirements.txt`) stay lean by excluding heavy offline ingestion packages (`edgartools`, `pandas`, `google-cloud-storage`) and dev tools (`pytest`, `ruff`). Install `pip install ".[ingest]"` for offline SEC dataset / vector-DB scripts, or `pip install ".[dev]"` (`make install`) for full dev + test tooling. The unused `uv.lock` file has been removed from the repository root.

> **Local Authentication Note:** Outside Cloud Run (`IAP_STRICT_MODE=false`), the UI can mint a local ES256 Workspace/Okta developer JWT via `POST /api/v1/auth/sso/dev-token` so all RBAC-protected `/api/v1/*` routes work locally without external IdP setup.

### Live Data & Gemini Configuration
Every request reads live GCP data (BigQuery `financial_metrics` / `filing_chunks` via `VECTOR_SEARCH`, Vertex AI Search, Vertex AI `text-embedding-005`, Cloud Firestore `analyst_sessions` / `analyst_feedback`, and the GCS corpus bucket), and every answer is written by live Gemini through the Google ADK `Runner` (SSE streaming over a persistent background `asyncio` event loop that caches `Gemini` clients for HTTP/2 keepalive reuse and executes sync tools via `ToolThreadPoolConfig`). `FinancialAnalystAgent` decides which deterministic engines each question needs — `calculate_period_variances`, `flag_material_variances` (with MD&A explanations), `compare_disclosures_across_years`, `sector_thematic_map_reduce`, and `query_looker_financial_explore` are agent-invoked ADK tools, never pre-computed — and emits the canonical intent in its structured `AnalystReport`; tool outputs are hydrated from ADK `session_state` before claim verification, so qualitative answers carry no unsolicited variance tables, anomaly flags, or Key Metrics. Active ADK agents are `FinancialAnalystAgent` (primary orchestrator) and `NarrativeEvaluatorAgent` — the legacy unused supervisor hierarchy (`RootOrchestrator` / `agent.py`, `ContextCacheAgent`, and `CodingAssistantAgent`) has been retired, while granular SPIFFE S2S per-agent workload identities remain enforced in `security/agent_identity.py`.

There is no local corpus, no synthesized filing text, and no template-synthesis fallback. If all findings in a live Gemini report fail claim-level verification on the first pass, one verification-guided regeneration attempt runs with exact rejection reasons; if still unverifiable, the answer is withheld (`answer_verified: false`, shown as **Answer Withheld** in the UI) rather than replaced. Upstream failures return HTTP `503`.

On interactive turns, the Gemini 3.5 Flash LLM-as-a-Judge runs asynchronously off the request path: initial turn telemetry records `evaluation_grade="pending"` and `evaluation_score=NULL` (never a fabricated placeholder score so BigQuery `AVG(evaluation_score)` in `v_session_analytics_summary` stays accurate), and once the background judge completes it records the real grade and score via `BigQueryTelemetrySink.record_async_evaluation_result(...)` (updating the local turn in place and streaming a schema-valid row to `session_telemetry_events`; pending evaluations are drained cleanly on shutdown via `flush_pending_evaluations`).

The Makefile targets (`dev`, `backend`, `start`, `test`, `eval-all`, `smoke`) set the GCP environment explicitly, because developer shells often export an unrelated `GCP_PROJECT_ID`. `test` and `eval-all` keep telemetry, session persistence, and feedback in memory so they don't write to the production telemetry tables.

```bash
gcloud auth application-default login
make dev                                          # project iron-radio-514217-q7, Vertex AI, Gemini on `global`
CORPUS_PROJECT_ID=<other-project> make dev        # point at another project

# Outside the Makefile, setting GEMINI_API_KEY (with GOOGLE_GENAI_USE_VERTEXAI unset) uses the
# Gemini API instead of Vertex AI. Optional overrides (defaults shown). Gemini models are served from the `global` endpoint in
# iron-radio-514217-q7; BigQuery / GCS / Model Armor / embeddings stay in GOOGLE_CLOUD_LOCATION.
# EDGAR_ANALYST_MODEL=gemini-3.8-flash  EDGAR_FLASH_MODEL=gemini-3.5-flash  EDGAR_GEMINI_LOCATION=global  EDGAR_TTFT_SLO_MS=3000
```

### Container Packaging & Smoke Tests
The image installs only the lean runtime package (`pip install .` — no `edgartools`, `pandas`, `google-cloud-storage`, `pytest`, or `ruff`), the built UI, and `prompts/`; no corpus data is baked in (the un-deployed LookML layer was removed — see [Observability](#observability-cloud-console-native-tracing-logs-metrics--bigquery)). At runtime every data read goes to BigQuery, Vertex AI Search, and GCS.

```bash
make smoke                          # In-process: GET /health (data_backend block) + UI served from /
SMOKE_LIVE_DATA=true make smoke     # ...plus a live BigQuery read through /api/v1/companies
make docker-build                   # Build the production image (sec-edgar-analyst:local)
make docker-smoke                   # Run the image and smoke-test /health + UI over HTTP
SMOKE_LIVE_DATA=true make docker-smoke   # Mount local ADC and include the live data check
```

`GET /health` includes `root_agent` (`FinancialAnalystAgent`), `active_agents` (`["FinancialAnalystAgent", "NarrativeEvaluatorAgent"]`), `data_backend` (project, BigQuery dataset/tables, Vertex AI Search engine, hybrid retrieval backends `["bigquery_vector_search", "vertex_ai_search"]`, GCS corpus bucket, and embedding model), `session_memory` (Firestore `analyst_sessions` persistence status), and `observability` blocks.

### Core API Endpoints
FastAPI routes are organized into modular routers under `src/edgar_capstone/api/routers/`. All `/api/v1/*` routes (except public SSO discovery/exchange/dev-token endpoints) enforce authenticated identity (IAP JWT assertion, Workspace/Okta OIDC JWT, or local ES256 dev token) and Cloud Identity group RBAC:

| Method & Route | Router | Purpose |
| :--- | :--- | :--- |
| `GET /health` | `health.py` | Service readiness, active ADK agents, `data_backend`, `session_memory`, and `observability` status. |
| `POST /api/v1/chat/stream` | `chat.py` | Primary SSE streaming endpoint. Emits immediate `connected` event (`< 50 ms`), live `progress` events (`scope_resolved`, `sector_map_start`, `sector_map_progress`, `synthesizing`), `15 s` keepalive heartbeats, and final `result` (or sanitized `error`) payload with `Cache-Control: no-cache` and `X-Accel-Buffering: no`. |
| `POST /api/v1/chat` | `chat.py` | Backward-compatible synchronous JSON conversational RAG endpoint (`run_conversational_rag_turn`). Enforces `FilingScopeMixin` (`start_year <= end_year`). |
| `GET /api/v1/companies` | `filings.py` | Dynamic ~501-issuer S&P 500 catalog & available fiscal years from live BigQuery `financial_metrics`. |
| `GET /api/v1/section` | `filings.py` | Loads full cached Item 1A / Item 7 10-K Markdown section from GCS for the split-pane Source Viewer (strictly validated query bounds). |
| `GET /api/v1/sessions` | `sessions.py` | Lists persistent multi-turn conversational sessions strictly scoped to the authenticated user's email (`analyst_sessions` collection in Firestore). |
| `DELETE /api/v1/sessions/{session_id}` | `sessions.py` | Deletes one session owned by the authenticated user (`404` if missing or owned by another user). |
| `POST /api/v1/feedback` | `feedback.py` | Persists thumbs-up/down ratings and analyst comments to Firestore (`analyst_feedback`). |
| `GET /api/v1/feedback` | `feedback.py` | Lists persisted analyst feedback ratings and audit comments (`audit:read`). |
| `GET /api/v1/telemetry/sessions` | `telemetry.py` | Returns session-length, company-scope, tool-usage, and BigQuery telemetry summaries (`audit:read`). |
| `GET /api/v1/auth/config` | `auth.py` | Public SSO/IAP discovery, RBAC policy, runtime SA, and per-agent SPIFFE identity metadata (no secrets). |
| `POST /api/v1/auth/sso/exchange` | `auth.py` | Validates a Workspace/Okta/IAP OIDC ID token for API Bearer access (emits audit logs on 401/403). |
| `POST /api/v1/auth/sso/dev-token` | `auth.py` | Local-only ES256 dev Workspace/Okta JWT broker (disabled in Cloud Run / `IAP_STRICT_MODE=true`; emits audit logs on 401/403). |
| `GET /api/v1/auth/me` | `auth.py` | Returns the verified principal for the active Bearer JWT / IAP session. |
| `POST /api/v1/auth/logout` | `auth.py` | Revokes the active Bearer JWT fingerprint and writes a logout audit event. |
| `GET /api/v1/auth/audit` | `auth.py` | Lists structured authentication and 10-K access audit events (`audit:read`). |
| `GET /api/v1/admin/rbac/policies` | `auth.py` | Admin-only RBAC governance and per-agent identity inspection (`rbac:admin`). |
| `GET /api/v1/secrets/status` | `auth.py` | Secret Manager binding metadata with redacted values (`secrets:inspect`). |
| `GET /` / `GET /{full_path:path}` | `spa.py` | Single-origin Terminal UI static bundle catch-all (registered last; explicitly returns `404` for `/api/*` and `/healthz`). |

> **Pruned Redundant / Test-Only Routes:** The following legacy endpoints have been removed and now return HTTP `404` or `405` (with `GET /healthz` returning `404` even when `frontend/dist` is built): `GET /healthz`, `POST /api/v1/section`, `GET /api/v1/auth/rbac`, `GET /api/v1/explore/yoy_variances`, `GET /api/v1/explore/audit_trail`, `GET /api/v1/looker/catalog`, `GET /api/v1/telemetry/context-latency`, `POST /api/v1/intent`, `POST /api/v1/evaluate`, `POST /api/v1/coding-assist`, `POST /api/v1/variances`, `POST /api/v1/search`, and `GET /api/v1/metrics`.

### Observability: Cloud Console-Native Tracing, Logs, Metrics & BigQuery
Each API request is one OpenTelemetry trace, exported to **Cloud Trace**. Every log line is structured JSON carrying the same trace ID and goes to **Cloud Logging**. On Cloud Run (`K_SERVICE` set) both turn on automatically, so Cloud Build and GitHub Actions don't need extra env vars. Locally, spans stay in-process unless you choose an exporter, and logs are readable text that still includes `[trace=… span=…]`. Full details are in [`docs/observability.md`](docs/observability.md).

```bash
# Print spans to stdout and emit Cloud Logging-style JSON logs locally
EDGAR_TRACE_EXPORTER=console EDGAR_LOG_FORMAT=json make backend

# Send local spans to Cloud Trace with ADC
gcloud auth application-default login
EDGAR_TRACE_EXPORTER=gcp GOOGLE_CLOUD_PROJECT=iron-radio-514217-q7 make backend

# Every response carries its trace ID and server-side latency
curl -si localhost:8000/api/v1/companies -H "Authorization: Bearer $TOKEN" | grep -i -E 'x-trace-id|server-timing'
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `EDGAR_TRACING_ENABLED` | `true` | Master switch for tracing (`OTEL_SDK_DISABLED=true` also turns it off). |
| `EDGAR_TRACE_EXPORTER` | `auto` | `auto` exports to Cloud Trace on Cloud Run and keeps spans in-process elsewhere. Other values: `gcp`, `console`, `otlp`, `none`. |
| `EDGAR_TRACE_SAMPLE_RATIO` | `1.0` | Head-sampling ratio. A sampled upstream parent (load balancer / Cloud Run) is always honored. |
| `EDGAR_TRACE_CAPTURE_CONTENT` | `false` | Default for ADK's `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`. Keeps prompts, 10-K text, and model output out of Cloud Trace. |
| `EDGAR_TRACE_EXCLUDED_URLS` | health + static assets | Comma-separated URL regexes that get no server span. |
| `EDGAR_TRACE_INSTRUMENT_HTTPX` | `true` | Client spans for outbound httpx calls (Vertex AI, BigQuery, Firestore, Model Armor, SDP, …). |
| `EDGAR_LOG_FORMAT` | `auto` | `json` on Cloud Run, `text` locally. |
| `EDGAR_LOG_LEVEL` | `INFO` | Root log level for app, uvicorn, and library loggers. |

`GET /health` includes an `observability` block showing the active exporter, sampling ratio, instrumentation state, and log format.

Operations run entirely in the Google Cloud Console:

- **Cloud Monitoring dashboards** (Terraform): *SEC EDGAR RAG Vector Database — Chunk Health & Ingestion Monitoring* and *SEC EDGAR Analyst — LLM Telemetry, Cost, Sessions & Security Audit*. The second covers API rate, status, and latency; LLM TTFT, latency, calls, cache hits, tokens, and approximate USD; turns by intent and fact-check; judge scores; security audit events; and logs panels for errors, audit, and LLM telemetry. `terraform output llm_observability_dashboard_url` prints the link.
- **Low-cardinality log-based metrics** `sec_edgar/*` (`api_request_latencies`, `llm_ttft_ms`, `llm_model_latency_ms`, `llm_calls`, `llm_input_tokens`, `llm_output_tokens`, `llm_cost_usd`, `llm_cache_savings_usd`, `session_turns`, `evaluation_scores`, `security_audit_events`). They never use `session_id`, `user_email`, `trace_id`, `call_id`, or tickers as labels.
- **Alert policies:** API p95 latency, HTTP 5xx errors, and elevated 401/403 denials or guardrail blocks, all notifying `monitoring_notification_channels`.
- **BigQuery:** the app streams `llm_telemetry_events`, `session_telemetry_events`, and `security_audit_logs` (each with `trace_id`). The SQL views `v_telemetry_cost_and_latency` (exact USD), `v_session_analytics_summary`, and `v_security_audit_summary` are queried from BigQuery Studio. Cloud Logging also exports the raw telemetry LogEntries to the dedicated `sec_edgar_ops_logs` dataset.

> **Intentional divergence — no Looker/LookML:** the reference spec's "BI dashboards exported to Looker" item is deliberately replaced by Cloud Monitoring dashboards plus version-controlled BigQuery SQL views. No Looker instance was ever deployed, so `looker/` and `lookml.py` were removed and `/health` no longer reports a LookML catalog. If executive reporting is needed later, attach Looker Studio to the BigQuery views (*Explore with Looker Studio*). The ADK tool `query_looker_financial_explore` already queries BigQuery directly and keeps its name until a follow-up rename PR.

### Golden Evals (`make eval-all`)
`make eval-all` runs the golden query–variance–explanation set in [`evals/golden_dataset.jsonl`](evals/golden_dataset.jsonl) through the chat pipeline and exits non-zero when any gate in [`evals/thresholds.yaml`](evals/thresholds.yaml) regresses. See [`evals/README.md`](evals/README.md) for the record schema and how to add items.

```bash
make eval-all                                # Live Gemini answers + Gemini 3.5 Flash judge + p95 TTFT gate on live GCP data; report in build/evals/
EVAL_WRITE_BQ=true make eval-all             # Also stream the run to BigQuery sec_edgar.eval_results
make load-golden-dataset                     # Replace BigQuery sec_edgar.golden_eval_set with the committed file
make refresh-golden-expectations             # Recompute golden variance numbers from the engine (review the diff!)
```

---

## Repository Structure

```text
├── docs/project_reference.md        # Target architecture and product specification
├── docs/observability.md            # Cloud Trace, Cloud Logging, log-based metrics, alerts, dashboards, BigQuery views & correlation guide
├── prompts/                         # Versioned agent prompts: registry.yaml (`latest` -> v5), v1/ (financial_analyst, narrative_evaluator, model_armor_policy), v5/financial_analyst.yaml
├── evals/                           # 63-item golden eval dataset (golden_dataset.jsonl across 6 categories), 17 regression gates (thresholds.yaml), schema guide
├── src/edgar_capstone/
│   ├── server.py                    # Slim FastAPI entrypoint (`create_app()`, lifespan flush/shutdown hooks, dynamic package version)
│   ├── api/                         # Modular FastAPI API layer (schemas, shared deps/error handlers, 8 APIRouters)
│   │   ├── dependencies.py          # Exception handlers (401/403/503), `audited_permissions`, runtime SA dependency
│   │   ├── schemas.py               # `FilingScopeMixin` (`start_year <= end_year`), Chat, SSO, and Feedback request schemas
│   │   └── routers/                 # `health.py`, `auth.py`, `filings.py`, `chat.py`, `sessions.py`, `telemetry.py`, `feedback.py`, `spa.py`
│   ├── common/                      # Zero-SDK shared foundations (lazy-import friendly)
│   │   ├── env.py                   # Fail-safe `env_bool` (empty/invalid preserves default), `split_csv_env`, numeric env parsers
│   │   ├── domain.py                # Tickers (`normalize_edgar_ticker`), CIK, XBRL tags, GICS sectors, `lookup_year_key`, `sha256_text`
│   │   ├── formatting.py            # USD/pct/bps formatters, percentile, quote normalization, resilient `parse_model_output`
│   │   ├── jsonl.py                 # Streaming `iter_jsonl` (`skip_corrupt`), `write_jsonl`, `write_json`
│   │   ├── concurrency.py           # `run_parallel_with_context` (propagates `contextvars` / OTel spans across worker threads)
│   │   ├── cache.py                 # Thread-safe `BoundedTTLCache`, `CacheMetrics`, `parse_10k_cache_key`
│   │   └── models.py                # Shared `FilingMetadataBase` schema
│   ├── gcp/                         # Shared lightweight GCP REST primitives (no heavy SDK imports)
│   │   ├── defaults.py              # Uniform env precedence (`GCP_PROJECT_ID` > `GOOGLE_CLOUD_PROJECT` > `GCLOUD_PROJECT` > `EDGAR_GCP_PROJECT_ID`), location, bucket, dataset, SEC UA
│   │   ├── rest_client.py           # `GcpRestClient`: thread-safe ADC refresh, pooled httpx, quota header, `Retry-After` & status translation
│   │   ├── bigquery.py              # Identifier injection guard, named params, recursive wire decoding, schema loader, `BigQueryRestClient`
│   │   ├── firestore.py             # Firestore REST v1 wire codec & `FirestoreCollectionStore`
│   │   ├── lro.py                   # `poll_operation`: per-poll token refresh, transient 429/5xx retries, immediate fail-fast on 403/404
│   │   ├── discovery_engine.py      # Discovery Engine / Vertex AI Search host & datastore path builders
│   │   └── paths.py                 # Repository/container asset & repo root resolution
│   ├── telemetry.py                 # Async BigQuery telemetry/audit sinks (w/ `trace_id`, LRU session cap, `record_async_evaluation_result`), Firestore feedback/session stores
│   ├── observability/               # OpenTelemetry -> Cloud Trace, trace-correlated JSON logs -> Cloud Logging
│   │   ├── tracing.py               # TracerProvider + CloudTraceSpanExporter, strict `traceparent`/`X-Cloud-Trace-Context` parsers, FastAPI/httpx instrumentation, `@traced`
│   │   ├── structured_logging.py    # Cloud Logging JSON formatter (reserved payload keys), non-mutating `TraceAwareTextFormatter`, uvicorn log routing
│   │   └── middleware.py            # Per-request log context, `X-Trace-Id` / `Server-Timing` headers, latency access log (emitted in `finally` even on 500s)
│   ├── agents/                      # PEP 562 lazy-loaded Google ADK agents, live Runner SSE engine, ReAct, map-reduce & scope validation
│   │   ├── adapters/                # Modular GCP tool adapters: `bigquery.py`, `vertex_search.py`, `gcs.py`, `registry.py`
│   │   ├── factories.py             # ADK agent factories (`build_edgar_agent`, `create_financial_analyst_agent` w/ `BuiltInPlanner` default thinking & `output_schema=AnalystReport`, `create_narrative_evaluator_agent`)
│   │   ├── intent.py                # Canonical intents; agent-emitted intent + tool-trace fallback (`resolve_turn_intent`)
│   │   ├── tool_runtime.py          # Turn-scoped session_state seeding/hydration of agent-invoked deterministic tool outputs + SSE tool context
│   │   ├── sector_tool.py           # Agent-invocable `sector_thematic_map_reduce` ADK tool (full-sector map-reduce + SSE progress)
│   │   ├── prompt_builders.py       # Compact prompt assembly, boundary-balanced evidence leads, judge citation selection
│   │   ├── evaluation.py            # Flash LLM-as-a-Judge evaluator (`evaluate_variance_explanation`, `citation_precision_score`)
│   │   ├── live_runtime.py          # Persistent background asyncio loop, cached ADK Gemini clients, SSE runner w/ `ToolThreadPoolConfig`
│   │   ├── chat_engine.py           # Conversational RAG turn: guardrails, scope validation, RAG evidence pre-pass, ReAct, tool-output hydration, telemetry
│   │   ├── scope_validation.py      # Dynamic 501-issuer `EntityCatalog` (dual-class CIK dedup, guarded `V`/Visa detection), all 11 GICS sectors, clarification & coverage notices
│   │   ├── sector_map_reduce.py     # Full-sector batched map-reduce (single BQ `ML.DISTANCE` + `QUALIFY` query, parallel Flash batches + quote verifier, SSE progress)
│   │   ├── react_loop.py            # Structured ReAct loop, single verification-guided live retry, async off-path Flash judge pool (`pending`/`None` -> `record_async_evaluation_result`) + clean atexit shutdown
│   │   ├── financial_analyst.py     # Thin backward-compatibility re-export shim (`DEFAULT_ANALYST_MODEL`, `DEFAULT_ANALYST_PROMPT_VERSION`, `create_financial_analyst_agent`)
│   │   ├── prompt_store.py          # Versioned prompt loader (local / GCS / Vertex AI) w/ `latest` registry resolution & `@retry_transient_errors` fallback
│   │   ├── context_memory.py        # Model routing, `low_thinking_config_for_model`, `ContextLatencyTracker`, user-scoped Firestore session memory & deterministic compaction
│   │   ├── callbacks.py             # Guardrails, first-chunk SSE TTFT capture & grounding callbacks
│   │   ├── citations.py             # Canonical inline 10-K citation tag formatter & excerpt cleanup helpers
│   │   ├── resilience.py            # `@with_tool_resilience` (ADK error dicts + `tool.<name>` spans) & `@retry_transient_errors` (propagating retry wrapper)
│   │   ├── schemas.py               # Pydantic schemas for metrics, variances, citations, `AnalystReport`, and `NarrativeEvaluationReport`
│   │   ├── structured_report.py     # Table-aware claim verification (honors SEC Markdown pipe tables & chunk scale), XBRL inline tags, deterministic coverage notices
│   │   ├── variance_engine.py       # Deterministic financial metric conversion & multi-period / YoY variance calculation engine
│   │   ├── hybrid_rag.py            # Parallel hybrid RAG retrieval combining BigQuery `VECTOR_SEARCH` and Vertex AI Search
│   │   ├── material_variances.py    # `flag_material_variances`: deterministic anomaly rules + MD&A explanations with `period_of_report` & XBRL provenance
│   │   ├── disclosure_deltas.py     # `compare_disclosures_across_years`: YoY wording deltas with both-year citations & chunk-fallback theme mention counts
│   │   └── tools.py                 # ADK `FunctionTool` facade re-exporting `adapters/*`, `variance_engine.py`, `hybrid_rag.py`, and pre-embedded 10-K passage ranking
│   ├── dataset/                     # SEC EDGAR 10-K XBRL + Markdown extraction pipeline (optional `[ingest]` extra for `edgartools`/`pandas`/`storage`)
│   │   ├── fiscal_years.py          # Fiscal year focus resolution & reconciliation helpers
│   │   ├── section_parser.py        # Item 1A / Item 7 Markdown cleaning, boundary truncation, and `rewrite_markdown_fiscal_year`
│   │   ├── financial_extractor.py   # XBRL income-statement & R&D concept extraction
│   │   ├── xbrl_facts.py            # SEC `companyfacts` R&D backfill (`BRK.B` -> `BRK-B` normalization)
│   │   ├── models.py                # Lightweight Pydantic schemas (zero heavy ingest SDK imports)
│   │   └── paths.py                 # Install-layout-aware data root resolution (source tree, `/app/data`, cwd)
│   ├── vector_db/                   # Lazy-loaded section-aware 10-K chunker, pooled Vertex AI 768-d embedder, chunk-health monitor
│   ├── evals/                       # Golden eval models, 17-gate scorers, runner (strict exception-path denominators), BigQuery I/O
│   └── security/                    # PEP 562 lazy-loaded auth, RBAC, SPIFFE S2S identities, Secret Manager, and Model Armor
│       ├── auth.py                  # IAP/OIDC JWT verification, strict Bearer-authoritative dual-header auth, audit logging
│       ├── jwt_codec.py             # Shared RFC 7515/7517 base64url, JWK, and ES256/RS256 codec helpers
│       ├── rbac.py                  # Cloud Identity Group RBAC & custom `sec-analyst-sa` governance
│       ├── agent_identity.py        # Retained granular SPIFFE S2S per-agent identities & attestation broker (`RootOrchestrator`, `FinancialAnalystAgent`, `ContextCacheManager`, `ModelArmorGuardrail`)
│       ├── secrets.py               # Secret Manager REST client, constant-time `RedactedSecret`, repo secret scanner
│       └── model_armor/             # Modular Model Armor v1 + Cloud SDP/DLP package (zero `import re`): `config.py`, `projection_filter.py`, `emulator.py`, `client.py`, `service.py`, `adk_plugin.py`
├── frontend/                        # React 19 + TypeScript + Tailwind v4 split-pane terminal UI w/ SSE streaming & modular inspector (`services/httpClient.ts`, `components/inspector/{CitationsTab,FilingReaderTab,XbrlFactTab}.tsx`; 14 Vitest tests)
├── scripts/                         # CLI entrypoints sharing `scripts/_cli.py`: dataset build, vector DB build (+ `--upload-only`), R&D backfill, BigQuery / Vertex AI Search publishing, container smoke test, secret scanner, golden evals
├── terraform/                       # GCP IaC: BigQuery (native views, `golden_eval_set`, `eval_results`), Firestore, Discovery Engine, Model Armor/SDP exclusion, Cloud Run, IAM/CI live access, IAP, Cloud Monitoring log-based metrics, alerts & 2 dashboards, ops-log BigQuery sink, opt-in VPC-SC & Cloud Build triggers
├── tests/                           # Live GCP + live Gemini pytest suite (`>=80%` coverage gate) plus fast offline `test_refactor_regressions.py`
├── requirements.txt                 # Lean runtime dependencies only (excludes dev and `[ingest]` extras; `uv.lock` removed)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + lean Python 3.11 runtime); no corpus data in the image
├── .dockerignore                    # Keeps secrets and gitignored heavy corpus artifacts out of the build context
├── cloudbuild.yaml                  # Cloud Build: lint + pytest-cov>=80 + build/smoke -> live eval-all gate (deploy mode) -> Cloud Run deploy
└── .github/workflows/deploy.yml     # GitHub Actions WIF CI/CD workflow (pytest-cov>=80, WIF live smoke test, main-only eval-all gate -> deploy)
```

---

## Sharing the App Internally

The app is never anonymous. IAP sits on the Cloud Run service, so anyone who opens the URL has to sign in with a Google account, and only principals in `iap_authorized_principals` get through.

1. **Choose who can open the app** in `terraform/variables.tf` or a tfvars file. Use groups or whole domains only; individual `user:` bindings are rejected:
   ```hcl
   iap_authorized_principals = [
     "domain:jackbetcher.altostrat.com",        # default: everyone in this Argolis domain
     "group:sec-edgar-analysts@example.com",     # a Google Group
   ]
   iap_default_app_role = "analyst"             # or "viewer"; "" admits nobody without an RBAC group
   ```
   The org policy `iam.allowedPolicyMemberDomains` decides which outside domains (for example `google.com` groups) can be granted access.
2. **Apply:** `terraform -chdir=terraform apply`.
3. **Share the link** from `terraform -chdir=terraform output -raw app_url` (the `https://sec-edgar-analyst-…run.app` URL). People who aren't authorized get an IAP "You don't have access" page, and the app never sees their requests.
4. **Revoke** by removing the principal and applying again. Users can sign out from the header; that clears their IAP session cookie.

---

## Remaining Gaps vs. `docs/project_reference.md`

Outstanding items remaining against [`docs/project_reference.md`](docs/project_reference.md). Section numbers (§) and objective IDs (B1–B3, T1–T3) refer to the reference spec. *(Intentional architectural divergences — such as retiring unused supervisor agents or keeping Vertex AI Vector Search undeployed in favor of BigQuery `VECTOR_SEARCH` — are deliberate and documented rather than tracked as gaps.)*

| Area | Spec | Priority | Status | Next Action |
| :--- | :--- | :--- | :--- | :--- |
| Chat-turn raw model `p95_ttft_ms` | §5 · B1 | P1 | Open | Iterate on raw Vertex AI `global` p95 TTFT (`<= 3,000 ms` target) without forcing a low thinking level on `FinancialAnalystAgent`. |
| Live observability & session verification | §5, §11 | P2 | In Progress | After Cloud Run deploys, confirm one end-to-end trace across Cloud Trace, Cloud Logging, BigQuery `trace_id`, and Firestore `analyst_sessions`. |
| Cloud Build triggers | §9 | P2 | Deferred | Authorize the 2nd-gen GitHub App OAuth connection and store a GitHub PAT in Secret Manager, then apply with `enable_cloudbuild_trigger = true`. |
| VPC Service Controls enforcement | §9 | P2 | Blocked | Needs an org-delegated Access Context Manager policy (`PERMISSION_DENIED` on org `1059565594977`). |

---

### Raw Model `p95_ttft_ms` Tail Latency *(§5 Latency · B1)*

**Requirement:** `p95 <= 3.0 s` TTFT for single- and peer-company queries; sector-wide thematic queries stream progress within `3.0 s`.

**Current state:**
- Interactive SSE streaming (`POST /api/v1/chat/stream`) emits its first status event in `< 50 ms`, and warm end-to-end turns complete in `~12–22 s` (`1,853–1,921 ms` warm raw TTFT) while preserving default native thinking (`include_thoughts=True`, no forced low thinking level) on `FinancialAnalystAgent`.
- Raw model-only `p95_ttft_ms` on the unprovisioned Vertex AI `global` endpoint still exceeds `3,000 ms` due to intermittent backend tail jitter (`evals/thresholds.yaml` keeps `p95_ttft_ms: 3000` strictly as the target metric).
- In `.github/workflows/deploy.yml`, a regression solely on `p95_ttft_ms` emits a non-blocking `::warning::` while all 17 quality, grounding, citation, selective-display, and safety gates remain hard-blocking deploy gates.

**Done when:** `p95_ttft_ms <= 3,000 ms` passes consistently with default analytical thinking (e.g., via Provisioned Throughput, regional model availability, or further retrieval/prefill optimization).

### Live Observability & Session Verification *(§5 Observability, §11)*

**Requirement:** End-to-end tracing through OpenTelemetry and Cloud Trace, structured logs correlated by trace ID, BigQuery telemetry joinable to traces, and durable multi-turn sessions in Firestore (`analyst_sessions`).

**Current state:** Terraform observability resources (`sec_edgar/api_request_latencies` metric, `api_latency_p95` alert policy, dashboard latency widget, `golden_eval_set`, `eval_results`) are applied.

**Done when:** On the deployed Cloud Run service, one live `/api/v1/chat` (and `/api/v1/chat/stream`) request produces a single trace in Cloud Trace with nested FastAPI, ReAct, tool, and model spans; Cloud Logging lines group under `trace="projects/iron-radio-514217-q7/traces/<X-Trace-Id>"`; BigQuery `llm_telemetry_events` / `session_telemetry_events` rows carry that `trace_id`; and a follow-up turn reloads scope from Firestore `analyst_sessions`.

### Cloud Build Triggers *(§9 CI/CD)*

**Requirement:** Cloud Build deploys automatically on successful merges.

**Current state:** `terraform/cloudbuild.tf` defines the 2nd-gen GitHub connection and PR/main triggers, but `enable_cloudbuild_trigger` defaults to `false` because authorizing the Google Cloud Build 2nd-gen GitHub App requires interactive browser OAuth and a GitHub PAT in Secret Manager. Deployments currently use keyless GitHub Actions WIF (`.github/workflows/deploy.yml`).

**Done when:** The 2nd-gen GitHub connection is authorized and applied with `enable_cloudbuild_trigger = true` (if Cloud Build triggers are desired alongside or instead of GitHub Actions).

### VPC Service Controls Enforcement *(§9 Data Protection)*

**Requirement:** A strict service perimeter locks down Vertex AI, Cloud Run, GCS, and BigQuery.

**Current state:** Defined in `terraform/vpc_service_controls.tf` (`enable_vpc_service_controls = false`, dry-run until `vpc_sc_enforce = true`). The sandbox account lacks org-level Access Context Manager permissions (`PERMISSION_DENIED` on org `1059565594977`).

**Done when:** An org admin delegates a scoped policy, Terraform applies `-var=enable_vpc_service_controls=true -var=vpc_sc_access_policy_id=<POLICY_NUMBER>`, dry-run violations are reviewed, and `vpc_sc_enforce = true` is enabled.

