# SEC EDGAR Natural Language Analyst

Conversational financial analysis workspace for querying SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A) alongside deterministic period-over-period financial metrics (Revenue, Operating Income, Net Income, and R&D expense).

This document covers how to run, test, and operate the app, and lists the few remaining gaps against [`docs/project_reference.md`](docs/project_reference.md).

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

> **Local Authentication Note:** Outside Cloud Run (`IAP_STRICT_MODE=false`), the UI can mint a local ES256 Workspace/Okta developer JWT via `POST /api/v1/auth/sso/dev-token` so all RBAC-protected `/api/v1/*` routes work locally without external IdP setup.

### Live Data & Gemini Configuration
Every request reads live GCP data (BigQuery `financial_metrics` / `filing_chunks` via `VECTOR_SEARCH`, Vertex AI Search, Vertex AI `text-embedding-005`, Cloud Firestore `analyst_sessions` / `analyst_feedback`, and the GCS corpus bucket), and every answer is written by live Gemini through the Google ADK `Runner` (SSE streaming over a persistent background `asyncio` event loop that caches `Gemini` clients for HTTP/2 keepalive reuse). There is no local corpus, no synthesized filing text, and no template-synthesis fallback. If all findings in a live Gemini report fail claim-level verification on the first pass, one verification-guided regeneration attempt runs with exact rejection reasons; if still unverifiable, the answer is withheld (`answer_verified: false`, shown as **Answer Withheld** in the UI) rather than replaced. Upstream failures return HTTP `503`.

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
The image contains the installed package, the built UI, `prompts/`, and `looker/`; no corpus data is baked in. At runtime every data read goes to BigQuery, Vertex AI Search, and GCS.

```bash
make smoke                          # In-process: GET /health (data_backend block) + UI served from /
SMOKE_LIVE_DATA=true make smoke     # ...plus a live BigQuery read through /api/v1/companies
make docker-build                   # Build the production image (sec-edgar-analyst:local)
make docker-smoke                   # Run the image and smoke-test /health + UI over HTTP
SMOKE_LIVE_DATA=true make docker-smoke   # Mount local ADC and include the live data check
```

`GET /health` includes `data_backend` (project, BigQuery dataset/tables, Vertex AI Search engine, hybrid retrieval backends `["bigquery_vector_search", "vertex_ai_search"]`, GCS corpus bucket, and embedding model), `session_memory` (Firestore `analyst_sessions` persistence status), and `observability` blocks.

### Core API Endpoints
All `/api/v1/*` routes enforce authenticated identity (IAP JWT assertion, Workspace/Okta OIDC JWT, or local ES256 dev token) and Cloud Identity group RBAC:

| Method & Route | Purpose |
| :--- | :--- |
| `POST /api/v1/chat/stream` | Primary SSE streaming endpoint. Emits immediate `connected` event (`< 50 ms`), live `progress` events (`scope_resolved`, `sector_map_start`, `sector_map_progress`, `synthesizing`), `15 s` keepalive heartbeats, and final `result` (or `error`) payload with `Cache-Control: no-cache` and `X-Accel-Buffering: no`. |
| `POST /api/v1/chat` | Backward-compatible synchronous JSON conversational RAG endpoint (`run_conversational_rag_turn`). |
| `GET /api/v1/sessions` | Lists persistent multi-turn conversational sessions strictly scoped to the authenticated user's email (`analyst_sessions` collection in Firestore). |
| `DELETE /api/v1/sessions/{session_id}` | Deletes one session owned by the authenticated user (`404` if missing or owned by another user). |
| `GET /api/v1/companies` | Dynamic ~501-issuer S&P 500 catalog & available fiscal years from live BigQuery `financial_metrics`. |
| `GET /api/v1/section` | Loads full cached Item 1A / Item 7 10-K Markdown section from GCS for the split-pane Source Viewer. |
| `POST /api/v1/feedback` | Persists thumbs-up/down ratings and analyst comments to Firestore (`analyst_feedback`). |
| `GET /health` | Service readiness, `data_backend`, `session_memory`, and `observability` status. |

### Observability: Latency Tracing & Structured Logs
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
├── docs/observability.md            # Tracing, structured logging, correlation surfaces & GCP lookup guide
├── prompts/                         # Versioned agent prompts: registry.yaml (`latest` -> v5) + v1/ (all agents), v2/–v5/ (FinancialAnalystAgent)
├── evals/                           # 43-item golden eval dataset (golden_dataset.jsonl across 6 categories), 17 regression gates (thresholds.yaml), schema guide
├── src/edgar_capstone/
│   ├── server.py                    # FastAPI gateway, auth/RBAC routes, SSE /api/v1/chat/stream, /api/v1/sessions, /api/v1/chat, shutdown flush hooks
│   ├── telemetry.py                 # Async BigQuery telemetry/audit sinks, FirestoreSessionStore (`analyst_sessions`), Firestore feedback & LookML catalog
│   ├── observability/               # OpenTelemetry -> Cloud Trace, trace-correlated JSON logs -> Cloud Logging
│   │   ├── tracing.py               # TracerProvider + CloudTraceSpanExporter, propagators, FastAPI/httpx instrumentation, @traced
│   │   ├── structured_logging.py    # Cloud Logging JSON formatter (trace/span IDs, Error Reporting), uvicorn log routing
│   │   └── middleware.py            # Per-request log context, X-Trace-Id / Server-Timing headers, latency access log (emitted in `finally` even on 500s)
│   ├── agents/                      # Google ADK agents, live Runner SSE engine, ReAct, map-reduce & scope validation
│   │   ├── live_runtime.py          # Persistent background asyncio loop, cached ADK Gemini clients, compact prompt assembly, boundary-balanced evidence leads
│   │   ├── chat_engine.py           # Conversational RAG turn: guardrails, scope validation, balanced retrieval, map-reduce, ReAct, telemetry
│   │   ├── scope_validation.py      # Dynamic 501-issuer EntityCatalog (with dual-class CIK dedup), all 11 GICS sectors, clarification, out-of-corpus & coverage notices
│   │   ├── sector_map_reduce.py     # Full-sector batched map-reduce (single BQ ML.DISTANCE + QUALIFY query, parallel Flash batches + quote verifier, SSE progress)
│   │   ├── react_loop.py            # Structured ReAct loop, single verification-guided live retry, async off-path Flash judge pool + clean atexit shutdown
│   │   ├── financial_analyst.py     # FinancialAnalystAgent (BuiltInPlanner w/ native default thinking, output_schema=AnalystReport, retrieval budget)
│   │   ├── prompt_store.py          # Versioned prompt loader (local / GCS / Vertex AI) + runtime model variables
│   │   ├── context_memory.py        # Model routing, low_thinking_config_for_model, ContextLatencyTracker, user-scoped Firestore session memory & deterministic compaction
│   │   ├── evaluation_and_coding.py # Flash LLM-as-a-Judge evaluator (incl. citation_precision_score) & SQL/Python/LookML coding assist
│   │   ├── callbacks.py             # Guardrails, first-chunk SSE TTFT capture & grounding callbacks
│   │   ├── structured_report.py     # Table-aware claim verification (honors SEC Markdown pipe tables & chunk scale), XBRL inline tags, deterministic coverage notices (no uncited Notes)
│   │   ├── material_variances.py    # flag_material_variances: deterministic anomaly rules + MD&A explanations with period_of_report & XBRL provenance
│   │   ├── disclosure_deltas.py     # compare_disclosures_across_years: YoY wording deltas with both-year citations
│   │   └── tools.py                 # Pooled httpx adapters for BigQuery (native financial_metrics + VECTOR_SEARCH + ML.DISTANCE), Vertex AI Search, & pre-embedded 10-K passage ranking
│   ├── dataset/                     # SEC EDGAR 10-K XBRL + Markdown extraction pipeline (edgartools)
│   │   ├── paths.py                 # Install-layout-aware data root resolution (source tree, /app/data, cwd)
│   ├── vector_db/                   # Section-aware 10-K chunker, pooled Vertex AI 768-d embedder (with bounded query cache & 5x retries), chunk-health monitor
│   ├── evals/                       # Golden eval models, 17-gate scorers, runner, BigQuery golden_eval_set / eval_results I/O
│   └── security/                    # IAP/OIDC JWT verification, Cloud Identity RBAC, SPIFFE S2S auth, Model Armor + SDP (service-account email exclusion, zero `import re`)
├── looker/                          # LookML semantic layer (manifest, model, 6 views, 2 LookML dashboards)
├── frontend/                        # React 19 + TypeScript + Tailwind v4 split-pane terminal UI w/ SSE streaming & XBRL Fact inspector (14 Vitest tests; see frontend/README.md)
├── scripts/                         # CLI entrypoints: dataset build, vector DB build (+ --upload-only), R&D backfill, BigQuery / Vertex AI Search publishing, container smoke test, secret scanner, golden evals (run_evals, load_golden_dataset, refresh_golden_expectations)
├── terraform/                       # GCP IaC: BigQuery (native views, golden_eval_set, eval_results), Firestore, Discovery Engine, Model Armor/SDP exclusion, Cloud Run, IAM/CI live access, IAP, Cloud Monitoring p95 latency alert & dashboard, opt-in VPC-SC & Cloud Build triggers
├── tests/                           # Live GCP + live Gemini pytest suite (`>=80%` coverage enforced via pytest-cov)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + Python 3.11 runtime); no corpus data in the image
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

Outstanding items remaining against [`docs/project_reference.md`](docs/project_reference.md). Section numbers (§) and objective IDs (B1–B3, T1–T3) refer to the reference spec.

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
- In `.github/workflows/deploy.yml`, a regression solely on `p95_ttft_ms` emits a non-blocking `::warning::` while all 16 quality, grounding, citation, and safety gates remain hard-blocking deploy gates.

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

