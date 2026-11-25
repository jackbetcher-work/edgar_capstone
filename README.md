# SEC EDGAR Natural Language Analyst

A conversational analysis workspace for S&P 500 Form 10-K filings. Analysts ask questions in plain English and get cited answers grounded in two sources:

- **Narrative text:** Item 1A (Risk Factors) and Item 7 (MD&A).
- **Financial metrics:** deterministic period-over-period calculations for Revenue, Operating Income, Net Income, and R&D expense, sourced from XBRL.

Answers are written by Gemini through the Google Agent Development Kit (ADK). Every claim is checked against the retrieved passages and computed figures before it reaches the user.

---

## How a Query Is Answered

```mermaid
flowchart TD
    UI["React Terminal UI"] -->|"POST /api/v1/chat/stream"| API["FastAPI"]
    API --> AUTH["Auth and RBAC: IAP or OIDC JWT"]
    AUTH --> PRE["Parallel pre-pass: Model Armor/SDP guardrails, Firestore session recall, catalog, optional ScopeContinuityAgent (Flash-Lite)"]
    PRE -->|blocked| OUT
    PRE --> SCOPE["Scope & subject-continuity resolution: tickers, sector, sub-industry, fiscal years, topic isolation"]
    SCOPE -->|"needs clarification or out of corpus"| OUT
    SCOPE --> RAG["Hybrid RAG evidence pre-pass"]
    RAG --> AGENT["FinancialAnalystAgent: Gemini via ADK Runner"]
    AGENT <-->|"calls only the tools it needs"| TOOLS["Deterministic tools: variances, anomaly flags, disclosure deltas, sector map-reduce, 10-K search"]
    AGENT --> VERIFY["Claim-level verification: one guided retry, otherwise answer withheld"]
    VERIFY --> RG["Response guardrails"]
    RG --> OUT["SSE result streamed to the UI"]

    subgraph GCP["Google Cloud data layer"]
        BQ[("BigQuery: financial_metrics, filing_chunks")]
        VAS[("Vertex AI Search")]
        GCS[("GCS 10-K corpus")]
        FS[("Firestore: sessions, feedback")]
        TEL[("BigQuery telemetry and audit tables")]
    end

    PRE -.-> FS
    PRE -.-> BQ
    SCOPE -.-> BQ
    RAG -.-> BQ
    RAG -.-> VAS
    TOOLS -.-> BQ
    TOOLS -.-> GCS
    RG -.-> FS
    RG -.-> TEL
```

Each turn runs through [`run_conversational_rag_turn`](src/edgar_capstone/agents/orchestration/chat_engine.py):

1. **Authenticate.** All `/api/v1/*` routes require an identity (Direct Cloud Run IAP assertion, Google Workspace OIDC JWT, or a local dev token). Access is controlled by Workspace domain allowlists and role-based permissions.
2. **Parallel pre-pass & prompt guardrails.** Model Armor (which runs Cloud Sensitive Data Protection server-side via configured DLP templates) and an SEC projection filter run once at the orchestration boundary concurrently with Firestore session recall and S&P 500 catalog loading. When L1 session memory and the catalog are warm, no local PII spans are present, and deterministic scope parsing finds no explicit entity, `ScopeContinuityAgent` (`gemini-3.5-flash-lite`) runs speculatively in parallel with Model Armor (discarded if Model Armor blocks, SDP redacts the prompt, or Firestore refreshes a newer scope). Guardrails block unsafe or projection requests before synthesis.
3. **Resolve scope & isolate subject changes.** [`scope_validation.py`](src/edgar_capstone/agents/orchestration/scope_validation.py) and [`scope_continuity.py`](src/edgar_capstone/agents/orchestration/scope_continuity.py) map the query to tickers, GICS sector, GICS sub-industry (all 126 catalog sub-industries expanded to constituent tickers), fiscal years, and 10-K section. On unresolved or ambiguous turns, `ScopeContinuityAgent` classifies continuation vs. subject switch (`continue_prior`, `new_tickers`, `new_sector`, `new_sub_industry`, `broad_corpus`, `unknown_entity`, `clarify`) with strict `EntityCatalog` validation and a conservative deterministic fallback. Subject switches advance `topic_epoch` to isolate ADK session history (`<session_id>::topic-<epoch>`) and omit unrelated prior-company memory from the prompt, while sector or corpus-wide turns clear `remembered_tickers`. Ambiguous or out-of-corpus questions end here with a clarification or coverage notice.
4. **Retrieve evidence.** [`hybrid_rag.py`](src/edgar_capstone/agents/tools/hybrid_rag.py) queries BigQuery `VECTOR_SEARCH` over `filing_chunks` and Vertex AI Search (Discovery Engine) in parallel (with per-company-year coverage slices), then balances hits across the tickers in scope.
5. **Reason.** [`react_loop.py`](src/edgar_capstone/agents/orchestration/react_loop.py) runs `FinancialAnalystAgent` via the ADK streaming runner ([`live_runtime.py`](src/edgar_capstone/agents/runtime/live_runtime.py)) and returns constrained `AnalystReport` JSON. The agent calls deterministic tools on demand based on the query.
6. **Verify.** [`structured_report.py`](src/edgar_capstone/agents/evaluation/structured_report.py) is the single deterministic verification gate, checking every finding against the retrieved passages and hydrated tool outputs before rendering fallback Markdown and returning `structured_report` JSON. If every finding fails, one guided regeneration runs with the rejection reasons; if that also fails, the answer is withheld (`answer_verified: false`).
7. **Response guardrails, citation deduplication, persistence, and telemetry.** The response is screened once by Model Armor and the SEC projection filter, exact/near-duplicate citations are collapsed once on the backend ([`citations.py`](src/edgar_capstone/agents/core/citations.py)), intent is finalized ([`intent.py`](src/edgar_capstone/agents/orchestration/intent.py)), the turn is saved to Firestore, and telemetry is streamed to BigQuery.

LLM-as-a-Judge scoring ([`judge.py`](src/edgar_capstone/agents/evaluation/judge.py)) runs both **asynchronously off the critical path** on verified live turns (sampled via `EDGAR_ASYNC_JUDGE_SAMPLE_RATE`, default `1.0`, streaming grades and scores to BigQuery `session_telemetry_events` and the `sec_edgar/evaluation_scores` Cloud Monitoring metric) and **synchronously offline** in the golden evaluation suite (`make eval-all`). Every live request reads GCP data and calls Gemini; upstream failures return HTTP `503`.

### Agents and Tools

| Component | Module | Role |
| :--- | :--- | :--- |
| `FinancialAnalystAgent` | [`runtime/factories.py`](src/edgar_capstone/agents/runtime/factories.py) | Primary ADK agent (`gemini-3.8-flash`, native thinking). Output is constrained to the `AnalystReport` schema. |
| `NarrativeEvaluatorAgent` | [`evaluation/judge.py`](src/edgar_capstone/agents/evaluation/judge.py) | LLM-as-a-Judge (`gemini-3.8-flash`, `ThinkingLevel.LOW`) scoring faithfulness, citation precision, and explanation quality asynchronously on sampled live turns (`EDGAR_ASYNC_JUDGE_SAMPLE_RATE`) and synchronously in `make eval-all`. |
| `SectorMapSummarizerAgent` | [`tools/sector_map_reduce.py`](src/edgar_capstone/agents/tools/sector_map_reduce.py) | Parallel batch summarizer (`gemini-3.5-flash-lite`) for the map phase of sector-wide thematic reviews. |
| `ScopeContinuityAgent` | [`orchestration/scope_continuity.py`](src/edgar_capstone/agents/orchestration/scope_continuity.py) | Fast structured scope & subject-continuity classifier (`gemini-3.5-flash-lite`, `ThinkingLevel.MINIMAL`) invoked only on unresolved/ambiguous turns (overlapped with Model Armor when warm) and strictly validated against `EntityCatalog`. |
| `calculate_period_variances` | [`tools/sec_tools.py`](src/edgar_capstone/agents/tools/sec_tools.py), [`tools/variance_engine.py`](src/edgar_capstone/agents/tools/variance_engine.py) | Deterministic multi-period and YoY variance engine over BigQuery `financial_metrics`. |
| `flag_material_variances` | [`tools/material_variances.py`](src/edgar_capstone/agents/tools/material_variances.py) | Rule-based XBRL anomaly flags paired with MD&A explanations and filing provenance. |
| `compare_disclosures_across_years` | [`tools/disclosure_deltas.py`](src/edgar_capstone/agents/tools/disclosure_deltas.py) | Deterministic YoY wording deltas (added, removed, modified) using word-trigram alignment with citations from both years. |
| `sector_thematic_map_reduce` | [`tools/sector_tool.py`](src/edgar_capstone/agents/tools/sector_tool.py), [`tools/sector_map_reduce.py`](src/edgar_capstone/agents/tools/sector_map_reduce.py) | Full-sector batched map-reduce over all issuers in a GICS sector. Streams progress over SSE. |
| `search_10k_chunks_rag` · `query_cached_10k_context` | [`tools/sec_tools.py`](src/edgar_capstone/agents/tools/sec_tools.py), [`tools/hybrid_rag.py`](src/edgar_capstone/agents/tools/hybrid_rag.py) | Targeted 10-K passage retrieval returning citable `chunk_id`s. |
| `query_analytics_view` | [`tools/sec_tools.py`](src/edgar_capstone/agents/tools/sec_tools.py) | Aggregated BigQuery financial & corpus views (`sector_benchmarks`, `peer_comparison`, `yoy_variances`, `chunk_health`). |

Prompts are Git-versioned in [`prompts/`](prompts/registry.yaml) and loaded through [`prompts/store.py`](src/edgar_capstone/agents/prompts/store.py). The analyst resolves `latest` to `v7`.

### Agents Package Architecture

[`src/edgar_capstone/agents/`](src/edgar_capstone/agents/__init__.py) is split into seven subpackages layered from zero-dependency foundations up to orchestration:

```text
core ──▶ adapters, prompts ──▶ tools ──▶ runtime, evaluation ──▶ orchestration
```

- **`core/`** — Shared Pydantic schemas (`schemas.py`), retry/exception wrappers and OTel tool spans (`resilience.py`), and citation formatting/deduplication helpers (`citations.py`). No dependencies on other `agents` subpackages.
- **`adapters/`** — Live GCP REST clients and thread-safe adapter registry (`bigquery.py`, `vertex_search.py`, `gcs.py`, `registry.py`).
- **`prompts/`** — Local Git-versioned prompt store (`store.py`) and dynamic payload/scope prompt builders (`builders.py`).
- **`tools/`** — ADK `FunctionTool` wrappers, deterministic analytical engines, and turn-scoped tool plumbing (`runtime.py`).
- **`runtime/`** — ADK `Runner` and persistent SSE event loop (`live_runtime.py`), agent factories (`factories.py`), lifecycle callbacks for telemetry and retrieval budgets (`callbacks.py`), and model routing, latency tracking, and session memory (`model_config.py`, `latency_tracker.py`, `context_memory.py`).
- **`evaluation/`** — Claim-level `AnalystReport` verification and fallback Markdown rendering (`structured_report.py`), plus the Flash LLM-as-a-Judge (`judge.py`) used asynchronously on sampled live turns and synchronously in golden evals.
- **`orchestration/`** — Conversational turn entry point and S&P 500 catalog cache (`chat_engine.py`), post-retrieval synthesis & guided verification retry (`react_loop.py`), scope/sub-industry/coverage validation (`scope_validation.py`), Flash-Lite subject-continuity classification (`scope_continuity.py`), and non-lexical intent resolution (`intent.py`).

Two conventions are worth knowing before editing `agents/`:

- **Turn-scoped tool state (`tools/runtime.py`):** Deterministic tools return compact JSON digests to Gemini to keep prompt prefill small, while writing their full payloads (XBRL provenance, flags, deltas, citations) into ADK `session_state`. `turn_scoped_state_seed()` resets those keys each turn so persistent chat sessions don't leak prior outputs, and `hydrate_tool_outputs()` reads them back before claim verification and UI rendering. Non-serializable per-turn handles (SSE progress callback, cached catalog profiles) travel via `tool_turn_context` (`contextvars`).
- **Cold-start imports (`__init__.py`):** `agents/__init__.py` contains no eager imports (and subpackages such as `tools/` and `evaluation/` use PEP 562 `__getattr__` lazy exports), keeping `google.adk`, `google.genai`, and ingest SDKs off the cold-start import path (enforced by `test_lazy_imports_avoid_heavy_sdks_on_cold_start`). Import symbols directly from their canonical subpackage modules.

### Storage Architecture

Data is split across Google Cloud storage and retrieval services by workload pattern:

| Service | Workload pattern | What it stores | Role |
| :--- | :--- | :--- | :--- |
| **BigQuery** | Analytical scans, vector search, append-only logs | Structured XBRL `financial_metrics`, 768-d embedded `filing_chunks` (`VECTOR_SEARCH`), and streaming telemetry, audit, and eval tables | Serverless SQL and vector analytics at S&P 500 scale |
| **Vertex AI Search** | Managed hybrid lexical + semantic retrieval | Indexed 10-K filings with auto query expansion, spell correction, and extractive segments | Pairs lexical recall and extractive snippets with BigQuery dense vector search in parallel for stronger grounding |
| **Cloud Storage (GCS)** | Immutable large text objects | Full Item 1A (Risk Factors) and Item 7 (MD&A) Markdown sections | Low-cost blob storage for serving full filing sections to the source viewer and cached section reads |
| **Cloud Firestore (Native)** | Low-latency mutable JSON CRUD and per-user state | `analyst_sessions` (multi-turn conversation memory) and `analyst_feedback` (thumbs-up/down ratings and comments) | Fast point `GET`/`PATCH`/`DELETE` and filtered queries; serverless scale-to-zero across stateless Cloud Run instances; schemaless nested JSON; SHA-256 `(user_id, session_id)` isolation with transactional read-modify-write concurrency via the official `google-cloud-firestore` SDK ([`firestore_stores.py`](src/edgar_capstone/firestore_stores.py)) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+ and `npm`
- `gcloud` with Application Default Credentials (`gcloud auth application-default login`). The principal needs to read BigQuery `sec_edgar`, the SEC corpus bucket, and Vertex AI Search, and to call Vertex AI (embeddings and Gemini).
- Terraform `>= 1.7` (only for infrastructure work)

### Common Commands

| Command | What it does |
| :--- | :--- |
| `make install` | Create `.venv`, install the package with dev extras, and install frontend packages |
| `make dev` | Run FastAPI (`:8000`) and the Vite UI (`:5173`) with hot reload |
| `make start` | Build the UI and serve the single-origin app from FastAPI (`:8000`) |
| `make test` | Secret scan, backend pytest (live GCP, `>= 80%` coverage gate), ESLint, Vitest |
| `make lint` / `make format` | Ruff, Terraform fmt/validate, ESLint (check or auto-fix) |
| `make check` | Every CI gate (`lint` + `test`) |
| `make eval-all` | Golden eval regression gates (see [Evaluation](#evaluation)) |
| `make smoke` | In-process smoke test of `/health` and the UI |
| `make docker-build` / `make docker-smoke` | Build the production image and smoke-test it over HTTP |
| `make help` | List all targets, including corpus publishing (`publish-financials`, `index-vertex-search`, `backfill-rd`) |

Run `make smoke` or `make docker-smoke` with `SMOKE_LIVE_DATA=true` to also check a live BigQuery read. For fast local iteration without GCP credentials, run `.venv/bin/pytest -m "not live"` (unit and typed fake-SDK integration tests; `@pytest.mark.live` tests hit live GCP and Gemini).

### Configuration

The Makefile pins the GCP environment (project `iron-radio-514217-q7`, `us-central1`, Gemini on the `global` endpoint), because developer shells often export an unrelated `GCP_PROJECT_ID`. `make test` and `make eval-all` keep telemetry and feedback in memory, so they don't write to production tables.

```bash
CORPUS_PROJECT_ID=<other-project> make dev   # point at another project
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `EDGAR_ANALYST_MODEL` | `gemini-3.8-flash` | Reasoning model for `FinancialAnalystAgent` |
| `EDGAR_FLASH_MODEL` | `gemini-3.8-flash` | LLM-as-a-Judge (`NarrativeEvaluatorAgent`, `ThinkingLevel.LOW`) |
| `EDGAR_ASYNC_JUDGE_SAMPLE_RATE` | `1.0` | Sampling probability in `[0.0, 1.0]` for off-critical-path async LLM-as-a-Judge scoring on verified live turns (`0.0` disables live background judging) |
| `EDGAR_LITE_MODEL` | `gemini-3.5-flash-lite` | Sector map-reduce (`SectorMapSummarizerAgent`), scope/continuity classification (`ScopeContinuityAgent`), and event compaction (`LlmEventSummarizer`) |
| `EDGAR_SCOPE_LLM_MODE` | `auto` | Controls `ScopeContinuityAgent`: `auto` (live Flash-Lite in production, deterministic fallback when scripted/offline), `live`, or `deterministic` |
| `EDGAR_GEMINI_LOCATION` | `global` | Vertex AI endpoint for Gemini. BigQuery, GCS, and embeddings use `GOOGLE_CLOUD_LOCATION`. |
| `GEMINI_API_KEY` | unset | Uses the Gemini API instead of Vertex AI when `GOOGLE_GENAI_USE_VERTEXAI` is unset |
| `IAP_STRICT_MODE` | `false` | When `false` (local), the UI can mint an ES256 dev JWT via `POST /api/v1/auth/sso/dev-token`. Always disabled on Cloud Run. |

### Packaging

The runtime install (`pip install .`, used by the `Dockerfile` and `requirements.txt`) is kept lean. Heavy packages are in optional extras:

- `pip install ".[ingest]"` adds the offline dataset, vector-DB, and Terraform-bootstrap build tools (`edgartools`, `pandas`, `google-cloud-service-usage`).
- `pip install ".[dev]"` adds test and lint tooling; `make install` uses this.

The container image packages the runtime library, the built UI, and `prompts/`; corpus data is read from GCP at runtime.

---

## API

Routers are in [`src/edgar_capstone/api/routers/`](src/edgar_capstone/api/routers/). All `/api/v1/*` routes require authentication except SSO discovery, exchange, and dev-token.

| Route | Purpose |
| :--- | :--- |
| `GET /health` | Readiness, plus status of the active agents, data backends, session memory, and observability |
| `POST /api/v1/chat/stream` | Main chat endpoint (SSE). Emits `connected`, `progress`, 15 s heartbeats, then `result` or `error`. |
| `POST /api/v1/chat` | Same pipeline as a single JSON response |
| `GET /api/v1/companies` | S&P 500 catalog and available fiscal years |
| `GET /api/v1/section` | Full Item 1A / Item 7 Markdown from GCS for the source viewer |
| `GET` · `DELETE /api/v1/sessions[/{id}]` | List or delete the caller's conversation sessions |
| `POST` · `GET /api/v1/feedback` | Submit or list analyst ratings and comments |
| `GET /api/v1/telemetry/sessions` | Session, tool-usage, and telemetry summaries (`audit:read`) |
| `GET /api/v1/auth/config` | Public SSO/IAP discovery and RBAC metadata |
| `POST /api/v1/auth/sso/exchange` · `/sso/dev-token` | Exchange an OIDC token, or mint a local dev token |
| `GET /api/v1/auth/me` · `POST /api/v1/auth/logout` | Current principal, or sign out |
| `GET /api/v1/auth/audit` | Authentication and filing-access audit events (`audit:read`) |
| `GET /api/v1/admin/rbac/policies` | RBAC policy and runtime service account inspection (`rbac:admin`) |
| `GET /*` | Serves the built React app |

---

## Observability

Each request is one OpenTelemetry trace exported to **Cloud Trace**. Logs are structured JSON with the same trace ID and go to **Cloud Logging**. Every response includes `X-Trace-Id` and `Server-Timing` headers.

On Cloud Run this is all enabled automatically. Locally, spans stay in-process unless you choose an exporter:

```bash
EDGAR_TRACE_EXPORTER=console EDGAR_LOG_FORMAT=json make backend   # print spans and JSON logs
EDGAR_TRACE_EXPORTER=gcp make backend                             # send spans to Cloud Trace
```

Monitoring is done entirely in the Cloud Console:

- **Dashboard:** Cloud Monitoring dashboard for API traffic/latency, LLM telemetry, cost, sessions, judge scores, and security audit.
- **Metrics:** log-based metrics under `sec_edgar/*`.
- **Alerts:** p95 latency, 5xx errors, and auth denials or guardrail blocks.
- **BigQuery views:** queried from BigQuery Studio.

See [`docs/observability.md`](docs/observability.md) for configuration variables, metric names, and how to correlate traces with logs and BigQuery rows.

---

## Evaluation

`make eval-all` runs the 30-item golden dataset ([`evals/golden_dataset.jsonl`](evals/golden_dataset.jsonl)) through the live chat pipeline. It exits non-zero if any gate in [`evals/thresholds.yaml`](evals/thresholds.yaml) regresses. The gates cover:

- Numeric agreement
- Citation coverage and grounding recall
- Routing and tool selection
- Judge scores
- Safety
- p95 TTFT

```bash
make eval-all                       # report written to build/evals/
EVAL_WRITE_BQ=true make eval-all    # also stream results to BigQuery sec_edgar.eval_results
make load-golden-dataset            # sync the dataset to BigQuery sec_edgar.golden_eval_set
make refresh-golden-expectations    # recompute expected variances from the engine (review the diff)
```

See [`evals/README.md`](evals/README.md) for the record schema and gate definitions.

---

## Deployment and Access

- **CI/CD:** [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) is the single CI/CD pipeline using keyless Workload Identity Federation (`sec-edgar-cicd-sa` with `roles/run.developer`). It runs lint, tests, and a container smoke test, an optional `eval-all` stage on `main` (defaults to `skip`; set `EVAL_GATE_MODE` or `eval_mode` to `advisory` or `blocking` to enable), then updates the Cloud Run service image (`gcloud run services update --image=...`) while Terraform owns service configuration.
- **Infrastructure:** everything is provisioned by Terraform; see [`terraform/README.md`](terraform/README.md).
- **Access:** IAP fronts the Cloud Run service, so the app is never anonymous. To share it:
  1. Add groups or domains to `iap_authorized_principals` in [`terraform/variables.tf`](terraform/variables.tf) or a tfvars file. Individual `user:` bindings are rejected.
     ```hcl
     iap_authorized_principals = [
       "domain:jackbetcher.altostrat.com",
       "group:sec-edgar-analysts@example.com",
     ]
     iap_default_app_role = "analyst"   # or "viewer"; "" admits nobody without an RBAC group
     ```
  2. Run `terraform -chdir=terraform apply`.
  3. Share the URL from `terraform -chdir=terraform output -raw app_url`.

  To revoke access, remove the principal and apply again.

---

## Repository Layout

```text
├── src/edgar_capstone/
│   ├── server.py            # FastAPI app factory and lifespan hooks
│   ├── api/                 # Routers, request schemas, auth dependencies
│   ├── agents/              # ADK agents, chat pipeline, ReAct loop, verification, deterministic tools
│   │   ├── core/            # Shared Pydantic schemas, retry/telemetry resilience, citation helpers
│   │   ├── adapters/        # BigQuery, Vertex AI Search, and GCS tool adapters
│   │   ├── prompts/         # Local Git-versioned prompt store and dynamic prompt builders
│   │   ├── tools/           # ADK tools, turn-scoped runtime, variance/delta/RAG/sector engines
│   │   ├── runtime/         # ADK Runner/streaming, agent factories, telemetry/budget callbacks, session memory
│   │   ├── evaluation/      # Structured report claim verification/rendering and Flash LLM-as-a-Judge
│   │   └── orchestration/   # Conversational RAG chat engine, intent, scope validation & continuity, synthesis loop
│   ├── security/            # Direct IAP / Google Workspace OIDC auth, RBAC, Model Armor, secret scanner
│   ├── observability/       # OpenTelemetry tracing, structured logging, request middleware
│   ├── telemetry.py         # BigQuery streaming telemetry and security audit sinks
│   ├── firestore_stores.py  # Cloud Firestore SDK session memory and analyst feedback stores
│   ├── gcp/                 # Official GCP SDK wrappers, error translation, defaults, and resource path helpers
│   ├── common/              # Shared helpers: env parsing, domain constants, formatting, caching
│   ├── dataset/             # Offline 10-K XBRL and Markdown extraction pipeline
│   ├── vector_db/           # 10-K chunker, Vertex AI embedder, chunk-health monitor
│   └── evals/               # Golden eval scorers and runner
├── frontend/                # React 19 + TypeScript + Tailwind split-pane UI (see frontend/README.md)
├── prompts/                 # Versioned agent prompts and registry
├── evals/                   # Golden dataset and regression thresholds
├── scripts/                 # CLIs: dataset/vector DB build, corpus publishing, evals, smoke test
├── terraform/               # GCP infrastructure as code
├── tests/                   # pytest suite (offline unit/integration + `@pytest.mark.live` GCP/Gemini tests)
└── docs/                    # Observability guide
```
