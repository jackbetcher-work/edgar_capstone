# SEC EDGAR Natural Language Analyst

Conversational financial analysis workspace for querying SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A) alongside deterministic period-over-period financial metrics (Revenue, Operating Income, and Net Income).

This document summarizes what is implemented in the repository today, how to run and verify it locally, and the implementation gaps remaining relative to [`docs/project_reference.md`](docs/project_reference.md).

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 22+ and `npm`
- Optional for infrastructure operations: Terraform `>= 1.5` and `gcloud`

### Install, Run, and Test
```bash
# 1. Create .venv, install Python package + dev dependencies, and install frontend packages
make install

# 2. Start FastAPI (:8000) and Vite React UI (:5173) together for local development
make dev

# 3. Or build the frontend bundle and serve the single-origin app from FastAPI (:8000)
make start

# 4. Run secret scan, backend pytest suite (62 tests), ESLint, and frontend Vitest suite (6 tests)
make test
```

> **Local Authentication Note:** Outside Cloud Run (`IAP_STRICT_MODE=false`), the UI can mint a local ES256 Workspace/Okta developer JWT via `POST /api/v1/auth/sso/dev-token` so all RBAC-protected `/api/v1/*` routes work locally without external IdP setup.

### Execution Modes & Live Gemini Configuration
Runtime execution is controlled via `EDGAR_EXECUTION_MODE` (`auto` by default) or overridden per request (`"execution_mode": "auto" | "live" | "deterministic"`):
- **`auto` (default):** Runs live Google ADK `Runner` SSE streaming when Vertex AI or a Gemini API key is configured; automatically falls back to deterministic local synthesis when offline (e.g., local dev without credentials or CI).
- **`live`:** Strictly requires live Gemini / Vertex AI execution and fails fast (HTTP `503 live_adk_unavailable`) if credentials are missing or upstream retries are exhausted.
- **`deterministic`:** Forces offline deterministic Python orchestration and template synthesis.

```bash
# Live mode via Vertex AI (uses Application Default Credentials)
gcloud auth application-default login
GOOGLE_GENAI_USE_VERTEXAI=true \
GOOGLE_CLOUD_PROJECT=iron-radio-514217-q7 \
GOOGLE_CLOUD_LOCATION=us-central1 \
EDGAR_EXECUTION_MODE=live \
make dev

# Live mode via Gemini API key
GEMINI_API_KEY="$YOUR_GEMINI_API_KEY" EDGAR_EXECUTION_MODE=live make dev

# Optional model & latency target overrides (defaults shown)
# EDGAR_ANALYST_MODEL=gemini-2.5-pro  EDGAR_FLASH_MODEL=gemini-3.5-flash  EDGAR_TTFT_SLO_MS=3000
```

---

## Repository Structure

```text
├── docs/project_reference.md        # Target architecture and product specification
├── src/edgar_capstone/
│   ├── server.py                    # FastAPI gateway, auth/RBAC routes, RAG & telemetry APIs
│   ├── agents/                      # Google ADK agents, live Runner SSE engine, ReAct & context tracking
│   │   ├── live_runtime.py          # ADK Runner (StreamingMode.SSE), Vertex/Gemini config & retry backoff
│   │   ├── react_loop.py            # Structured ReAct loop + post-generation numeric/citation lock
│   │   ├── context_memory.py        # Pro/Flash model routing, ContextLatencyTracker, session compaction
│   │   ├── evaluation_and_coding.py # Flash LLM-as-a-Judge evaluator & SQL/Python/LookML coding assist
│   │   ├── callbacks.py             # Guardrails, first-chunk SSE TTFT capture & grounding callbacks
│   │   └── tools.py                 # Resilient ADK FunctionTools for metrics, variances, RAG & 10-K cache
│   ├── dataset/                     # SEC EDGAR 10-K XBRL + Markdown extraction pipeline (edgartools)
│   ├── vector_db/                   # Section-aware 10-K chunker, 768-d embedder, chunk-health monitor
│   └── security/                    # IAP/OIDC JWT verification, Cloud Identity RBAC, SPIFFE S2S auth
├── frontend/                        # React 19 + TypeScript + Tailwind v4 split-pane terminal UI
├── scripts/                         # CLI entrypoints: dataset build, vector DB build, secret scanner
├── terraform/                       # GCP IaC: BigQuery, Vector/Discovery Search, Cloud Run, IAM, IAP
├── tests/                           # Automated pytest suite (69 tests across agents, live ADK, GCP adapters, APIs, IaC)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + Python 3.11 runtime)
├── cloudbuild.yaml                  # Google Cloud Build test -> build -> deploy pipeline
└── .github/workflows/deploy.yml     # GitHub Actions WIF CI/CD workflow
```

---

## Current State of the Repository

### 1. Curated SEC 10-K Dataset & Vector Pipeline (`dataset/`, `vector_db/`, `scripts/`)
- **10-K Extraction (`build-sec-dataset`):** Extracts XBRL financial statements (`Revenue`, `Operating Income`, `Net Income`) and cleans Item 1A / Item 7 Markdown via `edgartools`, writing local manifests and uploading to GCS (`gs://fde-sec-edgar-sandbox-dev-sec-corpus`).
- **Materialized Corpus (`data/curated_dataset/`):** Currently generated in `sample` mode covering **18 S&P 500 companies** across 8 GICS sectors from **FY2020–FY2026** (**114 company-year profiles**). `data/sp500_constituents.csv` and `--mode sp500` are included for full-index runs.
- **Chunking, Embeddings & Health Monitoring (`build-vector-db`):** Splits filings by section headings into **12,180 metadata-tagged chunks** (`5,556` Item 1A + `6,624` Item 7) with 768-dimensional embeddings (`vertex` mode via `text-embedding-005` or `deterministic` local hash mode) and evaluates chunk/embedding health metrics (`chunk_health_report.json`).

### 2. ADK Agents, Live Runner Execution & Model/Context Strategy (`agents/`)
- **Agent Hierarchy & Dual-Model Routing:**
  - **`gemini-2.5-pro`** handles financial reasoning and orchestration in `RootOrchestrator` and `FinancialAnalystAgent` (both configured with ADK `PlanReActPlanner`).
  - **`gemini-3.5-flash`** is assigned to `NarrativeEvaluatorAgent` (4-axis rubric evaluation), `CodingAssistantAgent` (BigQuery SQL, Python audit scripts, LookML snippets), `ContextCacheAgent` (10-K section cache warmup), and ADK `LlmEventSummarizer` context compaction.
- **Versioned Prompt Store & Dynamic Loader (`prompts/`, `agents/prompt_store.py`):**
  - Agent instructions and guardrail policy templates (`root_orchestrator`, `financial_analyst`, `context_cache_agent`, `narrative_evaluator`, `coding_assistant`, `model_armor_policy`) are stored in a versioned catalog (`prompts/registry.yaml` + `prompts/v1/*.yaml`) with SHA-256 integrity hashing and dynamic ADK `InstructionProvider` resolution across `local` (`prompts/`), Google Cloud Storage (`gs://...`), and Vertex AI Prompt Management backends (`EDGAR_PROMPT_STORE_BACKEND`).
- **Live ADK `Runner` Streaming & Grounding Fallback (`live_runtime.py`):**
  - Executes ADK `Agent`s inside `create_edgar_adk_app` (`ModelArmorSafetyPlugin`, `ContextCacheConfig`, and `EventsCompactionConfig`) via `google.adk.runners.Runner` (`StreamingMode.SSE`, `InMemorySessionService`, `InMemoryMemoryService`) with `HttpRetryOptions` backoff (`429`, `500`, `502`, `503`, `504`) against Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`) or Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`).
  - Captures real wall-clock streaming Time-to-First-Thought / Time-to-First-Token (`observed_ttft_ms`) on the first partial SSE chunk (`llm_response.partial == True`) in `after_model_fact_check_callback`, reconciles Gemini `usage_metadata` token counts on the final chunk, enforces post-generation `verify_narrative_against_calculations` grounding locks, and falls back deterministically when offline (`EDGAR_EXECUTION_MODE=auto`).
- **Tools, Live GCP Adapters & Resilience (`tools.py`, `resilience.py`):** Explicit ADK `FunctionTool` definitions (`MetricFetchTool`, `LookerQueryTool`, `VarianceCalculationTool`, `HybridSearchTool`, `ContextCacheTool`, `QueryCached10KTool`) backed by live GCP client adapters (`BigQueryAdapter`, `VertexAISearchAdapter`, `VertexVectorSearchAdapter`, `GcsCorpusAdapter`, and `VertexAIEmbedder` `text-embedding-005`) with `EDGAR_DATA_BACKEND` (`"auto"` | `"gcp"` | `"local"`) dispatch and `@with_tool_resilience` HTTP 429 exponential backoff and timeout handling.
- **Structured ReAct Execution Loop (`react_loop.py`, `chat_engine.py`):** Evaluates deterministic period variances first, queries cached 10-K sections + metadata-filtered hybrid search for causal explanations, synthesizes inline-cited Markdown (`[TICKER FYxxxx | Section | chunk_id](sec_filing_url)`) via live `FinancialAnalystAgent` or offline template synthesis, and runs post-generation numeric and citation fact-checks (`verify_narrative_against_calculations`).
- **Context Latency & Session Memory (`context_memory.py`, `callbacks.py`):**
  - `ContextLatencyTracker` decomposes every model call into context preparation + prefill latency vs. decode latency, tracks cached vs. uncached context tokens, estimates KV-cache prefill latency savings, and monitors TTFT against the `<= 3000 ms` target (`GET /api/v1/telemetry/context-latency`).
  - `ConversationMemoryManager` maintains multi-turn session state for follow-up entity/period recall and compacts older turns above a configurable token threshold.
  - `create_edgar_adk_app` configures an ADK `App` with `ContextCacheConfig` and `EventsCompactionConfig`.

### 3. FastAPI Backend & Split-Pane Frontend (`server.py`, `frontend/`)
- **Backend API:** Exposes `/health`, `/api/v1/chat`, `/api/v1/variances`, `/api/v1/search`, `/api/v1/metrics`, `/api/v1/explore/{explore_name}`, `/api/v1/section`, `/api/v1/companies`, `/api/v1/evaluate`, `/api/v1/coding-assist`, `/api/v1/telemetry/context-latency`, `/api/v1/feedback`, `/api/v1/secrets/status`, and `/api/v1/auth/*`. `POST /api/v1/chat`, `POST /api/v1/evaluate`, and `POST /api/v1/coding-assist` accept per-request `execution_mode` (`"auto"` | `"live"` | `"deterministic"`), returning HTTP `503` (`live_adk_unavailable`) if `"live"` is requested without credentials; `/health`, `/api/v1/chat`, and `/api/v1/telemetry/context-latency` surface live `adk_runtime` backend status. Serves the compiled React bundle from `/` in production.
- **Split-Pane Terminal UI:** Built with React 19, TypeScript, Tailwind CSS v4, and Vite. Left pane renders conversational queries, scope filters, deterministic variance tables, and interactive citation chips; right pane displays retrieved 10-K chunks and full cached Item 1A / Item 7 filing sections, with graceful offline fallback data if the backend is unreachable.

### 4. Security, Model Armor Guardrails, Infrastructure & CI/CD (`security/`, `terraform/`, `cloudbuild.yaml`)
- **AuthN / AuthZ:** Verifies IAP (`ES256`) and Google Workspace / Okta OIDC (`RS256`) JWTs, enforces Cloud Identity group RBAC (`admins`, `analysts`, `viewers` with `user:` bindings disallowed in Terraform), runs under custom runtime SA `sec-analyst-sa`, and enforces least-privilege S2S tool boundaries across four per-agent service accounts (`root-orchestrator-agent-sa`, `financial-analyst-agent-sa`, `context-cache-agent-sa`, `model-armor-guardrail-sa`) with SPIFFE IDs.
- **Managed Model Armor, Cloud SDP & Projection Filters (`security/model_armor.py`, `terraform/model_armor.tf`):** Replaces regex prompt-injection filters with regional Google Cloud Model Armor v1 (`sanitizeUserPrompt` & `sanitizeModelResponse` on `sec-edgar-analyst-armor`), Google Cloud Sensitive Data Protection (`content:deidentify` via `sec-edgar-sdp-inspect` & `sec-edgar-sdp-deidentify`), SEC forward-looking financial projection / MNPI leak filters, and ADK `ModelArmorSafetyPlugin` (`BasePlugin`) with session-poisoning prevention.
- **Terraform & CI/CD:** Provisions BigQuery tables and SQL views (`financial_metrics`, `filing_chunks`, `v_financial_variances_yoy`, `v_filing_chunk_coverage`, `v_chunk_health_metrics`), Vertex AI Vector Search, Vertex AI Search (Discovery Engine), Google Cloud Model Armor & Cloud SDP templates, Secret Manager, Cloud Monitoring chunk-health dashboards, Cloud Run v2, IAP backend service, and Workload Identity Federation. `terraform/cicd_and_secrets.tf`, `cloudbuild.yaml`, and `.github/workflows/deploy.yml` run secret scanning (`verify_no_secrets.py`), tests, container builds, and Cloud Run deployments wired with `GOOGLE_GENAI_USE_VERTEXAI=true`, `EDGAR_MODEL_ARMOR_MODE=gcp`, `MODEL_ARMOR_TEMPLATE_ID=sec-edgar-analyst-armor`, `EDGAR_PROMPT_STORE_BACKEND=auto`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `EDGAR_EXECUTION_MODE=auto`.

---

## Implementation Gaps vs. `docs/project_reference.md`

Live Gemini (`gemini-2.5-pro` & `gemini-3.5-flash`), Google ADK `Runner` SSE streaming execution, Google Cloud Model Armor + SDP/projection filters, and versioned prompt loading (`prompts/` + GCS + Vertex AI) are now wired up end-to-end across the backend and Cloud Run configuration. The remaining gaps relative to [`docs/project_reference.md`](docs/project_reference.md) involve replacing **local cache manifests and in-memory telemetry stores** with remote persistence and perimeter controls:

| Reference Area | Current Repo State | Remaining Gap to Close |
| :--- | :--- | :--- |
| **Vertex AI Context Caching API**<br>*(Secs. 2, 6, 7)* | Live GCP client adapters in `tools.py` now query BigQuery (`financial_metrics`, `v_financial_variances_yoy`, `v_chunk_health_metrics`, `VECTOR_SEARCH` on `filing_chunks`), Vertex AI Search (`sec-edgar-10k-hybrid-search` / `sec-edgar-10k-corpus-ds`), Vertex AI Vector Search (`:findNeighbors` on `sec-edgar-10k-chunks-endpoint`), and GCS on cache miss, **while** parsed 10-K sections remain cached in `_SESSION_10K_CONTEXT_CACHE`. | Register full 10-K Markdown sections directly as remote `CachedContent` resources via the Vertex AI Context Caching API (`google.genai.Client.caches`) with managed TTLs across Cloud Run instances. |
| **Container Corpus Packaging**<br>*(Secs. 4, 6)* | `tools.py` queries BigQuery, Vertex AI Search / Vector Search, and GCS directly in Cloud Run (`EDGAR_DATA_BACKEND=auto` / `gcp`), while local `Dockerfile` builds still omit the gitignored `bigquery_filing_chunks.jsonl` and `unstructured/*.md` files. | Optionally hydrate local fallback artifacts from GCS on startup when running offline container smoke tests without BigQuery / Vertex Search access. |
| **Dataset Breadth & Metrics**<br>*(Secs. 3, 7)* | 18 S&P 500 issuers (114 profiles, FY2020–FY2026) are materialized; structured XBRL extraction covers `Revenue`, `Operating Income`, and `Net Income`. | Run `build-sec-dataset --mode sp500` + `build-vector-db --embedding-backend vertex` to populate all ~500 S&P 500 issuers; add structured XBRL fields (e.g., `R&D` expense referenced in Section 3's example query) if quantitative R&D variance math is required. |
| **VPC Service Controls & IAP Load Balancer**<br>*(Sec. 8)* | `iap_and_sso.tf` creates the Serverless NEG and IAP-enabled `google_compute_backend_service`, and `vector_search.tf` sets `public_endpoint_enabled = true`. No VPC-SC resources exist. | Add `google_access_context_manager_service_perimeter` covering Vertex AI, Cloud Run, GCS, and BigQuery; switch Vector Search to private endpoint / PSC; add external HTTPS forwarding rule, URL map, and managed TLS certificate in front of the IAP backend service. |
| **OpenTelemetry, Cloud Trace & Structured JSON Logs**<br>*(Secs. 2, 4, 6, 10)* | `cloudtrace.googleapis.com` is enabled in Terraform and latency is tracked in Python, but no `opentelemetry-*` packages or Cloud Logging JSON formatters are installed. | Instrument FastAPI, ADK callbacks, and outbound GCP calls with OpenTelemetry spans exported to Cloud Trace, and emit structured JSON logs with `logging.googleapis.com/trace` correlation IDs. |
| **Persistent Telemetry, Audit, Feedback & Looker BI**<br>*(Secs. 6, 10)* | Context latency (`_GLOBAL_LATENCY_TRACKER`), security audit events (`_SECURITY_AUDIT_LOG`), and user feedback (`_FEEDBACK_AUDIT_LOG`) live in process memory and reset on container restart; Looker explores are mocked in Python. | Stream token/cost/latency telemetry and audit logs to BigQuery sink tables, persist user feedback to Firestore or Cloud SQL, and add real LookML model/dashboard definitions. |
| **Golden Eval Dataset & `make eval-all` Gate**<br>*(Secs. 4, 9)* | 62 `pytest` tests and the `NarrativeEvaluatorAgent` rubric exist, but there is no BigQuery golden dataset, no `make eval-all` target, and no automated `>= 92%` grounding-recall or `>80%` coverage gate in CI. | Seed a 20+ item query–variance–explanation golden dataset in BigQuery, add `make eval-all` to run `gemini-3.5-flash` LLM-as-a-Judge + recall benchmarks, and enforce `pytest --cov --cov-fail-under=80` in `cloudbuild.yaml`. |
