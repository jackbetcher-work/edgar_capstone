# SEC EDGAR Natural Language Analyst

Conversational financial analysis workspace for querying SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A) alongside deterministic period-over-period financial metrics (Revenue, Operating Income, and Net Income).

This document summarizes what is implemented in the repository today, how to run and verify it locally, and the implementation gaps remaining relative to [`docs/project_reference.md`](docs/project_reference.md).

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 22+ and `npm`
- Optional for infrastructure operations: Terraform `>= 1.7` and `gcloud`

### Install, Run, and Test
```bash
# 1. Create .venv, install Python package + dev dependencies, and install frontend packages
make install

# 2. Start FastAPI (:8000) and Vite React UI (:5173) together for local development
make dev

# 3. Or build the frontend bundle and serve the single-origin app from FastAPI (:8000)
make start

# 4. Run secret scan, backend pytest suite (147 tests), ESLint, and frontend Vitest suite (7 tests)
make test

# 5. Run the CI lint/format gates (ruff check + format --check, terraform fmt/validate, ESLint),
#    auto-fix them, or run every CI gate at once
make lint
make format
make check
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
EDGAR_GEMINI_LOCATION=global \
EDGAR_EXECUTION_MODE=live \
make dev

# Live mode via Gemini API key
GEMINI_API_KEY="$YOUR_GEMINI_API_KEY" EDGAR_EXECUTION_MODE=live make dev

# Optional model, Gemini endpoint & latency target overrides (defaults shown). In iron-radio-514217-q7
# the Gemini models are only served from the `global` endpoint, so set EDGAR_GEMINI_LOCATION=global
# (Cloud Run does this via Terraform); BigQuery / GCS / Model Armor stay in GOOGLE_CLOUD_LOCATION.
# EDGAR_ANALYST_MODEL=gemini-3.8-flash  EDGAR_FLASH_MODEL=gemini-3.5-flash  EDGAR_GEMINI_LOCATION=global  EDGAR_TTFT_SLO_MS=3000
```

### Container Packaging & Offline Smoke Tests
The image bakes only the lightweight, git-tracked manifests (`bigquery_financials.jsonl`, `chunk_health_report.json`, `dataset_catalog.json`, `vector_db_catalog.json`). The heavy, gitignored artifacts (`manifests/bigquery_filing_chunks.jsonl` and `unstructured/<TICKER>/<FY>_10k.md`) are left out of the Docker build context by `.dockerignore`. In Cloud Run they are read from BigQuery, Vertex AI Search, and GCS. For offline runs they can be downloaded from GCS when the app starts.

```bash
make smoke                   # Offline API smoke test in-process against the local checkout (no Docker)
make docker-build            # Build the production image (sec-edgar-analyst:local)
make docker-smoke            # Run the image with no GCP access; smoke-test the baked corpus over HTTP
make docker-smoke-hydrated   # Mount local ADC, hydrate from GCS on startup (EDGAR_CORPUS_HYDRATION=required), assert real chunks + 10-K Markdown
make hydrate-corpus          # Download the same artifacts into a fresh local checkout (HYDRATION_TICKERS="NVDA AAPL" to limit)
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `EDGAR_CORPUS_HYDRATION` | `off` | `off` = no GCS calls. `auto` = best effort; the app still starts with synthesized fallbacks if hydration fails. `required` = startup aborts if hydration fails. |
| `EDGAR_CORPUS_HYDRATION_SCOPES` | `chunks,unstructured` | Any of `chunks`, `search_metadata`, `manifests`, `unstructured`, `structured`, `vector_search`, or `all`. |
| `EDGAR_CORPUS_HYDRATION_TICKERS` | *(all)* | Limits `unstructured` / `structured` downloads to the listed tickers (e.g. `NVDA,AAPL`). |
| `EDGAR_CORPUS_HYDRATION_OVERWRITE` / `_MAX_WORKERS` | `false` / `8` | Re-download files that already exist / number of parallel downloads. |
| `EDGAR_DATASET_DIR` / `EDGAR_DATA_ROOT` | `/app/data/curated_dataset` / `/app/data` in the image | Overrides where the curated corpus is read from. |

`GET /health` now includes a `corpus` block. It shows the resolved dataset directory, where profiles, chunks, and 10-K sections are coming from (`*_manifest` / `markdown_files` vs. `synthesized_from_profiles`), and the result of the last hydration run.

> [!IMPORTANT]
> `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/` now holds the full-index manifests, including `bigquery_filing_chunks.jsonl` (342,945 chunks), so `make docker-smoke-hydrated` can hydrate with the default `chunks,unstructured` scopes. The Vector Search embeddings (`vector_search/embeddings/`, about 6 GB locally) are still not in the bucket, and BigQuery hasn't been loaded (see gap 1).

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

`GET /health` now includes an `observability` block showing the active exporter, sampling ratio, instrumentation state, and log format.

---

## Repository Structure

```text
├── docs/project_reference.md        # Target architecture and product specification
├── docs/observability.md            # Tracing, structured logging, correlation surfaces & GCP lookup guide
├── prompts/                         # Versioned agent prompts: registry.yaml + v1/ (all agents), v2/ & v3/ (FinancialAnalystAgent)
├── src/edgar_capstone/
│   ├── server.py                    # FastAPI gateway, auth/RBAC routes, RAG & telemetry APIs
│   ├── telemetry.py                 # Streaming BigQuery telemetry/audit sinks, Firestore feedback & LookML catalog
│   ├── observability/               # OpenTelemetry -> Cloud Trace, trace-correlated JSON logs -> Cloud Logging
│   │   ├── tracing.py               # TracerProvider + CloudTraceSpanExporter, propagators, FastAPI/httpx instrumentation, @traced
│   │   ├── structured_logging.py    # Cloud Logging JSON formatter (trace/span IDs, Error Reporting), uvicorn log routing
│   │   └── middleware.py            # Per-request log context, X-Trace-Id / Server-Timing headers, latency access log
│   ├── agents/                      # Google ADK agents, live Runner SSE engine, ReAct & context tracking
│   │   ├── live_runtime.py          # ADK Runner (StreamingMode.SSE), Vertex/Gemini config & retry backoff
│   │   ├── chat_engine.py           # /api/v1/chat turn: scope, guardrails, retrieval, ReAct, judge, telemetry
│   │   ├── react_loop.py            # Structured ReAct loop, agent-retrieval merge, structured-report verification
│   │   ├── financial_analyst.py     # FinancialAnalystAgent (BuiltInPlanner, output_schema=AnalystReport, retrieval budget)
│   │   ├── prompt_store.py          # Versioned prompt loader (local / GCS / Vertex AI) + runtime model variables
│   │   ├── context_memory.py        # Pro/Flash model routing, ContextLatencyTracker, session compaction
│   │   ├── evaluation_and_coding.py # Flash LLM-as-a-Judge evaluator & SQL/Python/LookML coding assist
│   │   ├── callbacks.py             # Guardrails, first-chunk SSE TTFT capture & grounding callbacks
│   │   ├── structured_report.py     # Claim-level verification + rendering of the analyst's structured AnalystReport
│   │   └── tools.py                 # Resilient ADK FunctionTools for metrics, variances, RAG & bounded 10-K section cache
│   ├── dataset/                     # SEC EDGAR 10-K XBRL + Markdown extraction pipeline (edgartools)
│   │   ├── paths.py                 # Install-layout-aware data root resolution (source tree, /app/data, cwd)
│   │   └── corpus_hydration.py      # GCS -> local corpus hydration (startup + CLI) and /health corpus status
│   ├── vector_db/                   # Section-aware 10-K chunker, batched Vertex AI / deterministic 768-d embedder, chunk-health monitor
│   └── security/                    # IAP/OIDC JWT verification, Cloud Identity RBAC, SPIFFE S2S auth
├── looker/                          # LookML semantic layer (manifest, model, 6 views, 2 LookML dashboards)
├── frontend/                        # React 19 + TypeScript + Tailwind v4 split-pane terminal UI (see frontend/README.md)
├── scripts/                         # CLI entrypoints: dataset build, vector DB build (+ --upload-only), corpus hydration, container smoke test, secret scanner
├── terraform/                       # GCP IaC: BigQuery + sinks, Firestore, Vector/Discovery Search, Cloud Run, IAM, IAP (direct or via HTTPS LB), VPC-SC (off by default), Cloud Build triggers, log-based latency metric
├── tests/                           # Automated pytest suite (147 tests across agents, live ADK, GCP adapters, telemetry, observability, APIs, IaC, container packaging)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + Python 3.11 runtime) with baked manifests + build-time corpus check
├── .dockerignore                    # Keeps secrets and gitignored heavy corpus artifacts out of the build context
├── cloudbuild.yaml                  # Cloud Build: parallel lint/test gates -> build (+ offline smoke test) -> push + deploy (main) / verify-only (PRs)
└── .github/workflows/deploy.yml     # GitHub Actions WIF CI/CD workflow
```

---

## Current State of the Repository

### 1. Curated SEC 10-K Dataset & Vector Pipeline (`dataset/`, `vector_db/`, `scripts/`)
- **10-K Extraction (`build-sec-dataset`):** Extracts XBRL financial statements (`Revenue`, `Operating Income`, `Net Income`) and cleans Item 1A / Item 7 Markdown via `edgartools`, writing local manifests and uploading to GCS (`gs://fde-sec-edgar-sandbox-dev-sec-corpus`).
- **Materialized Corpus (`data/curated_dataset/`):** The full S&P 500 index (`build-sec-dataset --sp500`): **501 tickers** from `data/sp500_constituents.csv` and **3,022 company-year profiles** for **FY2020–FY2026**, with structured financials and Item 1A / Item 7 Markdown for each. The structured JSON and Markdown are uploaded to GCS (`dataset_catalog.json`: `uploaded_to_gcs: true`, no failed tickers).
- **Chunking, Embeddings & Health Monitoring (`build-vector-db`):**
  - Splits filings by section headings into **342,945 metadata-tagged chunks** (154,383 Item 1A + 188,562 Item 7).
  - Embeds them as 768-dimensional `text-embedding-005` vectors (`--embedding-backend auto | vertex_ai | deterministic`; this build used `vertex_ai`). The Vertex AI embedder batches by count and total characters (`--max-batch-chars`, default 60,000), halves a batch the API rejects for token limits, retries with jittered backoff, and runs requests in parallel (`--embedding-workers`).
  - Writes `chunk_health_report.json`: health score 99.76 (`HEALTHY`), 100% embedding coverage, and 2,975 of 3,022 filings with both sections.
  - `--upload-only` publishes already-staged vector DB artifacts to GCS without re-chunking or re-embedding.
- **Container Corpus Packaging & Hydration (`dataset/paths.py`, `dataset/corpus_hydration.py`, `hydrate-sec-corpus`):** Data paths are now resolved the same way in a source checkout, an editable install, and the container's non-editable `/opt/venv` install. Before this fix, the image looked for its corpus under `site-packages` and never found the baked `/app/data` manifests. Gitignored chunks and 10-K Markdown can be downloaded from GCS at FastAPI startup (`EDGAR_CORPUS_HYDRATION`) or with the CLI. Downloads are atomic, existing files are skipped, ticker filters are supported, and paths that would escape the dataset directory are rejected.

### 2. ADK Agents, Live Runner Execution & Model/Context Strategy (`agents/`)
- **Agent Hierarchy & Dual-Model Routing:**
  - **`gemini-3.8-flash`** handles financial reasoning and orchestration in `RootOrchestrator` (ADK `PlanReActPlanner`) and `FinancialAnalystAgent` (native thinking via `BuiltInPlanner`; see below). Both are served from the `global` Vertex endpoint (`EDGAR_GEMINI_LOCATION`). This departs from the reference spec, which names Gemini 2.5 Pro: that model returns HTTP 404 in this project. Override it with `EDGAR_ANALYST_MODEL`. The prompts render whichever model is configured at runtime, and `gemini-3.8-flash` has its own cost profile ($0.75 input / $0.075 cached / $3.75 output per 1M tokens, introductory Vertex pricing until 2027-01-01).
  - **`gemini-3.5-flash`** is assigned to `NarrativeEvaluatorAgent` (4-axis rubric evaluation), `CodingAssistantAgent` (BigQuery SQL, Python audit scripts, LookML snippets), `ContextCacheAgent` (10-K section cache warmup), and ADK `LlmEventSummarizer` context compaction.
- **Versioned Prompt Store & Dynamic Loader (`prompts/`, `agents/prompt_store.py`):**
  - Agent instructions and guardrail policy templates (`root_orchestrator`, `financial_analyst`, `context_cache_agent`, `narrative_evaluator`, `coding_assistant`, `model_armor_policy`) are stored in a versioned catalog (`prompts/registry.yaml` + `prompts/v1/*.yaml`) with SHA-256 integrity hashing and dynamic ADK `InstructionProvider` resolution across `local` (`prompts/`), Google Cloud Storage (`gs://...`), and Vertex AI Prompt Management backends (`EDGAR_PROMPT_STORE_BACKEND`).
  - Most agents load `v1` (`EDGAR_PROMPT_VERSION`). `FinancialAnalystAgent` follows the registry's `latest` for its prompt, currently `v3` (structured output); `v2` added targeted retrieval. Older versions stay loadable by number.
  - The instruction provider renders the configured model names (`EDGAR_ANALYST_MODEL`, `EDGAR_FLASH_MODEL`) at runtime, overriding the defaults recorded in each prompt file.
- **Live ADK `Runner` Streaming & Grounding Fallback (`live_runtime.py`):**
  - Executes ADK `Agent`s inside `create_edgar_adk_app` (`ModelArmorSafetyPlugin`, `ContextCacheConfig`, and `EventsCompactionConfig`) via `google.adk.runners.Runner` (`StreamingMode.SSE`, `InMemorySessionService`, `InMemoryMemoryService`) with `HttpRetryOptions` backoff (`429`, `500`, `502`, `503`, `504`) against Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`) or Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`).
  - Captures real wall-clock streaming Time-to-First-Thought / Time-to-First-Token (`observed_ttft_ms`) on the first partial SSE chunk (`llm_response.partial == True`) in `after_model_fact_check_callback`, reconciles Gemini `usage_metadata` token counts on the final chunk, enforces post-generation `verify_narrative_against_calculations` grounding locks, and falls back deterministically when offline (`EDGAR_EXECUTION_MODE=auto`).
- **Tools, Live GCP Adapters & Resilience (`tools.py`, `resilience.py`):** Explicit ADK `FunctionTool` definitions (`MetricFetchTool`, `LookerQueryTool`, `VarianceCalculationTool`, `HybridSearchTool`, `ContextCacheTool`, `QueryCached10KTool`) backed by live GCP client adapters (`BigQueryAdapter`, `VertexAISearchAdapter`, `VertexVectorSearchAdapter`, `GcsCorpusAdapter`, and `VertexAIEmbedder` `text-embedding-005`) with `EDGAR_DATA_BACKEND` (`"auto"` | `"gcp"` | `"local"`) dispatch and `@with_tool_resilience` HTTP 429 exponential backoff and timeout handling. `@with_tool_resilience` also runs every tool inside a `tool.<name>` OpenTelemetry span that records attempts, `tool.retry` events, and error status.
- **Structured ReAct Execution Loop (`react_loop.py`, `chat_engine.py`):**
  - Evaluates deterministic period variances first.
  - Pre-queries 10-K passages and metadata-filtered hybrid search; these become evidence leads for the live agent and feed the offline fallback.
  - Lets the live agent retrieve the rest itself.
  - Produces the answer from the live `FinancialAnalystAgent`'s verified structured report, or from offline template synthesis. Either way the answer is Markdown with inline `[TICKER FYxxxx | Section | chunk_id](sec_filing_url)` citations.
  - Runs post-generation numeric and citation fact-checks (`verify_narrative_against_calculations`).
- **Context Latency, Cost Telemetry & Session Memory (`context_memory.py`, `telemetry.py`, `callbacks.py`):**
  - `ContextLatencyTracker` decomposes every model call into context preparation + prefill latency vs. decode latency, computes exact USD input/output cost and cached-token discount savings (`gemini-3.8-flash`, `gemini-3.5-flash`, and legacy `gemini-2.5-pro` profiles), monitors TTFT against the `<= 3000 ms` target, and streams per-call records directly to BigQuery (`sec_edgar.llm_telemetry_events`) via `BigQueryTelemetrySink` (`tabledata.insertAll` + structured Cloud Logging sink records).
  - `BigQueryTelemetrySink` also streams conversational turn/session metrics (`session_telemetry_events`: cumulative session length, companies analyzed, reports generated, ADK tool usage, and latency/cost) and authentication / 10-K access audit trails (`security_audit_logs`), with `"auto"` | `"gcp"` | `"local"` fallback (`EDGAR_TELEMETRY_BACKEND`).
  - Every row in all three tables carries the request's `trace_id`, so BigQuery telemetry joins back to Cloud Trace and Cloud Logging. The Cloud Logging copies of these sink records are written as top-level `jsonPayload` fields (`component`, `log_sink_target`, `event_payload`), which is what the `google_logging_project_sink` filter matches. Before this change they were plain INFO strings with no configured handler, so they never reached stdout.
  - `ConversationMemoryManager` maintains multi-turn session state for follow-up entity/period recall and compacts older turns above a configurable token threshold.
  - `create_edgar_adk_app` configures an ADK `App` with `ContextCacheConfig` and `EventsCompactionConfig`.
- **Agent-Driven 10-K Retrieval (`live_runtime.py`, `tools.py`, `prompts/v3/financial_analyst.yaml`):**
  - `FinancialAnalystAgent` decides what 10-K evidence each question needs and fetches it. No full sections are forced into its context window.
  - The synthesis request carries:
    - the authoritative variance payload;
    - the variance movers;
    - up to 8 short evidence leads (`chunk_id` plus a 240-character excerpt from the deterministic pre-pass);
    - suggested retrieval targets;
    - a retrieval brief explaining how to call the tools.
  - The agent's 10-K tools are `query_cached_10k_context` (ranked passages from one company-year section, each with its `chunk_id`) and `search_10k_chunks_rag` (metadata-filtered hybrid search across companies and fiscal years). `load_cached_10k_section`, which loads a whole section, stays with `ContextCacheAgent` and the source-viewer API.
  - Passages the agent retrieves are merged into the turn's citations, which feed the source viewer, the fact-check, and the judge. `adk_runtime.agent_retrieval` reports:
    - the retrieval calls made;
    - the chunks the agent pulled;
    - the cited chunk IDs;
    - any cited IDs that don't resolve.
  - `query_cached_10k_context` now ranks the complete section rather than the first 50k characters.
  - Section tools report `content_source` (`gcs` | `local_markdown` | `synthesized_fallback`). Placeholder text carries a "do not cite" warning.
  - The analyst uses the model's native thinking (`BuiltInPlanner`) instead of `PlanReActPlanner`. PlanReAct's text `/*ACTION*/` tool snippets made Gemini 3.x write tool calls as prose. As a backstop, a live answer that still contains planning text or an unexecuted call falls back to deterministic synthesis.
  - The judge now sees the full excerpt of every passage the report references, not just the first 8 retrieved.
  - Retrieval is capped at 6 tool calls per question (`EDGAR_ANALYST_RETRIEVAL_BUDGET`). After that, retrieval tools tell the agent to write its answer instead of running into `max_llm_calls`.
- **Structured, Verifiable Analyst Output (`schemas.AnalystReport`, `structured_report.py`, `prompts/v3/financial_analyst.yaml`):**
  - `FinancialAnalystAgent` answers with constrained JSON (ADK `output_schema=AnalystReport`) instead of free-form Markdown. The report has four parts:
    - a one-sentence `headline`;
    - `key_metrics` as references (ticker + metric) only, with the figures filled in from the deterministic engine;
    - up to 6 `findings`, each with `chunk_ids` and a verbatim `evidence_quote`;
    - free-text `notes` for any other details.
  - Before rendering, every finding is checked: its chunk IDs must have been retrieved this turn, the quote must appear in a cited passage, and every figure in the statement must appear in the quote or the deterministic payload. Findings that fail are dropped and listed in `report_verification`.
  - Figures in the headline or notes that can't be traced are removed. If no finding verifies, the turn falls back to deterministic synthesis.
  - The verified report is rendered as compact Markdown (headline, key metrics, findings with evidence quotes, "Notes (not source-verified)") and also returned as `structured_report` JSON.
  - The response guardrail inspects the raw, unverified report too, so a price target dropped from the notes is still blocked and audited. Blocked structured answers are replaced with a schema-valid report so ADK validation doesn't hide the block.
  - Caching: ADK's App-level `ContextCacheConfig` caches the static prefix (system instruction and tool declarations) plus the growing conversation, so passages fetched early in a turn are served from cache on later calls in the same turn. The earlier forced full-section `CachedContent` bundle (`tenk_context_cache.py`) was removed.
  - The process-local `_SESSION_10K_CONTEXT_CACHE` is a bounded LRU + TTL map (`EDGAR_10K_SECTION_CACHE_MAX_ENTRIES=256`, `EDGAR_10K_SECTION_CACHE_TTL_SECONDS=3600`).

### 3. FastAPI Backend, Persistent Feedback, LookML Semantic Layer & Split-Pane Frontend (`server.py`, `looker/`, `frontend/`)
- **Backend API & Firestore Feedback Persistence:** Exposes:
  - health: `/health`, `/healthz`;
  - chat and data: `/api/v1/chat`, `/api/v1/variances`, `/api/v1/search`, `/api/v1/intent`, `/api/v1/metrics`, `/api/v1/explore/{explore_name}`, `/api/v1/looker/catalog`, `/api/v1/section` (GET/POST), `/api/v1/companies`;
  - evaluation and coding: `/api/v1/evaluate`, `/api/v1/coding-assist`;
  - telemetry: `/api/v1/telemetry/context-latency`, `/api/v1/telemetry/sessions`;
  - feedback: `/api/v1/feedback` (GET/POST);
  - auth and admin: `/api/v1/auth/{config,sso/exchange,sso/dev-token,me,logout,audit,rbac}`, `/api/v1/admin/rbac/policies`, `/api/v1/secrets/status`.

  `/api/v1/chat` responses include the rendered `answer_markdown` plus `structured_report`, `report_verification`, and `adk_runtime.agent_retrieval`. `FirestoreFeedbackStore` persists analyst thumbs-up/down ratings and audit comments to Google Cloud Firestore Native (`analyst_feedback` collection via REST v1 `documents` & `:runQuery`, controlled by `EDGAR_FEEDBACK_BACKEND`). Serves the compiled React bundle from `/` in production.
- **LookML Semantic Layer & Dashboards (`looker/`):** Defines `manifest.lkml`, model `sec_edgar_analytics.model.lkml` (7 explores: `yoy_variances`, `sector_benchmarks`, `peer_comparison`, `chunk_health`, `llm_telemetry_and_cost`, `session_analytics`, `security_audit_logs`), 6 `.view.lkml` files mapped to BigQuery tables/views (the three telemetry views now have a `trace_id` dimension that links to Cloud Trace), and 2 `.dashboard.lookml` executive dashboards (`sec_edgar_observability_and_cost` and `sec_edgar_financial_variances`), introspected via `GET /api/v1/looker/catalog` and queried through `LookerQueryTool`.
- **Split-Pane Terminal UI:** Built with React 19, TypeScript, Tailwind CSS v4, and Vite. Left pane renders conversational queries, scope filters, deterministic variance tables, and interactive citation chips; right pane displays retrieved 10-K chunks and full cached Item 1A / Item 7 filing sections, with graceful offline fallback data if the backend is unreachable. Live answers arrive as the compact rendered structured report (headline, key metrics, quote-backed findings, notes). Details are in [`frontend/README.md`](frontend/README.md).

### 4. Security, Model Armor Guardrails, Infrastructure & CI/CD (`security/`, `terraform/`, `cloudbuild.yaml`)
- **AuthN / AuthZ:** Verifies IAP (`ES256`) and Google Workspace / Okta OIDC (`RS256`) JWTs, enforces Cloud Identity group RBAC (`admins`, `analysts`, `viewers` with `user:` bindings disallowed in Terraform), runs under custom runtime SA `sec-analyst-sa`, and enforces least-privilege S2S tool boundaries across four per-agent service accounts (`root-orchestrator-agent-sa`, `financial-analyst-agent-sa`, `context-cache-agent-sa`, `model-armor-guardrail-sa`) with SPIFFE IDs.
- **Managed Model Armor, Cloud SDP & Projection Filters (`security/model_armor.py`, `terraform/model_armor.tf`):** Replaces regex prompt-injection filters with regional Google Cloud Model Armor v1 (`sanitizeUserPrompt` & `sanitizeModelResponse` on `sec-edgar-analyst-armor`), Google Cloud Sensitive Data Protection (`content:deidentify` via `sec-edgar-sdp-inspect` & `sec-edgar-sdp-deidentify`), SEC forward-looking financial projection / MNPI leak filters, and ADK `ModelArmorSafetyPlugin` (`BasePlugin`) with session-poisoning prevention.
- **Terraform & CI/CD:** Provisions BigQuery analytical tables, streaming telemetry/audit sink tables (`llm_telemetry_events`, `session_telemetry_events`, `security_audit_logs` with table-scoped custom append role `secEdgarTelemetrySinkAppend` + Cloud Logging sink) and SQL views (`v_financial_variances_yoy`, `v_filing_chunk_coverage`, `v_chunk_health_metrics`, `v_telemetry_cost_and_latency`), Firestore Native feedback database (`analyst_feedback_db`), Vertex AI Vector Search, Vertex AI Search (Discovery Engine), Google Cloud Model Armor & Cloud SDP templates, Secret Manager, Cloud Monitoring chunk-health dashboards, Cloud Run v2 with IAP enabled on the service itself (`cloud_run_access_mode = "iap_direct"`, the default; see [Sharing the App Internally](#sharing-the-app-internally)) or, optionally, an IAP backend service behind a global external HTTPS load balancer (`cloud_run_access_mode = "load_balancer"`; `load_balancer.tf`: managed TLS certificate for `lb_domain` or `<ip>.nip.io`, `RESTRICTED` SSL policy, URL map, HTTPS proxy, and 80→443 redirect), a VPC Service Controls perimeter definition (`vpc_service_controls.tf`, dry-run by default), and Workload Identity Federation. `terraform/cloudbuild.tf` manages a 2nd-gen GitHub connection plus two Cloud Build triggers, both running as the dedicated least-privilege `sec-edgar-cloudbuild-sa`: merges to `main` run `cloudbuild.yaml` with `_PIPELINE_MODE=deploy`, and pull requests run it with `_PIPELINE_MODE=verify` (no push or deploy). `cloudbuild.yaml` first runs four quality gates in parallel: `lint-python` (`ruff check` + `ruff format --check`), `lint-terraform` (`terraform fmt -check` + `validate`), `frontend` (ESLint + Vitest), and `test`. `.github/workflows/deploy.yml` runs the same gates on every push; its build and deploy jobs are opt-in via the `DEPLOY_WITH_GITHUB_ACTIONS` repository variable, so a merge never deploys twice. The pipelines run secret scanning (`verify_no_secrets.py`), tests, container builds, an offline in-image smoke test before pushing (`scripts/container_smoke_test.py --in-process --expect-corpus baked`), and Cloud Run deployments wired with `GOOGLE_GENAI_USE_VERTEXAI=true`, `EDGAR_MODEL_ARMOR_MODE=gcp`, `MODEL_ARMOR_TEMPLATE_ID=sec-edgar-analyst-armor`, `EDGAR_PROMPT_STORE_BACKEND=auto`, `EDGAR_TELEMETRY_BACKEND=gcp`, `EDGAR_FEEDBACK_BACKEND=gcp`, `EDGAR_ANALYST_MODEL=gemini-3.8-flash`, `EDGAR_FLASH_MODEL=gemini-3.5-flash`, `EDGAR_GEMINI_LOCATION=global`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `EDGAR_EXECUTION_MODE=auto`. No Cloud Build triggers exist in the project yet (`enable_cloudbuild_trigger` defaults to `false`).

### 5. Observability: End-to-End Latency Tracing & Structured Logs (`observability/`, `terraform/observability.tf`)
- **OpenTelemetry → Cloud Trace (`observability/tracing.py`):**
  - At import time, `server.py` installs a global `TracerProvider` (GCP resource detection, `ParentBased` ratio sampling) with `BatchSpanProcessor(CloudTraceSpanExporter)`. Buffered spans are flushed in the FastAPI lifespan shutdown hook.
  - Propagation reads both `X-Cloud-Trace-Context` and W3C `traceparent`, so the server span continues the trace started by the HTTPS load balancer / Cloud Run front end.
- **Span coverage for one `/api/v1/chat` request:**
  - FastAPI server span (`opentelemetry-instrumentation-fastapi`).
  - Custom `@traced` spans: `auth.verify_jwt`, `chat.rag_turn`, `guardrail.inspect_user_prompt` / `inspect_model_response`, `chat.resolve_scope`, `orchestrator.classify_intent`, `chat.retrieve_citations`, `react.execute` → `react.evaluate_variances` / `react.query_cached_10k` / `react.format_grounded_output`, `adk.run_agent_stream`, `evaluation.narrative_judge`, and `coding.assist`.
  - A `tool.<name>` span for every ADK tool.
  - ADK's built-in `invoke_agent` / `call_llm` / `execute_tool` spans.
  - httpx client spans for every outbound GCP REST call (Vertex AI Gemini, BigQuery, Firestore, Model Armor, SDP, Secret Manager, IAM Credentials).
  - Each recorded model call adds an `llm.call` span event with `gen_ai.request.model`, input/output/cached/thinking tokens, `edgar.ttft_ms`, latency, and cost.
  - `_run_coro_sync` now copies `contextvars` into its worker thread, so live ADK runs stay children of the request span instead of starting orphan traces.
- **Structured JSON logs → Cloud Logging (`observability/structured_logging.py`):**
  - One root handler formats app, uvicorn, and library logs as single-line Cloud Logging JSON. Fields include `severity`, `message`, `time`, `logging.googleapis.com/trace` (`projects/<project>/traces/<trace_id>`), `logging.googleapis.com/spanId`, `logging.googleapis.com/trace_sampled`, `sourceLocation`, plain `trace_id` / `span_id`, and any `extra={"json_fields": {...}}`.
  - Exceptions logged at ERROR or above get the Error Reporting `@type` and `serviceContext`.
  - Health probes are excluded from tracing, but their logs still pick up the inbound trace header, so they line up with Cloud Run's request logs.
- **Request telemetry (`observability/middleware.py`):**
  - `RequestTelemetryMiddleware` binds `request_id` / method / path to every log line during the request.
  - It returns `X-Trace-Id`, `X-Request-Id`, and `Server-Timing: app;dur=…`. These are exposed via CORS for the Vite UI.
  - It writes one `http_request_completed` access log per request with `http_route`, `http_status`, and `latency_ms`, replacing uvicorn's access log.
  - `/api/v1/chat` responses include an `observability` block with `trace_id` and a Cloud Trace console deep link.
- **Infrastructure:**
  - `terraform/observability.tf` adds the log-based distribution metric `logging.googleapis.com/user/sec_edgar/api_request_latencies`: per-route / method / status latency with trace exemplars, for p50/p95/p99 dashboards and alerting.
  - The three telemetry tables get an additive, nullable `trace_id` column.
  - Already in place: `cloudtrace.googleapis.com` and `logging.googleapis.com` are enabled, and `sec-analyst-sa` holds `roles/cloudtrace.agent` and `roles/logging.logWriter`.
- **Dependencies:** `opentelemetry-exporter-gcp-trace`, `-propagator-gcp`, `-resourcedetector-gcp`, `-instrumentation-fastapi`, and `-instrumentation-httpx` were added to `pyproject.toml`, `requirements.txt`, and `uv.lock`.
- **Tests (`tests/test_observability.py`, 12 tests):**
  - An in-memory span exporter checks the full chat-turn span hierarchy, propagation of upstream `traceparent` and `X-Cloud-Trace-Context`, response headers, and that health probes are excluded from tracing.
  - Other tests cover the JSON log format and reserved fields, uvicorn log routing, error-path tool spans, context propagation across worker threads, and `trace_id` on BigQuery telemetry rows.

> [!TIP]
> Spans are exported in batches on a background thread. Under Cloud Run's default request-based CPU allocation, an idle instance may hold spans until its next request or until shutdown. Deploy with `--no-cpu-throttling` if you need spans to appear within seconds even under low traffic.

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

> [!NOTE]
> As of 2026-11-16, Cloud Run serves revision `sec-edgar-analyst-00011-kct`, built from commit `3d743f9`. None of the later work is deployed yet: the IAP-session sign-in, observability, `gemini-3.8-flash` on `global`, agent-driven retrieval, and structured output. Until a new image is built and deployed, people who pass IAP still see the Okta/Bearer sign-in panel.

---

## Implementation Gaps vs. `docs/project_reference.md`

These pieces are wired up end-to-end across the backend, Terraform, and Cloud Run: live Gemini (`gemini-3.8-flash` and `gemini-3.5-flash`), Google ADK `Runner` SSE streaming, Model Armor with SDP and projection filters, versioned prompt loading, streaming BigQuery telemetry and audit sinks, Firestore feedback, LookML models and dashboards, and OpenTelemetry tracing plus trace-correlated JSON logging (live Cloud Trace check pending; see gap 3).

**Closed since the last revision (removed from the gap list):**
- **Container corpus packaging** *(Secs. 4, 6)*: data paths resolve correctly for every install layout, the corpus can be hydrated from GCS at startup, `/health` reports where the corpus comes from, and Cloud Build and GitHub Actions run an offline smoke test inside the image before `docker push`. The one leftover step, publishing the vector DB artifacts, is now part of gap 1 below.
- **External HTTPS load balancer in front of IAP** *(Sec. 8)*: `terraform/load_balancer.tf` provisions a global IP, a Google-managed certificate, a `RESTRICTED` TLS policy, a URL map, an HTTPS proxy, a 443 forwarding rule, and an 80→443 redirect. It is now optional (`cloud_run_access_mode = "load_balancer"`); the default is IAP directly on Cloud Run (see gap 2).
- **10-K context for synthesis** *(Secs. 2, 6, 7)*: now agent-driven retrieval (see "Agent-Driven 10-K Retrieval" above), replacing the forced full-section `CachedContent` bundle.
  - Context caching comes from ADK's App-level `ContextCacheConfig` over the instruction, tools, and retrieved passages within a turn. It needs `serviceconsumermanagement.googleapis.com` (in `terraform/apis.tf`); the first cache in a project waits about 2 minutes while Vertex provisions it.
  - The financial reasoning model is `gemini-3.8-flash` on the `global` endpoint (`EDGAR_GEMINI_LOCATION=global`), because this project can't reach `gemini-2.5-pro` (HTTP 404).
  - **Live comparison on 2026-11-16** (NVDA FY2024–FY2026 export-control question):
    - The forced 34k-token bundle took about 27.7 s to first token.
    - With agent-driven retrieval across five runs, the first request was about 7.8k tokens and first token arrived in about 3.0–4.0 s; one run took 6.9 s. Time to first token here is the first streamed thought.
    - The agent made 5–6 targeted retrieval calls covering FY2024, FY2025, and FY2026. Every cited `chunk_id` resolved, and the fact-check passed.
    - Within a turn, up to about 32k context tokens were served from ADK's cache.
    - End-to-end turns take about 65–78 s, because of multiple model calls and long reports.
  - **Structured output, live on 2026-11-16** (same question, 3 runs after adding the retrieval budget):
    - All 16 findings verified, and every cited chunk resolved.
    - Answers were 4.1–4.5k characters, down from 10.6–16k as free-form Markdown.
    - 4 retrieval calls per run; time to first token about 3.2–3.6 s; turns took 58–63 s.
    - The judge scored every run 1.0.
  - **Correction:** an earlier note here called the "$4.5 billion H20 charge" an out-of-corpus hallucination. That was wrong. The FY2026 10-K text stores it Markdown-escaped as `$4\.5 billion`, which a plain-text search missed. The structured run cites it with a verbatim quote from `NVDA_2026_item_7_0002`.
- **Cloud Build trigger and lint/format gates** *(Sec. 8: CI/CD)*:
  - `terraform/cloudbuild.tf` manages the GitHub connection, the `main` deploy trigger, the PR verify trigger, and the dedicated `sec-edgar-cloudbuild-sa` service account. Set `enable_cloudbuild_trigger = true` after the one-time GitHub App and token setup described in [`terraform/README.md`](terraform/README.md).
  - `cloudbuild.yaml` gates the build on `ruff check` / `ruff format --check`, `terraform fmt` / `validate`, ESLint + Vitest, the secret scan, and pytest.
  - `make lint`, `make format`, and `make check` run the same checks locally.
  - GitHub Actions deploys only when opted in.

### Gap Summary

| # | Reference Area | Status | Main Blocker / Next Action |
| :-- | :--- | :--- | :--- |
| 1 | **Corpus publication and full S&P 500 breadth** *(Secs. 3, 7)* | Partly done | Full-index extraction, chunking, and Vertex embeddings are done. Next: load BigQuery, publish the Vector Search embeddings, import into Vertex AI Search, and add an R&D metric. |
| 2 | **VPC Service Controls and private networking** *(Sec. 8)* | Partly blocked | IAP-only access is done. The perimeter Terraform is written but can't be applied from this sandbox: the only available account has no org-level Access Context Manager rights. Still doable here: Private Service Connect for Vector Search. |
| 3 | **OpenTelemetry, Cloud Trace, and trace-correlated JSON logs** *(Secs. 2, 4, 6, 10)* | Implemented; live check pending | Code, Terraform, and tests are done (see [`docs/observability.md`](docs/observability.md)). Next: deploy, then check one live `/api/v1/chat` trace in Cloud Trace. |
| 4 | **Golden eval dataset, `make eval-all`, and coverage gate** *(Secs. 4, 9)* | Not started | No golden set, eval runner, coverage measurement, or CI gate. |

---

### 1. Corpus Publication and Full S&P 500 Breadth *(Secs. 3, 7)*

**Reference requirement:** Item 1A and Item 7 Markdown for the **full S&P 500** from 2020 onward, stored in `gs://fde-sec-edgar-sandbox-dev-sec-corpus/`, searchable through Vertex AI Search, Vector Search, and BigQuery. Section 3's example query ("Compare NVIDIA and Apple's **R&D spend** from 2024–2026") assumes R&D data is available.

**Done:**
- **Extraction:** all 501 tickers, 3,022 FY2020–FY2026 profiles, with no failed tickers. Structured JSON and Markdown are in GCS.
- **Chunking and embedding:** 342,945 chunks embedded with `text-embedding-005` (`embedding_backend: vertex_ai`). Chunk health is 99.76 (`HEALTHY`).
- **Manifests in GCS:** `bigquery_filing_chunks.jsonl`, `bigquery_financials.jsonl`, `vertex_ai_search_metadata.jsonl`, and the catalogs are in `manifests/`.

**Still open:**
- **BigQuery isn't loaded.** `sec_edgar.filing_chunks` has 0 rows and `sec_edgar.financial_metrics` still holds the 114-row sample. `vector_db_catalog.json` reports `loaded_to_bigquery: false` and `published_monitoring_metrics: false`.
- **Vector Search has no embeddings.** `vector_search/embeddings/` (about 6 GB locally) isn't in GCS, so the index has nothing to serve.
- **Vertex AI Search is empty.** Nothing imports documents into the `sec-edgar-10k-corpus-ds` data store; there is no `documents:import` call anywhere in `src/`, `scripts/`, or `terraform/`.
- **No R&D metric.** XBRL extraction covers only `Revenue`, `Operating Income`, and `Net Income`, so Section 3's R&D example query can't be answered quantitatively.

**Work to close:**
1. Load BigQuery: rerun `scripts/build_vector_db.py --upload-only` without `--no-load-bigquery` (or start the load job for `manifests/bigquery_filing_chunks.jsonl`), and reload `financial_metrics` from the 3,022-row `bigquery_financials.jsonl`. Publish the chunk-health metrics too.
2. Upload `vector_search/embeddings/` to GCS. Then set `seed_vertex_vector_index = true`, run `terraform apply`, and confirm `:findNeighbors` returns results from `sec-edgar-10k-chunks-endpoint`.
3. Add a Discovery Engine `documents:import` step that reads from GCS (`vertex_ai_search_metadata.jsonl` plus the `unstructured/` Markdown), either in `build_vector_db.py` or a new `scripts/index_vertex_search.py`, and poll the import operation until it completes.
4. Add an R&D metric:
   - Map `us-gaap_ResearchAndDevelopmentExpense` (and `..._ExcludingAcquiredInProcessCost`) in `dataset/extractor.py`.
   - Update `FinancialDataProfile`, `terraform/schemas/financial_metrics.json`, `v_financial_variances_yoy`, the LookML views, `VarianceCalculationTool`, `AnalystMetricReference`, and the frontend variance table.
   - Issuers that don't report R&D should get `null`, not `0`.

**Done when:**
- `sec_edgar.filing_chunks` and `financial_metrics` hold the full index.
- `vector_db_catalog.json` shows every upload/load flag set to `true`.
- Vector Search and Vertex AI Search both return hits for a ticker outside the old 18-company sample.
- The NVDA vs. AAPL R&D query returns a deterministic variance table with citations.

### 2. VPC Service Controls and Private Networking *(Sec. 8)*

**Reference requirement:** "A strict service perimeter locks down Vertex AI, Cloud Run, GCS, and BigQuery"; TLS 1.3 in transit; GMEK at rest.

**Done (code, Terraform, and tests):**
- **No anonymous or public access.** By default (`cloud_run_access_mode = "iap_direct"`), IAP is turned on for the Cloud Run service itself (`iap_enabled = true`). Every request to the `run.app` URL needs a Google sign-in from a principal in `iap_authorized_principals`, and only the IAP service agent holds `roles/run.invoker`. There's no load balancer, static IP, or certificate to run. The old LB path is still available with `cloud_run_access_mode = "load_balancer"`, and its HTTPS proxy now uses a `RESTRICTED` / TLS 1.2+ SSL policy.
- **IAP-session sign-in in the app.** When there's no `Authorization` header, the backend accepts the `X-Goog-IAP-JWT-Assertion` that IAP injects. It must be ES256, issued by `https://cloud.google.com/iap`, and carry the Cloud Run audience `/projects/<number>/locations/<region>/services/<name>`, which Terraform writes to `sec-edgar-iap-audience`. IAP assertions carry no group claims, so users who match no RBAC group get `IAP_ASSERTION_DEFAULT_ROLE` (`analyst` by default; `viewer` or empty are the other options). Admin still requires a group. The frontend signs in from the IAP session silently, and Sign Out clears the IAP cookie.
- **Perimeter definition.** `terraform/vpc_service_controls.tf` restricts `aiplatform`, `artifactregistry`, `bigquery`, `discoveryengine`, `dlp`, `firestore`, `modelarmor`, `run`, `secretmanager`, and `storage`. Its ingress rules allow:
  - by identity, from any source: operators (`vpc_sc_admin_identities`), the CI/CD and Cloud Build SAs, the runtime and per-agent SAs, and the Cloud Logging → BigQuery sink writer;
  - from any identity: invocations of `run.googleapis.com`. VPC-SC can't evaluate IAM identities on Cloud Run invocations, so IAP decides who gets in. Set `vpc_sc_app_ingress_access_levels` to an IP-based access level to also restrict by network.

  It deploys as a dry-run spec (`use_explicit_dry_run_spec = true`) unless `vpc_sc_enforce = true`.
- Plan-level tests are in `tests/test_deployment_pipeline.py`. Auth tests in `tests/test_iap_okta_auth.py` cover IAP-only sessions, wrong-audience and forged assertions, and default-role handling. A Vitest case covers the silent IAP sign-in.

**Can't be done in this sandbox:** every VPC-SC perimeter lives in an Access Context Manager policy, and every policy belongs to an organization. That includes policies scoped to a single project. Creating one requires `accesscontextmanager.policies.create` granted at the org level; Access Context Manager roles granted on a project have no effect. Org `1059565594977` is administered by Argolis-managed Google groups. `admin@jackbetcher.altostrat.com`, the only account available, holds no org roles beyond `resourcemanager.projectCreator` (`testIamPermissions` fails for `accesscontextmanager.policies.create` and `.list`). It also lacks `orgpolicy.policy.set` and `iam.denypolicies.create`, so org-policy and IAM-deny substitutes are out too. `vpc_service_controls.tf` stays off (`enable_vpc_service_controls = false`). In an org where someone can create a scoped policy and delegate it, apply it with:

```bash
terraform -chdir=terraform apply \
  -var=enable_vpc_service_controls=true \
  -var=vpc_sc_access_policy_id=<POLICY_NUMBER>
```

**Controls in place here instead:**
- IAP-only access to Cloud Run, with only the IAP service agent holding `run.invoker`.
- Public access prevention and uniform bucket-level access on the corpus and Terraform state buckets.
- Least-privilege runtime and per-agent service accounts.
- Model Armor and SDP on prompts and responses.
- Security audit logs routed to BigQuery.

These limit who can reach the app and data, but unlike a perimeter they don't stop a stolen credential from being used outside the project.

**Remaining:**
1. *(Only if a scoped access policy becomes available.)* Apply in dry-run. Review the dry-run violations in Logs Explorer (`protoPayload.metadata.dryRun=true`, `@type=...VpcServiceControlAuditMetadata`), add any missing ingress rules, then set `vpc_sc_enforce = true`.
2. Replace the public Vector Search endpoint (`public_endpoint_enabled = true`) with Private Service Connect: VPC/subnet, Cloud Run Direct VPC egress (all traffic, `restricted.googleapis.com`), PSC forwarding rule and DNS, and point `VertexVectorSearchAdapter` at the private `:findNeighbors` host. Once Cloud Run egress goes through the VPC, drop the runtime and agent SAs from the identity-based ingress rule.
3. Data at rest already uses Google-managed encryption (GMEK), so nothing is needed there.

**Done when:** the perimeter is enforced; signed-in users reach the app through IAP; and BigQuery or GCS calls from outside the perimeter by an unlisted identity are denied with `VPC_SERVICE_CONTROLS` errors.

### 3. OpenTelemetry, Cloud Trace, and Trace-Correlated JSON Logs *(Secs. 2, 4, 6, 10)*

**Reference requirement:** End-to-end latency tracing through OpenTelemetry and Cloud Trace, plus FastAPI structured JSON logs with trace IDs exported to Cloud Logging.

**Current state:** implemented in code, Terraform, and tests. See [Current State § 5](#5-observability-end-to-end-latency-tracing--structured-logs-observability-terraformobservabilitytf) and [`docs/observability.md`](docs/observability.md). This was verified locally (the test suite, plus a uvicorn run in Cloud Run JSON mode with upstream `traceparent` continuation), but not yet against live Cloud Trace. The deployed revision predates this work (see [Sharing the App Internally](#sharing-the-app-internally)).

**Remaining:**
1. Run `terraform apply`. It only adds things: the `sec_edgar/api_request_latencies` metric and the nullable BigQuery `trace_id` columns.
2. Deploy, send one `/api/v1/chat` request, and confirm in Cloud Trace / Logs Explorer (`trace="projects/iron-radio-514217-q7/traces/<X-Trace-Id>"`).
3. Optionally, add a p95 latency alert policy on the new log-based metric once notification channels exist.

**Done when:** one `/api/v1/chat` request shows a single Cloud Trace with nested FastAPI, ReAct, tool, and model spans; that request's Cloud Logging entries are grouped under the trace; and its BigQuery telemetry rows carry the same `trace_id`.

### 4. Golden Eval Dataset, `make eval-all`, and Coverage Gate *(Secs. 4, 9)*

**Reference requirement:** A golden dataset of 20+ query–variance–explanation mappings in BigQuery; Gemini 3.5 Flash LLM-as-a-Judge regression checks through `make eval-all` before deployment; >80% pytest coverage; ≥92% grounding recall; 100% numeric agreement and citation coverage; p95 TTFT ≤ 3.0 s; 100% safety compliance.

**Current state:**
- 147 pytest tests with mocked LLM and Search APIs.
- `NarrativeEvaluatorAgent` (4-axis rubric) and `/api/v1/evaluate`.
- `verify_narrative_against_calculations` enforces numeric and citation locks for each response.
- No golden dataset or BigQuery eval tables, no `make eval-all` target, and `pytest-cov` is not installed, so coverage is unmeasured.
- Grounding recall, p95 TTFT, and safety compliance aren't benchmarked in CI. p95 TTFT appears only in the Looker dashboard.

**Work to close:**
1. Write `evals/golden_dataset.jsonl` with at least 20 items covering single-company longitudinal, multi-company peer comparison, and thematic-shift queries (for example, AI-risk wording from 2024 to 2026). Fields: `query`, `tickers`, `fiscal_years`, `expected_variances`, `expected_chunk_ids` / sections, `reference_explanation`, and `category`.
2. Provision `sec_edgar.golden_eval_set` and `sec_edgar.eval_results` in Terraform, and add a loader script.
3. Write `scripts/run_evals.py` to compute these metrics against the thresholds:
   - numeric agreement (100%)
   - citation coverage (100%)
   - grounding recall: expected chunks retrieved and cited (≥ 92%)
   - Flash judge scores for faithfulness, explanation quality, and thematic consistency
   - Model Armor red-team prompts blocked (100%)
   - p95 TTFT in live mode (≤ 3000 ms)

   Write the results to `eval_results` and exit non-zero on any regression.
4. Add `make eval-all`, deterministic by default and `EVAL_MODE=live` for Gemini/Vertex.
5. Add `pytest-cov` to the `dev` extras and run `pytest --cov=edgar_capstone --cov-fail-under=80` in `make test`, `cloudbuild.yaml`, and GitHub Actions. Run `make eval-all` as a gate before the deploy stage.

**Done when:** CI fails if coverage drops below 80% or any eval threshold regresses, and each deploy has a matching `eval_results` row in BigQuery.
