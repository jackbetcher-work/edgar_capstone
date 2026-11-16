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

# 4. Run secret scan, backend pytest suite (109 tests), ESLint, and frontend Vitest suite (6 tests)
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
EDGAR_EXECUTION_MODE=live \
make dev

# Live mode via Gemini API key
GEMINI_API_KEY="$YOUR_GEMINI_API_KEY" EDGAR_EXECUTION_MODE=live make dev

# Optional model & latency target overrides (defaults shown)
# EDGAR_ANALYST_MODEL=gemini-2.5-pro  EDGAR_FLASH_MODEL=gemini-3.5-flash  EDGAR_TTFT_SLO_MS=3000
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
> `gs://fde-sec-edgar-sandbox-dev-sec-corpus/` currently has no `manifests/bigquery_filing_chunks.jsonl`. The vector DB artifacts were never uploaded because of an upload bug that is now fixed. Publish the locally staged artifacts once with `.venv/bin/python scripts/build_vector_db.py --upload-only`. Until then, use `HYDRATION_SCOPES=unstructured`. The local tools will chunk the hydrated 10-K Markdown on the fly, using deterministic embeddings.

---

## Repository Structure

```text
├── docs/project_reference.md        # Target architecture and product specification
├── src/edgar_capstone/
│   ├── server.py                    # FastAPI gateway, auth/RBAC routes, RAG & telemetry APIs
│   ├── telemetry.py                 # Streaming BigQuery telemetry/audit sinks, Firestore feedback & LookML catalog
│   ├── agents/                      # Google ADK agents, live Runner SSE engine, ReAct & context tracking
│   │   ├── live_runtime.py          # ADK Runner (StreamingMode.SSE), Vertex/Gemini config & retry backoff
│   │   ├── react_loop.py            # Structured ReAct loop + post-generation numeric/citation lock
│   │   ├── context_memory.py        # Pro/Flash model routing, ContextLatencyTracker, session compaction
│   │   ├── evaluation_and_coding.py # Flash LLM-as-a-Judge evaluator & SQL/Python/LookML coding assist
│   │   ├── callbacks.py             # Guardrails, first-chunk SSE TTFT capture & grounding callbacks
│   │   └── tools.py                 # Resilient ADK FunctionTools for metrics, variances, RAG & 10-K cache
│   ├── dataset/                     # SEC EDGAR 10-K XBRL + Markdown extraction pipeline (edgartools)
│   │   ├── paths.py                 # Install-layout-aware data root resolution (source tree, /app/data, cwd)
│   │   └── corpus_hydration.py      # GCS -> local corpus hydration (startup + CLI) and /health corpus status
│   ├── vector_db/                   # Section-aware 10-K chunker, 768-d embedder, chunk-health monitor
│   └── security/                    # IAP/OIDC JWT verification, Cloud Identity RBAC, SPIFFE S2S auth
├── looker/                          # LookML semantic layer (manifest, model, 6 views, 2 LookML dashboards)
├── frontend/                        # React 19 + TypeScript + Tailwind v4 split-pane terminal UI
├── scripts/                         # CLI entrypoints: dataset build, vector DB build (+ --upload-only), corpus hydration, container smoke test, secret scanner
├── terraform/                       # GCP IaC: BigQuery + sinks, Firestore, Vector/Discovery Search, Cloud Run, IAM, IAP + HTTPS LB, Cloud Build triggers
├── tests/                           # Automated pytest suite (109 tests across agents, live ADK, GCP adapters, telemetry, APIs, IaC, container packaging)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + Python 3.11 runtime) with baked manifests + build-time corpus check
├── .dockerignore                    # Keeps secrets and gitignored heavy corpus artifacts out of the build context
├── cloudbuild.yaml                  # Cloud Build: parallel lint/test gates -> build (+ offline smoke test) -> push + deploy (main) / verify-only (PRs)
└── .github/workflows/deploy.yml     # GitHub Actions WIF CI/CD workflow
```

---

## Current State of the Repository

### 1. Curated SEC 10-K Dataset & Vector Pipeline (`dataset/`, `vector_db/`, `scripts/`)
- **10-K Extraction (`build-sec-dataset`):** Extracts XBRL financial statements (`Revenue`, `Operating Income`, `Net Income`) and cleans Item 1A / Item 7 Markdown via `edgartools`, writing local manifests and uploading to GCS (`gs://fde-sec-edgar-sandbox-dev-sec-corpus`).
- **Materialized Corpus (`data/curated_dataset/`):** Currently generated in `sample` mode covering **18 S&P 500 companies** across 8 GICS sectors from **FY2020–FY2026** (**114 company-year profiles**). `data/sp500_constituents.csv` and `build-sec-dataset --sp500` are included for full-index runs.
- **Chunking, Embeddings & Health Monitoring (`build-vector-db`):** Splits filings by section headings into **12,180 metadata-tagged chunks** (`5,556` Item 1A + `6,624` Item 7) with 768-dimensional embeddings (`vertex` mode via `text-embedding-005` or `deterministic` local hash mode) and evaluates chunk/embedding health metrics (`chunk_health_report.json`). `--upload-only` publishes already-staged vector DB artifacts to GCS without re-chunking or re-embedding.
- **Container Corpus Packaging & Hydration (`dataset/paths.py`, `dataset/corpus_hydration.py`, `hydrate-sec-corpus`):** Data paths are now resolved the same way in a source checkout, an editable install, and the container's non-editable `/opt/venv` install. Before this fix, the image looked for its corpus under `site-packages` and never found the baked `/app/data` manifests. Gitignored chunks and 10-K Markdown can be downloaded from GCS at FastAPI startup (`EDGAR_CORPUS_HYDRATION`) or with the CLI. Downloads are atomic, existing files are skipped, ticker filters are supported, and paths that would escape the dataset directory are rejected.

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
- **Context Latency, Cost Telemetry & Session Memory (`context_memory.py`, `telemetry.py`, `callbacks.py`):**
  - `ContextLatencyTracker` decomposes every model call into context preparation + prefill latency vs. decode latency, computes exact USD input/output cost and 75% cached-token discount savings (`gemini-2.5-pro` & `gemini-3.5-flash`), monitors TTFT against the `<= 3000 ms` target, and streams per-call records directly to BigQuery (`sec_edgar.llm_telemetry_events`) via `BigQueryTelemetrySink` (`tabledata.insertAll` + structured Cloud Logging sink records).
  - `BigQueryTelemetrySink` also streams conversational turn/session metrics (`session_telemetry_events`: cumulative session length, companies analyzed, reports generated, ADK tool usage, and latency/cost) and authentication / 10-K access audit trails (`security_audit_logs`), with `"auto"` | `"gcp"` | `"local"` fallback (`EDGAR_TELEMETRY_BACKEND`).
  - `ConversationMemoryManager` maintains multi-turn session state for follow-up entity/period recall and compacts older turns above a configurable token threshold.
  - `create_edgar_adk_app` configures an ADK `App` with `ContextCacheConfig` and `EventsCompactionConfig`.

### 3. FastAPI Backend, Persistent Feedback, LookML Semantic Layer & Split-Pane Frontend (`server.py`, `looker/`, `frontend/`)
- **Backend API & Firestore Feedback Persistence:** Exposes `/health`, `/api/v1/chat`, `/api/v1/variances`, `/api/v1/search`, `/api/v1/metrics`, `/api/v1/explore/{explore_name}`, `/api/v1/looker/catalog`, `/api/v1/section`, `/api/v1/companies`, `/api/v1/evaluate`, `/api/v1/coding-assist`, `/api/v1/telemetry/context-latency`, `/api/v1/telemetry/sessions`, `/api/v1/feedback`, `/api/v1/secrets/status`, and `/api/v1/auth/*`. `FirestoreFeedbackStore` persists analyst thumbs-up/down ratings and audit comments to Google Cloud Firestore Native (`analyst_feedback` collection via REST v1 `documents` & `:runQuery`, controlled by `EDGAR_FEEDBACK_BACKEND`). Serves the compiled React bundle from `/` in production.
- **LookML Semantic Layer & Dashboards (`looker/`):** Defines `manifest.lkml`, model `sec_edgar_analytics.model.lkml` (7 explores: `yoy_variances`, `sector_benchmarks`, `peer_comparison`, `chunk_health`, `llm_telemetry_and_cost`, `session_analytics`, `security_audit_logs`), 6 `.view.lkml` files mapped to BigQuery tables/views, and 2 `.dashboard.lookml` executive dashboards (`sec_edgar_observability_and_cost` and `sec_edgar_financial_variances`), introspected via `GET /api/v1/looker/catalog` and queried through `LookerQueryTool`.
- **Split-Pane Terminal UI:** Built with React 19, TypeScript, Tailwind CSS v4, and Vite. Left pane renders conversational queries, scope filters, deterministic variance tables, and interactive citation chips; right pane displays retrieved 10-K chunks and full cached Item 1A / Item 7 filing sections, with graceful offline fallback data if the backend is unreachable.

### 4. Security, Model Armor Guardrails, Infrastructure & CI/CD (`security/`, `terraform/`, `cloudbuild.yaml`)
- **AuthN / AuthZ:** Verifies IAP (`ES256`) and Google Workspace / Okta OIDC (`RS256`) JWTs, enforces Cloud Identity group RBAC (`admins`, `analysts`, `viewers` with `user:` bindings disallowed in Terraform), runs under custom runtime SA `sec-analyst-sa`, and enforces least-privilege S2S tool boundaries across four per-agent service accounts (`root-orchestrator-agent-sa`, `financial-analyst-agent-sa`, `context-cache-agent-sa`, `model-armor-guardrail-sa`) with SPIFFE IDs.
- **Managed Model Armor, Cloud SDP & Projection Filters (`security/model_armor.py`, `terraform/model_armor.tf`):** Replaces regex prompt-injection filters with regional Google Cloud Model Armor v1 (`sanitizeUserPrompt` & `sanitizeModelResponse` on `sec-edgar-analyst-armor`), Google Cloud Sensitive Data Protection (`content:deidentify` via `sec-edgar-sdp-inspect` & `sec-edgar-sdp-deidentify`), SEC forward-looking financial projection / MNPI leak filters, and ADK `ModelArmorSafetyPlugin` (`BasePlugin`) with session-poisoning prevention.
- **Terraform & CI/CD:** Provisions BigQuery analytical tables, streaming telemetry/audit sink tables (`llm_telemetry_events`, `session_telemetry_events`, `security_audit_logs` with table-scoped custom append role `secEdgarTelemetrySinkAppend` + Cloud Logging sink) and SQL views (`v_financial_variances_yoy`, `v_filing_chunk_coverage`, `v_chunk_health_metrics`, `v_telemetry_cost_and_latency`), Firestore Native feedback database (`analyst_feedback_db`), Vertex AI Vector Search, Vertex AI Search (Discovery Engine), Google Cloud Model Armor & Cloud SDP templates, Secret Manager, Cloud Monitoring chunk-health dashboards, Cloud Run v2 (ingress `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`), an IAP backend service behind a global external HTTPS load balancer (`load_balancer.tf`: managed TLS certificate for `lb_domain` or `<ip>.nip.io`, URL map, HTTPS proxy, and 80→443 redirect), and Workload Identity Federation. `terraform/cloudbuild.tf` manages a 2nd-gen GitHub connection plus two Cloud Build triggers, both running as the dedicated least-privilege `sec-edgar-cloudbuild-sa`: merges to `main` run `cloudbuild.yaml` with `_PIPELINE_MODE=deploy`, and pull requests run it with `_PIPELINE_MODE=verify` (no push or deploy). `cloudbuild.yaml` first runs four quality gates in parallel: `lint-python` (`ruff check` + `ruff format --check`), `lint-terraform` (`terraform fmt -check` + `validate`), `frontend` (ESLint + Vitest), and `test`. `.github/workflows/deploy.yml` runs the same gates on every push; its build and deploy jobs are opt-in via the `DEPLOY_WITH_GITHUB_ACTIONS` repository variable, so a merge never deploys twice. The pipelines run secret scanning (`verify_no_secrets.py`), tests, container builds, an offline in-image smoke test before pushing (`scripts/container_smoke_test.py --in-process --expect-corpus baked`), and Cloud Run deployments wired with `GOOGLE_GENAI_USE_VERTEXAI=true`, `EDGAR_MODEL_ARMOR_MODE=gcp`, `MODEL_ARMOR_TEMPLATE_ID=sec-edgar-analyst-armor`, `EDGAR_PROMPT_STORE_BACKEND=auto`, `EDGAR_TELEMETRY_BACKEND=gcp`, `EDGAR_FEEDBACK_BACKEND=gcp`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `EDGAR_EXECUTION_MODE=auto`.

---

## Implementation Gaps vs. `docs/project_reference.md`

These pieces are wired up end-to-end across the backend, Terraform, and Cloud Run: live Gemini (`gemini-2.5-pro` and `gemini-3.5-flash`), Google ADK `Runner` SSE streaming, Model Armor with SDP and projection filters, versioned prompt loading, streaming BigQuery telemetry and audit sinks, Firestore feedback, and LookML models and dashboards.

**Closed since the last revision (removed from the gap list):**
- **Container corpus packaging** *(Secs. 4, 6)*: data paths resolve correctly for every install layout, the corpus can be hydrated from GCS at startup, `/health` reports where the corpus comes from, and Cloud Build and GitHub Actions run an offline smoke test inside the image before `docker push`. The one leftover step, publishing the vector DB artifacts, is now part of gap 1 below.
- **External HTTPS load balancer in front of IAP** *(Sec. 8)*: `terraform/load_balancer.tf` provisions a global IP, a Google-managed certificate, a URL map, an HTTPS proxy, a 443 forwarding rule, and an 80→443 redirect. Cloud Run ingress is `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`, so the load balancer is the only public entry point.
- **Cloud Build trigger and lint/format gates** *(Sec. 8: CI/CD)*:
  - `terraform/cloudbuild.tf` manages the GitHub connection, the `main` deploy trigger, the PR verify trigger, and the dedicated `sec-edgar-cloudbuild-sa` service account. Set `enable_cloudbuild_trigger = true` after the one-time GitHub App and token setup described in [`terraform/README.md`](terraform/README.md).
  - `cloudbuild.yaml` gates the build on `ruff check` / `ruff format --check`, `terraform fmt` / `validate`, ESLint + Vitest, the secret scan, and pytest.
  - `make lint`, `make format`, and `make check` run the same checks locally.
  - GitHub Actions deploys only when opted in.

### Gap Summary

| # | Reference Area | Status | Main Blocker / Next Action |
| :-- | :--- | :--- | :--- |
| 1 | **Corpus publication and full S&P 500 breadth** *(Secs. 3, 7)* | In progress | The S&P 500 extraction is running. Next: Vertex embeddings, GCS/BigQuery/Search ingestion, and an R&D metric. |
| 2 | **Vertex AI Context Caching for 10-K sections** *(Secs. 2, 6, 7)* | Partial | 10-K sections are cached only in each process's memory. They need to become shared, remote `CachedContent` resources. |
| 3 | **VPC Service Controls and private networking** *(Sec. 8)* | Not started | No service perimeter. The Vector Search endpoint is public. No TLS policy. |
| 4 | **OpenTelemetry, Cloud Trace, and trace-correlated JSON logs** *(Secs. 2, 4, 6, 10)* | Not started | No span exporter or tracer provider. App logs are not JSON and carry no trace IDs. |
| 5 | **Golden eval dataset, `make eval-all`, and coverage gate** *(Secs. 4, 9)* | Not started | No golden set, eval runner, coverage measurement, or CI gate. |

---

### 1. Corpus Publication and Full S&P 500 Breadth *(Secs. 3, 7)*

**Reference requirement:** Item 1A and Item 7 Markdown for the **full S&P 500** from 2020 onward, stored in `gs://fde-sec-edgar-sandbox-dev-sec-corpus/`, searchable through Vertex AI Search, Vector Search, and BigQuery. Section 3's example query ("Compare NVIDIA and Apple's **R&D spend** from 2024–2026") assumes R&D data is available.

**Current state:**
- The committed catalogs still describe the 18-issuer sample: `dataset_catalog.json` has 114 profiles, and `vector_db_catalog.json` has 12,180 chunks.
- A `scripts/build_sec_dataset.py --sp500` run is in progress locally. At the time of this revision, 155 ticker directories (940 structured JSON and 940 Markdown files) are on disk. The catalog is rewritten when the run finishes.
- `vector_db_catalog.json` reports `embedding_backend: deterministic` along with `uploaded_to_gcs: false`, `loaded_to_bigquery: false`, and `published_monitoring_metrics: false`. Hash embeddings make Vector Search and BigQuery `VECTOR_SEARCH` semantically meaningless.
- The bucket's `manifests/` folder holds only `bigquery_financials.jsonl`, `dataset_catalog.json`, and `vertex_ai_search_metadata.jsonl`. `bigquery_filing_chunks.jsonl` and `vector_search/embeddings/` are missing.
- The Vector Search index reads embeddings from `contents_delta_uri`, which Terraform attaches only after embeddings exist in GCS. Nothing imports documents into the `sec-edgar-10k-corpus-ds` Discovery Engine data store; there is no `documents:import` call anywhere in `src/`, `scripts/`, or `terraform/`.
- XBRL extraction covers only `Revenue`, `Operating Income`, and `Net Income`.

**Work to close:**
1. Let the `--sp500` run finish. Then check `failed_tickers`, `warnings`, and `uploaded_to_gcs` in `dataset_catalog.json`, and re-run failed tickers with `--tickers ... --force`.
2. Rebuild the chunks with real embeddings: `scripts/build_vector_db.py --embedding-backend vertex --force-reembed`. Leave out `--local-only`, `--no-load-bigquery`, and `--no-publish-metrics` so the run uploads to GCS, loads `sec_edgar.filing_chunks`, and publishes chunk-health metrics.
3. Attach the embeddings to the Vector Search index by setting `seed_vertex_vector_index = true` and running `terraform apply`. Then confirm `:findNeighbors` returns results from `sec-edgar-10k-chunks-endpoint`.
4. Add a Discovery Engine `documents:import` step that reads from GCS, using `vertex_ai_search_metadata.jsonl` plus the `unstructured/` Markdown. Add it to `build_vector_db.py` or a new `scripts/index_vertex_search.py`, and poll the import operation until it completes.
5. Add an R&D metric. Map `us-gaap_ResearchAndDevelopmentExpense` (and `..._ExcludingAcquiredInProcessCost`) in `dataset/extractor.py`, then update `FinancialDataProfile`, `terraform/schemas/financial_metrics.json`, `v_financial_variances_yoy`, the LookML views, `VarianceCalculationTool`, and the frontend variance table. Issuers that don't report R&D should get `null`, not `0`.
6. Refresh the corpus counts in this README and in `chunk_health_report.json`, then remove the `[!IMPORTANT]` upload note above.

**Done when:** `make docker-smoke-hydrated` passes with the default `chunks,unstructured` scopes; both catalogs report about 500 issuers, `embedding_backend: vertex`, and every upload/load flag set to `true`; Vertex AI Search returns hits for a non-sample ticker; and the NVDA vs. AAPL R&D query returns a deterministic variance table with citations.

### 2. Vertex AI Context Caching for 10-K Sections *(Secs. 2, 6, 7)*

**Reference requirement:** The Context Cache Manager checks the cache, loads the 10-K from GCS on a miss, and manages a **Vertex AI Context Cache** so that Gemini answers against cached 10-K context ("Real-time narrative synthesis using Gemini Context Caching").

**Current state:**
- `create_edgar_adk_app` sets the ADK `ContextCacheConfig` (`min_tokens=2048`, `ttl_seconds=1800`, `cache_intervals=10`). This lets ADK cache each agent's request prefix. 10-K sections are not registered as their own cache resources.
- `QueryCached10KTool` stores parsed sections in a module-level dict, `_SESSION_10K_CONTEXT_CACHE` in `tools.py`. That cache is per process: it isn't shared across Cloud Run instances, it's lost on cold start, and it has no TTL or eviction. With about 500 issuers × 7 years × 2 sections, it grows without bound.
- `context_memory.py` already parses `cached_content_token_count` and computes the 75% discount, so the telemetry is ready once real cache hits happen.

**Work to close:**
1. Add a `CachedContentManager` (e.g. `agents/context_cache_manager.py`) that registers each `{TICKER}_{FY}_{section}` Markdown section with `google.genai.Client.caches.create(model=..., config=CreateCachedContentConfig(display_name=cache_id, contents=[...], system_instruction=..., ttl="1800s"))`.
2. Make cache lookups work across instances. Either keep a `cache_id → cachedContents/{name}` mapping (with `expire_time`) in Firestore, or resolve by `display_name` through `caches.list()`. Extend the TTL with `caches.update` on each hit, and add a bounded LRU in front as a local L1 cache.
3. Pass `cached_content=<name>` in the synthesis step's `GenerateContentConfig`. Either extend ADK's cache manager or call the model directly for the grounded synthesis step, so the 10-K cache doesn't conflict with ADK's prefix caching. A cache can only be used with the model it was created for, so key caches by model.
4. Fall back to inline context when a section is under the model's minimum cacheable token count, or when the cache returns 404 or has expired. Delete caches when `/api/v1/section` content changes.
5. Grant `aiplatform.cachedContents.*` (via `roles/aiplatform.user`) to `context-cache-agent-sa` and `financial-analyst-agent-sa`, and add cache hit/miss/create counts to `llm_telemetry_events` and the LookML cost dashboard.
6. Write tests that mock `client.caches`, covering create, hit, expiry, fallback, and cross-instance reuse.

**Done when:** two Cloud Run instances answering the same company-year question reuse one `CachedContent` resource; telemetry shows `cached_content_token_count > 0` and non-zero cache savings; and nothing remains in process-local memory that can grow without bound.

### 3. VPC Service Controls and Private Networking *(Sec. 8)*

**Reference requirement:** "A strict service perimeter locks down Vertex AI, Cloud Run, GCS, and BigQuery"; TLS 1.3 in transit; GMEK at rest.

**Current state:**
- There are no `google_access_context_manager_*` resources.
- `terraform/vector_search.tf` sets `public_endpoint_enabled = true` on `sec-edgar-10k-chunks-endpoint`.
- The HTTPS proxy has no `google_compute_ssl_policy`.
- Data at rest uses Google-managed encryption (GMEK) by default, so that part of the requirement is already met.

**Work to close:**
1. Create or import an Access Context Manager access policy. This is an org-level resource and needs Access Context Manager Admin on the Argolis org. Add access levels for administrators and CI identities (the Cloud Build SA and the GitHub WIF principal).
2. Add a `google_access_context_manager_service_perimeter` around the project. Restrict `aiplatform`, `run`, `storage`, `bigquery`, `discoveryengine`, `secretmanager`, `firestore`, `modelarmor`, `dlp`, and `artifactregistry` (`.googleapis.com`). Add ingress/egress rules for IAP, Cloud Build, and the Cloud Logging → BigQuery sink. Roll out in **dry-run** mode first (`use_explicit_dry_run_spec = true`), review the violation logs, then enforce.
3. Replace the public Vector Search endpoint with a Private Service Connect endpoint. Add a VPC/subnet, Cloud Run Direct VPC egress, and the PSC forwarding rule and DNS, then point `VertexVectorSearchAdapter` at the private `:findNeighbors` host.
4. Attach a `google_compute_ssl_policy` (`profile = "RESTRICTED"`, `min_tls_version = "TLS_1_2"`) to the HTTPS proxy. Google front ends negotiate TLS 1.3 by default; LB SSL policies can't enforce a 1.3-only minimum, so document that limitation.
5. Add Terraform plan tests in `tests/test_deployment_pipeline.py` that check the perimeter, the restricted services, and that no public endpoints remain.

**Done when:** the perimeter is enforced; the smoke test passes through the HTTPS load balancer; and BigQuery or GCS calls from outside the perimeter are denied with `VPC_SERVICE_CONTROLS` errors.

### 4. OpenTelemetry, Cloud Trace, and Trace-Correlated JSON Logs *(Secs. 2, 4, 6, 10)*

**Reference requirement:** End-to-end latency tracing through OpenTelemetry and Cloud Trace, plus FastAPI structured JSON logs with trace IDs exported to Cloud Logging.

**Current state:**
- `cloudtrace.googleapis.com` is enabled, and `sec-analyst-sa` already has `roles/cloudtrace.agent`.
- `opentelemetry-api` and `opentelemetry-sdk` are installed only as dependencies of `google-adk`. ADK creates spans internally, but no `TracerProvider` or exporter is configured, so no spans leave the process.
- Only the telemetry/audit sink records are JSON. Application loggers use plain stdlib formatting and include no `logging.googleapis.com/trace` field.

**Work to close:**
1. Add these dependencies to `pyproject.toml` and `requirements.txt`: `opentelemetry-exporter-gcp-trace`, `opentelemetry-propagator-gcp`, `opentelemetry-instrumentation-fastapi`, and `opentelemetry-instrumentation-requests` / `-httpx`.
2. Create an `observability.py` module that initializes a `TracerProvider` with `BatchSpanProcessor(CloudTraceSpanExporter())` at FastAPI startup. Control it with `EDGAR_TRACING_BACKEND` (`auto | gcp | local | off`) and `EDGAR_TRACE_SAMPLE_RATIO`. Set the `CloudTraceFormatPropagator` and the W3C `traceparent` propagator so the trace context from the load balancer is carried through.
3. Instrument FastAPI and outbound HTTP calls (BigQuery, Vertex AI Search, Vector Search, GCS, Model Armor, Firestore). Add custom spans for each ReAct step: variance calculation, 10-K cache lookup, hybrid search, synthesis, and fact-check. Give the spans attributes such as `ticker`, `fiscal_year`, `model`, token counts, and `ttft_ms`.
4. Add a JSON log formatter that writes `severity`, `message`, `logging.googleapis.com/trace` (`projects/{project}/traces/{trace_id}`), `logging.googleapis.com/spanId`, and `logging.googleapis.com/trace_sampled` for every logger.
5. Add a `trace_id` column to `llm_telemetry_events`, `session_telemetry_events`, and `security_audit_logs` (schemas, sinks, and LookML) so BigQuery telemetry joins to Cloud Trace.
6. Add tests that use an in-memory span exporter to check the span hierarchy and log-record format.

**Done when:** one `/api/v1/chat` request shows a single Cloud Trace with nested FastAPI, ReAct, tool, and model spans; that request's Cloud Logging entries are grouped under the trace; and its BigQuery telemetry rows carry the same `trace_id`.

### 5. Golden Eval Dataset, `make eval-all`, and Coverage Gate *(Secs. 4, 9)*

**Reference requirement:** A golden dataset of 20+ query–variance–explanation mappings in BigQuery; Gemini 3.5 Flash LLM-as-a-Judge regression checks through `make eval-all` before deployment; >80% pytest coverage; ≥92% grounding recall; 100% numeric agreement and citation coverage; p95 TTFT ≤ 3.0 s; 100% safety compliance.

**Current state:**
- 109 pytest tests with mocked LLM and Search APIs.
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
