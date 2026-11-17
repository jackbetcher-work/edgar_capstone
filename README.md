# SEC EDGAR Natural Language Analyst

Conversational financial analysis workspace for querying SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A) alongside deterministic period-over-period financial metrics (Revenue, Operating Income, Net Income, and R&D expense).

This document covers how to run, test, and operate the app, and lists the implementation gaps that remain against [`docs/project_reference.md`](docs/project_reference.md).

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
│   │   ├── financial_analyst.py     # FinancialAnalystAgent (BuiltInPlanner w/ bounded thinking_budget=512, output_schema=AnalystReport, retrieval budget)
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
├── tests/                           # Live GCP + live Gemini pytest suite (`>=80%` enforced via pytest-cov; 87.40% measured across 219 tests)
├── Dockerfile                       # Multi-stage non-root build (Node 22 UI + Python 3.11 runtime); no corpus data & no unused google-cloud-aiplatform dependency
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

## Implementation Gaps vs. `docs/project_reference.md`

Outstanding work needed to meet the reference specification. Section numbers (§) and objective IDs (B1–B3, T1–T3) refer to [`docs/project_reference.md`](docs/project_reference.md).

### Gap Summary

| # | Area | Spec | Priority | Status | Next Action |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | Sector-wide thematic scaling (map-reduce) | §7 · T2 | P0 | Not started | Fan out retrieval/summarization per company-year, then aggregate. |
| 2 | Citations for numeric claims | §8 · T3 | P0 | Not started | Attach `accession_number` + XBRL concept citations to metrics and variances. |
| 3 | Uncited "Notes" section in analyst reports | §8 · T3 | P0 | Gap | Remove the notes section or require citations for every note. |
| 4 | Clarification, out-of-corpus, and missing-year handling | §7, §8 · B1 | P1 | Not started | Add a clarification path and explicit coverage statements. |
| 5 | Fiscal-period display in comparisons | §8 | P1 | Gap | Show `period_of_report` in variance tables and peer comparisons. |
| 6 | Persistent session memory | §7 · B1 | P1 | Gap | Replace `InMemorySessionService` with a durable session store. |
| 7 | Golden dataset coverage | §10 | P1 | Partial | Add peer, thematic, anomaly, and out-of-corpus items; analyst review. |
| 8 | Missing evaluation gates | §5, §10 | P1 | Not started | Add citation precision, retrieval precision@10, filter correctness, and wording-delta gates. |
| 9 | Coverage and eval gates in CI | §9, §10 | P1 | Not started | Add `pytest-cov` (>80%) and `make eval-all` as deploy gates. |
| 10 | Cloud Build triggers | §9 | P2 | Not applied | Apply with `enable_cloudbuild_trigger = true`. |
| 11 | Vector Search index deployment | §8 · T1 | P2 | Not applied | Run a full `terraform apply` to deploy the index to its PSC endpoint. |
| 12 | VPC Service Controls enforcement | §9 | P2 | Blocked | Needs an org-delegated Access Context Manager policy. |
| 13 | Live observability verification | §5, §11 | P2 | Not verified | Confirm one end-to-end `/api/v1/chat` trace in Cloud Trace. |
| 14 | TLS 1.3 in transit | §9 | P2 | Gap | Raise the minimum TLS version or relax the spec. |
| 15 | Chat-turn latency on live retrieval | §5 · B1 | P1 | Gap | Cut the ~105 s turn and 4.3 s TTFT toward the 3.0 s target (parallel retrieval, fewer embedding calls, judge off the critical path). |
| 16 | CI access to live GCP | §9 | P1 | Not applied | Apply `terraform/ci_live_test_access.tf` so the CI service accounts can run the live test suite. |
| 17 | Model Armor template redacts service-account emails | §9 | P2 | Gap | Exempt `*.iam.gserviceaccount.com` in the Model Armor SDP template. |

---

### 1. Sector-Wide Thematic Scaling *(§7 Architecture Principles · T2)*

**Requirement:** Sector-wide queries (e.g., "all tech companies in the S&P 500") use a fan-out / map-reduce pattern: summarize per company and year, then aggregate into a sector narrative. Progress streams within the 3.0 s TTFT target.

**Gap:** A sector query retrieves about 16 chunks for the whole sector in `_retrieve_balanced_citations`: `top_k` of 10, plus 3 per boundary year. The live agent can make at most 6 more retrieval calls (`EDGAR_ANALYST_RETRIEVAL_BUDGET`). The Disclosure Delta pipeline (`compare_disclosures_across_years`) only compares the 5 companies most relevant to the theme (`EDGAR_DISCLOSURE_DELTA_MAX_COMPANIES`). The answer therefore rests on a handful of issuers, not the 76 Information Technology constituents.

**Next actions:**
1. When the scope is a GICS sector, enumerate the in-scope tickers and fiscal years, then run per-company-year retrieval and disclosure deltas concurrently with bounded parallelism.
2. Aggregate the per-company summaries into sector-level trends, keeping the citations underneath each one.
3. Stream progress events, such as "12/70 companies analyzed", so the TTFT target holds.

**Done when:** A sector-wide thematic query covers every in-scope issuer that has filings and reports how many it analyzed. Each aggregated claim is traceable to company-level citations.

### 2. Citations for Numeric Claims *(§8 Citation Model · T3)*

**Requirement:** Reported figures cite `accession_number` plus the XBRL concept (`metric_provenance`). Calculated values cite the reported inputs they were derived from.

**Gap:** Key metrics and variance tables come from the deterministic engine but carry no citation. `accession_number` and `metric_provenance` exist in BigQuery `financial_metrics`. Material variance flags already carry them (`source.accession_number`, `source.xbrl_concept`), but key metrics, variance tables, and narrative numbers do not.

**Next actions:**
1. Return `accession_number`, `metric_provenance`, and `sec_filing_url` with each metric record and variance input.
2. Render a citation for each number in `key_metrics` and `VarianceTableCard`, and open the filing or XBRL fact from the source pane.
3. Extend the citation-coverage scorer to count numeric claims.

**Done when:** Every number in a response links to its source filing and XBRL concept, and citation coverage stays at 100% with numeric claims included.

### 3. Uncited "Notes" Section *(§8 Citation Model · T3)*

**Requirement:** 100% of narrative claims are cited. A response with an uncited claim fails verification and is regenerated, or the claim is removed.

**Gap:** `structured_report.py` renders a "Notes (not source-verified)" section that contains free-text claims with no citations.

**Next actions:** Drop the notes field from `AnalystReport`, or require `chunk_ids` on each note and verify them the same way as findings.

**Done when:** No rendered answer contains uncited narrative text.

### 4. Clarification, Out-of-Corpus, and Missing-Year Handling *(§7 Conversational Behavior, §8 Coverage · B1)*

**Requirement:**
- Ambiguous scope triggers a clarifying question.
- Unsupported metrics, filing types, non-S&P 500 issuers, or years without filings get an explicit coverage statement.
- When a requested year is missing for an issuer, the response says so.

**Gap:**
- Scope resolution and routing (`classify_query_intent`) are regex-based and always produce an answer. No clarification path exists.
- Out-of-corpus requests and missing years get no dedicated handling. This matters now: FY2026 10-Ks exist for only 84 of about 500 issuers, because most calendar-year filers file in Q1 2027.

**Next actions:**
1. Add a scope-validation step that returns a clarification turn when tickers, period, or metric can't be resolved.
2. Check requested tickers, years, and metrics against corpus coverage, and prepend a coverage statement for anything missing.

**Done when:** The clarification and out-of-corpus golden items (gap 7) pass, and every response that has missing coverage states it explicitly.

### 5. Fiscal-Period Display *(§8 Fiscal Year Convention)*

**Requirement:** Year ranges are fiscal years, and peer comparisons show period-end dates so misaligned fiscal calendars are visible (e.g., NVIDIA FY ends in January, Apple's in September).

**Gap:** `period_of_report` appears only in the source viewer's section header. Variance tables and peer-comparison narratives don't show it.

**Next actions:** Include `period_of_report` in the variance payload and add it to `VarianceTableCard` column headers and peer-comparison text.

**Done when:** Every company-year figure shown to the analyst carries its fiscal period-end date.

### 6. Persistent Session Memory *(§7 Conversational Behavior, Architecture Principles · B1)*

**Requirement:** Session scope persists across turns so follow-ups ("now add Microsoft") refine the prior query, while Cloud Run scales across instances.

**Gap:** Sessions use ADK `InMemorySessionService` / `InMemoryMemoryService` (`context_memory.py`). Follow-up context is lost when a request lands on a different Cloud Run instance or an instance restarts.

**Next actions:** Move to a durable session store, such as Vertex AI Agent Engine sessions or Firestore, keyed by authenticated user and session ID.

**Done when:** A multi-turn conversation keeps its scope across instance restarts and with `min-instances > 1`.

### 7. Golden Dataset Coverage *(§10)*

**Requirement:** Minimum golden items per category:

| Category | Required | Current | Shortfall |
| :--- | :--- | :--- | :--- |
| Single-company longitudinal | ≥ 8 | 9 | — |
| Multi-company peer comparison (incl. R&D) | ≥ 8 | 7 | 1 |
| Thematic trend tracking with annotated wording deltas | ≥ 8 | 7 (no delta annotations) | 1 + annotations |
| Anomaly explanation | ≥ 4 | 0 | 4 |
| Negative / out-of-corpus and clarification | ≥ 4 | 0 (the 8 `safety_red_team` items test Model Armor only) | 4 |

**Next actions:**
1. Add the missing items and annotate thematic items with expected added, removed, and modified passages. Add anomaly items whose expected flags (`flag_id`) and explaining MD&A sections can be scored against `material_variance_flags`.
2. Have an analyst review all seed queries, `expected_sections`, and reference explanations, then run `make load-golden-dataset`.
3. Re-check retrieval of start-year filings on live GCP: in the last run against the old local corpus, `msft_longitudinal_fy24_fy26` didn't retrieve or cite the MSFT FY2024 filing.

**Done when:** Every category meets its minimum and the dataset is analyst-approved and loaded into BigQuery.

### 8. Missing Evaluation Gates *(§5, §10)*

**Requirement:** These success criteria need gates in [`evals/thresholds.yaml`](evals/thresholds.yaml):
- citation precision ≥ 95%
- retrieval precision@10 ≥ 85%, with 100% metadata-filter correctness
- wording-delta accuracy ≥ 90%
- zero unsupported claims

**Gap:** None of these is scored. The current gates cover numeric agreement, citation coverage, grounding recall, routing, judge scores, safety, and p95 TTFT. The judge thresholds are uncalibrated assumptions, and no live p95 TTFT baseline exists.

**Next actions:**
1. Add scorers and gates for citation precision (judge plus spot audit), retrieval precision@10, filter correctness, wording-delta accuracy, and unsupported-claim count. For wording-delta accuracy, compare each response's `disclosure_deltas` classifications (added / removed / modified per year pair) with the annotated golden passages from gap 7.
2. Run `make eval-all` (now always live) to calibrate the judge thresholds and record a TTFT baseline.

**Done when:** Every §5 criterion that can be measured has a gate, and a live baseline run is recorded in `sec_edgar.eval_results`.

### 9. Coverage and Eval Gates in CI *(§9, §10)*

**Requirement:** >80% pytest coverage, with golden-dataset regression checks gating deployment.

**Gap:** `pytest-cov` isn't installed, so coverage is unmeasured. `make eval-all` is not wired into `cloudbuild.yaml` or `.github/workflows/deploy.yml`.

**Next actions:**
1. Add `pytest-cov` to the `dev` extras and run `pytest --cov=edgar_capstone --cov-fail-under=80` in `make test`, Cloud Build, and GitHub Actions.
2. Add a `python scripts/run_evals.py --write-bigquery` step before the deploy stage.

**Done when:** CI fails if coverage drops below 80% or any eval gate regresses, and every deploy has a matching `eval_results` row.

### 10. Cloud Build Triggers *(§9 CI/CD)*

**Requirement:** Cloud Build deploys automatically on successful merges.

**Gap:** `terraform/cloudbuild.tf` defines the GitHub connection and the merge/PR triggers, but `enable_cloudbuild_trigger` defaults to `false`, and no triggers exist in the project.

**Next actions:** Authorize the 2nd-gen GitHub connection, then run `terraform -chdir=terraform apply -var=enable_cloudbuild_trigger=true`.

**Done when:** A merge to `main` runs `cloudbuild.yaml` in deploy mode and a pull request runs it in verify mode.

### 11. Vector Search Index Deployment *(§8 Retrieval Strategy · T1)*

**Requirement:** Vector search over `filing_chunks` embeddings is part of hybrid retrieval.

**Gap:** At the last apply, the index was not deployed to its endpoint (`deploy_vertex_vector_index` was held at `false`). Vector retrieval therefore falls back to BigQuery `VECTOR_SEARCH` / Vertex AI Search.

**Next actions:** Run a full `terraform -chdir=terraform apply`. It deploys the index and moves the endpoint to Private Service Connect, which replaces the endpoint and takes 20–40 minutes. Then redeploy so Cloud Run gets `VERTEX_VECTOR_SEARCH_PSC_ADDRESS`.

**Done when:** Live hybrid search returns neighbors from the deployed Vector Search index over PSC.

### 12. VPC Service Controls Enforcement *(§9 Data Protection)*

**Requirement:** A strict service perimeter locks down Vertex AI, Cloud Run, GCS, and BigQuery.

**Gap:** The perimeter is defined in `vpc_service_controls.tf` but not applied: `enable_vpc_service_controls = false`, and it deploys as dry-run unless `vpc_sc_enforce = true`. Applying it requires an organization-owned Access Context Manager policy, and the sandbox account has no org-level rights.

**Next actions:** Get a delegated scoped policy, run `terraform -chdir=terraform apply -var=enable_vpc_service_controls=true -var=vpc_sc_access_policy_id=<POLICY_NUMBER>`, review the dry-run violations (`protoPayload.metadata.dryRun=true`), then set `vpc_sc_enforce = true`.

**Done when:** The perimeter is enforced with no unexpected violations.

### 13. Live Observability Verification *(§5 Observability, §11)*

**Requirement:** End-to-end tracing through OpenTelemetry and Cloud Trace, structured logs correlated by trace ID, and BigQuery telemetry joinable to traces.

**Gap:** This has not been confirmed against live Cloud Trace. It isn't confirmed either that the latest `terraform apply` has created the `sec_edgar/api_request_latencies` log-based metric, the nullable `trace_id` columns, or the `golden_eval_set` / `eval_results` tables. No p95 latency alert policy exists.

**Next actions:**
1. Run `terraform -chdir=terraform apply` and deploy.
2. Send one `/api/v1/chat` request and confirm it in Cloud Trace and Logs Explorer (`trace="projects/iron-radio-514217-q7/traces/<X-Trace-Id>"`).
3. Add a p95 latency alert policy on the log-based metric once notification channels exist.

**Done when:** One `/api/v1/chat` request shows a single trace with nested FastAPI, ReAct, tool, and model spans; its logs are grouped under that trace; and its BigQuery telemetry rows carry the same `trace_id`.

### 14. TLS 1.3 in Transit *(§9 Data Protection)*

**Requirement:** 100% of data is encrypted in transit using TLS 1.3.

**Gap:** The load-balancer SSL policy is `RESTRICTED`, which allows TLS 1.2 and above. The default `iap_direct` mode uses Google front-end TLS defaults, which also accept TLS 1.2.

**Next actions:** Set the SSL policy's `min_tls_version` to `TLS_1_3` in `load_balancer.tf` and use `load_balancer` mode where TLS 1.3 is mandatory. Otherwise, amend §9 to "TLS 1.2+".

**Done when:** The spec and the deployed TLS policy agree.

### 15. Chat-Turn Latency on Live Retrieval *(§5 Latency · B1)*

**Requirement:** p95 TTFT ≤ 3.0 s for single- and peer-company queries.

**Gap:** With every read going to live GCP, one peer-comparison turn ("Compare NVIDIA and Apple's R&D spend from 2024-2026") took about 105 s end to end, with a 4.3 s TTFT. Measured stages:
- hybrid search: about 10 s
- anomaly flags with MD&A explanations: about 16 s
- cached 10-K passage ranking: about 18 s (embeds every paragraph of each section; now cached per section after the first query)
- live analyst synthesis plus the live Flash judge: about 51 s

**Next actions:**
1. Run the independent retrieval steps (hybrid search, variance flags, section ranking) concurrently.
2. Use the `filing_chunks` embeddings already stored in BigQuery for section ranking instead of re-embedding paragraphs.
3. Move the LLM-as-a-Judge call off the request path (score asynchronously and write to telemetry).
4. Deploy the Vector Search index (gap 11) so hybrid search stops waiting on fallbacks.

**Done when:** `make eval-all` reports p95 TTFT ≤ 3,000 ms and a typical turn completes in well under 30 s.

### 16. CI Access to Live GCP *(§9 CI/CD)*

**Requirement:** The test suite runs against live GCP data and live Gemini in Cloud Build and GitHub Actions.

**Gap:** `terraform/ci_live_test_access.tf` grants the CI service accounts (`sec-edgar-cicd-sa`, `sec-edgar-cloudbuild-sa`) read access to BigQuery `sec_edgar`, the corpus bucket, Discovery Engine, Vertex AI, Model Armor, and DLP, but it hasn't been applied. The GitHub Actions build-job smoke test also skips the live data check, and the Cloud Build in-image live smoke test relies on the build network's metadata server; neither has been run yet.

**Next actions:** Run `terraform -chdir=terraform apply`, then trigger one PR build and confirm the `test` step and the smoke test pass.

**Done when:** A PR build runs the full live test suite green.

### 17. Model Armor Template Redacts Service-Account Emails *(§9 Data Protection)*

**Requirement:** PII is redacted from prompts and responses; service-account identifiers used in operational answers stay intact.

**Gap:** The live Model Armor template's basic SDP treats `*.iam.gserviceaccount.com` addresses as `EMAIL_ADDRESS` and redacts them.

**Next actions:** Add an exclusion for service-account emails to the SDP inspect configuration used by the `sec-edgar-analyst-armor` template (`terraform/model_armor.tf`).

**Done when:** A prompt containing a service-account email keeps it, while personal emails are still redacted.

