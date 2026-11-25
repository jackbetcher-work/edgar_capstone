# Terraform — SEC EDGAR Natural Language Analyst Infrastructure

Provisions the stack in project `iron-radio-514217-q7` (`us-central1`), with remote state in `gs://iron-radio-514217-q7-tfstate`:

- the GCS corpus bucket and the BigQuery data, vector, telemetry, and evaluation foundation;
- Vertex AI Search (Discovery Engine) unstructured datastore and search engine;
- Cloud Firestore (Native) for persistent multi-turn sessions and analyst feedback;
- Google Cloud Model Armor and Sensitive Data Protection (Cloud DLP) inspect/de-identify templates;
- a single custom least-privilege runtime service account (`sec-analyst-sa`), Cloud Run v2, and Direct Cloud Run IAP;
- Artifact Registry and keyless GitHub Actions Workload Identity Federation (`sec-edgar-cicd-sa`);
- Cloud Monitoring dashboard, low-cardinality log-based metrics, alert policies, and BigQuery SQL semantic views.

Requires Terraform `>= 1.7`.

## Provisioned Resources

- **Remote State Bucket** (`gs://iron-radio-514217-q7-tfstate`): Bootstrapped with object versioning, uniform bucket-level access, and enforced public access prevention.
- **SEC 10-K Corpus Bucket (`gs://fde-sec-edgar-sandbox-dev-sec-corpus`, `terraform/storage.tf`):** Managed regional (`us-central1`) GCS bucket (`google_storage_bucket.sec_corpus`) with uniform bucket-level access, enforced public access prevention, object versioning, 7-day soft delete, and noncurrent-version / multipart lifecycle cleanup.
- **BigQuery Dataset (`sec_edgar`, `us-central1`, `terraform/bigquery.tf`, `terraform/evals.tf`):**
  - `financial_metrics`: Native table partitioned by `fiscal_year` (`2015–2035`) and clustered by `gics_sector, ticker`, optionally seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_financials.jsonl`.
  - `filing_chunks`: Metadata-filtered RAG table partitioned by `fiscal_year` and clustered by `gics_sector, ticker, section_id` with `embedding` (`ARRAY<FLOAT64>`, 768-d) ready for `VECTOR_SEARCH`, optionally seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_filing_chunks.jsonl`.
  - `v_financial_variances_yoy`: Deterministic YoY variance SQL view built over native `financial_metrics` (`revenue`, `operating_income`, `net_income`, `research_and_development` deltas and percentages, `rd_intensity_pct`, plus `metric_provenance`).
  - `v_filing_chunk_coverage`: Per-filing (`profile_id`, `ticker`, `fiscal_year`) diagnostic view over native `financial_metrics` and `filing_chunks` reporting exact `item_1a_chunks`, `item_7_chunks`, `valid_embedding_chunks`, and `filing_chunk_status`.
  - `v_chunk_health_metrics`: Chunk health monitoring SQL view over native `financial_metrics` and `filing_chunks` computing ROLLUP health metrics by `(gics_sector, section_id)` including `embedding_coverage_ratio`, `filing_coverage_ratio`, token/character percentiles (`p50`, `p95`), anomaly counts (`missing_embedding`, `dimension_mismatch`, `undersized`, `oversized`, `missing_metadata`), and composite `health_score` (`0–100`) & `health_status` (`HEALTHY`, `DEGRADED`, `UNHEALTHY`).
  - `golden_eval_set` & `eval_results` (`terraform/evals.tf`): Golden dataset ground-truth table (`30` rows across 6 categories) and regression eval run history table storing all 18 gate metrics.
- **BigQuery Streaming Telemetry & Audit Sinks (`terraform/telemetry.tf`, `terraform/schemas/*.json`):**
  - `llm_telemetry_events`: Day-partitioned (`recorded_at`), clustered (`model_name, agent_name, task_category`) sink table capturing per-call cached/uncached input tokens, output/thoughts tokens, USD token cost, cached-token discount savings, prefill/decode/TTFT latency, and the request `trace_id`.
  - `session_telemetry_events`: Day-partitioned (`recorded_at`), clustered (`intent, execution_mode, user_id`) sink table capturing conversational session length, cumulative companies analyzed, report generation counts, ADK tool invocations, and turn latency/cost.
  - `security_audit_logs`: Day-partitioned (`recorded_at`), clustered (`event_type, status, user_email`) sink table capturing IAP / Google Workspace auth and 10-K filing access audit trails.
  - `v_telemetry_cost_and_latency`: SQL semantic view aggregating daily model/agent token volume, exact USD cost, cached-token savings, cache hit rate, p95 latency, and TTFT SLO compliance (source of truth for cumulative USD; query from BigQuery Studio).
  - `v_session_analytics_summary`: Daily × `intent` × `execution_mode` rollup over `session_telemetry_events`.
  - `v_security_audit_summary`: Daily × `event_type` × `status` × `identity_provider` rollup over `security_audit_logs` (total, `denied_401` / `denied_403`, blocked, distinct principals, IAP-verified, distinct traces).
  - `google_project_iam_custom_role.bigquery_telemetry_sink_writer` (`secEdgarTelemetrySinkAppend`): Least-privilege custom role (`bigquery.tables.get`, `bigquery.tables.updateData`) bound via `google_bigquery_table_iam_member` strictly on the 3 append-only sink tables (keeping `financial_metrics` and `filing_chunks` strictly read-only for `sec-analyst-sa`).
- **Google Cloud Firestore Persistent Feedback & Session Store (`terraform/telemetry.tf`):**
  - `google_firestore_database.analyst_feedback_db`: Regional `FIRESTORE_NATIVE` database (`(default)`) persisting both analyst thumbs-up/down ratings (`analyst_feedback` collection) and multi-turn conversational sessions (`analyst_sessions` collection, keyed by deterministic SHA-256 of `(user_id, session_id)`), with `roles/datastore.user` granted to `sec-analyst-sa`.
- **Model Armor & Cloud SDP Templates (`terraform/model_armor.tf`):**
  - `sec-edgar-analyst-armor` (`google_model_armor_template`), `sec-edgar-sdp-inspect` (`google_data_loss_prevention_inspect_template`), and `sec-edgar-sdp-deidentify` (`google_data_loss_prevention_deidentify_template`).
  - Model Armor executes the Cloud DLP inspect and de-identify templates server-side via `sdp_settings.advanced_config` in a single RPC (no separate application-side DLP call).
  - `sec-edgar-sdp-inspect` uses `min_likelihood = "LIKELY"` and exempts `.*@.*\.iam\.gserviceaccount\.com$` and `.*@(sec\.gov|.*\.googleapis\.com)$` so service-account emails stay intact while personal emails, SSNs, phones, and credentials are redacted.
- **Vertex AI Search / Discovery Engine (`terraform/vector_search.tf`):**
  - `sec-edgar-10k-corpus-ds` (`google_discovery_engine_data_store`) & `sec-edgar-10k-hybrid-search` (`google_discovery_engine_search_engine`): Unstructured 10-K search datastore and Enterprise LLM-enabled search app queried in parallel with BigQuery `VECTOR_SEARCH` over native `sec_edgar.filing_chunks`.
- **Gemini Models & APIs (`terraform/variables.tf`, `terraform/apis.tf`):**
  - `financial_reasoning_model` (default `gemini-3.8-flash`), `coding_and_evaluation_model` (`gemini-3.8-flash`, `ThinkingLevel.LOW`), `compaction_and_mapping_model` (`gemini-3.5-flash-lite`), `gemini_location` (`global`), and `async_judge_sample_rate` (`1.0`, validated in `[0.0, 1.0]`) become Cloud Run's `EDGAR_ANALYST_MODEL`, `EDGAR_FLASH_MODEL`, `EDGAR_LITE_MODEL`, `EDGAR_GEMINI_LOCATION`, and `EDGAR_ASYNC_JUDGE_SAMPLE_RATE`.
  - Gemini models are served from the `global` endpoint (`var.gemini_location`), while BigQuery, GCS, Model Armor, and embeddings run in `var.region`.
  - `serviceconsumermanagement.googleapis.com` is enabled because Vertex AI context caching needs it to provision its tenant project.
- **Observability (`terraform/observability.tf`):**
  - `google_logging_metric.api_request_latencies`: log-based distribution metric (`logging.googleapis.com/user/sec_edgar/api_request_latencies`) built from `http_request_completed` access logs, labeled by route, method, and status, with trace exemplars.
  - `google_logging_metric.telemetry_distributions` (`for_each`): `sec_edgar/llm_ttft_ms`, `llm_model_latency_ms`, `llm_cost_usd`, `llm_cache_savings_usd`, and `evaluation_scores`. LLM token counts and exact USD totals live in BigQuery (`llm_telemetry_events` / `v_telemetry_cost_and_latency`).
  - `google_logging_metric.telemetry_counters` (`for_each`, DELTA INT64): `sec_edgar/llm_calls`, `session_turns`, and `security_audit_events`.
  - Alert policies (all → `monitoring_notification_channels`): `api_latency_p95` (`30,000 ms`), `api_5xx_errors` (`5` per 5 min), and `security_audit_blocks` (`20` per 5 min).
  - Dashboard: `llm_observability_dashboard` (*SEC EDGAR Analyst — LLM Telemetry, Cost, Sessions & Security Audit*). Corpus chunk health is queried live via BigQuery views `v_chunk_health_metrics` and `v_filing_chunk_coverage`.
  - See [`docs/observability.md`](../docs/observability.md) for metric names, labels, and trace correlation.
- **Custom Application Service Account (`sec-analyst-sa`, `terraform/iam.tf`):**
  - Single custom runtime service account attached to `google_cloud_run_v2_service.sec_edgar_analyst` (default Compute Engine SAs are never used).
  - Granted least-privilege read-only BigQuery (`roles/bigquery.dataViewer` on `sec_edgar`, `roles/bigquery.jobUser`), append-only telemetry writes (`secEdgarTelemetrySinkAppend` on the 3 telemetry/audit tables), GCS (`roles/storage.objectViewer` on `sec_corpus`), Vertex AI (`roles/aiplatform.user`), Discovery Engine (`roles/discoveryengine.viewer`), Model Armor (`roles/modelarmor.user`), DLP (`roles/dlp.user`), Firestore (`roles/datastore.user`), and observability writers (`roles/logging.logWriter`, `roles/cloudtrace.agent`, `roles/telemetry.writer`).
- **Direct Cloud Run IAP (`terraform/iap_and_sso.tf`):**
  - `iap_enabled = true` directly on the Cloud Run v2 service (no external load balancer).
  - Only the Google-managed IAP service agent holds `roles/run.invoker`.
  - `google_iap_web_cloud_run_service_iam_member.cloud_run_authorized_accessors` grants `roles/iap.httpsResourceAccessor` to each `iap_authorized_principals` entry (`group:` or `domain:`; individual `user:` bindings are rejected by variable validation).
  - Expected IAP JWT audience (`/projects/<number>/locations/<region>/services/<name>`) is passed to Cloud Run via `INTERNAL_IAP_AUDIENCE`.
- **Artifact Registry, Keyless GitHub Actions WIF & Cloud Run (`terraform/cicd_and_secrets.tf`, `terraform/ci_live_test_access.tf`):**
  - `google_artifact_registry_repository.sec_edgar_repo`: Docker container repository (`us-central1-docker.pkg.dev/iron-radio-514217-q7/sec-edgar-repo`).
  - `google_iam_workload_identity_pool.github_pool` & `github_provider`: Keyless OIDC authentication for GitHub Actions (`sec-edgar-cicd-sa`, holding `roles/run.developer`, repo-scoped Artifact Registry writer, `roles/iam.serviceAccountUser` on `sec-analyst-sa`, and live-test read access + append-only `eval_results` access).
  - `google_cloud_run_v2_service.sec_edgar_analyst`: Managed by Terraform (`lifecycle.ignore_changes` preserves CI/CD image updates from `.github/workflows/deploy.yml`).

## Usage

```bash
# 1. Bootstrap the versioned GCS remote state bucket (idempotent)
../.venv/bin/python ../scripts/bootstrap_tfstate_bucket.py

# 2. Initialize, validate, plan, and apply the Terraform stack
# (If gs://fde-sec-edgar-sandbox-dev-sec-corpus already exists in GCP, import it once first:
#  terraform -chdir=terraform import 'google_storage_bucket.sec_corpus[0]' fde-sec-edgar-sandbox-dev-sec-corpus)
cd terraform
terraform init
terraform validate
terraform plan
terraform apply

# 3. Populate curated SEC 10-K filings & vector DB manifests into the Terraform-managed bucket and BigQuery
# (Requires the optional `[ingest]` extra for edgartools/pandas: `pip install -e ".[ingest]"` or `".[dev]"`)
../.venv/bin/python ../scripts/build_sec_dataset.py --sp500 --no-create-bucket
../.venv/bin/python ../scripts/build_vector_db.py
# (Already-built artifacts: publish without re-embedding)
../.venv/bin/python ../scripts/build_vector_db.py --upload-only

# 4. Backfill R&D metrics, load native financial_metrics, and import Vertex AI Search
cd .. && make backfill-rd publish-financials index-vertex-search
```

## Example Metadata-Filtered RAG Query (`VECTOR_SEARCH`)

Once chunks and embeddings are populated into `sec_edgar.filing_chunks`, pre-filtered vector search runs natively in SQL:

```sql
SELECT
  base.ticker,
  base.company_name,
  base.fiscal_year,
  base.section_id,
  base.section_title,
  base.chunk_text,
  base.sec_filing_url,
  distance
FROM VECTOR_SEARCH(
  (
    SELECT *
    FROM `iron-radio-514217-q7.sec_edgar.filing_chunks`
    WHERE gics_sector = 'Information Technology'
      AND fiscal_year BETWEEN 2024 AND 2026
      AND section_id = 'item_1a'
  ),
  'embedding',
  (SELECT @query_embedding AS embedding),
  top_k => 10,
  distance_type => 'COSINE'
);
```

## Example Chunk Health Monitoring Query (`v_chunk_health_metrics`)

```sql
SELECT
  gics_sector,
  section_id,
  total_chunks,
  distinct_profiles,
  expected_profiles,
  filing_coverage_ratio,
  embedding_coverage_ratio,
  avg_token_estimate,
  p95_token_estimate,
  undersized_chunk_count,
  oversized_chunk_count,
  missing_metadata_count,
  health_score,
  health_status
FROM `iron-radio-514217-q7.sec_edgar.v_chunk_health_metrics`
ORDER BY
  IF(gics_sector = 'ALL_SECTORS', 0, 1),
  gics_sector,
  section_id;
```
