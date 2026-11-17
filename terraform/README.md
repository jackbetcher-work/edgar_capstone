# Terraform — SEC EDGAR Natural Language Analyst Infrastructure

Provisions the whole stack in project `iron-radio-514217-q7` (`us-central1`), with remote state in `gs://iron-radio-514217-q7-tfstate`:

- the GCS corpus bucket and the BigQuery data, vector, and telemetry foundation;
- Vertex AI Vector Search and Vertex AI Search;
- Firestore feedback, Model Armor, and Sensitive Data Protection;
- IAM and per-agent identities, Secret Manager, Cloud Run, and IAP (directly on the service, or behind an HTTPS load balancer);
- an optional VPC Service Controls perimeter;
- Cloud Build triggers;
- Cloud Monitoring dashboards and the request-latency metric.

Requires Terraform `>= 1.7` (`for_each` in `import` blocks).

## Provisioned Resources

- **Remote State Bucket** (`gs://iron-radio-514217-q7-tfstate`): Bootstrapped with object versioning, uniform bucket-level access, and enforced public access prevention.
- **SEC 10-K Corpus Bucket (`gs://fde-sec-edgar-sandbox-dev-sec-corpus`, `terraform/storage.tf`):** Managed regional (`us-central1`) GCS bucket (`google_storage_bucket.sec_corpus`) with uniform bucket-level access, enforced public access prevention, object versioning, 7-day soft delete, and noncurrent-version / multipart lifecycle cleanup.
- **BigQuery Dataset (`sec_edgar`, `us-central1`):**
  - `financial_metrics`: Native table partitioned by `fiscal_year` (`2015–2035`) and clustered by `gics_sector, ticker`, automatically seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_financials.jsonl`.
  - `financial_metrics_ext`: Live GCS external table over `manifests/bigquery_financials.jsonl`.
  - `filing_chunks`: Metadata-filtered RAG table partitioned by `fiscal_year` and clustered by `gics_sector, ticker, section_id` with `embedding` (`ARRAY<FLOAT64>`, 768-d) ready for `VECTOR_SEARCH`, automatically seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_filing_chunks.jsonl`.
  - `filing_chunks_ext`: Live GCS external table over `manifests/bigquery_filing_chunks.jsonl`.
  - `unstructured_10k_objects`: BigQuery Object Table indexing `gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/*`.
  - `v_financial_variances_yoy`: Deterministic YoY variance SQL view (`revenue`, `operating_income`, `net_income`, `research_and_development` deltas and percentages, plus `rd_intensity_pct`).
  - `v_filing_chunk_coverage`: Per-filing (`profile_id`, `ticker`, `fiscal_year`) diagnostic view reporting exact `item_1a_chunks`, `item_7_chunks`, `valid_embedding_chunks`, and `filing_chunk_status`.
  - `v_chunk_health_metrics`: Chunk health monitoring SQL view computing ROLLUP health metrics by `(gics_sector, section_id)` including `embedding_coverage_ratio`, `filing_coverage_ratio`, token/character percentiles (`p50`, `p95`), anomaly counts (`missing_embedding`, `dimension_mismatch`, `undersized`, `oversized`, `missing_metadata`), and composite `health_score` (`0–100`) & `health_status` (`HEALTHY`, `DEGRADED`, `UNHEALTHY`).
- **BigQuery Streaming Telemetry & Audit Sinks + Cloud Logging Sink (`terraform/telemetry.tf`, `terraform/schemas/*.json`):**
  - `llm_telemetry_events`: Day-partitioned (`recorded_at`), clustered (`model_name, agent_name, task_category`) sink table capturing per-call cached/uncached input tokens, output/thoughts tokens, USD token cost, cached-token discount savings, prefill/decode/TTFT latency, and the request `trace_id`.
  - `session_telemetry_events`: Day-partitioned (`recorded_at`), clustered (`intent, execution_mode, user_id`) sink table capturing conversational session length, cumulative companies analyzed, report generation counts, ADK tool invocations, and turn latency/cost.
  - `security_audit_logs`: Day-partitioned (`recorded_at`), clustered (`event_type, status, user_email`) sink table capturing IAP/Workspace/Okta SSO authentication and 10-K filing access audit trails.
  - `v_telemetry_cost_and_latency`: Analytical SQL view aggregating daily model/agent token volume, USD cost, cached-token savings, cache hit rate, p95 latency, and TTFT SLO compliance for Looker BI.
  - `google_project_iam_custom_role.bigquery_telemetry_sink_writer` (`secEdgarTelemetrySinkAppend`): Least-privilege custom role (`bigquery.tables.get`, `bigquery.tables.updateData`) bound via `google_bigquery_table_iam_member` strictly on the 3 append-only sink tables (keeping dataset-wide `financial_metrics` and `filing_chunks` read-only for `sec-analyst-sa`).
  - `google_logging_project_sink.telemetry_and_audit_to_bigquery`: Project-level Cloud Logging sink exporting structured `sec-edgar-telemetry-sink` JSON logs to BigQuery partitioned tables.
- **Google Cloud Firestore Persistent Feedback Store (`terraform/telemetry.tf`):**
  - `google_firestore_database.analyst_feedback_db`: Regional `FIRESTORE_NATIVE` database (`(default)`) persisting analyst thumbs-up/down ratings and audit comments (`analyst_feedback` collection), with `roles/datastore.user` granted to `sec-analyst-sa`.
- **Vertex AI Vector Search (`terraform/vector_search.tf`):**
  - `sec-edgar-10k-chunks-index` (`google_vertex_ai_index`): 768-dimensional `COSINE_DISTANCE` Tree-AH ANN index backed by `gs://fde-sec-edgar-sandbox-dev-sec-corpus/vector_search/embeddings` with string `restricts` (`ticker`, `cik`, `gics_sector`, `gics_sub_industry`, `section_id`, `form_type`, `profile_id`), `numeric_restricts` (`fiscal_year`), and `crowding_tag` (`ticker`).
  - `sec-edgar-10k-chunks-endpoint` (`google_vertex_ai_index_endpoint`):
    - With `vector_search_endpoint_access = "private_service_connect"` (the default), it's a PSC-only endpoint with no public domain, accepting connections from this project only (`project_allowlist`).
    - With `vector_search_endpoint_access = "public"`, it's the public REST endpoint instead.
    - Changing an existing endpoint between the two modes replaces it and redeploys the index.
  - `seed_vertex_vector_index` (default `true`) points the index at `vector_search/embeddings` in the bucket, where the full-index embeddings (342,945 vectors) are published.
- **Vertex AI Search / Discovery Engine (`terraform/vector_search.tf`):**
  - `sec-edgar-10k-corpus-ds` (`google_discovery_engine_data_store`) & `sec-edgar-10k-hybrid-search` (`google_discovery_engine_search_engine`): Unstructured 10-K search datastore and Enterprise LLM-enabled search app.
- **Google Cloud Monitoring Chunk Health View (`terraform/monitoring.tf`):**
  - Custom Metric Descriptors under `custom.googleapis.com/sec_edgar/chunk_health/*`:
    - `health_score` (0–100 composite chunk health gauge)
    - `embedding_coverage_ratio` (valid 768-d embedding coverage ratio)
    - `filing_coverage_ratio` (dual-section `item_1a` + `item_7` filing coverage ratio)
    - `total_chunks` (chunk counts labeled by `gics_sector` and `section_id`)
    - `token_count_mean` (mean token count per chunk labeled by `section_id`)
    - `anomaly_chunk_count` (anomaly counts labeled by `anomaly_type`)
  - `chunk_health_dashboard` (`google_monitoring_dashboard`): 12-column Cloud Monitoring dashboard with KPI scorecards, GICS sector/section chunk distribution charts, anomaly breakdown bars, token length trends, BigQuery query execution telemetry, and a structured logs panel (`jsonPayload.event_type="sec_edgar_chunk_health_evaluation"`).
- **BigQuery Cloud Resource Connection (`vertex-ai-rag-conn`):** Granted `roles/aiplatform.user` and `roles/storage.objectViewer` on the SEC corpus bucket.
- **Gemini models & APIs (`terraform/variables.tf`, `terraform/apis.tf`):**
  - `financial_reasoning_model` (default `gemini-3.8-flash`), `coding_and_evaluation_model` (`gemini-3.5-flash`), and `gemini_location` (`global`) become Cloud Run's `EDGAR_ANALYST_MODEL`, `EDGAR_FLASH_MODEL`, and `EDGAR_GEMINI_LOCATION`. The Cloud Build triggers pass the same values.
  - The project can't reach `gemini-2.5-pro`, and both Flash models are served only from the `global` endpoint. BigQuery, GCS, Model Armor, and embeddings stay in `var.region`.
  - `serviceconsumermanagement.googleapis.com` is enabled because Vertex AI context caching needs it to provision its tenant project.
- **Observability (`terraform/observability.tf`):**
  - `google_logging_metric.api_request_latencies` is a log-based distribution metric (`logging.googleapis.com/user/sec_edgar/api_request_latencies`). It is built from the app's `http_request_completed` access logs, labeled by route, method, and status, with trace exemplars.
  - The three telemetry tables carry a nullable `trace_id` column. See [`docs/observability.md`](../docs/observability.md).
- **Role-Based Access Control via Cloud Identity Groups (`terraform/iam.tf`, `terraform/iap_and_sso.tf`):**
  - `google_cloud_identity_group.rbac_groups`: Optional provisioning of Google Cloud Identity security groups (`sec-edgar-admins`, `sec-edgar-analysts`, `sec-edgar-viewers`), plus variable validation strictly prohibiting individual `user:` bindings.
  - Group-scoped IAM bindings (`group:<group-id>@<domain>`) across IAP (`roles/iap.httpsResourceAccessor`), BigQuery (`roles/bigquery.dataViewer` / `roles/bigquery.metadataViewer`), Discovery Engine (`roles/discoveryengine.viewer`), and Cloud Run/Logging/Monitoring observability for Admins.
- **App access through IAP (`terraform/iap_and_sso.tf`, `terraform/cicd_and_secrets.tf`, `terraform/load_balancer.tf`):**
  - `cloud_run_access_mode = "iap_direct"` (default): `iap_enabled = true` on the Cloud Run service (google-beta provider). Users open the `app_url` output (`run.app`) and sign in with Google. `google_iap_web_cloud_run_service_iam_member` grants `roles/iap.httpsResourceAccessor` to each `iap_authorized_principals` entry (`group:` or `domain:`). No load balancer is created.
  - `cloud_run_access_mode = "load_balancer"`: ingress `internal-and-cloud-load-balancing` behind the external HTTPS LB, with IAP on the serverless-NEG backend service and a `RESTRICTED` / TLS 1.2+ SSL policy.
  - In both modes only the IAP service agent has `roles/run.invoker`. The IAP JWT audience for the active mode is written to the `sec-edgar-iap-audience` secret.
- **Private networking for Vector Search (`terraform/network.tf`, PSC mode only):**
  - `google_compute_network.sec_edgar` (`sec-edgar-analyst-vpc`, custom mode) and `google_compute_subnetwork.sec_edgar_private` (`private_subnet_cidr`, default `10.20.0.0/24`, Private Google Access on).
  - `google_compute_address.vector_search_psc` (an internal IP) and `google_compute_forwarding_rule.vector_search_psc`. The forwarding rule is a PSC consumer endpoint (`load_balancing_scheme = ""`) targeting the deployed index's `private_endpoints[0].service_attachment`. It's only created once the index is deployed (`deploy_vertex_vector_index`).
  - The Cloud Run service gets Direct VPC egress into the private subnet (`egress = "PRIVATE_RANGES_ONLY"`) plus `VERTEX_VECTOR_SEARCH_PSC_ADDRESS`. The app queries `<ip>:10000` over gRPC; the IP is also exported as the `vector_search_psc_address` output.
- **VPC Service Controls (`terraform/vpc_service_controls.tf`):** off by default (`enable_vpc_service_controls`). When on, it creates a regular perimeter around the project that restricts the data and model APIs (`vpc_sc_restricted_services`). Ingress is allowed by identity for operators, CI/CD, runtime/agent SAs, and the log sink writer, and from any identity for Cloud Run invocations (IAP gates those). It's dry-run until `vpc_sc_enforce = true`. It needs an Access Context Manager policy: `vpc_sc_access_policy_id` (scoped policy delegated by an org admin) or `create_vpc_sc_scoped_access_policy = true` (needs org-level permissions).
- **Custom Application Service Account (`sec-analyst-sa@iron-radio-514217-q7.iam.gserviceaccount.com` / logical project alias `sec-analyst-sa@fde-sec-edgar-sandbox-dev.iam.gserviceaccount.com`):**
  - Enforced via Terraform variable validation and attached directly to `google_cloud_run_v2_service.sec_edgar_analyst` (default Compute Engine SAs are never used).
  - Granted least-privilege read-only BigQuery (`roles/bigquery.dataViewer` + `roles/bigquery.jobUser` + `roles/bigquery.connectionUser`), GCS (`roles/storage.objectViewer`), Vertex AI (`roles/aiplatform.user`), Discovery Engine (`roles/discoveryengine.viewer`), Secret Manager (`roles/secretmanager.secretAccessor`), Cloud Monitoring (`roles/monitoring.metricWriter`), Cloud Trace (`roles/cloudtrace.agent`), and Cloud Logging (`roles/logging.logWriter`).
- **Granular Per-Agent Identities (`terraform/iam.tf`):**
  - `root-orchestrator-agent-sa`: `RootOrchestrator` identity (`roles/aiplatform.user`, `roles/run.invoker`, telemetry writers; zero BigQuery/GCS/Secret Manager access).
  - `financial-analyst-agent-sa`: `FinancialAnalystAgent` identity (dataset-scoped `roles/bigquery.dataViewer`, `roles/bigquery.jobUser`, `roles/discoveryengine.viewer`, `roles/aiplatform.user`; zero raw GCS corpus or Secret Manager access).
  - `context-cache-agent-sa`: `ContextCacheManager` / `ContextCacheAgent` identity (bucket-scoped `roles/storage.objectViewer` on `sec_corpus_bucket`, `roles/aiplatform.user`; zero BigQuery, Discovery Engine, or Secret Manager access).
  - `model-armor-guardrail-sa`: `ModelArmorGuardrail` identity (`roles/aiplatform.user` and `roles/logging.logWriter` only; zero data plane access).
  - Directed S2S `roles/iam.serviceAccountTokenCreator` edges (`sec-analyst-sa` -> agents, `RootOrchestrator` -> sub-agents, `FinancialAnalystAgent` -> `ContextCacheAgent` / `ModelArmorGuardrail`).
- **Google Cloud Secret Manager & Keyless CI/CD Pipeline (`terraform/cicd_and_secrets.tf`):**
  - `google_secret_manager_secret.managed_secrets`: Provisions `sec-edgar-user-agent`, `sec-edgar-gemini-api-key`, and `sec-edgar-iap-audience` with automatic replication and zero plaintext secret values in IaC.
  - `google_artifact_registry_repository.sec_edgar_repo`: Docker container repository (`us-central1-docker.pkg.dev/iron-radio-514217-q7/sec-edgar-repo`).
  - `google_iam_workload_identity_pool.github_pool` & `google_iam_workload_identity_pool_provider.github_provider`: Keyless OIDC authentication for GitHub Actions push pipelines (`sec-edgar-cicd-sa@iron-radio-514217-q7.iam.gserviceaccount.com`).
  - `google_cloud_run_v2_service.sec_edgar_analyst`: Cloud Run service running `sec-analyst-sa` with secrets mounted directly from Secret Manager (`value_source.secret_key_ref`). Bootstraps cleanly in a single-stage `terraform apply` via `var.cloud_run_initial_image` (`us-docker.pkg.dev/cloudrun/container/hello`) while preserving CI/CD container deployments via `lifecycle.ignore_changes`.
- **Cloud Build Triggers (`terraform/cloudbuild.tf`, opt-in with `enable_cloudbuild_trigger = true`):**
  - `google_cloudbuildv2_connection.github` and `google_cloudbuildv2_repository.edgar_capstone`: a 2nd-gen GitHub connection in `var.region` that links `var.github_repository`. To reuse a connection created in the console instead, set `create_cloudbuild_github_connection = false`.
  - `google_cloudbuild_trigger.main_deploy`: on pushes or merges matching `cloudbuild_trigger_branch_regex` (default `^main$`), runs `cloudbuild.yaml` with `_PIPELINE_MODE=deploy`. That covers the lint and test gates, container build and smoke test, push to Artifact Registry, and Cloud Run deploy.
  - `google_cloudbuild_trigger.pull_request_verify`: on pull requests into `main`, runs the same file with `_PIPELINE_MODE=verify`, which skips the push and deploy. PRs from external contributors need a `/gcbrun` comment first.
  - `google_service_account.cloudbuild_runner` (`sec-edgar-cloudbuild-sa`) is a dedicated least-privilege identity. Its roles are `roles/logging.logWriter`, `roles/run.developer`, Artifact Registry writer on the app repository only, `roles/iam.serviceAccountUser` on `sec-analyst-sa`, and Secret Accessor on the three managed secrets.
  - Triggers pass `_REGION`, `_GAR_REPOSITORY`, `_SERVICE_NAME`, `_RUNTIME_SA`, `_BIGQUERY_DATASET`, `_SEC_CORPUS_BUCKET`, `_ANALYST_MODEL`, `_FLASH_MODEL`, and `_GEMINI_LOCATION` from Terraform, so the pipeline always matches the deployed config. Changes to Markdown and `docs/**` are ignored.
  - One-time setup, done outside Terraform so the GitHub token never enters state:
    ```bash
    # a. Install the "Google Cloud Build" GitHub App on the repo; note the installation ID
    # b. Store a GitHub PAT (repo, read:user, read:org) in Secret Manager
    printf '%s' "$GITHUB_PAT" | gcloud secrets create sec-edgar-github-oauth-token \
      --project=iron-radio-514217-q7 --replication-policy=automatic --data-file=-
    # c. terraform.tfvars
    #    enable_cloudbuild_trigger             = true
    #    cloudbuild_github_app_installation_id = <installation-id>
    terraform apply
    ```
  - Not enabled in the project yet: no triggers exist, so deploys currently go through manual `gcloud builds submit` (or GitHub Actions with `DEPLOY_WITH_GITHUB_ACTIONS=true`).

## Usage

```bash
# 1. Bootstrap the versioned GCS remote state bucket (idempotent)
../.venv/bin/python ../scripts/bootstrap_tfstate_bucket.py

# 2. Initialize, validate, plan, and apply the full Terraform stack in a single stage
# (If gs://fde-sec-edgar-sandbox-dev-sec-corpus already exists in GCP, import it once first:
#  terraform -chdir=terraform import 'google_storage_bucket.sec_corpus[0]' fde-sec-edgar-sandbox-dev-sec-corpus)
# (On a brand-new GCP project without Secret Manager versions yet, pass -var="seed_placeholder_secret_versions=true" once)
cd terraform
terraform init
terraform validate
terraform plan
terraform apply

# 3. Populate curated SEC 10-K filings & vector DB manifests into the Terraform-managed bucket and BigQuery
../.venv/bin/python ../scripts/build_sec_dataset.py --sp500 --no-create-bucket
# (Embeddings always come from Vertex AI text-embedding-005; the build fails if it is unavailable)
../.venv/bin/python ../scripts/build_vector_db.py
# (Already-built artifacts: publish without re-embedding)
../.venv/bin/python ../scripts/build_vector_db.py --upload-only

# 4. Backfill R&D onto older profiles, load native financial_metrics, and import Vertex AI Search
cd .. && make backfill-rd publish-financials index-vertex-search
```

`seed_vertex_vector_index` defaults to `true`, so the `terraform apply` in step 2 attaches `vector_search/embeddings/` to the Vector Search index. In a fresh project, pass `-var=seed_vertex_vector_index=false` until the embeddings are uploaded.

Current data state (2026-11-16): the full index (501 tickers, 3,022 profiles, 342,945 Vertex-embedded chunks) is published. `filing_chunks` and `financial_metrics` (now with `research_and_development`) are loaded in BigQuery, `sec-edgar-10k-corpus-ds` holds all 3,022 filings, and the Vector Search index reads `vector_search/embeddings/`. The index is deployed to its (Private Service Connect) endpoint by the next full `terraform apply`.

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
