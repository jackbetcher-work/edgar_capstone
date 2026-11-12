# Terraform — SEC EDGAR Hybrid RAG Vector Database, BigQuery & Chunk Health Monitoring

Provisions the BigQuery data & vector search foundation, Vertex AI Vector Search (Matching Engine), Vertex AI Search (Discovery Engine), and the Google Cloud Monitoring chunk health dashboard in project `iron-radio-514217-q7` (`us-central1`) with remote Terraform state stored in `gs://iron-radio-514217-q7-tfstate`.

## Provisioned Resources

- **Remote State Bucket** (`gs://iron-radio-514217-q7-tfstate`): Bootstrapped with object versioning, uniform bucket-level access, and enforced public access prevention.
- **BigQuery Dataset (`sec_edgar`, `us-central1`):**
  - `financial_metrics`: Native table partitioned by `fiscal_year` (`2015–2035`) and clustered by `gics_sector, ticker`, automatically seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_financials.jsonl`.
  - `financial_metrics_ext`: Live GCS external table over `manifests/bigquery_financials.jsonl`.
  - `filing_chunks`: Metadata-filtered RAG table partitioned by `fiscal_year` and clustered by `gics_sector, ticker, section_id` with `embedding` (`ARRAY<FLOAT64>`, 768-d) ready for `VECTOR_SEARCH`, automatically seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_filing_chunks.jsonl`.
  - `filing_chunks_ext`: Live GCS external table over `manifests/bigquery_filing_chunks.jsonl`.
  - `unstructured_10k_objects`: BigQuery Object Table indexing `gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/*`.
  - `v_financial_variances_yoy`: Deterministic YoY variance SQL view (`revenue`, `operating_income`, `net_income` deltas and percentages).
  - `v_filing_chunk_coverage`: Per-filing (`profile_id`, `ticker`, `fiscal_year`) diagnostic view reporting exact `item_1a_chunks`, `item_7_chunks`, `valid_embedding_chunks`, and `filing_chunk_status`.
  - `v_chunk_health_metrics`: Chunk health monitoring SQL view computing ROLLUP health metrics by `(gics_sector, section_id)` including `embedding_coverage_ratio`, `filing_coverage_ratio`, token/character percentiles (`p50`, `p95`), anomaly counts (`missing_embedding`, `dimension_mismatch`, `undersized`, `oversized`, `missing_metadata`), and composite `health_score` (`0–100`) & `health_status` (`HEALTHY`, `DEGRADED`, `UNHEALTHY`).
- **Vertex AI Vector Search (`terraform/vector_search.tf`):**
  - `sec-edgar-10k-chunks-index` (`google_vertex_ai_index`): 768-dimensional `COSINE_DISTANCE` Tree-AH ANN index backed by `gs://fde-sec-edgar-sandbox-dev-sec-corpus/vector_search/embeddings` with string `restricts` (`ticker`, `cik`, `gics_sector`, `gics_sub_industry`, `section_id`, `form_type`, `profile_id`), `numeric_restricts` (`fiscal_year`), and `crowding_tag` (`ticker`).
  - `sec-edgar-10k-chunks-endpoint` (`google_vertex_ai_index_endpoint`): Public Index Endpoint for low-latency metadata-filtered vector retrieval.
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
- **Runtime Service Account (`sec-analyst-sa@iron-radio-514217-q7.iam.gserviceaccount.com`):** Granted least-privilege BigQuery, GCS, Vertex AI (`roles/aiplatform.user`), Discovery Engine (`roles/discoveryengine.viewer`), Secret Manager (`roles/secretmanager.secretAccessor`), Cloud Monitoring (`roles/monitoring.metricWriter`), and Cloud Logging (`roles/logging.logWriter`) roles.
- **Google Cloud Secret Manager & Keyless CI/CD Pipeline (`terraform/cicd_and_secrets.tf`):**
  - `google_secret_manager_secret.managed_secrets`: Provisions `sec-edgar-user-agent`, `sec-edgar-gemini-api-key`, and `sec-edgar-iap-audience` with automatic replication and zero plaintext secret values in IaC.
  - `google_artifact_registry_repository.sec_edgar_repo`: Docker container repository (`us-central1-docker.pkg.dev/iron-radio-514217-q7/sec-edgar-repo`).
  - `google_iam_workload_identity_pool.github_pool` & `google_iam_workload_identity_pool_provider.github_provider`: Keyless OIDC authentication for GitHub Actions push pipelines (`sec-edgar-cicd-sa@iron-radio-514217-q7.iam.gserviceaccount.com`).
  - `google_cloud_run_v2_service.sec_edgar_analyst`: Cloud Run service running `sec-analyst-sa` with secrets mounted directly from Secret Manager (`value_source.secret_key_ref`).

## Usage

```bash
# 1. Build and populate the metadata-tagged vector DB manifests & chunk health metrics
../.venv/bin/python ../scripts/build_vector_db.py

# 2. Bootstrap the versioned GCS remote state bucket (idempotent)
../.venv/bin/python ../scripts/bootstrap_tfstate_bucket.py

# 3. Initialize, validate, plan, and apply Terraform
cd terraform
terraform init
terraform validate
terraform plan
terraform apply
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
