# ------------------------------------------------------------------------------
# Live-GCP read access for CI test runs
# ------------------------------------------------------------------------------
# There is no offline/local data mode: the pytest suite, the golden evals, and the
# container smoke test read BigQuery `sec_edgar`, Vertex AI Search, Vertex AI
# embeddings, and the GCS SEC corpus bucket, and call Gemini + Model Armor on
# Vertex AI. The CI identities that run them get least-privilege READ access here:
#   - GitHub Actions: the keyless WIF service account (cicd_runner)
#   - Cloud Build:    the trigger service account (cloudbuild_runner), when enabled
# Both are already listed as trusted ingress identities in vpc_service_controls.tf.
# Telemetry and feedback stay in memory during tests (EDGAR_TELEMETRY_BACKEND=local,
# EDGAR_FEEDBACK_BACKEND=local), so no write roles are granted.

locals {
  ci_live_test_principals = merge(
    { github_actions = "serviceAccount:${google_service_account.cicd_runner.email}" },
    local.cloudbuild_enabled ? {
      cloud_build = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
    } : {}
  )
}

# Run query jobs (financial metrics, YoY views, VECTOR_SEARCH over filing_chunks).
resource "google_project_iam_member" "ci_live_test_bq_job_user" {
  for_each = local.ci_live_test_principals

  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = each.value
}

# Read tables and views in the `sec_edgar` dataset only.
resource "google_bigquery_dataset_iam_member" "ci_live_test_dataset_viewer" {
  for_each = local.ci_live_test_principals

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = each.value
}

# Read the corpus bucket (10-K Markdown sections and the GCS-backed external tables).
resource "google_storage_bucket_iam_member" "ci_live_test_corpus_viewer" {
  for_each = local.ci_live_test_principals

  bucket = local.sec_corpus_bucket_name
  role   = "roles/storage.objectViewer"
  member = each.value
}

# Query Vertex AI Search (Discovery Engine) serving configs.
resource "google_project_iam_member" "ci_live_test_discoveryengine_viewer" {
  for_each = local.ci_live_test_principals

  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = each.value
}

# Vertex AI predictions: text-embedding-005, Gemini, and Vector Search queries.
resource "google_project_iam_member" "ci_live_test_vertex_user" {
  for_each = local.ci_live_test_principals

  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = each.value
}

# Model Armor prompt/response sanitization (the runtime calls it whenever
# GOOGLE_GENAI_USE_VERTEXAI=true, which CI sets).
resource "google_project_iam_member" "ci_live_test_modelarmor_user" {
  for_each = local.ci_live_test_principals

  project = var.project_id
  role    = "roles/modelarmor.user"
  member  = each.value
}

# Cloud Sensitive Data Protection (DLP) content:deidentify, which the guardrail calls on every
# prompt and response alongside Model Armor's basic SDP.
resource "google_project_iam_member" "ci_live_test_dlp_user" {
  for_each = local.ci_live_test_principals

  project = var.project_id
  role    = "roles/dlp.user"
  member  = each.value
}
