# ------------------------------------------------------------------------------
# Least-Privilege Service Account for SEC Analyst Backend & RAG Pipeline
# ------------------------------------------------------------------------------
resource "google_service_account" "sec_analyst" {
  count = var.create_analyst_service_account ? 1 : 0

  project      = var.project_id
  account_id   = var.analyst_service_account_id
  display_name = "SEC EDGAR Natural Language Analyst Service Account"
  description  = "Runtime identity for querying BigQuery financial metrics, metadata-filtered RAG, and reading SEC 10-K GCS artifacts."

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "sec_analyst_bq_job_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_bq_connection_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/bigquery.connectionUser"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_bigquery_dataset_iam_member" "sec_analyst_dataset_editor" {
  count = var.create_analyst_service_account ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_storage_bucket_iam_member" "sec_analyst_corpus_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  bucket = var.sec_corpus_bucket
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}
