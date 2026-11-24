# ------------------------------------------------------------------------------
# Custom Application Runtime Service Account (sec-analyst-sa)
# ------------------------------------------------------------------------------

locals {
  runtime_service_account_email              = var.create_analyst_service_account ? google_service_account.sec_analyst[0].email : "${var.analyst_service_account_id}@${var.project_id}.iam.gserviceaccount.com"
  runtime_service_account_project_name_email = "${var.analyst_service_account_id}@${var.project_name}.iam.gserviceaccount.com"
}

# ------------------------------------------------------------------------------
# 1. Custom Application Runtime Service Account (sec-analyst-sa@<project>.iam.gserviceaccount.com)
# ------------------------------------------------------------------------------
resource "google_service_account" "sec_analyst" {
  count = var.create_analyst_service_account ? 1 : 0

  project      = var.project_id
  account_id   = var.analyst_service_account_id
  display_name = "SEC EDGAR Natural Language Analyst Service Account"
  description  = "Custom runtime identity (${var.analyst_service_account_id}@${var.project_name}.iam.gserviceaccount.com) for the Cloud Run application; default Compute SAs are prohibited."

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

resource "google_bigquery_dataset_iam_member" "sec_analyst_dataset_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_storage_bucket_iam_member" "sec_analyst_corpus_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  bucket = local.sec_corpus_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_vertex_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_discoveryengine_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_monitoring_metric_writer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_log_writer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_trace_agent" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_telemetry_writer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/telemetry.writer"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_modelarmor_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/modelarmor.user"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_dlp_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/dlp.user"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}
