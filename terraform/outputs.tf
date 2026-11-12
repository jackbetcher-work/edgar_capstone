output "project_id" {
  description = "Google Cloud Project ID."
  value       = var.project_id
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID hosting SEC 10-K financial metrics and RAG tables."
  value       = google_bigquery_dataset.sec_edgar.dataset_id
}

output "bigquery_dataset_location" {
  description = "Geographic location of the BigQuery dataset."
  value       = google_bigquery_dataset.sec_edgar.location
}

output "financial_metrics_table_fqn" {
  description = "Fully-qualified native BigQuery financial metrics table ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.financial_metrics.table_id}"
}

output "financial_metrics_external_table_fqn" {
  description = "Fully-qualified GCS-backed external BigQuery financial metrics table ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.financial_metrics_ext.table_id}"
}

output "filing_chunks_table_fqn" {
  description = "Fully-qualified BigQuery metadata-filtered RAG chunks & embeddings table ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.filing_chunks.table_id}"
}

output "unstructured_objects_table_fqn" {
  description = "Fully-qualified BigQuery Object Table over unstructured 10-K Markdown files in GCS."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.unstructured_10k_objects.table_id}"
}

output "yoy_variance_view_fqn" {
  description = "Fully-qualified BigQuery view computing deterministic YoY variances."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_financial_variances_yoy.table_id}"
}

output "vertex_ai_connection_name" {
  description = "Full resource name of the BigQuery Vertex AI Cloud Resource Connection."
  value       = google_bigquery_connection.vertex_ai.name
}

output "vertex_ai_connection_service_account" {
  description = "Google-managed service account attached to the BigQuery Vertex AI connection."
  value       = google_bigquery_connection.vertex_ai.cloud_resource[0].service_account_id
}

output "analyst_service_account_email" {
  description = "Email of the least-privilege SEC Analyst runtime service account."
  value       = var.create_analyst_service_account ? google_service_account.sec_analyst[0].email : null
}
