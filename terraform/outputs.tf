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

output "filing_chunks_external_table_fqn" {
  description = "Fully-qualified GCS-backed external BigQuery filing chunks table ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.filing_chunks_ext.table_id}"
}

output "chunk_health_metrics_view_fqn" {
  description = "Fully-qualified BigQuery chunk health monitoring view ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_chunk_health_metrics.table_id}"
}

output "filing_chunk_coverage_view_fqn" {
  description = "Fully-qualified BigQuery per-filing chunk coverage diagnostic view ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_filing_chunk_coverage.table_id}"
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

output "vertex_vector_search_index_id" {
  description = "Resource ID of the Vertex AI Vector Search (Matching Engine) 10-K chunks index."
  value       = var.enable_vertex_vector_search ? google_vertex_ai_index.sec_10k_chunks[0].id : null
}

output "vertex_vector_search_endpoint_id" {
  description = "Resource ID of the Vertex AI Vector Search Index Endpoint."
  value       = var.enable_vertex_vector_search ? google_vertex_ai_index_endpoint.sec_10k_chunks_endpoint[0].id : null
}

output "vertex_ai_search_datastore_id" {
  description = "Resource ID of the Vertex AI Search (Discovery Engine) 10-K unstructured datastore."
  value       = var.enable_vertex_ai_search ? google_discovery_engine_data_store.sec_10k_corpus[0].id : null
}

output "vertex_ai_search_engine_id" {
  description = "Resource ID of the Vertex AI Search (Discovery Engine) hybrid search engine."
  value       = var.enable_vertex_ai_search ? google_discovery_engine_search_engine.sec_10k_hybrid_search[0].id : null
}

output "chunk_health_monitoring_dashboard_id" {
  description = "Resource ID of the Google Cloud Monitoring dashboard for RAG vector DB chunk health."
  value       = google_monitoring_dashboard.chunk_health_dashboard.id
}

output "analyst_service_account_email" {
  description = "Email of the custom SEC Analyst runtime service account (sec-analyst-sa@<project>.iam.gserviceaccount.com)."
  value       = local.runtime_service_account_email
}

output "analyst_service_account_project_name_email" {
  description = "Logical project-name alias of the custom runtime service account (sec-analyst-sa@<project_name>.iam.gserviceaccount.com)."
  value       = local.runtime_service_account_project_name_email
}

output "cloud_identity_rbac_groups" {
  description = "Group-based RBAC access definitions managed via Google Cloud Identity Groups."
  value = {
    admins   = sort(local.rbac_admin_members)
    analysts = sort(local.rbac_analyst_members)
    viewers  = sort(local.rbac_viewer_members)
  }
}

output "agent_service_account_emails" {
  description = "Granular per-agent GCP service accounts enforcing Least Privilege for S2S communication."
  value = (
    var.create_agent_service_accounts
    ? { for k, sa in google_service_account.agent_identities : local.agent_identities[k].agent_name => sa.email }
    : {}
  )
}

output "cicd_service_account_email" {
  description = "Email of the keyless CI/CD runner service account used by GitHub push pipelines."
  value       = google_service_account.cicd_runner.email
}

output "workload_identity_provider_name" {
  description = "Full resource name of the GitHub Actions OIDC Workload Identity Provider."
  value       = google_iam_workload_identity_pool_provider.github_provider.name
}

output "artifact_registry_repository_uri" {
  description = "Docker repository URI in Google Artifact Registry."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sec_edgar_repo.repository_id}"
}

output "secret_manager_secret_ids" {
  description = "List of Secret Manager secret IDs managed for the deployment pipeline and Cloud Run service."
  value       = sort([for s in google_secret_manager_secret.managed_secrets : s.secret_id])
}

output "cloud_run_service_uri" {
  description = "Deployed Google Cloud Run v2 service URI."
  value       = var.enable_cloud_run_service ? google_cloud_run_v2_service.sec_edgar_analyst[0].uri : null
}

output "iap_backend_service_audience" {
  description = "Expected IAP JWT 'aud' claim string for the IAP-protected Cloud Run backend service."
  value = (
    var.enable_iap_backend_service && var.enable_cloud_run_service
    ? "/projects/${data.google_project.current.number}/global/backendServices/${google_compute_backend_service.sec_edgar_iap_backend[0].generated_id}"
    : null
  )
}
