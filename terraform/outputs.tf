output "project_id" {
  description = "Google Cloud Project ID."
  value       = var.project_id
}

output "sec_corpus_bucket_name" {
  description = "Managed GCS bucket name storing curated SEC 10-K filings, manifests, and vector embeddings."
  value       = local.sec_corpus_bucket_name
}

output "sec_corpus_bucket_url" {
  description = "gs:// URI of the SEC 10-K corpus GCS bucket."
  value       = var.create_sec_corpus_bucket ? google_storage_bucket.sec_corpus[0].url : "gs://${var.sec_corpus_bucket}"
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

output "filing_chunks_table_fqn" {
  description = "Fully-qualified BigQuery metadata-filtered RAG chunks & embeddings table ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.filing_chunks.table_id}"
}

output "chunk_health_metrics_view_fqn" {
  description = "Fully-qualified BigQuery chunk health monitoring view ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_chunk_health_metrics.table_id}"
}

output "filing_chunk_coverage_view_fqn" {
  description = "Fully-qualified BigQuery per-filing chunk coverage diagnostic view ID."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_filing_chunk_coverage.table_id}"
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

output "chunk_health_monitoring_dashboard_url" {
  description = "Cloud Console URL of the RAG vector DB chunk health dashboard."
  value       = local.chunk_health_dashboard_console_url
}

output "llm_observability_dashboard_id" {
  description = "Resource ID of the Cloud Monitoring dashboard for LLM telemetry, cost, sessions, and security audit."
  value       = google_monitoring_dashboard.llm_observability_dashboard.id
}

output "llm_observability_dashboard_url" {
  description = "Cloud Console URL of the LLM telemetry, cost, sessions, and security audit dashboard."
  value       = "https://console.cloud.google.com/monitoring/dashboards/builder/${element(split("/", google_monitoring_dashboard.llm_observability_dashboard.id), 3)}?project=${var.project_id}"
}

output "observability_log_based_metrics" {
  description = "Names of the low-cardinality sec_edgar/* log-based metrics backing the Cloud Monitoring dashboards and alert policies."
  value = sort(concat(
    [google_logging_metric.api_request_latencies.name],
    [for m in google_logging_metric.telemetry_distributions : m.name],
    [for m in google_logging_metric.telemetry_counters : m.name],
  ))
}

output "observability_alert_policy_ids" {
  description = "Resource names of the Cloud Monitoring alert policies (p95 latency, 5xx errors, auth/guardrail blocks)."
  value = {
    api_latency_p95       = google_monitoring_alert_policy.api_latency_p95.name
    api_5xx_errors        = google_monitoring_alert_policy.api_5xx_errors.name
    security_audit_blocks = google_monitoring_alert_policy.security_audit_blocks.name
  }
}

output "analyst_service_account_email" {
  description = "Email of the custom SEC Analyst runtime service account (sec-analyst-sa@<project>.iam.gserviceaccount.com)."
  value       = local.runtime_service_account_email
}

output "analyst_service_account_project_name_email" {
  description = "Logical project-name alias of the custom runtime service account (sec-analyst-sa@<project_name>.iam.gserviceaccount.com)."
  value       = local.runtime_service_account_project_name_email
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

output "cloud_run_target_image_uri" {
  description = "Target Artifact Registry image repository URI pushed by CI/CD pipelines for Cloud Run."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sec_edgar_repo.repository_id}/${var.cloud_run_service_name}"
}

output "cloud_run_service_uri" {
  description = "Deployed Google Cloud Run v2 service URI."
  value       = var.enable_cloud_run_service ? google_cloud_run_v2_service.sec_edgar_analyst[0].uri : null
}

output "iap_backend_service_audience" {
  description = "Expected IAP JWT 'aud' claim for Direct Cloud Run IAP."
  value       = local.iap_jwt_audience
}

output "iap_authorized_principals" {
  description = "Principals that can sign in to the app through IAP (roles/iap.httpsResourceAccessor)."
  value       = sort(tolist(local.effective_iap_group_principals))
}

output "model_armor_template_name" {
  description = "Full resource name of the regional Google Cloud Model Armor guardrail template."
  value       = var.enable_model_armor ? google_model_armor_template.sec_edgar_analyst_armor[0].name : null
}

output "sdp_inspect_template_id" {
  description = "Resource ID of the Google Cloud Sensitive Data Protection (DLP) inspect template."
  value       = var.enable_model_armor ? google_data_loss_prevention_inspect_template.sec_edgar_sdp_inspect[0].id : null
}

output "sdp_deidentify_template_id" {
  description = "Resource ID of the Google Cloud Sensitive Data Protection (DLP) de-identification template."
  value       = var.enable_model_armor ? google_data_loss_prevention_deidentify_template.sec_edgar_sdp_deidentify[0].id : null
}

output "llm_telemetry_table_fqn" {
  description = "Fully-qualified BigQuery streaming sink table for per-call LLM token, cost, cached-token savings, and latency telemetry."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.llm_telemetry_events.table_id}"
}

output "session_telemetry_table_fqn" {
  description = "Fully-qualified BigQuery streaming sink table for conversational session length, companies analyzed, report counts, and ADK tool usage."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.session_telemetry_events.table_id}"
}

output "security_audit_logs_table_fqn" {
  description = "Fully-qualified BigQuery streaming sink table for IAP/Okta/Workspace auth and 10-K filing access audit logs."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.security_audit_logs.table_id}"
}

output "telemetry_cost_and_latency_view_fqn" {
  description = "Fully-qualified BigQuery SQL semantic view aggregating exact daily LLM token cost, cached-token savings, and p95 latency (source of truth for cumulative USD totals)."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_telemetry_cost_and_latency.table_id}"
}

output "session_analytics_summary_view_fqn" {
  description = "Fully-qualified BigQuery SQL view rolling up interactive turns vs async LLM-as-a-Judge evaluations by day, intent, and execution mode."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_session_analytics_summary.table_id}"
}

output "security_audit_summary_view_fqn" {
  description = "Fully-qualified BigQuery SQL view rolling up security audit events (denials, guardrail blocks, principals) by day, event type, status, and identity provider."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_security_audit_summary.table_id}"
}

output "firestore_feedback_database_name" {
  description = "Firestore Native database resource name persisting analyst thumbs-up/down feedback and comments."
  value       = var.enable_firestore_feedback ? google_firestore_database.analyst_feedback_db[0].name : null
}

output "app_url" {
  description = "IAP-protected Cloud Run URL to share with authorized users."
  value       = local.cloud_run_direct_iap ? google_cloud_run_v2_service.sec_edgar_analyst[0].uri : null
}
