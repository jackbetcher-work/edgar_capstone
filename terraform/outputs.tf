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

output "cloud_run_target_image_uri" {
  description = "Target Artifact Registry image repository URI pushed by CI/CD pipelines for Cloud Run."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sec_edgar_repo.repository_id}/${var.cloud_run_service_name}"
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
  description = "Expected IAP JWT 'aud' claim: the Cloud Run service path in iap_direct mode, the backend service path in load_balancer mode."
  value       = local.iap_jwt_audience
}

output "cloud_run_access_mode" {
  description = "How users reach the app: iap_direct (IAP on the run.app URL) or load_balancer (external HTTPS LB + IAP)."
  value       = var.cloud_run_access_mode
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
  description = "Fully-qualified BigQuery SQL view rolling up security audit events (denials, guardrail blocks, principals, SPIFFE attestations) by day, event type, status, and identity provider."
  value       = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.v_security_audit_summary.table_id}"
}

output "ops_logs_dataset_id" {
  description = "Dedicated BigQuery dataset receiving the Cloud Logging -> BigQuery project sink (null when enable_bigquery_logging_sink = false)."
  value       = var.enable_bigquery_logging_sink ? google_bigquery_dataset.sec_edgar_ops_logs[0].dataset_id : null
}

output "logging_sink_writer_identity" {
  description = "Writer identity of sec-edgar-telemetry-audit-bq-sink (granted roles/bigquery.dataEditor on the ops-logs dataset only)."
  value       = var.enable_bigquery_logging_sink ? google_logging_project_sink.telemetry_and_audit_to_bigquery[0].writer_identity : null
}

output "firestore_feedback_database_name" {
  description = "Firestore Native database resource name persisting analyst thumbs-up/down feedback and comments."
  value       = var.enable_firestore_feedback ? google_firestore_database.analyst_feedback_db[0].name : null
}

output "lb_ip_address" {
  description = "Static global IPv4 address of the HTTPS load balancer (point your DNS A record here if lb_domain is set)."
  value       = local.enable_https_lb ? google_compute_global_address.sec_edgar_lb_ip[0].address : null
}

output "app_url" {
  description = "IAP-protected URL to share with authorized users: the Cloud Run run.app URL (iap_direct) or the HTTPS load balancer URL (load_balancer)."
  value = (
    local.cloud_run_direct_iap
    ? google_cloud_run_v2_service.sec_edgar_analyst[0].uri
    : (local.enable_https_lb ? "https://${local.lb_domain}" : null)
  )
}

output "vpc_sc_perimeter_name" {
  description = "Resource name of the VPC Service Controls perimeter, or null when it isn't managed."
  value       = local.vpc_sc_enabled ? google_access_context_manager_service_perimeter.sec_edgar[0].name : null
}

output "vpc_sc_mode" {
  description = "VPC Service Controls state: disabled, dry_run, or enforced."
  value       = local.vpc_sc_enabled ? (var.vpc_sc_enforce ? "enforced" : "dry_run") : "disabled"
}

output "vertex_vector_search_deployed_index_id" {
  description = "Deployed index ID on the Vector Search endpoint (VERTEX_VECTOR_SEARCH_DEPLOYED_INDEX_ID)."
  value       = length(google_vertex_ai_index_endpoint_deployed_index.sec_10k_chunks_deployed) > 0 ? google_vertex_ai_index_endpoint_deployed_index.sec_10k_chunks_deployed[0].deployed_index_id : null
}

output "cloudbuild_service_account_email" {
  description = "Email of the dedicated Cloud Build trigger service account (null when enable_cloudbuild_trigger = false)."
  value       = var.enable_cloudbuild_trigger ? google_service_account.cloudbuild_runner[0].email : null
}

output "cloudbuild_trigger_ids" {
  description = "IDs of the Terraform-managed Cloud Build triggers."
  value = {
    main_deploy         = var.enable_cloudbuild_trigger ? google_cloudbuild_trigger.main_deploy[0].trigger_id : null
    pull_request_verify = var.enable_cloudbuild_trigger && var.enable_cloudbuild_pr_trigger ? google_cloudbuild_trigger.pull_request_verify[0].trigger_id : null
  }
}

output "vector_search_psc_address" {
  description = "Internal IP of the Vector Search Private Service Connect endpoint (null in public mode or before the index is deployed)."
  value       = local.vector_search_psc_endpoint ? google_compute_address.vector_search_psc[0].address : null
}
