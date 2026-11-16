# ------------------------------------------------------------------------------
# Streaming BigQuery Telemetry & Audit Sink Tables, Cloud Logging Sink, &
# Persistent Firestore Analyst Feedback Database
# ------------------------------------------------------------------------------

# 1. LLM Token / Cost / Latency Streaming Sink Table (DAY-partitioned & clustered)
resource "google_bigquery_table" "llm_telemetry_events" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "llm_telemetry_events"
  description         = "Streaming LLM telemetry sink capturing per-call token breakdowns, USD cost, cached-token discount savings, TTFT, and prefill/decode latency across Gemini 3.8 Flash and Gemini 3.5 Flash."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/llm_telemetry_events.json")

  time_partitioning {
    type  = "DAY"
    field = "recorded_at"
  }

  clustering = ["model_name", "agent_name", "task_category"]
}

# 2. Conversational Session, Company Scope & Tool Usage Sink Table
resource "google_bigquery_table" "session_telemetry_events" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "session_telemetry_events"
  description         = "Streaming conversational session sink tracking session length, companies analyzed, report generation counts, ADK tool invocations, turn latency, and turn USD cost."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/session_telemetry_events.json")

  time_partitioning {
    type  = "DAY"
    field = "recorded_at"
  }

  clustering = ["intent", "execution_mode", "session_id"]
}

# 3. Security, RBAC & 10-K Access Audit Trail Sink Table
resource "google_bigquery_table" "security_audit_logs" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "security_audit_logs"
  description         = "Immutable streaming audit sink recording IAP/Okta/Workspace authentication events, per-agent S2S SPIFFE attestations, and 10-K report/section access trails."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/security_audit_logs.json")

  time_partitioning {
    type  = "DAY"
    field = "recorded_at"
  }

  clustering = ["event_type", "status", "user_email"]
}

# 4. Analytical BI View for Looker: Daily Token Cost, Cache Savings & p95 Latency
resource "google_bigquery_table" "v_telemetry_cost_and_latency" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_telemetry_cost_and_latency"
  description         = "Looker BI semantic view aggregating LLM token usage, USD cost, cached-token discount savings, cache hit rates, and average/p95 TTFT & end-to-end latency by day, model, agent, and workload."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      SELECT
        DATE(recorded_at) AS telemetry_date,
        model_name,
        agent_name,
        task_category,
        COUNT(*) AS call_count,
        SUM(total_context_tokens) AS total_context_tokens,
        SUM(cached_context_tokens) AS cached_context_tokens,
        SUM(uncached_context_tokens) AS uncached_context_tokens,
        SUM(output_tokens + COALESCE(thoughts_tokens, 0)) AS output_tokens,
        ROUND(SAFE_DIVIDE(COUNTIF(cache_hit), COUNT(*)), 4) AS cache_hit_rate,
        ROUND(AVG(total_model_latency_ms), 3) AS avg_total_latency_ms,
        ROUND(APPROX_QUANTILES(total_model_latency_ms, 100)[OFFSET(95)], 3) AS p95_total_latency_ms,
        ROUND(AVG(ttft_ms), 3) AS avg_ttft_ms,
        ROUND(APPROX_QUANTILES(ttft_ms, 100)[OFFSET(95)], 3) AS p95_ttft_ms,
        ROUND(SAFE_DIVIDE(COUNTIF(within_ttft_slo), COUNT(*)), 4) AS ttft_slo_compliance_rate,
        ROUND(SUM(total_cost_usd), 8) AS total_cost_usd,
        ROUND(SUM(uncached_baseline_cost_usd), 8) AS uncached_baseline_cost_usd,
        ROUND(SUM(cached_token_savings_usd), 8) AS cached_token_savings_usd
      FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.llm_telemetry_events.table_id}`
      GROUP BY 1, 2, 3, 4
    SQL
  }
}

# 5. Least-Privilege Table-Scoped IAM for Streaming Append Only on Sink Tables
# Keeps sec-analyst-sa strictly read-only on financial_metrics and filing_chunks.
resource "google_project_iam_custom_role" "bigquery_telemetry_sink_writer" {
  project     = var.project_id
  role_id     = "secEdgarTelemetrySinkAppend"
  title       = "SEC EDGAR Telemetry & Audit Sink Append Writer"
  description = "Least-privilege append-only streaming permission (tables.get + tables.updateData) restricted exclusively to telemetry and audit sink tables."
  permissions = [
    "bigquery.tables.get",
    "bigquery.tables.updateData",
  ]
}

locals {
  telemetry_sink_table_ids = {
    llm_telemetry     = google_bigquery_table.llm_telemetry_events.table_id
    session_telemetry = google_bigquery_table.session_telemetry_events.table_id
    security_audit    = google_bigquery_table.security_audit_logs.table_id
  }
}

resource "google_bigquery_table_iam_member" "runtime_sa_telemetry_sink_append" {
  for_each = var.create_analyst_service_account ? local.telemetry_sink_table_ids : {}

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  table_id   = each.value
  role       = google_project_iam_custom_role.bigquery_telemetry_sink_writer.name
  member     = "serviceAccount:${local.runtime_service_account_email}"
}

# 6. Cloud Logging Project Sink Routing Structured Telemetry & Audit Logs to BigQuery
resource "google_logging_project_sink" "telemetry_and_audit_to_bigquery" {
  count = var.enable_bigquery_logging_sink ? 1 : 0

  project                = var.project_id
  name                   = "sec-edgar-telemetry-audit-bq-sink"
  description            = "Streams structured SEC EDGAR telemetry, session analytics, and 10-K security audit logs from Cloud Logging into partitioned BigQuery tables."
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.sec_edgar.dataset_id}"
  filter                 = "jsonPayload.component=\"sec-edgar-telemetry-sink\""
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_bigquery_dataset_iam_member" "logging_sink_bq_writer" {
  count = var.enable_bigquery_logging_sink ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = google_project_iam_custom_role.bigquery_telemetry_sink_writer.name
  member     = google_logging_project_sink.telemetry_and_audit_to_bigquery[0].writer_identity
}

# 7. Google Cloud Firestore Native Database for Persistent Analyst Feedback
resource "google_firestore_database" "analyst_feedback_db" {
  count = var.enable_firestore_feedback ? 1 : 0

  project                     = var.project_id
  name                        = var.firestore_database_id
  location_id                 = var.region
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"
  deletion_policy             = var.deletion_protection ? "ABANDON" : "DELETE"

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "runtime_sa_firestore_feedback_user" {
  count = var.create_analyst_service_account && var.enable_firestore_feedback ? 1 : 0

  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${local.runtime_service_account_email}"
}
