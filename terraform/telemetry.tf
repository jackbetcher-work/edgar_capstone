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

# 4. SQL Semantic Views (queried from BigQuery Studio): exact USD cost, cache
#    savings, latency, session analytics, and audit rollups. These replace the removed
#    BI semantic layer; Cloud Monitoring log-based metrics (observability.tf) give the
#    real-time, approximate view of the same signals.

# 4a. Daily Token Cost, Cache Savings & p95 Latency
resource "google_bigquery_table" "v_telemetry_cost_and_latency" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_telemetry_cost_and_latency"
  description         = "SQL semantic view aggregating exact LLM token usage, USD cost, cached-token discount savings, cache hit rates, and average/p95 TTFT & end-to-end latency by day, model, agent, and workload (source of truth for cumulative USD totals)."
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

# 4b. Daily Session Analytics: interactive turns vs async Flash judge evaluations
# Interactive turns (execution_mode != 'async_evaluation') carry latency, cost, and
# evaluation_grade = 'pending' / evaluation_score = NULL. The async judge appends a
# separate row (execution_mode = 'async_evaluation', 0 ms latency, 0 USD) with the real
# score. Every turn/latency/cost aggregate is restricted to interactive rows and every
# evaluation aggregate to scored async rows, so 0 ms judge rows never dilute turn
# latency and pending (NULL) scores never enter AVG(evaluation_score).
resource "google_bigquery_table" "v_session_analytics_summary" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_session_analytics_summary"
  description         = "SQL semantic view rolling up conversational turns by day, intent, and execution mode, separating interactive turns from async LLM-as-a-Judge evaluation rows (turn counts/latency exclude 0 ms async rows; AVG(evaluation_score) excludes pending NULL scores)."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      SELECT
        DATE(recorded_at) AS session_date,
        intent,
        execution_mode,
        COUNTIF(execution_mode != 'async_evaluation') AS interactive_turn_count,
        COUNTIF(execution_mode = 'async_evaluation') AS async_evaluation_count,
        COUNT(DISTINCT IF(execution_mode != 'async_evaluation', session_id, NULL)) AS distinct_sessions,
        COUNT(DISTINCT IF(execution_mode != 'async_evaluation', user_id, NULL)) AS distinct_users,
        COUNTIF(execution_mode != 'async_evaluation' AND status = 'success') AS successful_turn_count,
        COUNTIF(execution_mode != 'async_evaluation' AND report_generated) AS reports_generated_count,
        COUNTIF(execution_mode != 'async_evaluation' AND fact_check_passed) AS fact_check_passed_count,
        ROUND(SAFE_DIVIDE(
          COUNTIF(execution_mode != 'async_evaluation' AND fact_check_passed),
          COUNTIF(execution_mode != 'async_evaluation')
        ), 4) AS fact_check_pass_rate,
        ROUND(AVG(IF(execution_mode != 'async_evaluation', companies_analyzed_count, NULL)), 3) AS avg_companies_per_turn,
        SUM(IF(execution_mode != 'async_evaluation', tool_invocation_count, 0)) AS tool_invocation_count,
        SUM(IF(execution_mode != 'async_evaluation', llm_call_count, 0)) AS llm_call_count,
        ROUND(AVG(IF(execution_mode != 'async_evaluation', ttft_ms, NULL)), 3) AS avg_turn_ttft_ms,
        ROUND(AVG(IF(execution_mode != 'async_evaluation', total_latency_ms, NULL)), 3) AS avg_turn_latency_ms,
        ROUND(APPROX_QUANTILES(IF(execution_mode != 'async_evaluation', total_latency_ms, NULL), 100)[SAFE_OFFSET(95)], 3) AS p95_turn_latency_ms,
        ROUND(SUM(IF(execution_mode != 'async_evaluation', total_cost_usd, 0)), 8) AS total_cost_usd,
        ROUND(SUM(IF(execution_mode != 'async_evaluation', cached_token_savings_usd, 0)), 8) AS cached_token_savings_usd,
        COUNTIF(execution_mode != 'async_evaluation' AND (evaluation_grade = 'pending' OR evaluation_score IS NULL)) AS pending_evaluation_turn_count,
        COUNTIF(execution_mode = 'async_evaluation' AND evaluation_score >= 0) AS scored_evaluation_count,
        ROUND(AVG(IF(execution_mode = 'async_evaluation' AND evaluation_score >= 0, evaluation_score, NULL)), 4) AS avg_evaluation_score,
        ROUND(APPROX_QUANTILES(IF(execution_mode = 'async_evaluation' AND evaluation_score >= 0, evaluation_score, NULL), 100)[SAFE_OFFSET(50)], 4) AS p50_evaluation_score
      FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.session_telemetry_events.table_id}`
      GROUP BY 1, 2, 3
    SQL
  }
}

# 4c. Daily Security Audit Rollup (auth denials, guardrail blocks, SPIFFE S2S attestations)
resource "google_bigquery_table" "v_security_audit_summary" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_security_audit_summary"
  description         = "SQL semantic view rolling up security audit events by day, event type, status, and identity provider: totals, 401/403 denials, guardrail blocks, distinct principals, and SPIFFE S2S attested events."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      SELECT
        DATE(recorded_at) AS audit_date,
        event_type,
        status,
        COALESCE(NULLIF(identity_provider, ''), 'none') AS identity_provider,
        COUNT(*) AS total_events,
        COUNTIF(STARTS_WITH(status, 'denied')) AS denied_events,
        COUNTIF(status = 'denied_401') AS denied_401_events,
        COUNTIF(status = 'denied_403') AS denied_403_events,
        COUNTIF(STARTS_WITH(status, 'blocked')) AS blocked_events,
        COUNT(DISTINCT NULLIF(user_email, 'unauthenticated')) AS distinct_principals,
        COUNTIF(spiffe_id IS NOT NULL AND spiffe_id != '') AS spiffe_attested_events,
        COUNTIF(iap_assertion_verified) AS iap_verified_events,
        COUNT(DISTINCT trace_id) AS distinct_traces
      FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.security_audit_logs.table_id}`
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

# 6. Cloud Logging Project Sink -> Dedicated Ops-Log BigQuery Dataset
# The sink writes into its own dataset (not sec_edgar) so its writer identity can hold
# roles/bigquery.dataEditor (incl. bigquery.tables.create, needed to auto-create the
# partitioned log tables) without any write path to financial_metrics / filing_chunks,
# which stay strictly read-only. The app's direct insertAll streaming into
# sec_edgar.*_events / security_audit_logs remains the analytical source of truth; this
# export is the durable, raw Cloud Logging copy (full LogEntry incl. trace + labels).
resource "google_bigquery_dataset" "sec_edgar_ops_logs" {
  count = var.enable_bigquery_logging_sink ? 1 : 0

  project                         = var.project_id
  dataset_id                      = var.ops_logs_dataset_id
  friendly_name                   = "SEC EDGAR Operational Log Export"
  description                     = "Cloud Logging sink destination (sec-edgar-telemetry-audit-bq-sink) for raw structured telemetry, session, and security audit LogEntries. Isolated from sec_edgar so the sink writer never gains write access to curated financial tables."
  location                        = var.region
  delete_contents_on_destroy      = !var.deletion_protection
  default_partition_expiration_ms = var.ops_logs_retention_days > 0 ? var.ops_logs_retention_days * 86400000 : null
  labels                          = var.labels

  depends_on = [google_project_service.required_apis]
}

resource "google_logging_project_sink" "telemetry_and_audit_to_bigquery" {
  count = var.enable_bigquery_logging_sink ? 1 : 0

  project                = var.project_id
  name                   = "sec-edgar-telemetry-audit-bq-sink"
  description            = "Streams structured SEC EDGAR telemetry, session analytics, and 10-K security audit logs from Cloud Logging into partitioned tables in the dedicated sec_edgar_ops_logs dataset."
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.sec_edgar_ops_logs[0].dataset_id}"
  filter                 = "jsonPayload.component=\"sec-edgar-telemetry-sink\""
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

# Dataset-scoped (never project-level) dataEditor for the sink's writer identity.
resource "google_bigquery_dataset_iam_member" "logging_sink_ops_logs_writer" {
  count = var.enable_bigquery_logging_sink ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar_ops_logs[0].dataset_id
  role       = "roles/bigquery.dataEditor"
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
