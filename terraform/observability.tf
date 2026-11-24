# ------------------------------------------------------------------------------
# End-to-end observability: Cloud Trace + structured Cloud Logging + Cloud
# Monitoring (log-based metrics, alert policies, dashboards) + BigQuery SQL views
#
# The FastAPI service exports OpenTelemetry spans to Cloud Trace (runtime SA has
# roles/cloudtrace.agent in iam.tf; cloudtrace.googleapis.com is enabled in
# apis.tf) and writes one structured `http_request_completed` JSON access log per
# request to stdout -> Cloud Logging, carrying `logging.googleapis.com/trace`.
# Every telemetry sink write (LLM call, session turn, async judge score, security
# audit event) is also emitted as a structured `sec-edgar-telemetry-sink` log line
# (telemetry.py:emit_cloud_logging_sink_record) with the same trace ID.
#
# Log-based metrics below turn those logs into low-cardinality Cloud Monitoring
# series. Labels are restricted to bounded enums (model, agent, task, intent,
# status, ...). session_id, user_email, trace_id, call_id, and ticker are NEVER
# metric labels: per-request drill-down goes through trace exemplars, the
# dashboard logs panels, and the BigQuery sink tables instead.
#
# Token usage & exact cumulative USD totals: LLM token counts are reported
# exclusively in BigQuery (`sec_edgar.llm_telemetry_events` /
# `v_telemetry_cost_and_latency`) rather than duplicated as Cloud Monitoring
# log-based metrics. Cloud Logging counter metrics count entries and cannot sum
# float fields, so cost/cache-savings remain approximate DISTRIBUTION metrics in
# Cloud Monitoring while exact cumulative USD lives in BigQuery (telemetry.tf).
# ------------------------------------------------------------------------------

resource "google_logging_metric" "api_request_latencies" {
  project     = var.project_id
  name        = "sec_edgar/api_request_latencies"
  description = "Server-side FastAPI request latency (ms) per route, method, and status, extracted from structured access logs emitted by RequestTelemetryMiddleware."
  filter = join(" AND ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.cloud_run_service_name}\"",
    "jsonPayload.event=\"http_request_completed\"",
  ])

  metric_descriptor {
    metric_kind  = "DELTA"
    value_type   = "DISTRIBUTION"
    unit         = "ms"
    display_name = "SEC EDGAR API Request Latency"

    labels {
      key         = "http_route"
      value_type  = "STRING"
      description = "Matched FastAPI route template, e.g. /api/v1/chat"
    }
    labels {
      key         = "http_method"
      value_type  = "STRING"
      description = "HTTP request method"
    }
    labels {
      key         = "http_status"
      value_type  = "INT64"
      description = "HTTP response status code"
    }
  }

  value_extractor = "EXTRACT(jsonPayload.latency_ms)"
  label_extractors = {
    http_route  = "EXTRACT(jsonPayload.http_route)"
    http_method = "EXTRACT(jsonPayload.http_method)"
    http_status = "EXTRACT(jsonPayload.http_status)"
  }

  # 5 ms .. ~38 min in 40 exponential buckets (covers sub-ms health checks to long live ADK turns).
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 40
      growth_factor      = 1.4
      scale              = 5
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_alert_policy" "api_latency_p95" {
  project      = var.project_id
  display_name = "SEC EDGAR API p95 Latency Exceeded (${var.api_latency_p95_alert_threshold_ms} ms)"
  combiner     = "OR"
  enabled      = true

  notification_channels = var.monitoring_notification_channels

  documentation {
    content   = "Triggered when p95 server-side request latency on `sec-edgar-analyst` exceeds ${var.api_latency_p95_alert_threshold_ms} ms over a 5-minute window. Follow the trace exemplars attached to `logging.googleapis.com/user/sec_edgar/api_request_latencies` directly into Cloud Trace."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Cloud Run /api/v1/chat p95 latency > ${var.api_latency_p95_alert_threshold_ms} ms"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"logging.googleapis.com/user/sec_edgar/api_request_latencies\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.labels.service_name=\"${var.cloud_run_service_name}\"",
        "metric.labels.http_route=monitoring.regex.full_match(\"/api/v1/chat(/stream)?\")",
      ])
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.api_latency_p95_alert_threshold_ms

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["metric.labels.http_route"]
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  user_labels = var.labels

  depends_on = [
    google_project_service.required_apis,
    google_logging_metric.api_request_latencies,
  ]
}

# ------------------------------------------------------------------------------
# Telemetry sink log-based metrics (jsonPayload.component="sec-edgar-telemetry-sink")
# ------------------------------------------------------------------------------
locals {
  telemetry_sink_base_filter = join(" AND ", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.cloud_run_service_name}\"",
    "jsonPayload.component=\"sec-edgar-telemetry-sink\"",
  ])

  # Bounded-enum label descriptions (shared by every telemetry metric). Booleans are
  # STRING labels holding "true" / "false" ("" when the payload field is null).
  telemetry_label_descriptions = {
    model_name        = "Gemini model ID (e.g. gemini-3.8-flash, gemini-3.5-flash-lite)"
    agent_name        = "ADK agent name (FinancialAnalystAgent, NarrativeEvaluatorAgent)"
    task_category     = "Workload category (financial_reasoning, evaluation, ...)"
    within_ttft_slo   = "Whether TTFT met the configured SLO (true/false)"
    cache_hit         = "Whether the call reused cached context tokens (true/false)"
    intent            = "Canonical analyst intent routed for the turn"
    execution_mode    = "Turn execution mode (e.g. live_adk); async judge rows are excluded from turn counts"
    status            = "Outcome status (turn: success/error/...; audit: success/allowed/denied_401/denied_403/blocked/...)"
    fact_check_passed = "Whether claim-level verification passed (true/false)"
    report_generated  = "Whether a structured analyst report was produced (true/false)"
    evaluation_grade  = "Async Flash LLM-as-a-Judge grade (pass/fail/...)"
    event_type        = "Security audit event type (conversational_rag_chat, api_authorization_denied, model_armor_prompt_blocked, ...)"
    identity_provider = "Authenticating identity provider (google_iap, okta, ...; empty for unauthenticated / guardrail events)"
  }

  # Exponential latency buckets: 5 ms .. ~38 min (same as api_request_latencies).
  latency_ms_buckets = { type = "exponential", num_finite_buckets = 40, growth_factor = 1.4, scale = 5, width = 0, offset = 0 }
  # Exponential USD buckets for sub-cent .. dollar LLM calls: $0.0001 .. ~$12.8 (+ overflow).
  usd_buckets = { type = "exponential", num_finite_buckets = 29, growth_factor = 1.5, scale = 0.0001, width = 0, offset = 0 }
  # Linear 0.0 .. 1.05 judge score buckets (width 0.05; a perfect 1.0 lands in [1.0, 1.05)).
  score_buckets = { type = "linear", num_finite_buckets = 21, growth_factor = 0, scale = 0, width = 0.05, offset = 0 }

  telemetry_distribution_metrics = {
    llm_ttft_ms = {
      display_name = "SEC EDGAR LLM Time-to-First-Token"
      description  = "Per-call Gemini time-to-first-token (ms) from llm_telemetry_events sink logs."
      unit         = "ms"
      target       = "llm_telemetry_events"
      extra_filter = ""
      value_field  = "ttft_ms"
      labels       = ["model_name", "agent_name", "task_category", "within_ttft_slo"]
      buckets      = local.latency_ms_buckets
    }
    llm_model_latency_ms = {
      display_name = "SEC EDGAR LLM End-to-End Model Latency"
      description  = "Per-call Gemini end-to-end model latency (prefill + decode, ms) from llm_telemetry_events sink logs."
      unit         = "ms"
      target       = "llm_telemetry_events"
      extra_filter = ""
      value_field  = "total_model_latency_ms"
      labels       = ["model_name", "agent_name", "task_category"]
      buckets      = local.latency_ms_buckets
    }
    llm_cost_usd = {
      display_name = "SEC EDGAR LLM Cost per Call (USD, approximate)"
      description  = "Per-call estimated Gemini cost in USD (distribution; exact cumulative totals live in BigQuery sec_edgar.v_telemetry_cost_and_latency)."
      unit         = "1"
      target       = "llm_telemetry_events"
      extra_filter = ""
      value_field  = "total_cost_usd"
      labels       = ["model_name", "task_category"]
      buckets      = local.usd_buckets
    }
    llm_cache_savings_usd = {
      display_name = "SEC EDGAR LLM Cached-Token Savings per Call (USD, approximate)"
      description  = "Per-call USD saved by cached-context token discounts (distribution; exact cumulative totals live in BigQuery sec_edgar.v_telemetry_cost_and_latency)."
      unit         = "1"
      target       = "llm_telemetry_events"
      extra_filter = ""
      value_field  = "cached_token_savings_usd"
      labels       = ["model_name", "task_category"]
      buckets      = local.usd_buckets
    }
    evaluation_scores = {
      display_name = "SEC EDGAR Async Judge Evaluation Score"
      description  = "Async Gemini Flash LLM-as-a-Judge overall evaluation score (0.0-1.0); pending (NULL) scores are excluded."
      unit         = "1"
      target       = "session_telemetry_events"
      extra_filter = "jsonPayload.event_payload.execution_mode=\"async_evaluation\" AND jsonPayload.event_payload.evaluation_score>=0"
      value_field  = "evaluation_score"
      labels       = ["intent", "evaluation_grade"]
      buckets      = local.score_buckets
    }
  }

  telemetry_counter_metrics = {
    llm_calls = {
      display_name = "SEC EDGAR LLM Calls"
      description  = "Count of Gemini model calls recorded to llm_telemetry_events."
      target       = "llm_telemetry_events"
      extra_filter = ""
      labels       = ["model_name", "agent_name", "task_category", "cache_hit", "within_ttft_slo"]
    }
    session_turns = {
      display_name = "SEC EDGAR Conversational Turns"
      description  = "Count of interactive conversational turns (async_evaluation judge rows excluded)."
      target       = "session_telemetry_events"
      extra_filter = "jsonPayload.event_payload.execution_mode!=\"async_evaluation\""
      labels       = ["intent", "execution_mode", "status", "fact_check_passed", "report_generated"]
    }
    security_audit_events = {
      display_name = "SEC EDGAR Security Audit Events"
      description  = "Count of security audit events (auth, RBAC denials, guardrail blocks, 10-K access)."
      target       = "security_audit_logs"
      extra_filter = ""
      labels       = ["event_type", "status", "identity_provider"]
    }
  }

  sec_edgar_user_metric_prefix = "logging.googleapis.com/user/sec_edgar"
  cloud_run_metric_filter      = "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${var.cloud_run_service_name}\""
  # PromQL selector for Cloud Run log-based metrics (resource labels are exposed as PromQL labels).
  cloud_run_promql_selector = "monitored_resource=\"cloud_run_revision\",service_name=\"${var.cloud_run_service_name}\""
  api_request_count_promql  = "logging_googleapis_com:user_sec_edgar_api_request_latencies_count"

  chunk_health_dashboard_console_url = "https://console.cloud.google.com/monitoring/dashboards/builder/${element(split("/", google_monitoring_dashboard.chunk_health_dashboard.id), 3)}?project=${var.project_id}"
}

resource "google_logging_metric" "telemetry_distributions" {
  for_each = local.telemetry_distribution_metrics

  project     = var.project_id
  name        = "sec_edgar/${each.key}"
  description = each.value.description
  filter = join(" AND ", compact([
    local.telemetry_sink_base_filter,
    "jsonPayload.log_sink_target=\"${each.value.target}\"",
    each.value.extra_filter,
  ]))

  metric_descriptor {
    metric_kind  = "DELTA"
    value_type   = "DISTRIBUTION"
    unit         = each.value.unit
    display_name = each.value.display_name

    dynamic "labels" {
      for_each = each.value.labels
      content {
        key         = labels.value
        value_type  = "STRING"
        description = local.telemetry_label_descriptions[labels.value]
      }
    }
  }

  value_extractor  = "EXTRACT(jsonPayload.event_payload.${each.value.value_field})"
  label_extractors = { for k in each.value.labels : k => "EXTRACT(jsonPayload.event_payload.${k})" }

  bucket_options {
    dynamic "exponential_buckets" {
      for_each = each.value.buckets.type == "exponential" ? [each.value.buckets] : []
      content {
        num_finite_buckets = exponential_buckets.value.num_finite_buckets
        growth_factor      = exponential_buckets.value.growth_factor
        scale              = exponential_buckets.value.scale
      }
    }
    dynamic "linear_buckets" {
      for_each = each.value.buckets.type == "linear" ? [each.value.buckets] : []
      content {
        num_finite_buckets = linear_buckets.value.num_finite_buckets
        width              = linear_buckets.value.width
        offset             = linear_buckets.value.offset
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_logging_metric" "telemetry_counters" {
  for_each = local.telemetry_counter_metrics

  project     = var.project_id
  name        = "sec_edgar/${each.key}"
  description = each.value.description
  filter = join(" AND ", compact([
    local.telemetry_sink_base_filter,
    "jsonPayload.log_sink_target=\"${each.value.target}\"",
    each.value.extra_filter,
  ]))

  metric_descriptor {
    metric_kind  = "DELTA"
    value_type   = "INT64"
    unit         = "1"
    display_name = each.value.display_name

    dynamic "labels" {
      for_each = each.value.labels
      content {
        key         = labels.value
        value_type  = "STRING"
        description = local.telemetry_label_descriptions[labels.value]
      }
    }
  }

  label_extractors = { for k in each.value.labels : k => "EXTRACT(jsonPayload.event_payload.${k})" }

  depends_on = [google_project_service.required_apis]
}

# ------------------------------------------------------------------------------
# Alert policies (wired to var.monitoring_notification_channels)
# ------------------------------------------------------------------------------

# HTTP 5xx server errors. REDUCE_COUNT on a DISTRIBUTION metric counts time series,
# not requests, so the request count comes from the PromQL `_count` series of
# sec_edgar/api_request_latencies (one sample per access log line).
resource "google_monitoring_alert_policy" "api_5xx_errors" {
  project      = var.project_id
  display_name = "SEC EDGAR API 5xx Server Errors (>= ${var.api_5xx_alert_threshold} / 5 min)"
  combiner     = "OR"
  enabled      = true

  notification_channels = var.monitoring_notification_channels

  documentation {
    content   = "Triggered when `${var.cloud_run_service_name}` returns at least ${var.api_5xx_alert_threshold} HTTP 5xx responses within 5 minutes (log-based metric `sec_edgar/api_request_latencies`, `http_status >= 500`). Open the **SEC EDGAR Analyst — LLM Telemetry, Cost, Sessions & Security Audit** dashboard: the *Operational errors* logs panel lists the failing requests, and each entry's `logging.googleapis.com/trace` links straight to its Cloud Trace waterfall."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Cloud Run HTTP 5xx responses >= ${var.api_5xx_alert_threshold} in 5 min"

    condition_prometheus_query_language {
      query               = "sum(increase(${local.api_request_count_promql}{${local.cloud_run_promql_selector},http_status=~\"5..\"}[5m])) >= ${var.api_5xx_alert_threshold}"
      duration            = "0s"
      evaluation_interval = "60s"
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  user_labels = var.labels

  depends_on = [
    google_project_service.required_apis,
    google_logging_metric.api_request_latencies,
  ]
}

# Elevated authentication denials (401/403) or guardrail blocks (Model Armor, SDP,
# projection filter, fail-closed) - a brute-force / prompt-injection signal.
resource "google_monitoring_alert_policy" "security_audit_blocks" {
  project      = var.project_id
  display_name = "SEC EDGAR Elevated Auth Denials / Guardrail Blocks (> ${var.security_block_alert_threshold} / 5 min)"
  combiner     = "OR"
  enabled      = true

  notification_channels = var.monitoring_notification_channels

  documentation {
    content   = "Triggered when more than ${var.security_block_alert_threshold} security audit events with status `denied_401`, `denied_403`, or `blocked` (incl. `blocked_fail_closed`) are recorded within 5 minutes (log-based metric `sec_edgar/security_audit_events`). Check the *Security audit logs* panel on the LLM telemetry dashboard, then query `sec_edgar.v_security_audit_summary` / `sec_edgar.security_audit_logs` in BigQuery Studio for principals and trace IDs."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Denied (401/403) or blocked audit events > ${var.security_block_alert_threshold} in 5 min"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"${local.sec_edgar_user_metric_prefix}/security_audit_events\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.labels.service_name=\"${var.cloud_run_service_name}\"",
        "metric.labels.status=monitoring.regex.full_match(\"denied_40[13]|blocked.*\")",
      ])
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.security_block_alert_threshold

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  user_labels = var.labels

  depends_on = [
    google_project_service.required_apis,
    google_logging_metric.telemetry_counters,
  ]
}

# ------------------------------------------------------------------------------
# Cloud Monitoring Dashboard — LLM Telemetry, Cost, Sessions & Security Audit
# (replaces the removed LookML observability dashboard; chunk_health_dashboard in
# monitoring.tf is unchanged)
# ------------------------------------------------------------------------------
locals {
  llm_dash_m  = local.sec_edgar_user_metric_prefix
  llm_dash_rf = local.cloud_run_metric_filter
  llm_dash_bq = "${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}"

  llm_dash_logs_query = join("\n", [
    "resource.type=\"cloud_run_revision\"",
    "resource.labels.service_name=\"${var.cloud_run_service_name}\"",
  ])
  llm_dash_logs_url      = "https://console.cloud.google.com/logs/query;query=${replace(urlencode(local.llm_dash_logs_query), "+", "%20")}?project=${var.project_id}"
  llm_dash_trace_url     = "https://console.cloud.google.com/traces/explorer?project=${var.project_id}"
  llm_dash_bq_studio_url = "https://console.cloud.google.com/bigquery?project=${var.project_id}&p=${var.project_id}&d=${google_bigquery_dataset.sec_edgar.dataset_id}&page=dataset"
  llm_dash_errors_url    = "https://console.cloud.google.com/errors?project=${var.project_id}"

  llm_dashboard_nav_markdown = <<-MD
    **Drill-down:** [Cloud Trace Explorer](${local.llm_dash_trace_url}) (filter `service.name: ${var.cloud_run_service_name}`) · [Cloud Logging Explorer](${local.llm_dash_logs_url}) · [Error Reporting](${local.llm_dash_errors_url}) · [Chunk Health Dashboard](${local.chunk_health_dashboard_console_url}) · [BigQuery Studio](${local.llm_dash_bq_studio_url})

    Every log line, BigQuery telemetry row (`trace_id`), and API response (`X-Trace-Id`) shares the same Cloud Trace ID. Top scorecards summarize rolling **7-day** totals/percentiles; charts below are aggregated **hourly** for readability on interactive traffic. Exact cumulative token & USD BigQuery SQL recipes are at the bottom of this dashboard.
  MD

  llm_dashboard_sql_markdown = <<-MD
    **LLM token reporting and exact cumulative USD totals live in BigQuery** (USD charts above come from approximate Cloud Logging distributions; copy into [BigQuery Studio](${local.llm_dash_bq_studio_url})):

    ```sql
    -- Exact daily LLM tokens, cost & cached-token savings
    SELECT telemetry_date, model_name, SUM(call_count) AS calls, SUM(total_context_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, ROUND(SUM(total_cost_usd), 4) AS cost_usd, ROUND(SUM(cached_token_savings_usd), 4) AS savings_usd
    FROM `${local.llm_dash_bq}.v_telemetry_cost_and_latency` WHERE telemetry_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) GROUP BY 1, 2 ORDER BY 1 DESC;
    -- Interactive turns vs async judge scores
    SELECT session_date, intent, SUM(interactive_turn_count) AS turns, ROUND(AVG(avg_turn_latency_ms)) AS avg_latency_ms, SUM(scored_evaluation_count) AS evals, ROUND(AVG(avg_evaluation_score), 3) AS avg_score
    FROM `${local.llm_dash_bq}.v_session_analytics_summary` GROUP BY 1, 2 ORDER BY 1 DESC;
    -- Security audit rollup
    SELECT audit_date, event_type, status, SUM(total_events) AS events, SUM(denied_events) AS denied, SUM(blocked_events) AS blocked, SUM(spiffe_attested_events) AS spiffe_attested
    FROM `${local.llm_dash_bq}.v_security_audit_summary` GROUP BY 1, 2, 3 ORDER BY 1 DESC, events DESC;
    ```
  MD
}

resource "google_monitoring_dashboard" "llm_observability_dashboard" {
  project = var.project_id

  dashboard_json = jsonencode({
    displayName = "SEC EDGAR Analyst — LLM Telemetry, Cost, Sessions & Security Audit"
    labels      = { sec_edgar = "" }
    mosaicLayout = {
      columns = 12
      tiles = [
        # Row 0: Compact Console navigation links (keeps KPI scorecards above the fold)
        {
          xPos   = 0
          yPos   = 0
          width  = 12
          height = 2
          widget = {
            title = "Observability Quick Links"
            text = {
              content = local.llm_dashboard_nav_markdown
              format  = "MARKDOWN"
            }
          }
        },

        # Row 1: Executive KPI Scorecards — Traffic, Chat Latency, TTFT & Judge Quality (7d rolling)
        {
          xPos   = 0
          yPos   = 2
          width  = 3
          height = 3
          widget = {
            title = "Chat Turns (7d)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_session_turns{${local.cloud_run_promql_selector}}[7d])) or vector(0)"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
            }
          }
        },
        {
          xPos   = 3
          yPos   = 2
          width  = 3
          height = 3
          widget = {
            title = "Chat API p95 Latency (7d)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "histogram_quantile(0.95, sum by (le) (increase(logging_googleapis_com:user_sec_edgar_api_request_latencies_bucket{${local.cloud_run_promql_selector},http_route=~\"/api/v1/chat(/stream)?\"}[7d]))) / 1000"
                unitOverride    = "s"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
              thresholds = [
                {
                  label     = "Slow Chat"
                  value     = 30.0
                  color     = "YELLOW"
                  direction = "ABOVE"
                },
                {
                  label     = "Critical Chat Latency"
                  value     = 60.0
                  color     = "RED"
                  direction = "ABOVE"
                }
              ]
            }
          }
        },
        {
          xPos   = 6
          yPos   = 2
          width  = 3
          height = 3
          widget = {
            title = "LLM TTFT p95 (7d)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "histogram_quantile(0.95, sum by (le) (increase(logging_googleapis_com:user_sec_edgar_llm_ttft_ms_bucket{${local.cloud_run_promql_selector}}[7d]))) / 1000"
                unitOverride    = "s"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
              thresholds = [
                {
                  label     = "Above TTFT Target"
                  value     = 3.0
                  color     = "YELLOW"
                  direction = "ABOVE"
                },
                {
                  label     = "High TTFT"
                  value     = 8.0
                  color     = "RED"
                  direction = "ABOVE"
                }
              ]
            }
          }
        },
        {
          xPos   = 9
          yPos   = 2
          width  = 3
          height = 3
          widget = {
            title = "Async Judge Mean Score (7d, 0-1)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_evaluation_scores_sum{${local.cloud_run_promql_selector}}[7d])) / clamp_min(sum(increase(logging_googleapis_com:user_sec_edgar_evaluation_scores_count{${local.cloud_run_promql_selector}}[7d])), 1)"
              }
              gaugeView = {
                lowerBound = 0.0
                upperBound = 1.0
              }
              thresholds = [
                {
                  label     = "Low Quality"
                  value     = 0.70
                  color     = "RED"
                  direction = "BELOW"
                },
                {
                  label     = "Below Target"
                  value     = 0.85
                  color     = "YELLOW"
                  direction = "BELOW"
                }
              ]
            }
          }
        },

        # Row 2: Executive KPI Scorecards — Cache Efficiency, Cost & Security (7d rolling)
        {
          xPos   = 0
          yPos   = 5
          width  = 3
          height = 3
          widget = {
            title = "LLM Cache Hit Rate (7d)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "100 * (sum(increase(logging_googleapis_com:user_sec_edgar_llm_calls{${local.cloud_run_promql_selector},cache_hit=\"true\"}[7d])) or vector(0)) / clamp_min(sum(increase(logging_googleapis_com:user_sec_edgar_llm_calls{${local.cloud_run_promql_selector}}[7d])), 1)"
                unitOverride    = "%"
              }
              gaugeView = {
                lowerBound = 0.0
                upperBound = 100.0
              }
            }
          }
        },
        {
          xPos   = 3
          yPos   = 5
          width  = 3
          height = 3
          widget = {
            title = "Est. LLM Spend (7d, USD)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_llm_cost_usd_sum{${local.cloud_run_promql_selector}}[7d])) or vector(0)"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 5
          width  = 3
          height = 3
          widget = {
            title = "Est. Cache Savings (7d, USD)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_llm_cache_savings_usd_sum{${local.cloud_run_promql_selector}}[7d])) or vector(0)"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
            }
          }
        },
        {
          xPos   = 9
          yPos   = 5
          width  = 3
          height = 3
          widget = {
            title = "Auth Denials & Guardrail Blocks (7d)"
            scorecard = {
              timeSeriesQuery = {
                prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_security_audit_events{${local.cloud_run_promql_selector},status=~\"denied_40[13]|blocked.*\"}[7d])) or vector(0)"
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
              thresholds = [
                {
                  label     = "Elevated Blocks"
                  value     = var.security_block_alert_threshold
                  color     = "RED"
                  direction = "ABOVE"
                }
              ]
            }
          }
        },

        # Row 3: API traffic by route & status class (hourly counts) + Chat-only latency
        {
          xPos   = 0
          yPos   = 8
          width  = 4
          height = 4
          widget = {
            title = "API Requests by Route (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "{{http_route}}"
                timeSeriesQuery = {
                  prometheusQuery = "sum by (http_route) (increase(${local.api_request_count_promql}{${local.cloud_run_promql_selector}}[1h]))"
                }
              }]
              yAxis = { label = "Requests / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 4
          yPos   = 8
          width  = 4
          height = 4
          widget = {
            title = "API Responses by Status Class (2xx / 4xx / 5xx per hour)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "STACKED_BAR"
                  legendTemplate = "2xx"
                  timeSeriesQuery = {
                    prometheusQuery = "sum(increase(${local.api_request_count_promql}{${local.cloud_run_promql_selector},http_status=~\"2..\"}[1h]))"
                  }
                },
                {
                  plotType       = "STACKED_BAR"
                  legendTemplate = "4xx"
                  timeSeriesQuery = {
                    prometheusQuery = "sum(increase(${local.api_request_count_promql}{${local.cloud_run_promql_selector},http_status=~\"4..\"}[1h]))"
                  }
                },
                {
                  plotType       = "STACKED_BAR"
                  legendTemplate = "5xx"
                  timeSeriesQuery = {
                    prometheusQuery = "sum(increase(${local.api_request_count_promql}{${local.cloud_run_promql_selector},http_status=~\"5..\"}[1h]))"
                  }
                }
              ]
              yAxis = { label = "Responses / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 8
          yPos   = 8
          width  = 4
          height = 4
          widget = {
            title = "Chat API Latency p50 / p95 (/api/v1/chat*, seconds)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "p50 (chat)"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.50, sum by (le) (increase(logging_googleapis_com:user_sec_edgar_api_request_latencies_bucket{${local.cloud_run_promql_selector},http_route=~\"/api/v1/chat(/stream)?\"}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                },
                {
                  plotType       = "LINE"
                  legendTemplate = "p95 (chat)"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.95, sum by (le) (increase(logging_googleapis_com:user_sec_edgar_api_request_latencies_bucket{${local.cloud_run_promql_selector},http_route=~\"/api/v1/chat(/stream)?\"}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                }
              ]
              yAxis = { label = "Latency (s)", scale = "LINEAR" }
              thresholds = [{
                label = "p95 Alert Threshold (30s)"
                value = var.api_latency_p95_alert_threshold_ms / 1000.0
              }]
            }
          }
        },

        # Row 4: LLM TTFT and end-to-end model latency (hourly quantiles in seconds, grouped by task)
        {
          xPos   = 0
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "LLM Time-to-First-Token p50 / p95 by Task (seconds)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "p50 {{task_category}}"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.50, sum by (le, task_category) (increase(logging_googleapis_com:user_sec_edgar_llm_ttft_ms_bucket{${local.cloud_run_promql_selector}}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                },
                {
                  plotType       = "LINE"
                  legendTemplate = "p95 {{task_category}}"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.95, sum by (le, task_category) (increase(logging_googleapis_com:user_sec_edgar_llm_ttft_ms_bucket{${local.cloud_run_promql_selector}}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                }
              ]
              yAxis = { label = "TTFT (s)", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "LLM End-to-End Model Latency p50 / p95 by Task (seconds)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "p50 {{task_category}}"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.50, sum by (le, task_category) (increase(logging_googleapis_com:user_sec_edgar_llm_model_latency_ms_bucket{${local.cloud_run_promql_selector}}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                },
                {
                  plotType       = "LINE"
                  legendTemplate = "p95 {{task_category}}"
                  timeSeriesQuery = {
                    prometheusQuery = "histogram_quantile(0.95, sum by (le, task_category) (increase(logging_googleapis_com:user_sec_edgar_llm_model_latency_ms_bucket{${local.cloud_run_promql_selector}}[1h]))) / 1000"
                    unitOverride    = "s"
                  }
                }
              ]
              yAxis = { label = "Latency (s)", scale = "LINEAR" }
            }
          }
        },

        # Row 5: LLM call volume, cache hit breakdown, TTFT SLO compliance (hourly buckets)
        {
          xPos   = 0
          yPos   = 16
          width  = 4
          height = 4
          widget = {
            title = "LLM Calls by Model (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "$${metric.label.model_name}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/llm_calls\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.model_name"]
                    }
                  }
                }
              }]
              yAxis = { label = "Calls / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 4
          yPos   = 16
          width  = 4
          height = 4
          widget = {
            title = "LLM Context Cache Hits vs Misses (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "cache_hit=$${metric.label.cache_hit}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/llm_calls\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.cache_hit"]
                    }
                  }
                }
              }]
              yAxis = { label = "Calls / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 8
          yPos   = 16
          width  = 4
          height = 4
          widget = {
            title = "LLM Calls Within TTFT SLO (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "within_ttft_slo=$${metric.label.within_ttft_slo}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/llm_calls\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.within_ttft_slo"]
                    }
                  }
                }
              }]
              yAxis = { label = "Calls / hr", scale = "LINEAR" }
            }
          }
        },

        # Row 6: USD spend vs cache savings + per-call cost by model
        {
          xPos   = 0
          yPos   = 20
          width  = 6
          height = 4
          widget = {
            title = "Estimated LLM Spend vs Cache Savings (USD / hour)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "Est. Spend (USD/hr)"
                  timeSeriesQuery = {
                    prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_llm_cost_usd_sum{${local.cloud_run_promql_selector}}[1h]))"
                  }
                },
                {
                  plotType       = "LINE"
                  legendTemplate = "Cache Savings (USD/hr)"
                  timeSeriesQuery = {
                    prometheusQuery = "sum(increase(logging_googleapis_com:user_sec_edgar_llm_cache_savings_usd_sum{${local.cloud_run_promql_selector}}[1h]))"
                  }
                }
              ]
              yAxis = { label = "USD / hour", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 20
          width  = 6
          height = 4
          widget = {
            title = "LLM Cost per Call by Model (Mean / p95, USD)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "Mean ($${metric.label.model_name})"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"${local.llm_dash_m}/llm_cost_usd\" ${local.llm_dash_rf}"
                      aggregation = {
                        alignmentPeriod    = "3600s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_MEAN"
                        groupByFields      = ["metric.label.model_name"]
                      }
                    }
                  }
                },
                {
                  plotType       = "LINE"
                  legendTemplate = "p95 ($${metric.label.model_name})"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"${local.llm_dash_m}/llm_cost_usd\" ${local.llm_dash_rf}"
                      aggregation = {
                        alignmentPeriod    = "3600s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                        groupByFields      = ["metric.label.model_name"]
                      }
                    }
                  }
                }
              ]
              yAxis = { label = "USD / call", scale = "LINEAR" }
            }
          }
        },

        # Row 7: Conversational turns and async judge quality (exact mean <= 1.0)
        {
          xPos   = 0
          yPos   = 24
          width  = 4
          height = 4
          widget = {
            title = "Interactive Turns by Intent (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "$${metric.label.intent}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/session_turns\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.intent"]
                    }
                  }
                }
              }]
              yAxis = { label = "Turns / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 4
          yPos   = 24
          width  = 4
          height = 4
          widget = {
            title = "Claim Verification Outcomes (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "fact_check_passed=$${metric.label.fact_check_passed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/session_turns\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.fact_check_passed"]
                    }
                  }
                }
              }]
              yAxis = { label = "Turns / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 8
          yPos   = 24
          width  = 4
          height = 4
          widget = {
            title = "Async Judge Mean Score by Intent (0.0 - 1.0)"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                legendTemplate = "$${metric.label.intent}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/evaluation_scores\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_MEAN"
                      groupByFields      = ["metric.label.intent"]
                    }
                  }
                }
              }]
              yAxis = { label = "Mean Score (0-1)", scale = "LINEAR" }
              thresholds = [{
                label = "Target Quality (0.85)"
                value = 0.85
              }]
            }
          }
        },

        # Row 8: Security audit events (all types vs denials/blocks only with matching threshold)
        {
          xPos   = 0
          yPos   = 28
          width  = 6
          height = 4
          widget = {
            title = "Security Audit Events by Event Type (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "$${metric.label.event_type}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/security_audit_events\" ${local.llm_dash_rf}"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.event_type"]
                    }
                  }
                }
              }]
              yAxis = { label = "Events / hr", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 28
          width  = 6
          height = 4
          widget = {
            title = "Auth Denials & Guardrail Blocks by Status (per hour)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "$${metric.label.status}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.llm_dash_m}/security_audit_events\" ${local.llm_dash_rf} metric.labels.status=monitoring.regex.full_match(\"denied_40[13]|blocked.*\")"
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.status"]
                    }
                  }
                }
              }]
              yAxis = { label = "Denials / Blocks / hr", scale = "LINEAR" }
              thresholds = [{
                label = "5-min Block Alert Threshold"
                value = var.security_block_alert_threshold
              }]
            }
          }
        },

        # Row 9: Embedded Cloud Logging panels
        {
          xPos   = 0
          yPos   = 32
          width  = 4
          height = 6
          widget = {
            title = "Operational Errors (severity >= ERROR on Cloud Run)"
            logsPanel = {
              filter        = "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${var.cloud_run_service_name}\" severity>=ERROR"
              resourceNames = ["projects/${var.project_id}"]
            }
          }
        },
        {
          xPos   = 4
          yPos   = 32
          width  = 4
          height = 6
          widget = {
            title = "Security Audit Logs (security_audit_logs sink)"
            logsPanel = {
              filter        = "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${var.cloud_run_service_name}\" jsonPayload.component=\"sec-edgar-telemetry-sink\" jsonPayload.log_sink_target=\"security_audit_logs\""
              resourceNames = ["projects/${var.project_id}"]
            }
          }
        },
        {
          xPos   = 8
          yPos   = 32
          width  = 4
          height = 6
          widget = {
            title = "LLM Telemetry Sink Logs (llm_telemetry_events sink)"
            logsPanel = {
              filter        = "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${var.cloud_run_service_name}\" jsonPayload.component=\"sec-edgar-telemetry-sink\" jsonPayload.log_sink_target=\"llm_telemetry_events\""
              resourceNames = ["projects/${var.project_id}"]
            }
          }
        },

        # Row 10: Cloud Logging -> BigQuery export health (sec-edgar-telemetry-audit-bq-sink)
        {
          xPos   = 0
          yPos   = 38
          width  = 12
          height = 3
          widget = {
            title = "Cloud Logging → BigQuery Sink Export Errors (should stay at 0)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                legendTemplate = "$${resource.label.name}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/exports/error_count\" resource.type=\"logging_sink\""
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["resource.label.name"]
                    }
                  }
                }
              }]
              yAxis = { label = "Errors / hr", scale = "LINEAR" }
            }
          }
        },

        # Row 11: BigQuery Studio SQL recipes (below charts so KPIs sit above the fold)
        {
          xPos   = 0
          yPos   = 41
          width  = 12
          height = 5
          widget = {
            title = "BigQuery Studio SQL Recipes (Exact Cumulative Tokens, Spend & Audit Rollups)"
            text = {
              content = local.llm_dashboard_sql_markdown
              format  = "MARKDOWN"
            }
          }
        }
      ]
    }
  })

  depends_on = [
    google_logging_metric.api_request_latencies,
    google_logging_metric.telemetry_distributions,
    google_logging_metric.telemetry_counters,
  ]
}
