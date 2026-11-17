# ------------------------------------------------------------------------------
# End-to-end latency observability: Cloud Trace + structured Cloud Logging
#
# The FastAPI service exports OpenTelemetry spans to Cloud Trace (runtime SA has
# roles/cloudtrace.agent in iam.tf; cloudtrace.googleapis.com is enabled in
# apis.tf) and writes one structured `http_request_completed` JSON access log per
# request to stdout -> Cloud Logging, carrying `logging.googleapis.com/trace`.
#
# This log-based distribution metric turns those access logs into a per-route
# latency histogram (p50/p95/p99 in Metrics Explorer / dashboards / alerting).
# Each point keeps trace exemplars, so a latency spike links straight to the
# trace that caused it.
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
