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
