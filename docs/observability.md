# Observability: End-to-End Latency Tracing & Structured Logs

The FastAPI service exports **OpenTelemetry spans to Cloud Trace** and writes **structured JSON logs carrying trace IDs** to stdout. Cloud Run ingests those logs into **Cloud Logging**. A single `/api/v1/chat` request produces one trace covering the Cloud Run front end (or the HTTPS load balancer in `load_balancer` access mode), FastAPI, guardrails, ReAct steps, tools, ADK model calls, and outbound GCP REST calls. That request's log lines appear grouped under the trace, and its BigQuery telemetry rows carry the same `trace_id`.

Code lives in [`src/edgar_capstone/observability/`](../src/edgar_capstone/observability):

| Module | Responsibility |
| :-- | :-- |
| `tracing.py` | `TracerProvider` + `CloudTraceSpanExporter`, GCP resource detection, sampling, propagators (`traceparent` + `X-Cloud-Trace-Context`), FastAPI/httpx auto-instrumentation, `@traced` / `start_span` helpers |
| `structured_logging.py` | `CloudLoggingJsonFormatter` (Cloud Logging special fields + Error Reporting), local trace-aware text formatter, uvicorn log routing |
| `middleware.py` | `RequestTelemetryMiddleware`: request log context, `X-Trace-Id` / `X-Request-Id` / `Server-Timing` response headers, one `http_request_completed` access log per request |

## Trace anatomy

```mermaid
flowchart TD
    LB["Cloud Run front end / optional HTTPS LB (traceparent, X-Cloud-Trace-Context)"] --> S["FastAPI server span: POST /api/v1/chat"]
    S --> A["auth.verify_jwt"]
    S --> T["chat.rag_turn"]
    T --> G["guardrail.inspect_user_prompt (+ Model Armor / SDP HTTP spans)"]
    T --> R["chat.resolve_scope / orchestrator.classify_intent"]
    T --> V["tool.calculate_period_variances"]
    T --> AN["anomaly.attach_explanations"]
    T --> C["chat.retrieve_citations / tool.search_10k_chunks_rag (+ BigQuery, Vertex AI Search HTTP spans)"]
    T --> X["react.execute"]
    X --> E["react.evaluate_variances"]
    X --> Q["react.query_cached_10k / tool.query_cached_10k_context"]
    X --> M["sector.map_reduce (for GICS sector thematic queries)"]
    X --> L["adk.run_agent_stream -> ADK invoke_agent / call_llm / execute_tool (+ Vertex AI Gemini HTTP spans)"]
    X -.->|Async off critical path| J["evaluation.narrative_judge"]
    T --> B["BigQuery insertAll HTTP spans (async pool -> telemetry + audit sinks)"]
    T --> FS["Firestore REST spans (analyst_sessions / analyst_feedback)"]
```

- **Server span:** `opentelemetry-instrumentation-fastapi`. It continues the upstream W3C `traceparent` (or the legacy `X-Cloud-Trace-Context`), so Cloud Run's own request span and the app spans share one trace. Health probes and static assets are excluded (`EDGAR_TRACE_EXCLUDED_URLS`).
- **App spans:** `@traced(...)` on the chat engine (`chat.rag_turn`, `chat.resolve_scope`, `chat.retrieve_citations`), intent router (`orchestrator.classify_intent`), anomaly explanation linker (`anomaly.attach_explanations`), ReAct steps (`react.execute`, `react.evaluate_variances`, `react.query_cached_10k`), sector map-reduce (`sector.map_reduce`), ADK runner (`adk.run_agent_stream`), Model Armor guardrail, JWT verification, and narrative evaluator. `with_tool_resilience` wraps every ADK `FunctionTool` in a `tool.<name>` span (returning structured error dicts on failure), while `retry_transient_errors` adds retry spans for propagating internal calls (such as `prompt_store.py`).
- **Model calls:** ADK's built-in `gcp.vertex.agent` spans (`invoke_agent`, `call_llm`, `execute_tool`) attach to the same global provider. Every recorded model call also adds an `llm.call` span event with `gen_ai.request.model`, input/output/cached/thinking tokens, `edgar.ttft_ms`, latency, and cost.
- **Outbound HTTP:** `opentelemetry-instrumentation-httpx` creates client spans for Vertex AI (google-genai), BigQuery, Firestore, Model Armor, SDP, Secret Manager, and IAM Credentials, and propagates context on them.
- **Thread hops:** `_run_coro_sync` (persistent background ADK `asyncio` event loop), `POST /api/v1/chat/stream` SSE worker threads, `sector.map_reduce` batch pools, and the off-critical-path background Flash judge pool all copy `contextvars` (`contextvars.copy_context().run` / `loop.create_task(coro, context=ctx)`), so live ADK runs, map-reduce batches, and background evaluations stay correlated under the parent request trace.
- **Error & mid-stream access logging:** `RequestTelemetryMiddleware` emits `http_request_completed` inside a `finally` block so latency metrics and trace-linked access logs are recorded even when an unhandled exception raises HTTP `500`. If an exception occurs mid-stream after headers are already flushed (e.g., SSE on `/api/v1/chat/stream`), the original HTTP status code is preserved and the log sets `unhandled_exception: true` and `stream_aborted: true` (alongside an `http_request_failed` exception log at `ERROR`).
- **Shutdown flushing:** FastAPI's lifespan shutdown hook flushes pending background judge futures (`flush_pending_evaluations`), async BigQuery streaming inserts (`flush_async_writes`), and OpenTelemetry trace spans (`flush_tracing`).

## Structured log format

Example `http_request_completed` access log line as written to stdout (`EDGAR_LOG_FORMAT=json`):

```json
{
  "severity": "INFO",
  "message": "POST /api/v1/chat -> 200 (704.8 ms)",
  "time": "2026-11-16T20:04:41.603000+00:00",
  "logger": "edgar_capstone.access",
  "logging.googleapis.com/sourceLocation": {"file": ".../middleware.py", "line": "178", "function": "_log_access"},
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00d73528256086a3",
  "logging.googleapis.com/trace": "projects/iron-radio-514217-q7/traces/4bf92f3577b34da6a3ce929d0e0e4736",
  "logging.googleapis.com/spanId": "00d73528256086a3",
  "logging.googleapis.com/trace_sampled": true,
  "request": {"request_id": "4bf92f3577b34da6a3ce929d0e0e4736", "method": "POST", "path": "/api/v1/chat"},
  "event": "http_request_completed",
  "http_method": "POST",
  "http_path": "/api/v1/chat",
  "http_route": "/api/v1/chat",
  "http_status": 200,
  "latency_ms": 704.77,
  "unhandled_exception": false,
  "stream_aborted": false,
  "user_agent": "Mozilla/5.0 ...",
  "remote_ip": "10.128.0.12"
}
```

- Every logger in the process is covered: app modules, `uvicorn` / `uvicorn.error`, and third-party libraries. They all go through one root handler. Uvicorn's own access log is turned off because the middleware's access log replaces it with trace IDs and route templates.
- To add structured fields, use `logger.info("msg", extra={"json_fields": {...}})`. Other non-standard `extra=` keys are included too.
- `logger.exception(...)` at ERROR or above adds the Error Reporting `@type` and `serviceContext`, so stack traces appear in Cloud Error Reporting.
- Telemetry sink records (`edgar_capstone.cloud_logging_sink`) now write `component`, `log_sink_target`, and `event_payload` as top-level `jsonPayload` fields. The existing `google_logging_project_sink` filter (`jsonPayload.component="sec-edgar-telemetry-sink"`) matches them.
- Locally (`auto` without `K_SERVICE`), logs are readable text that still includes `[trace=… span=…]`.

## Correlation surfaces

| Where | What |
| :-- | :-- |
| Response headers | `X-Trace-Id`, `X-Request-Id`, `Server-Timing: app;dur=…` (exposed through CORS for the Vite dev UI) |
| `/api/v1/chat` body | `observability.trace_id`, `span_id`, `cloud_trace_resource`, and a `cloud_trace_url` console deep link |
| BigQuery | New nullable `trace_id` column in `llm_telemetry_events`, `session_telemetry_events`, and `security_audit_logs` (Terraform schemas, sink rows, and LookML `trace_id` dimensions linking to Cloud Trace) |
| Cloud Monitoring | Log-based distribution metric `logging.googleapis.com/user/sec_edgar/api_request_latencies` (per `http_route` / `http_method` / `http_status`, with trace exemplars), `api_latency_p95` alert policy (`30,000 ms` default threshold) in `terraform/observability.tf`, and Row 5 `API Request Latency Distribution (p50 / p95 / p99 ms)` widget on `chunk_health_dashboard` (`terraform/monitoring.tf`) |
| `/health` | `observability.tracing` (exporter, sampling, instrumentation state) and `observability.logging` |

## Configuration

| Variable | Default | Purpose |
| :-- | :-- | :-- |
| `EDGAR_TRACING_ENABLED` | `true` | Master switch (`OTEL_SDK_DISABLED=true` also disables tracing) |
| `EDGAR_TRACE_EXPORTER` | `auto` | `auto` (Cloud Trace on Cloud Run, in-process only elsewhere) \| `gcp` \| `console` \| `otlp` \| `none` |
| `EDGAR_TRACE_SAMPLE_RATIO` | `1.0` | Head sampling ratio. A sampled upstream parent is always honored. |
| `EDGAR_TRACE_CAPTURE_CONTENT` | `false` | Default for ADK `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`. Keeps prompts, 10-K text, and model output out of Cloud Trace. |
| `EDGAR_TRACE_EXCLUDED_URLS` | health + static assets | Comma-separated regexes that get no server span |
| `EDGAR_TRACE_INSTRUMENT_HTTPX` | `true` | Outbound httpx client spans |
| `EDGAR_LOG_FORMAT` | `auto` | `json` on Cloud Run, `text` locally |
| `EDGAR_LOG_LEVEL` | `INFO` | Root log level |
| `OTEL_SERVICE_NAME` / `K_SERVICE`, `COMMIT_SHA` | — | `service.name` / `service.version` resource attributes |

IAM and APIs were already in place: `cloudtrace.googleapis.com` and `logging.googleapis.com` are enabled in `terraform/apis.tf`, and `sec-analyst-sa` holds `roles/cloudtrace.agent` and `roles/logging.logWriter` in `terraform/iam.tf`. No Cloud Build or GitHub Actions env changes are needed, because `auto` detects Cloud Run.

> [!TIP]
> Spans are exported in batches on a background thread. With Cloud Run's default request-based CPU allocation, an idle instance may hold spans until the next request or shutdown, when the lifespan hook flushes them. If you need spans to show up within seconds even under low traffic, deploy with `--no-cpu-throttling` (instance-based billing).

## Local usage

```bash
# Spans printed to stdout, JSON logs locally
EDGAR_TRACE_EXPORTER=console EDGAR_LOG_FORMAT=json make backend

# Send local spans to Cloud Trace with ADC
gcloud auth application-default login
EDGAR_TRACE_EXPORTER=gcp GOOGLE_CLOUD_PROJECT=iron-radio-514217-q7 make backend

curl -si -X POST localhost:8000/api/v1/chat -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query":"Compare NVDA and AAPL revenue 2024-2026"}' \
  | grep -i -E 'x-trace-id|server-timing'
```

## Finding a request in GCP

- **Cloud Trace:** Trace Explorer, filter on `service.name: sec-edgar-analyst` or paste the `X-Trace-Id`.
- **Cloud Logging:** `trace="projects/iron-radio-514217-q7/traces/<TRACE_ID>"`, or `jsonPayload.event="http_request_completed" AND jsonPayload.latency_ms > 10000` to find slow requests.
- **BigQuery:** `SELECT * FROM sec_edgar.llm_telemetry_events WHERE trace_id = '<TRACE_ID>'`.

Tests: [`tests/test_observability.py`](../tests/test_observability.py) checks the span hierarchy with an in-memory exporter, upstream `traceparent` / `X-Cloud-Trace-Context` propagation, response headers, the JSON log format and trace fields, uvicorn routing, and `trace_id` on telemetry rows.
