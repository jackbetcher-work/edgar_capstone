"""End-to-end OpenTelemetry distributed tracing exported to Google Cloud Trace.

A single request flows through these layers, and each one becomes a span in the
same trace, so Cloud Trace shows where the latency goes and populates the
**Inputs/Outputs** tab for LLM generations:

    Cloud Run / HTTPS LB front end (`traceparent` / `X-Cloud-Trace-Context`)
      -> FastAPI server span            (opentelemetry-instrumentation-fastapi)
        -> app spans                    (`@traced` / `start_span`: guardrails, scope
                                         resolution, RAG retrieval, ReAct loop, tools)
          -> Google ADK agent/LLM/tool spans (`gcp.vertex.agent` tracer, global provider)
          -> Google GenAI SDK spans & events (opentelemetry-instrumentation-google-genai:
                                              `gen_ai.input.messages`, `gen_ai.output.messages`,
                                              `gen_ai.client.inference.operation.details`)
          -> outbound HTTP client spans (opentelemetry-instrumentation-httpx: Vertex AI
                                         Gemini, BigQuery, Firestore, Model Armor, ...)

On Cloud Run (or when `EDGAR_TRACE_EXPORTER=gcp`), spans and GenAI OTel log events
are exported over OTLP/HTTP to Google's Telemetry API (`telemetry.googleapis.com/v1/traces`
and `telemetry.googleapis.com/v1/logs`) via ADK's `google_cloud.get_gcp_exporters(...)`
with Vertex AI Agent Engine auto-configuration (`GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY`),
avoiding Cloud Trace v2's 256-byte attribute truncation so Cloud Trace Explorer's
**Inputs/Outputs** tab renders full prompts, system instructions, tool calls, and outputs.

Runtime configuration (all optional):

* `EDGAR_TRACING_ENABLED` (default `true`; `OTEL_SDK_DISABLED=true` also turns it off).
* `EDGAR_TRACE_EXPORTER` = `auto` | `gcp` | `console` | `otlp` | `none`. `auto` exports to
  Google Cloud (`telemetry.googleapis.com`) on Cloud Run (`K_SERVICE` is set) and keeps spans
  in-process locally, so trace IDs still show up in logs and response headers without needing
  credentials.
* `EDGAR_TRACE_SAMPLE_RATIO` (default `1.0`). A sampled upstream parent (e.g. the Cloud
  Run front end) is always honoured; otherwise this ratio decides.
* `OTEL_SEMCONV_STABILITY_OPT_IN` (default `gen_ai_latest_experimental`). Enables OTel GenAI
  experimental semantic conventions (`gen_ai.input.messages` / `gen_ai.output.messages`).
* `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (default `SPAN_AND_EVENT`, or
  `NO_CONTENT` when `EDGAR_TRACE_CAPTURE_CONTENT=false`). Controls prompt/response capture
  for Cloud Trace Explorer's Inputs/Outputs tab.
* `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT` (default `true` when content capture is enabled).
  Emits `gen_ai.client.inference.operation.details` OTel log events.
* `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` (default `true` when content capture is enabled).
* `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY` (default `true`). Enables Vertex AI Agent
  Engine OTel telemetry headers and auto-configuration.
* `EDGAR_TRACE_EXCLUDED_URLS`: comma-separated regexes of URLs that get no server span
  (health probes and static assets by default).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from opentelemetry import _logs, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from edgar_capstone.common.env import env_bool
from edgar_capstone.gcp.defaults import DEFAULT_PROJECT_ID, resolve_gcp_project_id

logger = logging.getLogger(__name__)

TRACER_NAME = "edgar_capstone"
DEFAULT_SERVICE_NAME = "sec-edgar-analyst"
DEFAULT_SERVICE_NAMESPACE = "sec-edgar"
DEFAULT_EXCLUDED_URLS = r"/health$,/healthz$,/assets/,/favicon\.ico$,/vite\.svg$"
# Fallback alias retained for backward compatibility; canonical default lives in gcp.defaults.
DEFAULT_TRACE_PROJECT_ID = DEFAULT_PROJECT_ID
ADK_CAPTURE_CONTENT_ENV = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"
OTEL_SEMCONV_STABILITY_OPT_IN_ENV = "OTEL_SEMCONV_STABILITY_OPT_IN"
OTEL_GENAI_CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
OTEL_GENAI_EMIT_EVENT_ENV = "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"
AGENT_ENGINE_ENABLE_TELEMETRY_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
DEFAULT_SEMCONV_STABILITY_OPT_IN = "gen_ai_latest_experimental"
DEFAULT_CAPTURE_CONTENT_MODE = "SPAN_AND_EVENT"

TraceExporterMode = Literal["auto", "gcp", "console", "otlp", "none"]
_VALID_EXPORTERS: frozenset[str] = frozenset({"auto", "gcp", "console", "otlp", "none"})
_VALID_CAPTURE_MODES: frozenset[str] = frozenset(
    {"NO_CONTENT", "EVENT_ONLY", "SPAN_ONLY", "SPAN_AND_EVENT"}
)
_MAX_ATTR_STR_LEN = 512

_TRACEPARENT_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_CLOUD_TRACE_CONTEXT_RE = re.compile(r"^([0-9a-fA-F]{32})(?:/(\d+))?(?:;o=([01]))?$")

_F = TypeVar("_F", bound=Callable[..., Any])


def _env_ratio(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.2f", name, raw, default)
        return default
    return min(1.0, max(0.0, value))


def running_on_cloud_run() -> bool:
    return bool(os.environ.get("K_SERVICE", "").strip())


def _resolve_capture_content_mode() -> str:
    raw = os.environ.get(OTEL_GENAI_CAPTURE_CONTENT_ENV, "").strip().upper()
    if raw:
        if raw in _VALID_CAPTURE_MODES:
            return raw
        logger.warning(
            "Ignoring invalid %s=%r (valid: %s); falling back to %s",
            OTEL_GENAI_CAPTURE_CONTENT_ENV,
            raw,
            ", ".join(sorted(_VALID_CAPTURE_MODES)),
            DEFAULT_CAPTURE_CONTENT_MODE,
        )
    if not env_bool("EDGAR_TRACE_CAPTURE_CONTENT", True):
        return "NO_CONTENT"
    return DEFAULT_CAPTURE_CONTENT_MODE


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Resolved OpenTelemetry tracing settings."""

    enabled: bool = True
    exporter: TraceExporterMode = "auto"
    resolved_exporter: TraceExporterMode = "none"
    sample_ratio: float = 1.0
    service_name: str = DEFAULT_SERVICE_NAME
    service_version: str = "unknown"
    project_id: str = DEFAULT_TRACE_PROJECT_ID
    environment: str = "local"
    capture_message_content: bool = True
    capture_message_content_mode: str = DEFAULT_CAPTURE_CONTENT_MODE
    semconv_stability_opt_in: str = DEFAULT_SEMCONV_STABILITY_OPT_IN
    emit_genai_events: bool = True
    agent_engine_telemetry_enabled: bool = True
    excluded_urls: str = DEFAULT_EXCLUDED_URLS
    instrument_httpx: bool = True
    instrument_requests: bool = True
    instrument_grpc: bool = True
    instrument_genai: bool = True

    @classmethod
    def from_env(cls) -> TracingConfig:
        enabled = env_bool("EDGAR_TRACING_ENABLED", True) and not env_bool(
            "OTEL_SDK_DISABLED", False
        )
        raw_exporter = os.environ.get("EDGAR_TRACE_EXPORTER", "auto").strip().lower() or "auto"
        if raw_exporter not in _VALID_EXPORTERS:
            logger.warning("Unknown EDGAR_TRACE_EXPORTER=%r; falling back to 'auto'", raw_exporter)
            raw_exporter = "auto"
        exporter: TraceExporterMode = raw_exporter  # type: ignore[assignment]
        resolved: TraceExporterMode = exporter
        on_cloud_run = running_on_cloud_run()
        if exporter == "auto":
            resolved = "gcp" if on_cloud_run else "none"

        capture_mode = _resolve_capture_content_mode()
        capture_enabled = capture_mode != "NO_CONTENT"
        semconv_opt_in = (
            os.environ.get(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, "").strip()
            or DEFAULT_SEMCONV_STABILITY_OPT_IN
        )
        emit_events_default = capture_mode in ("EVENT_ONLY", "SPAN_AND_EVENT")

        return cls(
            enabled=enabled,
            exporter=exporter,
            resolved_exporter=resolved,
            sample_ratio=_env_ratio("EDGAR_TRACE_SAMPLE_RATIO", 1.0),
            service_name=(
                os.environ.get("OTEL_SERVICE_NAME")
                or os.environ.get("K_SERVICE")
                or DEFAULT_SERVICE_NAME
            ).strip(),
            service_version=(
                os.environ.get("COMMIT_SHA") or os.environ.get("K_REVISION") or "unknown"
            ).strip(),
            project_id=resolve_gcp_project_id(),
            environment=(
                os.environ.get("EDGAR_ENVIRONMENT") or ("cloud_run" if on_cloud_run else "local")
            ).strip(),
            capture_message_content=capture_enabled,
            capture_message_content_mode=capture_mode,
            semconv_stability_opt_in=semconv_opt_in,
            emit_genai_events=env_bool(OTEL_GENAI_EMIT_EVENT_ENV, emit_events_default),
            agent_engine_telemetry_enabled=env_bool(AGENT_ENGINE_ENABLE_TELEMETRY_ENV, True),
            excluded_urls=os.environ.get("EDGAR_TRACE_EXCLUDED_URLS", DEFAULT_EXCLUDED_URLS),
            instrument_httpx=env_bool("EDGAR_TRACE_INSTRUMENT_HTTPX", True),
            instrument_requests=env_bool("EDGAR_TRACE_INSTRUMENT_REQUESTS", True),
            instrument_grpc=env_bool("EDGAR_TRACE_INSTRUMENT_GRPC", True),
            instrument_genai=env_bool("EDGAR_TRACE_INSTRUMENT_GENAI", True),
        )


@dataclass(frozen=True, slots=True)
class TraceCorrelation:
    """Hex trace/span identifiers used to correlate logs, responses, and Cloud Trace."""

    trace_id: str
    span_id: str
    sampled: bool

    def cloud_trace_resource(self, project_id: str) -> str:
        return f"projects/{project_id}/traces/{self.trace_id}"


class _TracingState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config: TracingConfig | None = None
        self.provider: TracerProvider | None = None
        self.logger_provider: LoggerProvider | None = None
        self.exporter_name: str = "none"
        self.otel_logging_enabled: bool = False
        self.fastapi_instrumented: bool = False
        self.httpx_instrumented: bool = False
        self.requests_instrumented: bool = False
        self.grpc_instrumented: bool = False
        self.genai_instrumented: bool = False
        self.propagators: tuple[str, ...] = ()
        self.last_error: str | None = None


_STATE = _TracingState()
_FALLBACK_TRACE: contextvars.ContextVar[TraceCorrelation | None] = contextvars.ContextVar(
    "edgar_fallback_trace", default=None
)


# ---------------------------------------------------------------------------
# Provider / exporter / propagator setup
# ---------------------------------------------------------------------------


def _apply_genai_otel_env_defaults(config: TracingConfig) -> None:
    """Apply OTel GenAI & Vertex AI Agent Engine defaults without overriding explicit env vars."""
    os.environ.setdefault("OTEL_SERVICE_NAME", config.service_name)
    os.environ.setdefault(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, config.semconv_stability_opt_in)
    os.environ.setdefault(OTEL_GENAI_CAPTURE_CONTENT_ENV, config.capture_message_content_mode)
    os.environ.setdefault(
        OTEL_GENAI_EMIT_EVENT_ENV,
        "true" if config.emit_genai_events else "false",
    )
    os.environ.setdefault(
        ADK_CAPTURE_CONTENT_ENV,
        "true" if config.capture_message_content else "false",
    )
    os.environ.setdefault(
        AGENT_ENGINE_ENABLE_TELEMETRY_ENV,
        "true" if config.agent_engine_telemetry_enabled else "false",
    )


def _build_resource(config: TracingConfig) -> Resource:
    base = Resource.create(
        {
            "service.name": config.service_name,
            "service.namespace": DEFAULT_SERVICE_NAMESPACE,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
            "gcp.project_id": config.project_id,
            "cloud.account.id": config.project_id,
        }
    )
    if config.resolved_exporter == "gcp":
        try:
            from google.adk.telemetry.google_cloud import get_gcp_resource

            base = base.merge(get_gcp_resource(config.project_id))
        except Exception as exc:  # pragma: no cover - optional ADK helper
            logger.debug("ADK get_gcp_resource unavailable; using static GCP resource: %s", exc)
    if not running_on_cloud_run():
        # The GCP detector probes the metadata server; skip it off-GCP to avoid startup stalls.
        return base
    try:
        from opentelemetry.resourcedetector.gcp_resource_detector import (
            GoogleCloudResourceDetector,
        )

        return base.merge(GoogleCloudResourceDetector().detect())
    except Exception as exc:  # pragma: no cover - depends on the GCP metadata server
        logger.warning("GCP resource detection failed; using static resource: %s", exc)
        return base


def _build_gcp_otlp_hooks(
    config: TracingConfig,
) -> tuple[list[SpanProcessor], list[LogRecordProcessor], str]:
    """Build ADK's native OTLP processors for `telemetry.googleapis.com` (traces + GenAI logs).

    Uses `google.adk.telemetry.google_cloud.get_gcp_exporters(...)` so traces and OTel log events
    (`gen_ai.client.inference.operation.details`) go over OTLP/HTTP to `telemetry.googleapis.com`
    with Vertex AI Agent Engine headers when `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true`.
    """
    try:
        import google.auth
        from google.adk.telemetry.google_cloud import (
            _CLOUD_PLATFORM_SCOPE,
            get_gcp_exporters,
        )
    except ImportError as exc:
        logger.debug("ADK GCP OTLP telemetry helper unavailable: %s", exc)
        return [], [], "none"

    credentials, detected_project = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    project_id = config.project_id or detected_project
    hooks = get_gcp_exporters(
        enable_cloud_tracing=True,
        enable_cloud_metrics=False,
        enable_cloud_logging=True,
        google_auth=(credentials, project_id),
    )
    span_processors = list(hooks.span_processors)
    log_processors = list(hooks.log_record_processors)
    if not span_processors and not log_processors:
        return [], [], "none"
    return span_processors, log_processors, "gcp_otlp"


def _build_exporter(config: TracingConfig) -> tuple[SpanExporter | None, str]:
    mode = config.resolved_exporter
    if mode == "none":
        return None, "none"
    if mode == "console":
        return ConsoleSpanExporter(service_name=config.service_name), "console"
    if mode == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
        except ImportError:
            logger.warning(
                "EDGAR_TRACE_EXPORTER=otlp requires opentelemetry-exporter-otlp-proto-http; "
                "spans will not be exported."
            )
            return None, "none"
        return OTLPSpanExporter(), "otlp"

    return None, "none"


def _configure_propagators() -> tuple[str, ...]:
    propagators: list[TextMapPropagator] = []
    installed_names: list[str] = []
    try:
        from opentelemetry.propagators.cloud_trace_propagator import CloudTraceFormatPropagator

        # Listed first so the W3C `traceparent` (added last) wins when both headers exist.
        propagators.append(CloudTraceFormatPropagator())
        installed_names.append("x-cloud-trace-context")
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("opentelemetry-propagator-gcp unavailable; X-Cloud-Trace-Context ignored")
    propagators.extend([W3CBaggagePropagator(), TraceContextTextMapPropagator()])
    installed_names.extend(["baggage", "traceparent"])
    set_global_textmap(CompositePropagator(propagators))
    return tuple(installed_names)


def _instrument_httpx(provider: TracerProvider) -> bool:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        logger.warning("opentelemetry-instrumentation-httpx unavailable; outbound HTTP untraced")
        return False
    instrumentor = HTTPXClientInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    return True


def _instrument_requests(provider: TracerProvider) -> bool:
    """Instrument `requests` so BigQuery, Cloud Storage, and Google OAuth HTTP calls emit spans."""
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except ImportError:
        logger.warning("opentelemetry-instrumentation-requests unavailable; SDK HTTP untraced")
        return False
    instrumentor = RequestsInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    return True


def _instrument_grpc(provider: TracerProvider) -> bool:
    """Instrument gRPC clients so Firestore, Model Armor, and Discovery Engine calls emit spans."""
    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
    except ImportError:
        logger.warning("opentelemetry-instrumentation-grpc unavailable; SDK gRPC untraced")
        return False
    instrumentor = GrpcInstrumentorClient()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    return True


def _instrument_google_genai(
    provider: TracerProvider,
    logger_provider: LoggerProvider | None,
) -> bool:
    """Instrument the Google GenAI SDK so LLM calls emit GenAI semconv spans and events."""
    try:
        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-google-genai unavailable; install google-adk[otel] "
            "to populate the Cloud Trace Explorer Inputs/Outputs tab for Gemini generations."
        )
        return False
    instrumentor = GoogleGenAiSdkInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        kwargs: dict[str, Any] = {"tracer_provider": provider}
        if logger_provider is not None:
            kwargs["logger_provider"] = logger_provider
        instrumentor.instrument(**kwargs)
    return True


def configure_tracing(
    config: TracingConfig | None = None,
    *,
    extra_span_exporters: Sequence[SpanExporter] = (),
    extra_log_record_processors: Sequence[LogRecordProcessor] = (),
) -> TracerProvider | None:
    """Install the global OpenTelemetry tracer & logger providers (idempotent).

    Returns the active SDK `TracerProvider`, or `None` when tracing is disabled.
    `extra_span_exporters` and `extra_log_record_processors` are attached synchronously
    (useful in tests).
    """
    with _STATE.lock:
        cfg = config or TracingConfig.from_env()

        if _STATE.provider is not None:
            for extra in extra_span_exporters:
                _STATE.provider.add_span_processor(SimpleSpanProcessor(extra))
            if _STATE.logger_provider is not None:
                for extra_log_proc in extra_log_record_processors:
                    _STATE.logger_provider.add_log_record_processor(extra_log_proc)
            return _STATE.provider

        _STATE.config = cfg
        if not cfg.enabled:
            logger.info("OpenTelemetry tracing disabled (EDGAR_TRACING_ENABLED/OTEL_SDK_DISABLED)")
            return None

        _apply_genai_otel_env_defaults(cfg)

        resource = _build_resource(cfg)
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            provider = existing
        else:
            ratio_sampler = TraceIdRatioBased(cfg.sample_ratio)
            provider = TracerProvider(
                resource=resource,
                sampler=ParentBased(
                    root=ratio_sampler,
                    remote_parent_not_sampled=ratio_sampler,
                ),
            )
            trace.set_tracer_provider(provider)

        existing_logger = _logs.get_logger_provider()
        if isinstance(existing_logger, LoggerProvider):
            logger_provider = existing_logger
        else:
            logger_provider = LoggerProvider(resource=resource)
            _logs.set_logger_provider(logger_provider)

        span_processors: list[SpanProcessor] = []
        log_processors: list[LogRecordProcessor] = []
        exporter_name = "none"

        if cfg.resolved_exporter == "gcp":
            try:
                span_processors, log_processors, exporter_name = _build_gcp_otlp_hooks(cfg)
            except Exception as exc:  # pragma: no cover - credential / environment specific
                _STATE.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("GCP OTLP telemetry exporter unavailable; spans stay local: %s", exc)
        elif cfg.resolved_exporter != "none":
            try:
                exporter, exporter_name = _build_exporter(cfg)
            except Exception as exc:  # pragma: no cover - credential / environment specific
                _STATE.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Trace exporter unavailable; spans stay local: %s", exc)
                exporter, exporter_name = None, "none"
            if exporter is not None:
                span_processors.append(BatchSpanProcessor(exporter))

        for proc in span_processors:
            provider.add_span_processor(proc)
        for extra in extra_span_exporters:
            provider.add_span_processor(SimpleSpanProcessor(extra))

        for log_proc in log_processors:
            logger_provider.add_log_record_processor(log_proc)
        for extra_log_proc in extra_log_record_processors:
            logger_provider.add_log_record_processor(extra_log_proc)

        _STATE.propagators = _configure_propagators()
        if cfg.instrument_httpx:
            _STATE.httpx_instrumented = _instrument_httpx(provider)
        if cfg.instrument_requests:
            _STATE.requests_instrumented = _instrument_requests(provider)
        if cfg.instrument_grpc:
            _STATE.grpc_instrumented = _instrument_grpc(provider)
        if cfg.instrument_genai:
            _STATE.genai_instrumented = _instrument_google_genai(provider, logger_provider)

        _STATE.provider = provider
        _STATE.logger_provider = logger_provider
        _STATE.exporter_name = exporter_name
        _STATE.otel_logging_enabled = bool(log_processors or extra_log_record_processors)
        logger.info(
            "OpenTelemetry tracing configured",
            extra={
                "json_fields": {
                    "event": "tracing_configured",
                    "trace_exporter": exporter_name,
                    "otel_logging_enabled": _STATE.otel_logging_enabled,
                    "genai_instrumented": _STATE.genai_instrumented,
                    "capture_message_content_mode": cfg.capture_message_content_mode,
                    "semconv_stability_opt_in": cfg.semconv_stability_opt_in,
                    "agent_engine_telemetry_enabled": cfg.agent_engine_telemetry_enabled,
                    "trace_sample_ratio": cfg.sample_ratio,
                    "service_name": cfg.service_name,
                    "service_version": cfg.service_version,
                    "trace_project_id": cfg.project_id,
                }
            },
        )
        return provider


def instrument_fastapi_app(app: Any) -> bool:
    """Attach OpenTelemetry server spans (with upstream context extraction) to a FastAPI app."""
    with _STATE.lock:
        cfg = _STATE.config or TracingConfig.from_env()
        if _STATE.provider is None or not cfg.enabled:
            return False
        if getattr(app, "_is_instrumented_by_opentelemetry", False):
            _STATE.fastapi_instrumented = True
            return True
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except ImportError:
            logger.warning("opentelemetry-instrumentation-fastapi unavailable; no server spans")
            return False
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=_STATE.provider,
            excluded_urls=cfg.excluded_urls,
            # Per-ASGI-message send/receive spans add noise without latency insight.
            exclude_spans=["receive", "send"],
        )
        _STATE.fastapi_instrumented = True
        return True


def flush_tracing(timeout_millis: int = 5000) -> bool:
    """Flush buffered spans and OTel log records (call on shutdown)."""
    with _STATE.lock:
        provider = _STATE.provider
        logger_provider = _STATE.logger_provider
    ok = True
    if provider is not None:
        try:
            ok = bool(provider.force_flush(timeout_millis=timeout_millis)) and ok
        except Exception as exc:  # pragma: no cover - exporter specific
            logger.warning("Failed to flush OpenTelemetry spans: %s", exc)
            ok = False
    if logger_provider is not None:
        try:
            ok = bool(logger_provider.force_flush(timeout_millis=timeout_millis)) and ok
        except Exception as exc:  # pragma: no cover - exporter specific
            logger.warning("Failed to flush OpenTelemetry log records: %s", exc)
            ok = False
    return ok


def get_tracer(name: str = TRACER_NAME) -> trace.Tracer:
    return trace.get_tracer(name)


def tracing_status_metadata() -> dict[str, Any]:
    with _STATE.lock:
        cfg = _STATE.config or TracingConfig.from_env()
        provider_ready = _STATE.provider is not None
        exporter_name = _STATE.exporter_name
        otel_logging_enabled = _STATE.otel_logging_enabled
        fastapi_instrumented = _STATE.fastapi_instrumented
        httpx_instrumented = _STATE.httpx_instrumented
        requests_instrumented = _STATE.requests_instrumented
        grpc_instrumented = _STATE.grpc_instrumented
        genai_instrumented = _STATE.genai_instrumented
        propagators = list(_STATE.propagators)
        last_error = _STATE.last_error
    return {
        "enabled": cfg.enabled and provider_ready,
        "exporter": exporter_name,
        "configured_exporter": cfg.exporter,
        "otel_logging_enabled": otel_logging_enabled,
        "sample_ratio": cfg.sample_ratio,
        "service_name": cfg.service_name,
        "service_version": cfg.service_version,
        "project_id": cfg.project_id,
        "fastapi_instrumented": fastapi_instrumented,
        "httpx_instrumented": httpx_instrumented,
        "requests_instrumented": requests_instrumented,
        "grpc_instrumented": grpc_instrumented,
        "genai_instrumented": genai_instrumented,
        "capture_message_content": cfg.capture_message_content,
        "capture_message_content_mode": cfg.capture_message_content_mode,
        "semconv_stability_opt_in": cfg.semconv_stability_opt_in,
        "emit_genai_events": cfg.emit_genai_events,
        "agent_engine_telemetry_enabled": cfg.agent_engine_telemetry_enabled,
        "propagators": propagators,
        "last_error": last_error,
    }


# ---------------------------------------------------------------------------
# Trace correlation helpers (logs, response headers, API payloads)
# ---------------------------------------------------------------------------


def parse_traceparent(header: str | None) -> TraceCorrelation | None:
    """Parse a W3C `traceparent` header value."""
    if not header:
        return None
    match = _TRACEPARENT_RE.match(header.strip().lower())
    if not match:
        return None
    version, trace_id, span_id, flags = match.groups()
    if version == "ff" or trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return TraceCorrelation(trace_id=trace_id, span_id=span_id, sampled=bool(int(flags, 16) & 1))


def parse_cloud_trace_context(header: str | None) -> TraceCorrelation | None:
    """Parse a legacy Google `X-Cloud-Trace-Context: TRACE_ID/SPAN_ID;o=1` header value."""
    if not header:
        return None
    match = _CLOUD_TRACE_CONTEXT_RE.match(header.strip())
    if not match:
        return None
    trace_id, span_dec, sampled = match.groups()
    trace_id = trace_id.lower()
    if trace_id == "0" * 32:
        return None
    if span_dec is None:
        span_hex = "0" * 16
    else:
        span_int = int(span_dec)
        if not (1 <= span_int < (1 << 64)):
            return None
        span_hex = format(span_int, "016x")
    return TraceCorrelation(trace_id=trace_id, span_id=span_hex, sampled=sampled == "1")


def extract_trace_correlation(headers: Mapping[str, str]) -> TraceCorrelation | None:
    """Best-effort trace context from inbound headers (used when no OTel span is active)."""
    return parse_traceparent(headers.get("traceparent")) or parse_cloud_trace_context(
        headers.get("x-cloud-trace-context")
    )


def bind_fallback_trace(correlation: TraceCorrelation | None) -> contextvars.Token:
    return _FALLBACK_TRACE.set(correlation)


def reset_fallback_trace(token: contextvars.Token) -> None:
    _FALLBACK_TRACE.reset(token)


def current_trace_correlation() -> TraceCorrelation | None:
    """Return IDs of the active OTel span, else the inbound header context, else `None`."""
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx.is_valid:
        return TraceCorrelation(
            trace_id=format(span_ctx.trace_id, "032x"),
            span_id=format(span_ctx.span_id, "016x"),
            sampled=bool(span_ctx.trace_flags.sampled),
        )
    return _FALLBACK_TRACE.get()


def cloud_trace_console_url(trace_id: str, project_id: str | None = None) -> str:
    project = project_id or resolve_gcp_project_id()
    return f"https://console.cloud.google.com/traces/list?project={project}&tid={trace_id}"


def current_trace_metadata() -> dict[str, Any] | None:
    """Trace IDs + Cloud Trace console deep link for embedding in API responses."""
    corr = current_trace_correlation()
    if corr is None:
        return None
    with _STATE.lock:
        configured_project = _STATE.config.project_id if _STATE.config else None
    project_id = configured_project or resolve_gcp_project_id()
    return {
        "trace_id": corr.trace_id,
        "span_id": corr.span_id,
        "sampled": corr.sampled,
        "cloud_trace_resource": corr.cloud_trace_resource(project_id),
        "cloud_trace_url": cloud_trace_console_url(corr.trace_id, project_id),
    }


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


def _coerce_attr(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_ATTR_STR_LEN else value[:_MAX_ATTR_STR_LEN] + "…"
    if isinstance(value, list | tuple | set | frozenset):
        items = [_coerce_attr(v) for v in value]
        items = [v for v in items if v is not None]
        if all(isinstance(v, str) for v in items):
            return items
        return [str(v) for v in items]
    return _coerce_attr(str(value))


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert arbitrary values into OpenTelemetry-legal attribute values (drops `None`)."""
    if not attributes:
        return {}
    clean: dict[str, Any] = {}
    for key, value in attributes.items():
        coerced = _coerce_attr(value)
        if coerced is not None:
            clean[str(key)] = coerced
    return clean


def set_span_attributes(span: Span | None = None, **attributes: Any) -> None:
    """Set attributes on `span` (or the current span) if it is recording."""
    target = span or trace.get_current_span()
    if target.is_recording():
        target.set_attributes(sanitize_attributes(attributes))


def add_span_event(name: str, **attributes: Any) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, sanitize_attributes(attributes))


def mark_span_error(message: str, span: Span | None = None, **attributes: Any) -> None:
    target = span or trace.get_current_span()
    if target.is_recording():
        target.set_status(Status(StatusCode.ERROR, message[:_MAX_ATTR_STR_LEN]))
        if attributes:
            target.set_attributes(sanitize_attributes(attributes))


@contextmanager
def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[Span]:
    """Start a child span of the current context; exceptions are recorded and re-raised."""
    with get_tracer().start_as_current_span(
        name,
        kind=kind,
        attributes=sanitize_attributes(attributes),
        record_exception=True,
        set_status_on_exception=True,
    ) as span:
        yield span


def traced(
    name: str | None = None,
    *,
    attributes: Mapping[str, Any] | None = None,
    record_args: Sequence[str] = (),
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[[_F], _F]:
    """Decorator wrapping a sync or async callable in a span.

    `record_args` names parameters whose values become `edgar.<param>` span attributes.
    The original signature is preserved so Google ADK `FunctionTool` introspection works.
    """

    def decorator(func: _F) -> _F:
        span_name = name or f"{func.__module__.removeprefix('edgar_capstone.')}.{func.__qualname__}"
        static_attrs = {
            "code.function": func.__qualname__,
            "code.namespace": func.__module__,
            **(attributes or {}),
        }
        signature = inspect.signature(func)

        def _span_attributes(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            if not record_args:
                return static_attrs
            attrs = dict(static_attrs)
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                return attrs
            for arg_name in record_args:
                if arg_name in bound.arguments:
                    attrs[f"edgar.{arg_name}"] = bound.arguments[arg_name]
            return attrs

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with start_span(span_name, attributes=_span_attributes(args, kwargs), kind=kind):
                    return await func(*args, **kwargs)

            async_wrapper.__signature__ = signature  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(span_name, attributes=_span_attributes(args, kwargs), kind=kind):
                return func(*args, **kwargs)

        wrapper.__signature__ = signature  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
