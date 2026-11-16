"""End-to-end OpenTelemetry distributed tracing exported to Google Cloud Trace.

A single request flows through these layers, and each one becomes a span in the
same trace, so Cloud Trace shows where the latency goes:

    Cloud Run / HTTPS LB front end (`traceparent` / `X-Cloud-Trace-Context`)
      -> FastAPI server span            (opentelemetry-instrumentation-fastapi)
        -> app spans                    (`@traced` / `start_span`: guardrails, scope
                                         resolution, RAG retrieval, ReAct loop, tools)
          -> Google ADK agent/LLM/tool spans (`gcp.vertex.agent` tracer, global provider)
          -> outbound HTTP client spans (opentelemetry-instrumentation-httpx: Vertex AI
                                         Gemini, BigQuery, Firestore, Model Armor, ...)

Runtime configuration (all optional):

* `EDGAR_TRACING_ENABLED` (default `true`; `OTEL_SDK_DISABLED=true` also turns it off).
* `EDGAR_TRACE_EXPORTER` = `auto` | `gcp` | `console` | `otlp` | `none`. `auto` exports to
  Cloud Trace on Cloud Run (`K_SERVICE` is set) and keeps spans in-process locally, so
  trace IDs still show up in logs and response headers without needing credentials.
* `EDGAR_TRACE_SAMPLE_RATIO` (default `1.0`). A sampled upstream parent (e.g. the Cloud
  Run front end) is always honoured; otherwise this ratio decides.
* `EDGAR_TRACE_CAPTURE_CONTENT` (default `false`). Sets the default for ADK's
  `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` so prompts, 10-K excerpts, and model output are
  kept out of Cloud Trace unless you opt in.
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

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

TRACER_NAME = "edgar_capstone"
DEFAULT_SERVICE_NAME = "sec-edgar-analyst"
DEFAULT_SERVICE_NAMESPACE = "sec-edgar"
DEFAULT_EXCLUDED_URLS = r"/health$,/healthz$,/assets/,/favicon\.ico$,/vite\.svg$"
# Fallback only; Cloud Run always injects GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT.
DEFAULT_TRACE_PROJECT_ID = "iron-radio-514217-q7"
ADK_CAPTURE_CONTENT_ENV = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"

TraceExporterMode = Literal["auto", "gcp", "console", "otlp", "none"]
_VALID_EXPORTERS: frozenset[str] = frozenset({"auto", "gcp", "console", "otlp", "none"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})
_MAX_ATTR_STR_LEN = 512

_TRACEPARENT_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_CLOUD_TRACE_CONTEXT_RE = re.compile(r"^([0-9a-fA-F]{32})(?:/(\d+))?(?:;o=([01]))?$")

_F = TypeVar("_F", bound=Callable[..., Any])


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return default


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


def resolve_gcp_project_id() -> str:
    """Return the GCP project that owns traces and logs for this process."""
    for env_name in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_TRACE_PROJECT_ID


def running_on_cloud_run() -> bool:
    return bool(os.environ.get("K_SERVICE", "").strip())


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
    capture_message_content: bool = False
    excluded_urls: str = DEFAULT_EXCLUDED_URLS
    instrument_httpx: bool = True

    @classmethod
    def from_env(cls) -> TracingConfig:
        enabled = _env_bool("EDGAR_TRACING_ENABLED", True) and not _env_bool(
            "OTEL_SDK_DISABLED", False
        )
        raw_exporter = os.environ.get("EDGAR_TRACE_EXPORTER", "auto").strip().lower() or "auto"
        if raw_exporter not in _VALID_EXPORTERS:
            logger.warning("Unknown EDGAR_TRACE_EXPORTER=%r; falling back to 'auto'", raw_exporter)
            raw_exporter = "auto"
        exporter: TraceExporterMode = raw_exporter  # type: ignore[assignment]
        resolved: TraceExporterMode = exporter
        if exporter == "auto":
            resolved = "gcp" if running_on_cloud_run() else "none"

        on_cloud_run = running_on_cloud_run()
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
            capture_message_content=_env_bool("EDGAR_TRACE_CAPTURE_CONTENT", False),
            excluded_urls=os.environ.get("EDGAR_TRACE_EXCLUDED_URLS", DEFAULT_EXCLUDED_URLS),
            instrument_httpx=_env_bool("EDGAR_TRACE_INSTRUMENT_HTTPX", True),
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
        self.exporter_name: str = "none"
        self.fastapi_instrumented: bool = False
        self.httpx_instrumented: bool = False
        self.last_error: str | None = None


_STATE = _TracingState()
_FALLBACK_TRACE: contextvars.ContextVar[TraceCorrelation | None] = contextvars.ContextVar(
    "edgar_fallback_trace", default=None
)


# ---------------------------------------------------------------------------
# Provider / exporter / propagator setup
# ---------------------------------------------------------------------------


def _build_resource(config: TracingConfig) -> Resource:
    base = Resource.create(
        {
            "service.name": config.service_name,
            "service.namespace": DEFAULT_SERVICE_NAMESPACE,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
            "gcp.project_id": config.project_id,
        }
    )
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

    # mode == "gcp"
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    except ImportError:
        logger.warning(
            "opentelemetry-exporter-gcp-trace is not installed; spans will not reach Cloud Trace."
        )
        return None, "none"
    exporter = CloudTraceSpanExporter(
        project_id=config.project_id,
        # Promote service/cloud/faas resource attributes to Cloud Trace span labels.
        resource_regex=r"^(service|cloud|faas|deployment|gcp)\..*",
    )
    return exporter, "gcp"


def _configure_propagators() -> None:
    propagators: list[TextMapPropagator] = []
    try:
        from opentelemetry.propagators.cloud_trace_propagator import CloudTraceFormatPropagator

        # Listed first so the W3C `traceparent` (added last) wins when both headers exist.
        propagators.append(CloudTraceFormatPropagator())
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("opentelemetry-propagator-gcp unavailable; X-Cloud-Trace-Context ignored")
    propagators.extend([W3CBaggagePropagator(), TraceContextTextMapPropagator()])
    set_global_textmap(CompositePropagator(propagators))


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


def configure_tracing(
    config: TracingConfig | None = None,
    *,
    extra_span_exporters: Sequence[SpanExporter] = (),
) -> TracerProvider | None:
    """Install the global OpenTelemetry tracer provider (idempotent).

    Returns the active SDK `TracerProvider`, or `None` when tracing is disabled.
    `extra_span_exporters` are attached with a synchronous processor (useful in tests).
    """
    with _STATE.lock:
        cfg = config or TracingConfig.from_env()

        if _STATE.provider is not None:
            for extra in extra_span_exporters:
                _STATE.provider.add_span_processor(SimpleSpanProcessor(extra))
            return _STATE.provider

        _STATE.config = cfg
        if not cfg.enabled:
            logger.info("OpenTelemetry tracing disabled (EDGAR_TRACING_ENABLED/OTEL_SDK_DISABLED)")
            return None

        # Keep raw prompts / 10-K passages / model output out of exported ADK spans by default.
        os.environ.setdefault(
            ADK_CAPTURE_CONTENT_ENV, "true" if cfg.capture_message_content else "false"
        )

        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            provider = existing
        else:
            ratio_sampler = TraceIdRatioBased(cfg.sample_ratio)
            provider = TracerProvider(
                resource=_build_resource(cfg),
                sampler=ParentBased(
                    root=ratio_sampler,
                    remote_parent_not_sampled=ratio_sampler,
                ),
            )
            trace.set_tracer_provider(provider)

        try:
            exporter, exporter_name = _build_exporter(cfg)
        except Exception as exc:  # pragma: no cover - credential / environment specific
            _STATE.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Cloud Trace exporter unavailable; spans stay local: %s", exc)
            exporter, exporter_name = None, "none"
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        for extra in extra_span_exporters:
            provider.add_span_processor(SimpleSpanProcessor(extra))

        _configure_propagators()
        if cfg.instrument_httpx:
            _STATE.httpx_instrumented = _instrument_httpx(provider)

        _STATE.provider = provider
        _STATE.exporter_name = exporter_name
        logger.info(
            "OpenTelemetry tracing configured",
            extra={
                "json_fields": {
                    "event": "tracing_configured",
                    "trace_exporter": exporter_name,
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
    """Flush buffered spans (call on shutdown; Cloud Run may freeze CPU between requests)."""
    provider = _STATE.provider
    if provider is None:
        return True
    try:
        return bool(provider.force_flush(timeout_millis=timeout_millis))
    except Exception as exc:  # pragma: no cover - exporter specific
        logger.warning("Failed to flush OpenTelemetry spans: %s", exc)
        return False


def get_tracer(name: str = TRACER_NAME) -> trace.Tracer:
    return trace.get_tracer(name)


def tracing_status_metadata() -> dict[str, Any]:
    cfg = _STATE.config or TracingConfig.from_env()
    return {
        "enabled": cfg.enabled and _STATE.provider is not None,
        "exporter": _STATE.exporter_name,
        "configured_exporter": cfg.exporter,
        "sample_ratio": cfg.sample_ratio,
        "service_name": cfg.service_name,
        "service_version": cfg.service_version,
        "project_id": cfg.project_id,
        "fastapi_instrumented": _STATE.fastapi_instrumented,
        "httpx_instrumented": _STATE.httpx_instrumented,
        "capture_message_content": cfg.capture_message_content,
        "propagators": ["x-cloud-trace-context", "baggage", "traceparent"],
        "last_error": _STATE.last_error,
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
    _version, trace_id, span_id, flags = match.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
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
    span_hex = format(int(span_dec), "016x")[-16:] if span_dec else "0" * 16
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
    project_id = (_STATE.config.project_id if _STATE.config else None) or resolve_gcp_project_id()
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
