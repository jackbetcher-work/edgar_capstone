"""Structured JSON logging for Cloud Logging with OpenTelemetry trace correlation.

On Cloud Run, anything written to stdout as one JSON object per line is ingested by
Cloud Logging as `jsonPayload`, with these special fields promoted to the LogEntry:

* `severity`: the log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
* `logging.googleapis.com/trace`: `projects/PROJECT_ID/traces/TRACE_ID`. Ties the line to
  its Cloud Trace trace and groups it under the parent request log in Logs Explorer.
* `logging.googleapis.com/spanId` / `logging.googleapis.com/trace_sampled`.
* `logging.googleapis.com/sourceLocation`: file, line, and function.

Error-level records that carry a traceback are also tagged with the Error Reporting
`@type` so they show up in Cloud Error Reporting.

Configuration:

* `EDGAR_LOG_FORMAT` = `auto` | `json` | `text`. `auto` picks JSON on Cloud Run and
  readable text (still with trace IDs) on a laptop.
* `EDGAR_LOG_LEVEL` (or `LOG_LEVEL`), default `INFO`.

Structured fields go through `extra={"json_fields": {...}}` (the same convention as
`google-cloud-logging`); any other non-standard `extra=` keys are included as well.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from edgar_capstone.observability.tracing import (
    current_trace_correlation,
    resolve_gcp_project_id,
    running_on_cloud_run,
)

LogFormat = Literal["json", "text"]

ERROR_REPORTING_TYPE = (
    "type.googleapis.com/google.devtools.clouderrorreporting.v1beta1.ReportedErrorEvent"
)
_TRACE_KEY = "logging.googleapis.com/trace"
_SPAN_KEY = "logging.googleapis.com/spanId"
_SAMPLED_KEY = "logging.googleapis.com/trace_sampled"
_SOURCE_KEY = "logging.googleapis.com/sourceLocation"

_RESERVED_PAYLOAD_KEYS = frozenset(
    {"severity", "message", "time", _TRACE_KEY, _SPAN_KEY, _SAMPLED_KEY, _SOURCE_KEY}
)
_STANDARD_RECORD_ATTRS = frozenset(
    set(vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys())
    | {"message", "asctime", "json_fields", "taskName", "color_message"}
)
_SEVERITY_BY_LEVEL = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "google.auth.transport.requests")

_REQUEST_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "edgar_request_log_context", default=None
)


def bind_request_context(**fields: Any) -> contextvars.Token:
    """Attach per-request fields (method, path, request_id, ...) to every log line."""
    current = dict(_REQUEST_CONTEXT.get() or {})
    current.update({k: v for k, v in fields.items() if v is not None})
    return _REQUEST_CONTEXT.set(current)


def reset_request_context(token: contextvars.Token) -> None:
    _REQUEST_CONTEXT.reset(token)


def get_request_context() -> dict[str, Any] | None:
    ctx = _REQUEST_CONTEXT.get()
    return dict(ctx) if ctx else None


def _severity_for(levelno: int) -> str:
    if levelno in _SEVERITY_BY_LEVEL:
        return _SEVERITY_BY_LEVEL[levelno]
    if levelno >= logging.CRITICAL:
        return "CRITICAL"
    if levelno >= logging.ERROR:
        return "ERROR"
    if levelno >= logging.WARNING:
        return "WARNING"
    if levelno >= logging.INFO:
        return "INFO"
    if levelno > logging.NOTSET:
        return "DEBUG"
    return "DEFAULT"


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    extras = {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
    }
    json_fields = getattr(record, "json_fields", None)
    if isinstance(json_fields, Mapping):
        extras.update(json_fields)
    return extras


class CloudLoggingJsonFormatter(logging.Formatter):
    """Format records as single-line Cloud Logging structured JSON with trace correlation."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
    ) -> None:
        super().__init__()
        self.project_id = project_id or resolve_gcp_project_id()
        self.service_name = (
            service_name or os.environ.get("K_SERVICE") or "sec-edgar-analyst"
        ).strip()
        self.service_version = (
            service_version or os.environ.get("COMMIT_SHA") or os.environ.get("K_REVISION") or ""
        ).strip()

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        severity = _severity_for(record.levelno)
        payload: dict[str, Any] = {
            "severity": severity,
            "message": message,
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
            _SOURCE_KEY: {
                "file": record.pathname,
                "line": str(record.lineno),
                "function": record.funcName,
            },
        }

        corr = current_trace_correlation()
        if corr is not None:
            payload["trace_id"] = corr.trace_id
            payload["span_id"] = corr.span_id
            payload[_TRACE_KEY] = corr.cloud_trace_resource(self.project_id)
            payload[_SPAN_KEY] = corr.span_id
            payload[_SAMPLED_KEY] = corr.sampled

        request_ctx = get_request_context()
        if request_ctx:
            payload["request"] = request_ctx

        if record.exc_info and record.levelno >= logging.ERROR:
            payload["@type"] = ERROR_REPORTING_TYPE
            service_ctx = {"service": self.service_name}
            if self.service_version:
                service_ctx["version"] = self.service_version
            payload["serviceContext"] = service_ctx

        for key, value in _record_extras(record).items():
            if key not in _RESERVED_PAYLOAD_KEYS:
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))


class TraceAwareTextFormatter(logging.Formatter):
    """Human-readable local format that still carries trace/span IDs and structured fields."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s [trace=%(trace_id)s span=%(span_id)s] %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        corr = current_trace_correlation()
        record.trace_id = corr.trace_id if corr else "-"
        record.span_id = corr.span_id if corr else "-"
        try:
            rendered = super().format(record)
        finally:
            del record.trace_id
            del record.span_id
        extras = _record_extras(record)
        if extras:
            rendered = f"{rendered} | {json.dumps(extras, default=str, separators=(',', ':'))}"
        return rendered


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever `sys.stdout` currently is (safe with pytest/uvicorn stream swaps)."""

    _edgar_structured_handler = True

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, _value: Any) -> None:
        return None


def resolve_log_format(log_format: str | None = None) -> LogFormat:
    raw = (log_format or os.environ.get("EDGAR_LOG_FORMAT") or "auto").strip().lower()
    if raw in ("json", "structured", "cloud", "gcp"):
        return "json"
    if raw in ("text", "plain", "console", "local"):
        return "text"
    return "json" if running_on_cloud_run() else "text"


def resolve_log_level(level: str | int | None = None) -> int:
    if isinstance(level, int):
        return level
    raw = level or os.environ.get("EDGAR_LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO"
    resolved = logging.getLevelName(str(raw).strip().upper())
    return resolved if isinstance(resolved, int) else logging.INFO


_CONFIGURED_FORMAT: LogFormat | None = None


def configure_logging(
    log_format: str | None = None,
    level: str | int | None = None,
    *,
    project_id: str | None = None,
    handler: logging.Handler | None = None,
) -> LogFormat:
    """Route root, app, and uvicorn loggers through one trace-aware handler (idempotent).

    Replaces any handler previously installed by this function and leaves foreign
    handlers (e.g. pytest's `caplog`) untouched. Uvicorn's own access log is silenced
    because `RequestTelemetryMiddleware` emits a richer, trace-correlated access log.
    """
    global _CONFIGURED_FORMAT

    fmt = resolve_log_format(log_format)
    lvl = resolve_log_level(level)

    out_handler = handler or _StdoutHandler()
    out_handler._edgar_structured_handler = True  # type: ignore[attr-defined]
    if fmt == "json":
        out_handler.setFormatter(CloudLoggingJsonFormatter(project_id=project_id))
    else:
        out_handler.setFormatter(TraceAwareTextFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_edgar_structured_handler", False):
            root.removeHandler(existing)
    root.addHandler(out_handler)
    root.setLevel(lvl)

    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(lvl, logging.WARNING))

    _CONFIGURED_FORMAT = fmt
    return fmt


def logging_status_metadata() -> dict[str, Any]:
    return {
        "format": _CONFIGURED_FORMAT or resolve_log_format(),
        "level": logging.getLevelName(logging.getLogger().level),
        "trace_correlation": True,
        "destination": "stdout -> Cloud Logging" if running_on_cloud_run() else "stdout",
    }
