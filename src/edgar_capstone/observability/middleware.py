"""ASGI middleware: trace-correlated access logs, latency headers, and request log context."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edgar_capstone.observability.structured_logging import (
    bind_request_context,
    reset_request_context,
)
from edgar_capstone.observability.tracing import (
    bind_fallback_trace,
    current_trace_correlation,
    extract_trace_correlation,
    reset_fallback_trace,
    set_span_attributes,
)

access_logger = logging.getLogger("edgar_capstone.access")

TRACE_ID_HEADER = "X-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"
DEFAULT_QUIET_PATHS: tuple[str, ...] = ("/health", "/healthz")


class RequestTelemetryMiddleware:
    """Per-request observability for the FastAPI service.

    Runs inside the OpenTelemetry FastAPI server span, so it:

    * binds `request_id`/method/path to every log line written while handling the request;
    * falls back to the inbound `traceparent` / `X-Cloud-Trace-Context` header for log
      correlation on routes that are excluded from tracing (health probes);
    * returns `X-Trace-Id`, `X-Request-Id`, and a `Server-Timing` header so a browser or
      `curl -i` can jump straight to the trace in Cloud Trace;
    * writes one structured `http_request_completed` access log line with `latency_ms`
      (the source of the `sec_edgar/api_request_latencies` log-based metric).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        quiet_paths: Iterable[str] = DEFAULT_QUIET_PATHS,
        access_log: bool = True,
    ) -> None:
        self.app = app
        self.quiet_paths = frozenset(quiet_paths)
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        headers = Headers(scope=scope)

        fallback_token = None
        if current_trace_correlation() is None:
            fallback_token = bind_fallback_trace(extract_trace_correlation(headers))

        corr = current_trace_correlation()
        raw_request_id = headers.get(REQUEST_ID_HEADER, "").strip()
        sanitized_request_id = "".join(ch for ch in raw_request_id if 0x20 <= ord(ch) <= 0x7E)[:128]
        request_id = sanitized_request_id or (corr.trace_id if corr else "") or uuid.uuid4().hex
        ctx_token = bind_request_context(request_id=request_id, method=method, path=path)
        set_span_attributes(**{"edgar.request_id": request_id})
        status_code = 500
        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                response_headers = MutableHeaders(scope=message)
                active = current_trace_correlation()
                if active is not None:
                    response_headers[TRACE_ID_HEADER] = active.trace_id
                response_headers[REQUEST_ID_HEADER] = request_id
                response_headers.append("Server-Timing", f'app;dur={elapsed_ms:.1f};desc="FastAPI"')
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                # The OTel server span ends right after the final body chunk is sent.
                set_span_attributes(
                    **{"edgar.request.latency_ms": round((time.perf_counter() - start) * 1000.0, 2)}
                )
            await send(message)

        failed = False
        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            failed = True
            # Preserve already-sent HTTP status (e.g. 200 on mid-stream SSE failure) rather than
            # misreporting 500 for headers already flushed to the wire.
            if not response_started:
                status_code = 500
            access_logger.exception(
                "Unhandled exception while serving %s %s",
                method,
                path,
                extra={
                    "json_fields": {
                        "event": "http_request_failed",
                        "http_status": status_code,
                        "response_started": response_started,
                    }
                },
            )
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            route_path = _route_template(scope) or path
            if self.access_log and path not in self.quiet_paths:
                self._log_access(
                    scope=scope,
                    headers=headers,
                    method=method,
                    path=path,
                    route=route_path,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    failed=failed,
                    response_started=response_started,
                )
            reset_request_context(ctx_token)
            if fallback_token is not None:
                reset_fallback_trace(fallback_token)

    @staticmethod
    def _log_access(
        *,
        scope: Scope,
        headers: Headers,
        method: str,
        path: str,
        route: str,
        status_code: int,
        latency_ms: float,
        failed: bool = False,
        response_started: bool = False,
    ) -> None:
        if failed or status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        client = scope.get("client")
        # Note: X-Forwarded-For left-most entry can be spoofed by untrusted clients unless
        # stripped by the edge load balancer. Cloud Run / GCLB appends `<client_ip>, <lb_ip>`
        # at the right-most hops; keep first entry for diagnostics alongside client socket IP.
        xff_parts = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
        remote_ip = xff_parts[0] if xff_parts else (client[0] if client else "")
        fields: dict[str, Any] = {
            "event": "http_request_completed",
            "http_method": method,
            "http_path": path,
            "http_route": route,
            "http_status": status_code,
            "latency_ms": latency_ms,
            "unhandled_exception": failed,
            "stream_aborted": failed and response_started,
            "user_agent": headers.get("user-agent", "")[:256],
            "remote_ip": remote_ip,
        }
        access_logger.log(
            level,
            "%s %s -> %d (%.1f ms)",
            method,
            path,
            status_code,
            latency_ms,
            extra={"json_fields": fields},
        )


def _route_template(scope: Scope) -> str | None:
    """Return the matched route template (e.g. `/api/v1/explore/{explore_name}`)."""
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    return str(route_path) if route_path else None
