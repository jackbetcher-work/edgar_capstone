"""Tests for OpenTelemetry tracing (Cloud Trace) and structured JSON logging (Cloud Logging)."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from edgar_capstone.agents.live_runtime import _run_coro_sync
from edgar_capstone.agents.resilience import with_tool_resilience
from edgar_capstone.agents.tools import calculate_period_variances
from edgar_capstone.observability import (
    CloudLoggingJsonFormatter,
    configure_logging,
    configure_tracing,
    current_trace_correlation,
    parse_cloud_trace_context,
    parse_traceparent,
    start_span,
    traced,
)
from edgar_capstone.observability.structured_logging import ERROR_REPORTING_TYPE
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app

UPSTREAM_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
UPSTREAM_SPAN_ID = "00f067aa0ba902b7"

_SPAN_EXPORTER = InMemorySpanExporter()
configure_tracing(extra_span_exporters=[_SPAN_EXPORTER])


@pytest.fixture()
def spans() -> Iterator[InMemorySpanExporter]:
    _SPAN_EXPORTER.clear()
    yield _SPAN_EXPORTER
    _SPAN_EXPORTER.clear()


@pytest.fixture()
def json_log_stream() -> Iterator[io.StringIO]:
    """Capture JSON lines formatted synchronously (inside the request's trace context)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(CloudLoggingJsonFormatter(project_id="test-project"))
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def _json_lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _bearer_headers() -> dict[str, str]:
    jwt_token, _ = issue_local_dev_workspace_sso_token(
        email="analyst@fde-workspace.corp",
        name="Lead SEC Analyst",
        groups=["sec-edgar-analysts"],
    )
    return {"Authorization": f"Bearer {jwt_token}"}


def _formatter_line(record_logger: logging.Logger, formatter: logging.Formatter, **kwargs) -> dict:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    record_logger.addHandler(handler)
    record_logger.propagate = False
    try:
        record_logger.setLevel(logging.DEBUG)
        msg = kwargs.pop("msg", "hello")
        if kwargs.pop("exception", False):
            try:
                raise ValueError("boom")
            except ValueError:
                record_logger.exception(msg, **kwargs)
        else:
            record_logger.info(msg, **kwargs)
    finally:
        record_logger.removeHandler(handler)
    return json.loads(stream.getvalue().strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Structured logging formatter
# ---------------------------------------------------------------------------


def test_json_formatter_emits_cloud_logging_trace_fields() -> None:
    formatter = CloudLoggingJsonFormatter(project_id="test-project")
    test_logger = logging.getLogger("edgar_capstone.tests.json_formatter")
    with start_span("unit.json_formatter") as span:
        ctx = span.get_span_context()
        payload = _formatter_line(
            test_logger,
            formatter,
            msg="variance computed",
            extra={"json_fields": {"ticker": "NVDA", "latency_ms": 12.5}, "intent": "peer"},
        )

    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    assert payload["severity"] == "INFO"
    assert payload["message"] == "variance computed"
    assert payload["logger"] == "edgar_capstone.tests.json_formatter"
    assert payload["logging.googleapis.com/trace"] == f"projects/test-project/traces/{trace_id}"
    assert payload["logging.googleapis.com/spanId"] == span_id
    assert payload["logging.googleapis.com/trace_sampled"] is True
    assert payload["trace_id"] == trace_id
    assert payload["logging.googleapis.com/sourceLocation"]["function"]
    assert payload["ticker"] == "NVDA"
    assert payload["latency_ms"] == 12.5
    assert payload["intent"] == "peer"
    assert payload["time"].endswith("+00:00")


def test_json_formatter_without_span_has_no_trace_and_reports_errors() -> None:
    formatter = CloudLoggingJsonFormatter(project_id="test-project", service_name="svc")
    test_logger = logging.getLogger("edgar_capstone.tests.json_formatter_err")
    payload = _formatter_line(test_logger, formatter, msg="tool failed", exception=True)

    assert "logging.googleapis.com/trace" not in payload
    assert payload["severity"] == "ERROR"
    assert "ValueError: boom" in payload["message"]
    assert payload["@type"] == ERROR_REPORTING_TYPE
    assert payload["serviceContext"]["service"] == "svc"


def test_json_formatter_cannot_override_reserved_keys() -> None:
    formatter = CloudLoggingJsonFormatter(project_id="test-project")
    test_logger = logging.getLogger("edgar_capstone.tests.json_formatter_reserved")
    payload = _formatter_line(
        test_logger,
        formatter,
        msg="real message",
        extra={"json_fields": {"severity": "DEBUG", "message": "spoofed"}},
    )
    assert payload["severity"] == "INFO"
    assert payload["message"] == "real message"


def test_configure_logging_routes_uvicorn_through_json_handler() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    try:
        fmt = configure_logging("json", "INFO", project_id="test-project", handler=handler)
        assert fmt == "json"
        logging.getLogger("uvicorn.error").info("Application startup complete.")
        logging.getLogger("uvicorn.access").info("GET / 200")  # superseded by our access log
        lines = _json_lines(stream)
        assert [ln["message"] for ln in lines] == ["Application startup complete."]
        assert lines[0]["logger"] == "uvicorn.error"
    finally:
        configure_logging()


def test_parse_trace_headers() -> None:
    w3c = parse_traceparent(f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01")
    assert w3c is not None and w3c.trace_id == UPSTREAM_TRACE_ID and w3c.sampled
    assert parse_traceparent("00-" + "0" * 32 + f"-{UPSTREAM_SPAN_ID}-01") is None
    assert parse_traceparent("garbage") is None

    gcp = parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID.upper()}/12345;o=1")
    assert gcp is not None
    assert gcp.trace_id == UPSTREAM_TRACE_ID
    assert gcp.span_id == format(12345, "016x")
    assert gcp.sampled is True
    assert parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID}") is not None
    assert parse_cloud_trace_context("not-a-trace") is None


# ---------------------------------------------------------------------------
# Span helpers & context propagation
# ---------------------------------------------------------------------------


def test_traced_decorator_preserves_signatures_and_records_args(
    spans: InMemorySpanExporter,
) -> None:
    @traced("unit.compute", record_args=("ticker", "fiscal_year"))
    def compute(ticker: str, fiscal_year: int, *, scale: float = 1.0) -> float:
        return fiscal_year * scale

    @traced()
    async def compute_async(value: int) -> int:
        return value * 2

    assert list(inspect.signature(compute).parameters) == ["ticker", "fiscal_year", "scale"]
    assert compute("NVDA", 2026, scale=2.0) == 4052.0
    assert asyncio.run(compute_async(21)) == 42

    finished = {s.name: s for s in spans.get_finished_spans()}
    assert finished["unit.compute"].attributes["edgar.ticker"] == "NVDA"
    assert finished["unit.compute"].attributes["edgar.fiscal_year"] == 2026
    assert any(name.endswith("compute_async") for name in finished)

    # ADK FunctionTool schema introspection still sees the real tool parameters.
    params = inspect.signature(calculate_period_variances).parameters
    assert {"tickers", "start_year", "end_year"} <= set(params)


def test_tool_resilience_span_marks_errors(spans: InMemorySpanExporter) -> None:
    @with_tool_resilience(tool_name="unit_failing_tool", max_retries=1)
    def failing_tool(tickers: str) -> dict:
        raise ValueError(f"bad tickers {tickers}")

    result = failing_tool("ZZZZ")
    assert result["status"] == "error"

    (span,) = [s for s in spans.get_finished_spans() if s.name == "tool.unit_failing_tool"]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["edgar.tool.status"] == "error"
    assert span.attributes["edgar.tool.error_type"] == "ValueError"
    assert span.attributes["edgar.tool.attempts"] == 1


def test_run_coro_sync_keeps_parent_span_across_worker_thread() -> None:
    async def inner() -> str | None:
        corr = current_trace_correlation()
        return corr.trace_id if corr else None

    async def outer() -> tuple[str, str | None]:
        with start_span("unit.outer") as span:
            parent_trace = format(span.get_span_context().trace_id, "032x")
            # Called from inside a running loop -> executes on a worker thread.
            return parent_trace, _run_coro_sync(inner())

    parent_trace, inner_trace = asyncio.run(outer())
    assert inner_trace == parent_trace


# ---------------------------------------------------------------------------
# FastAPI end-to-end: server span, child spans, headers, and access logs
# ---------------------------------------------------------------------------


def test_chat_request_continues_upstream_trace_end_to_end(
    spans: InMemorySpanExporter,
    json_log_stream: io.StringIO,
) -> None:
    client = TestClient(app)
    headers = {
        **_bearer_headers(),
        "traceparent": f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01",
    }
    resp = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "query": "Compare NVIDIA and Apple's revenue and operating income from 2024 to 2026",
            "tickers": "NVDA,AAPL",
            "start_year": 2024,
            "end_year": 2026,
            "execution_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-Trace-Id"] == UPSTREAM_TRACE_ID
    assert resp.headers["Server-Timing"].startswith("app;dur=")
    body = resp.json()
    assert body["observability"]["trace_id"] == UPSTREAM_TRACE_ID
    assert UPSTREAM_TRACE_ID in body["observability"]["cloud_trace_url"]

    trace_spans = [
        s
        for s in spans.get_finished_spans()
        if format(s.context.trace_id, "032x") == UPSTREAM_TRACE_ID
    ]
    by_id = {s.context.span_id: s for s in trace_spans}
    (server_span,) = [s for s in trace_spans if s.kind == SpanKind.SERVER]
    assert format(server_span.parent.span_id, "016x") == UPSTREAM_SPAN_ID
    assert server_span.attributes["edgar.chat.intent"] == body["intent"]
    assert server_span.attributes["edgar.request.latency_ms"] > 0

    names = {s.name for s in trace_spans}
    expected = {
        "auth.verify_jwt",
        "chat.rag_turn",
        "guardrail.inspect_user_prompt",
        "chat.resolve_scope",
        "orchestrator.classify_intent",
        "tool.calculate_period_variances",
        "chat.retrieve_citations",
        "react.execute",
    }
    assert expected <= names, expected - names

    def _descends_from_server(span) -> bool:
        current = span
        while current.parent is not None and current.parent.span_id in by_id:
            current = by_id[current.parent.span_id]
            if current is server_span:
                return True
        return False

    rag_turn = next(s for s in trace_spans if s.name == "chat.rag_turn")
    assert _descends_from_server(rag_turn)
    assert rag_turn.attributes["edgar.tickers"] == "NVDA,AAPL"
    tool_span = next(s for s in trace_spans if s.name == "tool.calculate_period_variances")
    assert _descends_from_server(tool_span)

    lines = _json_lines(json_log_stream)
    access = [ln for ln in lines if ln.get("event") == "http_request_completed"]
    assert len(access) == 1
    assert access[0]["http_route"] == "/api/v1/chat"
    assert access[0]["http_status"] == 200
    assert access[0]["latency_ms"] > 0
    assert access[0]["trace_id"] == UPSTREAM_TRACE_ID
    assert (
        access[0]["logging.googleapis.com/trace"]
        == f"projects/test-project/traces/{UPSTREAM_TRACE_ID}"
    )
    assert access[0]["request"]["path"] == "/api/v1/chat"
    # Every app log emitted while serving the request is correlated with the same trace.
    assert all(ln.get("trace_id") == UPSTREAM_TRACE_ID for ln in lines)

    # BigQuery telemetry rows (mirrored to the Cloud Logging sink) carry the same trace_id.
    sink_rows = {
        ln["log_sink_target"]: ln["event_payload"]
        for ln in lines
        if ln.get("component") == "sec-edgar-telemetry-sink"
    }
    assert {"llm_telemetry_events", "session_telemetry_events", "security_audit_logs"} <= set(
        sink_rows
    )
    assert all(row["trace_id"] == UPSTREAM_TRACE_ID for row in sink_rows.values())

    # Model calls are visible on the trace with model, token, and TTFT attributes.
    llm_events = [ev for s in trace_spans for ev in s.events if ev.name == "llm.call"]
    assert llm_events
    assert all("gen_ai.request.model" in ev.attributes for ev in llm_events)
    assert all("edgar.ttft_ms" in ev.attributes for ev in llm_events)


def test_x_cloud_trace_context_header_is_honoured(spans: InMemorySpanExporter) -> None:
    client = TestClient(app)
    resp = client.get(
        "/api/v1/companies",
        headers={
            **_bearer_headers(),
            "X-Cloud-Trace-Context": f"{UPSTREAM_TRACE_ID}/987654321;o=1",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-Trace-Id"] == UPSTREAM_TRACE_ID
    server_spans = [s for s in spans.get_finished_spans() if s.kind == SpanKind.SERVER]
    assert any(format(s.context.trace_id, "032x") == UPSTREAM_TRACE_ID for s in server_spans)


def test_health_probe_is_untraced_but_still_log_correlated(
    spans: InMemorySpanExporter,
) -> None:
    client = TestClient(app)
    resp = client.get(
        "/health", headers={"traceparent": f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01"}
    )
    assert resp.status_code == 200
    # Falls back to the inbound header so Cloud Run request logs still line up.
    assert resp.headers["X-Trace-Id"] == UPSTREAM_TRACE_ID
    assert not [s for s in spans.get_finished_spans() if s.kind == SpanKind.SERVER]
    observability = resp.json()["observability"]
    assert observability["tracing"]["enabled"] is True
    assert observability["tracing"]["fastapi_instrumented"] is True
    assert observability["logging"]["trace_correlation"] is True


def test_new_trace_started_without_upstream_context(spans: InMemorySpanExporter) -> None:
    client = TestClient(app)
    resp = client.get(
        "/api/v1/companies",
        headers={**_bearer_headers(), "Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    trace_id = resp.headers["X-Trace-Id"]
    assert len(trace_id) == 32 and trace_id != UPSTREAM_TRACE_ID
    assert resp.headers["X-Request-Id"] == trace_id
    # The browser UI may read the trace ID cross-origin (Vite dev server -> FastAPI).
    assert "x-trace-id" in resp.headers["access-control-expose-headers"].lower()
