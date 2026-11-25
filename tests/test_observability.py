"""Tests for OpenTelemetry tracing (Cloud Trace) and structured JSON logging (Cloud Logging)."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from edgar_capstone.agents.core.resilience import with_tool_resilience
from edgar_capstone.agents.runtime.live_runtime import _run_coro_sync
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
from edgar_capstone.observability.structured_logging import (
    ERROR_REPORTING_TYPE,
    TraceAwareTextFormatter,
    _level_name_for,
    _severity_for,
    resolve_log_level,
)
from edgar_capstone.observability.tracing import (
    TraceCorrelation,
    TracingConfig,
    _build_exporter,
    _env_ratio,
    bind_fallback_trace,
    reset_fallback_trace,
    sanitize_attributes,
)
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app
from tests.structured_report_helpers import ScriptedAnalystLlm, patch_http_chat_llms

UPSTREAM_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
UPSTREAM_SPAN_ID = "00f067aa0ba902b7"


class _GatedInMemorySpanExporter(InMemorySpanExporter):
    """Records spans only while `enabled` is True so unrelated test modules don't accumulate memory."""

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    def export(self, spans):  # type: ignore[override]
        if not self.enabled:
            return SpanExportResult.SUCCESS
        return super().export(spans)


_SPAN_EXPORTER = _GatedInMemorySpanExporter()
configure_tracing(extra_span_exporters=[_SPAN_EXPORTER])


@pytest.fixture()
def spans() -> Iterator[InMemorySpanExporter]:
    _SPAN_EXPORTER.clear()
    _SPAN_EXPORTER.enabled = True
    try:
        yield _SPAN_EXPORTER
    finally:
        _SPAN_EXPORTER.enabled = False
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
        level = kwargs.pop("level", "info")
        if kwargs.pop("exception", False):
            try:
                raise ValueError("boom")
            except ValueError:
                record_logger.exception(msg, **kwargs)
        elif level == "error":
            record_logger.error(msg, **kwargs)
        else:
            record_logger.info(msg, **kwargs)
    finally:
        record_logger.removeHandler(handler)
    return json.loads(stream.getvalue().strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Structured logging formatters, severity mapping & Error Reporting
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


def test_error_logs_follow_cloud_logging_and_error_reporting_conventions() -> None:
    """Plain ERROR logs stay normal entries; exceptions emit Error Reporting with immutable location."""
    formatter = CloudLoggingJsonFormatter(
        project_id="test-project", service_name="svc", service_version="rev-1"
    )
    test_logger = logging.getLogger("edgar_capstone.tests.json_formatter_err")

    # Outside any span or fallback trace: no trace key emitted
    no_span = _formatter_line(test_logger, formatter, msg="tool failed", exception=True)
    assert "logging.googleapis.com/trace" not in no_span
    assert no_span["@type"] == ERROR_REPORTING_TYPE

    trace_token = bind_fallback_trace(
        TraceCorrelation(trace_id="a" * 32, span_id="b" * 16, sampled=True)
    )
    try:
        plain = _formatter_line(
            test_logger,
            formatter,
            msg="GET /api/v1/chat -> 503",
            level="error",
            extra={"json_fields": {"context": "should-not-win"}},
        )
        with_exc = _formatter_line(
            test_logger,
            formatter,
            msg="tool failed",
            exception=True,
            extra={"json_fields": {"context": "should-not-win"}},
        )
    finally:
        reset_fallback_trace(trace_token)

    for payload in (plain, with_exc):
        assert payload["severity"] == "ERROR"
        assert payload["logging.googleapis.com/trace"] == f"projects/test-project/traces/{'a' * 32}"
        assert payload["logging.googleapis.com/spanId"] == "b" * 16
        assert payload["logging.googleapis.com/trace_sampled"] is True
        assert payload["serviceContext"] == {"service": "svc", "version": "rev-1"}

    assert "@type" not in plain
    assert plain["context"] == "should-not-win"

    assert with_exc["@type"] == ERROR_REPORTING_TYPE
    assert "ValueError: boom" in with_exc["message"]
    location = with_exc["context"]["reportLocation"]
    assert location["functionName"] == "_formatter_line"
    assert isinstance(location["lineNumber"], int)


def test_formatters_protect_reserved_keys_and_do_not_mutate_log_record() -> None:
    rec = logging.LogRecord("edgar.test", logging.INFO, __file__, 10, "hello %s", ("world",), None)
    rec.json_fields = {
        "trace_id": "hijacked",
        "logger": "spoofed",
        "severity": "FAKE",
        "message": "spoofed",
        "custom_field": 99,
    }
    before = dict(rec.__dict__)
    token = bind_fallback_trace(
        TraceCorrelation(trace_id=UPSTREAM_TRACE_ID, span_id=UPSTREAM_SPAN_ID, sampled=True)
    )
    try:
        text_out = TraceAwareTextFormatter().format(rec)
        assert UPSTREAM_TRACE_ID in text_out
        assert dict(rec.__dict__) == before

        json_out = json.loads(CloudLoggingJsonFormatter(project_id="proj-x").format(rec))
        assert json_out["trace_id"] == UPSTREAM_TRACE_ID
        assert json_out["logger"] == "edgar.test"
        assert json_out["severity"] == "INFO"
        assert json_out["message"] == "hello world"
        assert json_out["custom_field"] == 99
    finally:
        reset_fallback_trace(token)


def test_severity_and_level_name_helpers() -> None:
    assert _severity_for(logging.NOTSET) == "DEFAULT"
    assert _severity_for(5) == "DEBUG"
    assert _severity_for(25) == "INFO"
    assert _severity_for(35) == "WARNING"
    assert _severity_for(45) == "ERROR"
    assert _severity_for(60) == "CRITICAL"

    assert _level_name_for(logging.NOTSET) == "NOTSET"
    assert _level_name_for(logging.WARNING) == "WARNING"
    assert _level_name_for(37) == "LEVEL_37"

    assert resolve_log_level(logging.DEBUG) == logging.DEBUG
    assert resolve_log_level("20") == logging.INFO
    assert resolve_log_level("WARNING") == logging.WARNING


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


def test_parse_trace_headers_and_bounds() -> None:
    w3c = parse_traceparent(f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01")
    assert w3c == TraceCorrelation(
        trace_id=UPSTREAM_TRACE_ID, span_id=UPSTREAM_SPAN_ID, sampled=True
    )
    assert parse_traceparent(f"ff-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01") is None
    assert parse_traceparent(f"00-{'0' * 32}-{UPSTREAM_SPAN_ID}-01") is None
    assert parse_traceparent(f"00-{UPSTREAM_TRACE_ID}-{'0' * 16}-01") is None
    assert parse_traceparent("garbage") is None

    gcp = parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID.upper()}/12345;o=1")
    assert gcp is not None
    assert gcp.trace_id == UPSTREAM_TRACE_ID
    assert gcp.span_id == format(12345, "016x")
    assert gcp.sampled is True
    assert parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID}") is not None
    assert parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID}/0;o=1") is None
    assert parse_cloud_trace_context(f"{UPSTREAM_TRACE_ID}/{1 << 64};o=1") is None
    assert parse_cloud_trace_context(f"{'0' * 32}/12;o=1") is None
    assert parse_cloud_trace_context("not-a-trace") is None


def test_tracing_config_env_ratio_exporter_and_attribute_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGAR_TEST_RATIO", raising=False)
    assert _env_ratio("EDGAR_TEST_RATIO", 0.5) == 0.5
    monkeypatch.setenv("EDGAR_TEST_RATIO", "bad-float")
    assert _env_ratio("EDGAR_TEST_RATIO", 0.5) == 0.5
    monkeypatch.setenv("EDGAR_TEST_RATIO", "1.75")
    assert _env_ratio("EDGAR_TEST_RATIO", 0.5) == 1.0
    monkeypatch.setenv("EDGAR_TEST_RATIO", "-0.25")
    assert _env_ratio("EDGAR_TEST_RATIO", 0.5) == 0.0

    exp_none, name_none = _build_exporter(TracingConfig(resolved_exporter="none"))
    assert exp_none is None and name_none == "none"

    exp_con, name_con = _build_exporter(TracingConfig(resolved_exporter="console"))
    assert exp_con is not None and name_con == "console"

    # Simulate missing optional OTLP package
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    exp_otlp, name_otlp = _build_exporter(TracingConfig(resolved_exporter="otlp"))
    assert exp_otlp is None and name_otlp == "none"

    # GCP mode uses ADK OTLP hooks in configure_tracing; _build_exporter handles console/otlp/none
    exp_gcp, name_gcp = _build_exporter(TracingConfig(resolved_exporter="gcp", project_id="p-1"))
    assert name_gcp == "none" and exp_gcp is None

    long_str = "x" * 700
    clean = sanitize_attributes(
        {
            "drop_me": None,
            "flag": True,
            "count": 3,
            "long": long_str,
            "str_list": ["NVDA", "AAPL", None],
            "mixed_list": [1, "AAPL"],
            "obj": SimpleNamespace(id=7),
        }
    )
    assert "drop_me" not in clean
    assert clean["flag"] is True and clean["count"] == 3
    assert clean["long"].endswith("…") and len(clean["long"]) == 513
    assert clean["str_list"] == ["NVDA", "AAPL"]
    assert clean["mixed_list"] == ["1", "AAPL"]
    assert "id=7" in clean["obj"]


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
            return parent_trace, _run_coro_sync(inner())

    parent_trace, inner_trace = asyncio.run(outer())
    assert inner_trace == parent_trace


# ---------------------------------------------------------------------------
# FastAPI end-to-end: server span, child spans, headers, and access logs
# ---------------------------------------------------------------------------


def test_chat_request_continues_upstream_trace_end_to_end(
    spans: InMemorySpanExporter,
    json_log_stream: io.StringIO,
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_http_chat_llms(
        monkeypatch,
        reasoning_llm=ScriptedAnalystLlm(
            tool_calls=[
                (
                    "calculate_period_variances",
                    {"tickers": "NVDA,AAPL", "start_year": 2024, "end_year": 2026},
                )
            ],
            headline="NVDA and AAPL expanded revenue and operating income from FY2024 to FY2026.",
            intent="multi_company_peer_comparison",
            key_metrics=[("NVDA", "revenue"), ("AAPL", "revenue")],
        ),
        skip_background_judge=True,
    )
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
        "orchestrator.infer_intent_from_tools",
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
    assert all(ln.get("trace_id") == UPSTREAM_TRACE_ID for ln in lines)

    sink_rows = {
        ln["log_sink_target"]: ln["event_payload"]
        for ln in lines
        if ln.get("component") == "sec-edgar-telemetry-sink"
    }
    assert {"llm_telemetry_events", "session_telemetry_events", "security_audit_logs"} <= set(
        sink_rows
    )
    assert all(row["trace_id"] == UPSTREAM_TRACE_ID for row in sink_rows.values())

    llm_events = [ev for s in trace_spans for ev in s.events if ev.name == "llm.call"]
    assert llm_events
    assert all("gen_ai.request.model" in ev.attributes for ev in llm_events)
    assert all("edgar.ttft_ms" in ev.attributes for ev in llm_events)


def test_x_cloud_trace_context_header_is_honoured(spans: InMemorySpanExporter) -> None:
    client = TestClient(app)
    resp = client.get(
        "/api/v1/auth/me",
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
    assert resp.headers["X-Trace-Id"] == UPSTREAM_TRACE_ID
    assert not [s for s in spans.get_finished_spans() if s.kind == SpanKind.SERVER]
    observability = resp.json()["observability"]
    assert observability["tracing"]["enabled"] is True
    assert observability["tracing"]["fastapi_instrumented"] is True
    assert observability["logging"]["trace_correlation"] is True


def test_new_trace_started_without_upstream_context(spans: InMemorySpanExporter) -> None:
    client = TestClient(app)
    resp = client.get(
        "/api/v1/auth/me",
        headers={**_bearer_headers(), "Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    trace_id = resp.headers["X-Trace-Id"]
    assert len(trace_id) == 32 and trace_id != UPSTREAM_TRACE_ID
    assert resp.headers["X-Request-Id"] == trace_id
    assert "x-trace-id" in resp.headers["access-control-expose-headers"].lower()


def test_genai_otel_env_and_gcp_otlp_exporter_auto_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies GenAI OTel env defaults, opt-outs, and ADK `get_gcp_exporters` OTLP hooks."""
    from edgar_capstone.observability.tracing import (
        ADK_CAPTURE_CONTENT_ENV,
        AGENT_ENGINE_ENABLE_TELEMETRY_ENV,
        OTEL_GENAI_CAPTURE_CONTENT_ENV,
        OTEL_GENAI_EMIT_EVENT_ENV,
        OTEL_SEMCONV_STABILITY_OPT_IN_ENV,
        _build_gcp_otlp_hooks,
        flush_tracing,
        tracing_status_metadata,
    )

    # Active app startup config populates GenAI Inputs/Outputs env vars and instruments GenAI SDK.
    status = tracing_status_metadata()
    assert status["genai_instrumented"] is True
    assert status["requests_instrumented"] is True
    assert status["grpc_instrumented"] is True
    assert status["capture_message_content"] is True
    assert status["capture_message_content_mode"] == "SPAN_AND_EVENT"
    assert status["semconv_stability_opt_in"] == "gen_ai_latest_experimental"
    assert status["emit_genai_events"] is True
    assert status["agent_engine_telemetry_enabled"] is True
    assert flush_tracing() is True

    # Explicit opt-out via EDGAR_TRACE_CAPTURE_CONTENT=false or NO_CONTENT
    monkeypatch.delenv(OTEL_GENAI_CAPTURE_CONTENT_ENV, raising=False)
    monkeypatch.delenv(OTEL_GENAI_EMIT_EVENT_ENV, raising=False)
    monkeypatch.setenv("EDGAR_TRACE_CAPTURE_CONTENT", "false")
    opt_out = TracingConfig.from_env()
    assert opt_out.capture_message_content is False
    assert opt_out.capture_message_content_mode == "NO_CONTENT"
    assert opt_out.emit_genai_events is False

    # Explicit OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT takes precedence
    monkeypatch.setenv(OTEL_GENAI_CAPTURE_CONTENT_ENV, "SPAN_ONLY")
    span_only = TracingConfig.from_env()
    assert span_only.capture_message_content is True
    assert span_only.capture_message_content_mode == "SPAN_ONLY"
    assert span_only.emit_genai_events is False

    # ADK's get_gcp_exporters wires both OTLP span and OTLP log processors to telemetry.googleapis.com
    captured_kwargs: dict[str, Any] = {}
    fake_span_proc = SimpleNamespace(name="otlp_span_proc")
    fake_log_proc = SimpleNamespace(name="otlp_log_proc")

    def fake_get_gcp_exporters(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            span_processors=[fake_span_proc],
            log_record_processors=[fake_log_proc],
        )

    import google.adk.telemetry.google_cloud as adk_gcp
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (SimpleNamespace(), "proj-1"))
    monkeypatch.setattr(adk_gcp, "get_gcp_exporters", fake_get_gcp_exporters)
    span_procs, log_procs, exp_name = _build_gcp_otlp_hooks(
        TracingConfig(resolved_exporter="gcp", project_id="proj-1")
    )
    assert exp_name == "gcp_otlp"
    assert span_procs == [fake_span_proc]
    assert log_procs == [fake_log_proc]
    assert captured_kwargs["enable_cloud_tracing"] is True
    assert captured_kwargs["enable_cloud_logging"] is True
    assert captured_kwargs["enable_cloud_metrics"] is False

    # Verify canonical env vars are exported in process environment for ADK & util-genai
    import os

    assert os.environ.get(OTEL_SEMCONV_STABILITY_OPT_IN_ENV) == "gen_ai_latest_experimental"
    assert os.environ.get(ADK_CAPTURE_CONTENT_ENV) == "true"
    assert os.environ.get(AGENT_ENGINE_ENABLE_TELEMETRY_ENV) == "true"


def test_genai_sdk_and_adk_populate_cloud_trace_inputs_and_outputs(
    spans: InMemorySpanExporter,
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies GenAI SDK spans/events and ADK tool spans carry Cloud Trace Inputs/Outputs attrs."""
    import httpx
    from google import genai
    from google.genai import types as genai_types
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    log_exporter = InMemoryLogRecordExporter()
    configure_tracing(extra_log_record_processors=[SimpleLogRecordProcessor(log_exporter)])

    # 1. Mocked google.genai.Client exercises GoogleGenAiSdkInstrumentor end-to-end (offline).
    def _mock_gemini_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "NVDA revenue rose sharply in FY2026."}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 24,
                    "candidatesTokenCount": 9,
                    "totalTokenCount": 33,
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "resp-test-1",
            },
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(_mock_gemini_handler))
    genai_client = genai.Client(
        api_key="test-key",
        http_options=genai_types.HttpOptions(httpx_client=mock_http),
    )
    with start_span("unit.genai_parent") as parent_span:
        parent_trace_id = format(parent_span.get_span_context().trace_id, "032x")
        reply = genai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Summarize NVIDIA FY2026 revenue growth.",
            config=genai_types.GenerateContentConfig(
                system_instruction="You are a financial analyst.",
                max_output_tokens=64,
            ),
        )
    assert "NVDA revenue rose sharply" in (reply.text or "")

    genai_spans = [
        s
        for s in spans.get_finished_spans()
        if format(s.context.trace_id, "032x") == parent_trace_id
        and s.attributes
        and s.attributes.get("gen_ai.operation.name") == "generate_content"
    ]
    assert len(genai_spans) == 1
    g_attrs = genai_spans[0].attributes or {}
    assert "Summarize NVIDIA FY2026 revenue growth." in str(g_attrs.get("gen_ai.input.messages"))
    assert "NVDA revenue rose sharply" in str(g_attrs.get("gen_ai.output.messages"))
    assert "You are a financial analyst." in str(g_attrs.get("gen_ai.system_instructions"))

    completion_events = [
        rec
        for rec in log_exporter.get_finished_logs()
        if rec.log_record.event_name == "gen_ai.client.inference.operation.details"
        and format(rec.log_record.trace_id, "032x") == parent_trace_id
    ]
    assert len(completion_events) == 1
    ev_attrs = completion_events[0].log_record.attributes or {}
    assert "gen_ai.input.messages" in ev_attrs
    assert "gen_ai.output.messages" in ev_attrs
    assert "gen_ai.system_instructions" in ev_attrs

    # 2. ADK tool execution spans also carry tool call arguments & results for Cloud Trace.
    patch_http_chat_llms(
        monkeypatch,
        reasoning_llm=ScriptedAnalystLlm(
            tool_calls=[
                (
                    "calculate_period_variances",
                    {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
                )
            ],
            headline="NVDA revenue expanded sharply from FY2024 to FY2026.",
            intent="longitudinal_single_company",
            key_metrics=[("NVDA", "revenue")],
        ),
        skip_background_judge=True,
    )
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat",
        headers={
            **_bearer_headers(),
            "traceparent": f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01",
        },
        json={
            "query": "Analyze NVIDIA revenue growth from 2024 to 2026",
            "tickers": "NVDA",
            "start_year": 2024,
            "end_year": 2026,
        },
    )
    assert resp.status_code == 200
    tool_spans = [
        s
        for s in spans.get_finished_spans()
        if format(s.context.trace_id, "032x") == UPSTREAM_TRACE_ID
        and s.attributes
        and s.attributes.get("gen_ai.operation.name") == "execute_tool"
    ]
    assert tool_spans
    assert "gen_ai.tool.call.arguments" in (tool_spans[0].attributes or {})
    assert "gen_ai.tool.call.result" in (tool_spans[0].attributes or {})
