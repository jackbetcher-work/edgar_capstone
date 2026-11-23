"""Tests for BigQuery Telemetry & Audit Streaming Sinks, Firestore Feedback Store, and /health."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.agents.core.schemas import ContextBreakdown
from edgar_capstone.agents.orchestration.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.runtime.latency_tracker import (
    ContextLatencyTracker,
    attribute_context_latency,
    get_context_latency_tracker,
)
from edgar_capstone.agents.runtime.model_config import get_model_strategy_metadata
from edgar_capstone.agents.tools import query_analytics_view
from edgar_capstone.security import (
    GOOGLE_IAP_ISSUER,
    GOOGLE_IAP_JWKS_URI,
    S2S_AGENT_ASSERTION_HEADER,
    AuthSettings,
    IAPJwtVerifier,
    JwksKeyStore,
    clear_security_audit_events,
    ec_public_key_to_jwk,
    get_agent_s2s_broker,
    get_auth_verifier,
    record_security_audit_event,
    sign_jwt_es256,
)
from edgar_capstone.server import app
from edgar_capstone.telemetry import (
    BigQueryTelemetrySink,
    FirestoreFeedbackStore,
    TelemetryConfig,
    TelemetryRateLimitError,
    TelemetrySinkError,
    configure_telemetry_stores,
    decode_firestore_document,
    encode_firestore_fields,
    reset_telemetry_stores,
)
from tests.structured_report_helpers import ScriptedAnalystLlm, patch_http_chat_llms

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IAP_AUDIENCE = "/projects/991122334455/global/backendServices/887766554433"


class _FakeGcpCredentials:
    """Synthetic OAuth2 credentials stub for offline `httpx.MockTransport` verification."""

    def __init__(self) -> None:
        self.valid = True
        self.token = "test-gcp-oauth-bearer"
        self.refresh_calls = 0

    def refresh(self, request: Any) -> None:
        self.refresh_calls += 1
        self.valid = True
        self.token = "test-gcp-oauth-bearer"


class _MissingCredentials:
    """Credentials stub raising `DefaultCredentialsError` to verify `auto` mode fallback."""

    valid = False
    token = None

    def refresh(self, request: Any) -> None:
        raise DefaultCredentialsError("ADC unavailable in test sandbox")


@pytest.fixture(autouse=True)
def _isolated_telemetry_state():
    """Ensure every test starts and ends with clean telemetry/feedback/audit singletons."""
    reset_telemetry_stores()
    clear_security_audit_events()
    get_context_latency_tracker().reset()
    yield
    reset_telemetry_stores()
    clear_security_audit_events()
    get_context_latency_tracker().reset()


@pytest.fixture()
def auth_client_env(session_idp_keys: dict[str, Any]):
    """Isolated IAP ES256 verifier supporting Admin, Analyst, Viewer, and Agent S2S principals."""
    ec_signing_key = session_idp_keys["iap_ec_key"]
    jwks_payload = {"keys": [ec_public_key_to_jwk(ec_signing_key.public_key(), kid="iap-telem-k1")]}

    def _jwks_transport(request: httpx.Request) -> httpx.Response:
        if str(request.url) == GOOGLE_IAP_JWKS_URI:
            return httpx.Response(200, json=jwks_payload)
        return httpx.Response(404, json={"error": "not_found"})

    jwks_store = JwksKeyStore(
        http_client=httpx.Client(transport=httpx.MockTransport(_jwks_transport))
    )
    settings = AuthSettings(
        audiences=(TEST_IAP_AUDIENCE,),
        okta_issuer_uri=None,
        okta_jwks_uri=None,
        okta_client_id=None,
        google_oauth_client_id=None,
        allowed_workspace_domains=("fde-workspace.corp",),
        cloud_identity_domain="fde-workspace.corp",
        required_analyst_groups=(),
        admin_workspace_groups=("sec-edgar-admins@fde-workspace.corp",),
        analyst_workspace_groups=("sec-edgar-analysts@fde-workspace.corp",),
        viewer_workspace_groups=("sec-edgar-viewers@fde-workspace.corp",),
        allow_gcp_service_accounts=True,
        strict_production_mode=True,
    )
    verifier = IAPJwtVerifier(settings=settings, jwks_store=jwks_store)

    def _mint_token(
        email: str,
        *,
        groups: list[str] | None = None,
        hosted_domain: str | None = "fde-workspace.corp",
    ) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": GOOGLE_IAP_ISSUER,
            "sub": f"accounts.google.com:{email}",
            "email": email,
            "aud": TEST_IAP_AUDIENCE,
            "iat": now,
            "exp": now + 900,
        }
        if hosted_domain is not None:
            claims["hd"] = hosted_domain
        if groups is not None:
            claims["groups"] = groups
        return sign_jwt_es256(claims, ec_signing_key, kid="iap-telem-k1")

    app.dependency_overrides[get_auth_verifier] = lambda: verifier
    try:
        yield {
            "client": TestClient(app),
            "mint_token": _mint_token,
        }
    finally:
        app.dependency_overrides.pop(get_auth_verifier, None)


# ---------------------------------------------------------------------------
# 1. Token Cost & Cached-Token Savings Attribution
# ---------------------------------------------------------------------------


def test_pro_and_flash_token_cost_and_cached_savings_math() -> None:
    """Verify exact USD token cost and cached-context discount math across model tiers."""
    meta = get_model_strategy_metadata()
    pricing = meta["pricing_per_1m_tokens_usd"]
    assert pricing["gemini-2.5-pro"]["uncached_input_usd_per_1m"] == pytest.approx(1.25)
    assert pricing["gemini-2.5-pro"]["cached_input_usd_per_1m"] == pytest.approx(0.3125)
    assert pricing["gemini-2.5-pro"]["output_usd_per_1m"] == pytest.approx(10.0)
    assert pricing["gemini-3.8-flash"]["uncached_input_usd_per_1m"] == pytest.approx(0.75)
    assert pricing["gemini-3.8-flash"]["cached_input_usd_per_1m"] == pytest.approx(0.075)
    assert pricing["gemini-3.8-flash"]["output_usd_per_1m"] == pytest.approx(3.75)
    assert pricing["gemini-3.5-flash-lite"]["uncached_input_usd_per_1m"] == pytest.approx(0.30)
    assert pricing["gemini-3.5-flash-lite"]["cached_input_usd_per_1m"] == pytest.approx(0.03)
    assert pricing["gemini-3.5-flash-lite"]["output_usd_per_1m"] == pytest.approx(2.50)
    assert pricing["gemini-3.5-flash"]["uncached_input_usd_per_1m"] == pytest.approx(1.50)
    assert pricing["gemini-3.5-flash"]["cached_input_usd_per_1m"] == pytest.approx(0.15)
    assert pricing["gemini-3.5-flash"]["output_usd_per_1m"] == pytest.approx(9.00)

    # 100k cached 10-K tokens + 20k uncached prompt/variance tokens + 2k output tokens on Gemini 2.5 Pro
    pro_breakdown = ContextBreakdown(
        system_instruction_tokens=2_000,
        memory_and_history_tokens=3_000,
        variance_payload_tokens=10_000,
        cached_10k_tokens=100_000,
        uncached_10k_and_tool_tokens=0,
        user_prompt_tokens=5_000,
        total_context_tokens=120_000,
        cached_context_tokens=100_000,
        uncached_context_tokens=20_000,
        cache_hit=True,
    )
    pro_rec = attribute_context_latency(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="final_synthesis",
        context_breakdown=pro_breakdown,
        output_tokens=1_500,
        thoughts_tokens=500,
        raw_inference_latency_ms=1450.0,
        context_prep_latency_ms=35.0,
        session_id="sess-cost-01",
        user_id="analyst@fde-workspace.corp",
    )
    # Expected Pro input: (20k/1M)*1.25 + (100k/1M)*0.3125 = 0.025 + 0.03125 = 0.05625
    # Uncached input baseline: (120k/1M)*1.25 = 0.15 -> savings = 0.15 - 0.05625 = 0.09375
    # Expected Pro output (1500 + 500 = 2000 tokens): (2k/1M)*10.0 = 0.02
    assert pro_rec.input_cost_usd == pytest.approx(0.05625)
    assert pro_rec.output_cost_usd == pytest.approx(0.02)
    assert pro_rec.total_cost_usd == pytest.approx(0.07625)
    assert pro_rec.uncached_baseline_cost_usd == pytest.approx(0.17)
    assert pro_rec.cached_token_savings_usd == pytest.approx(0.09375)
    assert pro_rec.session_id == "sess-cost-01"
    assert pro_rec.user_id == "analyst@fde-workspace.corp"

    # 40k cached + 10k uncached + 1k output on Gemini 3.5 Flash-Lite
    flash_breakdown = ContextBreakdown(
        system_instruction_tokens=2_000,
        cached_10k_tokens=40_000,
        user_prompt_tokens=8_000,
        total_context_tokens=50_000,
        cached_context_tokens=40_000,
        uncached_context_tokens=10_000,
        cache_hit=True,
    )
    tracker = ContextLatencyTracker()
    r1 = tracker.record_call(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        context_breakdown=pro_breakdown,
        output_tokens=1_500,
        thoughts_tokens=500,
        raw_inference_latency_ms=1200.0,
        session_id="sess-cost-01",
        user_id="analyst@fde-workspace.corp",
    )
    r2 = tracker.record_call(
        agent_name="SectorMapSummarizerAgent",
        model_name="gemini-3.5-flash-lite",
        task_category="context_compaction",
        context_breakdown=flash_breakdown,
        output_tokens=1_000,
        raw_inference_latency_ms=420.0,
        session_id="sess-cost-01",
        user_id="analyst@fde-workspace.corp",
    )
    # Expected Flash-Lite input: (10k/1M)*0.30 + (40k/1M)*0.03 = 0.003 + 0.0012 = 0.0042
    # Baseline input: (50k/1M)*0.30 = 0.015 -> savings = 0.0108; output = (1k/1M)*2.50 = 0.0025
    assert r2.total_cost_usd == pytest.approx(0.0067)
    assert r2.cached_token_savings_usd == pytest.approx(0.0108)

    summary = tracker.summarize()
    assert summary.total_calls == 2
    assert summary.total_output_tokens == 2_500
    assert summary.total_thoughts_tokens == 500
    assert summary.total_cost_usd == pytest.approx(r1.total_cost_usd + r2.total_cost_usd)
    assert summary.total_cached_token_savings_usd == pytest.approx(0.09375 + 0.0108)
    assert summary.total_uncached_baseline_cost_usd == pytest.approx(0.17 + 0.0175)
    expected_pct = round(((0.09375 + 0.0108) / (0.17 + 0.0175)) * 100.0, 2)
    assert summary.cache_cost_savings_pct == pytest.approx(expected_pct)
    assert summary.by_model["gemini-2.5-pro"]["total_cost_usd"] == pytest.approx(0.07625)
    assert summary.by_model["gemini-3.5-flash-lite"]["cached_token_savings_usd"] == pytest.approx(
        0.0108
    )


# ---------------------------------------------------------------------------
# 2. BigQuery Streaming Sink (`insertAll`), Cloud Logging & SQL Readback
# ---------------------------------------------------------------------------


def test_bigquery_streaming_insert_all_for_llm_session_and_audit_tables(
    caplog: pytest.LogCaptureFixture,
    fake_gcp_adapters: Any,
) -> None:
    """Verify `tabledata.insertAll` streaming and SQL readback across all 3 BigQuery sink tables."""
    streamed_requests: dict[str, list[dict[str, Any]]] = {
        "llm_telemetry_events": [],
        "session_telemetry_events": [],
        "security_audit_logs": [],
    }
    executed_queries: list[dict[str, Any]] = []

    def _bq_transport(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-gcp-oauth-bearer"
        url_path = request.url.path
        body = json.loads(request.content.decode("utf-8"))

        if url_path.endswith("/insertAll"):
            table_name = url_path.split("/tables/")[1].split("/")[0]
            streamed_requests[table_name].append(body)
            return httpx.Response(200, json={"kind": "bigquery#tableDataInsertAllResponse"})

        if url_path.endswith("/queries"):
            executed_queries.append(body)
            sql = body.get("query", "")
            if "security_audit_logs" in sql:
                return httpx.Response(
                    200,
                    json={
                        "schema": {
                            "fields": [
                                {"name": "event_id", "type": "STRING"},
                                {"name": "recorded_at", "type": "STRING"},
                                {"name": "recorded_at_epoch", "type": "FLOAT64"},
                                {"name": "event_type", "type": "STRING"},
                                {"name": "status", "type": "STRING"},
                                {"name": "resource", "type": "STRING"},
                                {"name": "tickers", "type": "STRING", "mode": "REPEATED"},
                                {"name": "user_email", "type": "STRING"},
                                {"name": "user_sub", "type": "STRING"},
                                {"name": "hosted_domain", "type": "STRING"},
                                {"name": "identity_provider", "type": "STRING"},
                                {"name": "roles", "type": "STRING", "mode": "REPEATED"},
                                {
                                    "name": "cloud_identity_groups",
                                    "type": "STRING",
                                    "mode": "REPEATED",
                                },
                                {"name": "agent_identity", "type": "STRING"},
                                {"name": "spiffe_id", "type": "STRING"},
                                {"name": "iap_assertion_verified", "type": "BOOL"},
                                {"name": "detail", "type": "STRING"},
                            ]
                        },
                        "rows": [
                            {
                                "f": [
                                    {"v": "aud-bq-001"},
                                    {"v": "2026-11-16T12:00:00Z"},
                                    {"v": "1794830400.0"},
                                    {"v": "filing_10k_section_read"},
                                    {"v": "allowed"},
                                    {"v": "/api/v1/section"},
                                    {"v": [{"v": "NVDA"}]},
                                    {"v": "analyst@fde-workspace.corp"},
                                    {"v": "accounts.google.com:analyst"},
                                    {"v": "fde-workspace.corp"},
                                    {"v": "google_iap"},
                                    {"v": [{"v": "analyst"}]},
                                    {"v": [{"v": "sec-edgar-analysts@fde-workspace.corp"}]},
                                    {"v": None},
                                    {"v": None},
                                    {"v": "true"},
                                    {"v": "FY2026 item_7"},
                                ]
                            }
                        ],
                    },
                )
            return httpx.Response(200, json={"schema": {"fields": []}, "rows": []})

        return httpx.Response(404, json={"error": "unexpected_url"})

    cfg = TelemetryConfig(
        telemetry_backend="gcp",
        feedback_backend="local",
        project_id="iron-radio-514217-q7",
        bq_dataset_id="sec_edgar",
    )
    sink = BigQueryTelemetrySink(
        cfg,
        http_client=httpx.Client(transport=httpx.MockTransport(_bq_transport)),
        credentials=_FakeGcpCredentials(),
    )
    configure_telemetry_stores(telemetry_sink=sink)

    llm_turn_1 = ScriptedAnalystLlm(
        tool_calls=[
            (
                "calculate_period_variances",
                {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
            )
        ],
        headline="NVDA FY2024-FY2026 revenue and operating income expanded strongly.",
        intent="longitudinal_single_company",
        key_metrics=[("NVDA", "revenue")],
    )
    llm_turn_2 = ScriptedAnalystLlm(
        tool_calls=[
            (
                "calculate_period_variances",
                {"tickers": "AAPL,MSFT", "start_year": 2024, "end_year": 2025},
            )
        ],
        headline="AAPL and MSFT delivered steady operating margin expansion.",
        intent="multi_company_peer_comparison",
        key_metrics=[("AAPL", "revenue"), ("MSFT", "revenue")],
    )

    with caplog.at_level(logging.INFO, logger="edgar_capstone.cloud_logging_sink"):
        # 1. Two deterministic scripted RAG turns on one session stream both LLM call telemetry
        #    and session telemetry into the mocked BigQuery sink (zero live LLM latency).
        turn_1 = run_conversational_rag_turn(
            query="Analyze NVDA FY2024 to FY2026 revenue and operating income variances.",
            tickers="NVDA",
            start_year=2024,
            end_year=2026,
            session_id="sess-bq-stream-1",
            user_id="analyst@fde-workspace.corp",
            reasoning_llm=llm_turn_1,
            skip_background_judge=True,
        )
        turn_2 = run_conversational_rag_turn(
            query="Compare AAPL and MSFT operating margin drivers from FY2024 to FY2025.",
            tickers="AAPL,MSFT",
            start_year=2024,
            end_year=2025,
            session_id="sess-bq-stream-1",
            user_id="analyst@fde-workspace.corp",
            reasoning_llm=llm_turn_2,
            skip_background_judge=True,
        )
        assert turn_1["status"] == "success" and turn_2["status"] == "success"
        assert turn_1["answer_verified"] is True and turn_2["answer_verified"] is True
        assert turn_1["execution_mode"] == "live_adk"

        # 2. Security / 10-K access audit log streaming
        record_security_audit_event(
            event_type="filing_10k_section_read",
            user=None,
            resource="/api/v1/section",
            tickers=["NVDA"],
            status_outcome="allowed",
            detail="FY2026 item_7",
        )

    # Verify LLM telemetry rows streamed with deterministic insertId == call_id
    assert len(streamed_requests["llm_telemetry_events"]) >= 2
    first_llm_req = streamed_requests["llm_telemetry_events"][0]
    assert first_llm_req["skipInvalidRows"] is False
    llm_row_envelope = first_llm_req["rows"][0]
    assert llm_row_envelope["insertId"] == llm_row_envelope["json"]["call_id"]
    assert llm_row_envelope["json"]["session_id"] == "sess-bq-stream-1"
    assert llm_row_envelope["json"]["total_cost_usd"] > 0.0

    # Verify multi-turn session analytics streamed with cumulative session tracking
    assert len(streamed_requests["session_telemetry_events"]) == 2
    sess_row_1 = streamed_requests["session_telemetry_events"][0]["rows"][0]["json"]
    sess_row_2 = streamed_requests["session_telemetry_events"][1]["rows"][0]["json"]
    assert sess_row_1["session_id"] == "sess-bq-stream-1"
    assert sess_row_1["session_cumulative_turns"] == 1
    assert sess_row_1["tickers_analyzed"] == ["NVDA"]
    assert sess_row_1["evaluation_grade"] == "pending"
    assert sess_row_1["evaluation_score"] is None
    assert sess_row_1["report_generated"] is True
    assert sess_row_1["execution_mode"] == "live_adk"
    assert len(sess_row_1["tool_invocations"]) >= 2

    assert sess_row_2["session_cumulative_turns"] == 2
    assert sess_row_2["session_cumulative_companies_count"] == 3  # NVDA, AAPL, MSFT
    assert sess_row_2["session_cumulative_reports_count"] == 2
    assert turn_1["context_and_memory"]["session_telemetry"]["turn_index"] == 1
    assert turn_2["context_and_memory"]["session_telemetry"]["turn_index"] == 2

    # Verify security audit log streamed to BigQuery
    assert len(streamed_requests["security_audit_logs"]) == 1
    audit_row = streamed_requests["security_audit_logs"][0]["rows"][0]["json"]
    assert audit_row["event_type"] == "filing_10k_section_read"
    assert audit_row["tickers"] == ["NVDA"]

    # Verify Cloud Logging structured sink records were also emitted (as jsonPayload fields)
    sink_logs = [
        r.json_fields for r in caplog.records if r.name == "edgar_capstone.cloud_logging_sink"
    ]
    assert all(item["component"] == "sec-edgar-telemetry-sink" for item in sink_logs)
    logged_targets = {item["log_sink_target"] for item in sink_logs}
    assert {
        "llm_telemetry_events",
        "session_telemetry_events",
        "security_audit_logs",
    } <= logged_targets

    # Verify BigQuery parameterized SQL readback for security audit logs
    queried_audits = sink.query_security_audit_logs(limit=10, event_type="filing_10k_section_read")
    assert len(queried_audits) == 1
    assert queried_audits[0]["event_id"] == "aud-bq-001"
    assert queried_audits[0]["tickers"] == ["NVDA"]
    assert queried_audits[0]["iap_assertion_verified"] is True
    assert len(executed_queries) == 1
    assert executed_queries[0]["parameterMode"] == "NAMED"

    status_meta = sink.status_metadata()
    assert status_meta["live_bigquery_streaming_enabled"] is True
    assert status_meta["streamed_row_counts"]["session_telemetry_events"] == 2
    assert status_meta["streamed_row_counts"]["security_audit_logs"] == 1


def test_bigquery_insert_errors_strict_vs_auto_fallback() -> None:
    """Verify `insertErrors` raises in strict `gcp` mode and falls back cleanly in `auto` mode."""

    def _rejecting_bq_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "kind": "bigquery#tableDataInsertAllResponse",
                "insertErrors": [
                    {"index": 0, "errors": [{"reason": "invalid", "message": "schema mismatch"}]}
                ],
            },
        )

    strict_sink = BigQueryTelemetrySink(
        TelemetryConfig(telemetry_backend="gcp"),
        http_client=httpx.Client(transport=httpx.MockTransport(_rejecting_bq_transport)),
        credentials=_FakeGcpCredentials(),
    )
    with pytest.raises(TelemetrySinkError, match="BigQuery insertAll rejected rows"):
        strict_sink.record_security_audit_log(
            {"event_id": "aud-fail-1", "event_type": "test_event"}
        )

    auto_sink = BigQueryTelemetrySink(
        TelemetryConfig(telemetry_backend="auto", gcp_auto_triggered=True),
        http_client=httpx.Client(transport=httpx.MockTransport(_rejecting_bq_transport)),
        credentials=_FakeGcpCredentials(),
    )
    res = auto_sink.record_security_audit_log(
        {"event_id": "aud-fallback-1", "event_type": "test_event"}
    )
    assert res["delivery"] == "local_buffer_fallback"
    assert auto_sink.status_metadata()["fallback_counts"]["security_audit_logs"] == 1

    # Also verify missing ADC credentials in auto mode falls back cleanly
    no_adc_sink = BigQueryTelemetrySink(
        TelemetryConfig(telemetry_backend="auto", gcp_auto_triggered=True),
        credentials=_MissingCredentials(),
    )
    res_no_adc = no_adc_sink.record_security_audit_log(
        {"event_id": "aud-noadc-1", "event_type": "test_event"}
    )
    assert res_no_adc["delivery"] == "local_buffer_fallback"


# ---------------------------------------------------------------------------
# 3. Google Cloud Firestore Feedback Persistence & RBAC Enforcement
# ---------------------------------------------------------------------------


def test_firestore_typed_value_encoding_and_roundtrip_decoding() -> None:
    """Verify typed Firestore REST v1 Value wire format (`timestampValue`, `doubleValue`, `arrayValue`, etc.)."""
    sample = {
        "feedback_id": "fbk-abc123456789",
        "rating": "up",
        "recorded_at": "2026-11-16T15:04:05Z",
        "recorded_at_epoch": 1794841445.125,
        "tickers": ["NVDA", "MSFT"],
        "verified": True,
        "turn_count": 3,
        "optional_note": None,
        "nested": {"grade": "A"},
    }
    encoded = encode_firestore_fields(sample)
    assert encoded["recorded_at"] == {"timestampValue": "2026-11-16T15:04:05Z"}
    assert encoded["recorded_at_epoch"] == {"doubleValue": 1794841445.125}
    assert encoded["tickers"] == {
        "arrayValue": {"values": [{"stringValue": "NVDA"}, {"stringValue": "MSFT"}]}
    }
    assert encoded["verified"] == {"booleanValue": True}
    assert encoded["turn_count"] == {"integerValue": "3"}
    assert encoded["optional_note"] == {"nullValue": None}

    decoded = decode_firestore_document(
        {
            "name": "projects/iron-radio-514217-q7/databases/(default)/documents/analyst_feedback/fbk-abc123456789",
            "fields": encoded,
        }
    )
    for k, v in sample.items():
        assert decoded[k] == v
    assert decoded["document_name"].endswith("/analyst_feedback/fbk-abc123456789")


def test_firestore_feedback_api_persistence_query_and_rbac(
    auth_client_env: dict[str, Any],
) -> None:
    """Verify `POST /api/v1/feedback` and `GET /api/v1/feedback` against Firestore REST v1 & RBAC."""
    stored_firestore_docs: list[dict[str, Any]] = []
    run_query_payloads: list[dict[str, Any]] = []

    def _firestore_transport(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-gcp-oauth-bearer"
        url_str = str(request.url)
        body = json.loads(request.content.decode("utf-8"))

        if "/documents/analyst_feedback" in url_str and request.method == "POST":
            doc_id = request.url.params.get("documentId")
            assert doc_id and doc_id.startswith("fbk-")
            doc_obj = {
                "name": (
                    "projects/iron-radio-514217-q7/databases/(default)"
                    f"/documents/analyst_feedback/{doc_id}"
                ),
                "fields": body["fields"],
            }
            stored_firestore_docs.append(doc_obj)
            return httpx.Response(200, json=doc_obj)

        if url_str.endswith(":runQuery") and request.method == "POST":
            run_query_payloads.append(body)
            sq = body.get("structuredQuery", {})
            rating_filter = (
                sq.get("where", {}).get("fieldFilter", {}).get("value", {}).get("stringValue")
            )
            docs = list(reversed(stored_firestore_docs))
            if rating_filter:
                docs = [
                    d
                    for d in docs
                    if d["fields"].get("rating", {}).get("stringValue") == rating_filter
                ]
            return httpx.Response(200, json=[{"document": d} for d in docs])

        return httpx.Response(404, json={"error": "unexpected_firestore_endpoint"})

    fs_store = FirestoreFeedbackStore(
        TelemetryConfig(
            telemetry_backend="local",
            feedback_backend="gcp",
            project_id="iron-radio-514217-q7",
            firestore_database_id="(default)",
            firestore_feedback_collection="analyst_feedback",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(_firestore_transport)),
        credentials=_FakeGcpCredentials(),
    )
    configure_telemetry_stores(feedback_store=fs_store)

    client: TestClient = auth_client_env["client"]
    mint_token = auth_client_env["mint_token"]

    analyst_jwt = mint_token(
        "analyst@fde-workspace.corp",
        groups=["sec-edgar-analysts@fde-workspace.corp"],
    )
    viewer_jwt = mint_token(
        "viewer@fde-workspace.corp",
        groups=["sec-edgar-viewers@fde-workspace.corp"],
    )

    # 1. Viewer lacks `feedback:submit` -> 403 Forbidden
    denied_post = client.post(
        "/api/v1/feedback",
        headers={"Authorization": f"Bearer {viewer_jwt}"},
        json={"message_id": "msg-viewer-1", "rating": "up"},
    )
    assert denied_post.status_code == 403

    # 2. Analyst submits thumbs-up and thumbs-down feedback -> persisted to Firestore
    up_resp = client.post(
        "/api/v1/feedback",
        headers={"Authorization": f"Bearer {analyst_jwt}"},
        json={
            "message_id": "msg-nvda-001",
            "rating": "up",
            "comment": "Exact GAAP ASC 606 revenue citation verified.",
            "query": "Explain NVDA FY2026 revenue variance",
            "session_id": "sess-ui-100",
            "tickers": ["nvda"],
            "intent": "variance_explanation",
        },
    )
    assert up_resp.status_code == 200
    up_json = up_resp.json()
    assert up_json["status"] == "success"
    assert up_json["persistence"]["backend"] == "firestore"
    assert up_json["persistence"]["database_id"] == "(default)"
    assert up_json["persistence"]["collection"] == "analyst_feedback"
    assert up_json["recorded"]["tickers"] == ["NVDA"]
    assert up_json["recorded"]["session_id"] == "sess-ui-100"

    down_resp = client.post(
        "/api/v1/feedback",
        headers={"Authorization": f"Bearer {analyst_jwt}"},
        json={
            "message_id": "msg-aapl-002",
            "rating": "down",
            "comment": "Include gross margin bps delta.",
            "tickers": ["AAPL"],
        },
    )
    assert down_resp.status_code == 200
    assert len(stored_firestore_docs) == 2

    # 3. Both Viewer and Analyst hold `audit:read` -> can list persisted feedback from Firestore `:runQuery`
    list_resp = client.get(
        "/api/v1/feedback",
        headers={"Authorization": f"Bearer {viewer_jwt}"},
    )
    assert list_resp.status_code == 200
    list_json = list_resp.json()
    assert list_json["source"] == "firestore"
    assert list_json["feedback_count"] == 2
    assert list_json["thumbs_up_count"] == 1
    assert list_json["thumbs_down_count"] == 1
    assert list_json["positive_rating_ratio"] == pytest.approx(0.5)

    # Filtered `:runQuery` by rating="up"
    up_only = client.get(
        "/api/v1/feedback?rating=up",
        headers={"Authorization": f"Bearer {analyst_jwt}"},
    )
    assert up_only.status_code == 200
    assert up_only.json()["feedback_count"] == 1
    assert up_only.json()["feedback"][0]["message_id"] == "msg-nvda-001"
    assert len(run_query_payloads) == 2

    # 4. S2S FinancialAnalystAgent service account lacks `audit:read` -> 403 on GET /api/v1/feedback & /api/v1/telemetry/sessions
    agent_jwt = mint_token(
        "financial-analyst-agent-sa@iron-radio-514217-q7.iam.gserviceaccount.com",
        hosted_domain=None,
    )
    s2s_assertion = get_agent_s2s_broker().mint_s2s_assertion(
        agent_name="FinancialAnalystAgent",
        target_service="looker-semantic-service",
        caller="RootOrchestrator",
    )
    agent_headers = {
        "Authorization": f"Bearer {agent_jwt}",
        S2S_AGENT_ASSERTION_HEADER: s2s_assertion,
    }
    assert client.get("/api/v1/feedback", headers=agent_headers).status_code == 403
    assert client.get("/api/v1/telemetry/sessions", headers=agent_headers).status_code == 403


def test_firestore_feedback_strict_vs_auto_fallback() -> None:
    """Verify Firestore HTTP 503 / missing ADC raises in strict `gcp` mode and falls back in `auto` mode."""

    def _failing_fs_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

    strict_fs = FirestoreFeedbackStore(
        TelemetryConfig(feedback_backend="gcp"),
        http_client=httpx.Client(transport=httpx.MockTransport(_failing_fs_transport)),
        credentials=_FakeGcpCredentials(),
    )
    with pytest.raises(TelemetrySinkError, match="HTTP 503"):
        strict_fs.record_feedback(
            message_id="m-strict",
            rating="up",
            user_email="analyst@fde-workspace.corp",
        )

    auto_fs = FirestoreFeedbackStore(
        TelemetryConfig(feedback_backend="auto", gcp_auto_triggered=True),
        http_client=httpx.Client(transport=httpx.MockTransport(_failing_fs_transport)),
        credentials=_FakeGcpCredentials(),
    )
    fallback_res = auto_fs.record_feedback(
        message_id="m-auto",
        rating="down",
        comment="Offline fallback check",
        user_email="analyst@fde-workspace.corp",
    )
    assert fallback_res["backend_used"] == "local_memory_fallback"
    listed = auto_fs.list_feedback(limit=10)
    assert listed["source"] == "local_memory_fallback"
    assert listed["feedback_count"] == 1
    assert listed["thumbs_down_count"] == 1


# ---------------------------------------------------------------------------
# 4. /health Surface, Pruned Routes & BigQuery-backed Explore Tool
# ---------------------------------------------------------------------------


def test_health_and_bigquery_explores(
    auth_client_env: dict[str, Any],
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `/health`, BigQuery-backed explore helper, and `/api/v1/telemetry/sessions`."""
    client: TestClient = auth_client_env["client"]
    analyst_jwt = auth_client_env["mint_token"](
        "analyst@fde-workspace.corp",
        groups=["sec-edgar-analysts@fde-workspace.corp"],
    )
    headers = {"Authorization": f"Bearer {analyst_jwt}"}

    health = client.get("/health")
    assert health.status_code == 200
    health_json = health.json()
    assert health_json["root_agent"] == "FinancialAnalystAgent"
    assert "lookml_semantic_layer" not in health_json
    for key in (
        "observability",
        "telemetry_sink",
        "feedback_store",
        "session_memory",
        "data_backend",
    ):
        assert key in health_json

    sec_resp = client.get(
        "/api/v1/section",
        headers=headers,
        params={"ticker": "NVDA", "fiscal_year": 2026, "section_id": "item_7", "max_chars": 2000},
    )
    assert sec_resp.status_code == 200

    # Scripted turn over fake adapters seeds LLM, session, and audit buffers deterministically.
    patch_http_chat_llms(
        monkeypatch,
        reasoning_llm=ScriptedAnalystLlm(
            tool_calls=[
                (
                    "calculate_period_variances",
                    {"tickers": "NVDA,MSFT", "start_year": 2024, "end_year": 2025},
                )
            ],
            headline="NVDA and MSFT posted strong FY2024-FY2025 revenue growth.",
            intent="multi_company_peer_comparison",
            key_metrics=[("NVDA", "revenue"), ("MSFT", "revenue")],
        ),
        skip_background_judge=True,
    )
    chat_resp = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "query": "Compare NVDA and MSFT revenue growth from FY2024 to FY2025.",
            "tickers": "NVDA,MSFT",
            "start_year": 2024,
            "end_year": 2025,
            "session_id": "sess-explore-check",
        },
    )
    assert chat_resp.status_code == 200
    chat_json = chat_resp.json()
    assert chat_json["status"] == "success"
    assert chat_json["answer_verified"] is True
    assert chat_json["evaluation"]["grade"] == "pending"
    assert chat_json["evaluation"]["overall_score"] is None

    for explore_slug in ("llm_telemetry", "session_analytics", "audit_trail"):
        exp_res = query_analytics_view(
            explore_name=explore_slug,
            tickers="ALL",
            gics_sector="ALL",
            start_year=2024,
            end_year=2026,
        )
        assert exp_res["status"] == "success"
        assert exp_res["row_count"] >= 1

    sess_resp = client.get(
        "/api/v1/telemetry/sessions?session_id=sess-explore-check",
        headers=headers,
    )
    assert sess_resp.status_code == 200
    sess_json = sess_resp.json()
    assert sess_json["event_count"] == 1
    assert sess_json["summary"]["distinct_companies_analyzed"] == ["MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# 5. BigQuery Schema Parity, SQL Trace ID, LRU Bounds & Async Streaming
# ---------------------------------------------------------------------------


def test_bigquery_telemetry_schemas_match_serialized_rows() -> None:
    """Verify BigQuery JSON table schemas match serialized Python row keys exactly."""
    schemas_dir = REPO_ROOT / "terraform" / "schemas"
    llm_schema = json.loads((schemas_dir / "llm_telemetry_events.json").read_text(encoding="utf-8"))
    sess_schema = json.loads(
        (schemas_dir / "session_telemetry_events.json").read_text(encoding="utf-8")
    )
    audit_schema = json.loads(
        (schemas_dir / "security_audit_logs.json").read_text(encoding="utf-8")
    )

    llm_schema_cols = {col["name"] for col in llm_schema}
    sess_schema_cols = {col["name"] for col in sess_schema}
    audit_schema_cols = {col["name"] for col in audit_schema}

    sink = BigQueryTelemetrySink(TelemetryConfig(telemetry_backend="local"))
    sample_rec = attribute_context_latency(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="synthesis",
        context_breakdown=ContextBreakdown(total_context_tokens=1000, uncached_context_tokens=1000),
        output_tokens=200,
        raw_inference_latency_ms=500.0,
        session_id="s1",
        user_id="u1",
    )
    llm_row = sink.record_llm_telemetry(sample_rec)["row"]
    sess_row = sink.record_session_turn_telemetry(
        session_id="s1",
        user_id="u1",
        intent="variance_explanation",
        execution_mode="live_adk",
        status_outcome="success",
        tickers_analyzed=["NVDA"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        report_generated=True,
        variance_reports_count=1,
        citations_count=2,
        agent_trace=[{"tool": "calculate_period_variances"}],
        turn_latency_records=[sample_rec],
        total_latency_ms=550.0,
        evaluation_grade="pending",
        evaluation_score=None,
    )["row"]
    async_row = sink.record_async_evaluation_result(
        session_id="s1",
        user_email="u1",
        intent="variance_explanation",
        evaluation_grade="pass",
        evaluation_score=0.85,
        evaluator_model="gemini-3.5-flash",
    )["row"]
    audit_row = sink.record_security_audit_log(
        {"event_id": "aud-1", "event_type": "filing_10k_section_read", "tickers": ["NVDA"]}
    )["row"]
    sink.close()

    assert set(llm_row.keys()) == llm_schema_cols
    assert set(sess_row.keys()) == sess_schema_cols
    assert set(async_row.keys()) == sess_schema_cols
    assert set(audit_row.keys()) == audit_schema_cols

    # Interactive rows start pending/NULL; async judge rows carry 0.0 ms latency so SQL views
    # can cleanly separate interactive turn latency from background evaluation scores.
    assert sess_row["evaluation_grade"] == "pending" and sess_row["evaluation_score"] is None
    assert async_row["execution_mode"] == "async_evaluation"
    assert async_row["total_latency_ms"] == 0.0
    assert async_row["evaluation_score"] == 0.85


def test_telemetry_sql_trace_id_retry_after_lru_and_lifecycle() -> None:
    """Verify SQL queries select trace_id, 429 Retry-After propagates, LRU & deque bounds hold."""
    captured_sql: list[str] = []

    def _sql_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_sql.append(str(body.get("query") or ""))
        return httpx.Response(200, json={"schema": {"fields": []}, "rows": []})

    mock_http = httpx.Client(transport=httpx.MockTransport(_sql_handler))
    gcp_cfg = TelemetryConfig(telemetry_backend="gcp", feedback_backend="local")
    gcp_sink = BigQueryTelemetrySink(
        config=gcp_cfg, http_client=mock_http, credentials=_FakeGcpCredentials()
    )
    gcp_sink.query_session_telemetry(limit=5)
    gcp_sink.query_security_audit_logs(limit=5)
    assert len(captured_sql) == 2
    assert all("trace_id" in q for q in captured_sql)
    gcp_sink.close()
    mock_http.close()

    rate_http = httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(429, headers={"Retry-After": "7.5"}, text="quota")
        )
    )
    rate_sink = BigQueryTelemetrySink(
        config=gcp_cfg, http_client=rate_http, credentials=_FakeGcpCredentials()
    )
    with pytest.raises(TelemetryRateLimitError) as exc_rl:
        rate_sink.query_session_telemetry(limit=5)
    assert exc_rl.value.retry_after_seconds == 7.5
    rate_sink.close()
    rate_http.close()

    local_sink = BigQueryTelemetrySink(
        config=TelemetryConfig(telemetry_backend="local"),
        max_sessions=3,
    )
    for sid in ("s1", "s2", "s3", "s1", "s4"):
        local_sink.record_session_turn_telemetry(
            session_id=sid,
            user_id="u@corp.com",
            intent="longitudinal_single_company",
            execution_mode="live_adk",
            status_outcome="success",
            tickers_analyzed=["NVDA"],
            gics_sector="ALL",
            start_year=2025,
            end_year=2026,
            report_generated=True,
            variance_reports_count=1,
            citations_count=1,
            agent_trace=[],
            turn_latency_records=[],
            total_latency_ms=10.0,
        )
    assert list(local_sink._sessions.keys()) == ["s3", "s1", "s4"]
    assert "s2" not in local_sink._sessions

    pooled = local_sink._get_http_client()
    assert not pooled.is_closed
    local_sink.close()
    assert local_sink._closed is True
    assert pooled.is_closed

    fb_store = FirestoreFeedbackStore(
        config=TelemetryConfig(feedback_backend="local"),
        max_local_records=3,
    )
    assert isinstance(fb_store._local_feedback, deque)
    assert fb_store._local_feedback.maxlen == 3
    for idx in range(5):
        fb_store.record_feedback(
            message_id=f"msg-{idx}",
            rating="up",
            user_email="analyst@corp.com",
        )
    assert len(fb_store._local_feedback) == 3
    assert [e["message_id"] for e in fb_store._local_feedback] == ["msg-2", "msg-3", "msg-4"]
    fb_store.close()


def test_bigquery_sink_async_background_streaming_flush_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify background thread-pool delivery, flush_async_writes, error tracking, and closed fallback."""
    cfg = TelemetryConfig(telemetry_backend="gcp", feedback_backend="local")
    sink = BigQueryTelemetrySink(config=cfg)
    delivered: list[tuple[str, int]] = []
    should_fail = [False]

    def _fake_stream_rows(table_id: str, rows: list[tuple[str, dict[str, Any]]]) -> None:
        if should_fail[0]:
            raise TelemetrySinkError("simulated bg failure")
        delivered.append((table_id, len(rows)))

    monkeypatch.setattr(sink, "stream_rows", _fake_stream_rows)
    try:
        res_ok = sink.record_security_audit_log(
            {"event_id": "bg-1", "event_type": "filing_10k_section_read", "tickers": ["NVDA"]}
        )
        assert res_ok["delivery"] == "bigquery_streaming_async"
        sink.flush_async_writes(timeout=5.0)
        assert delivered == [("security_audit_logs", 1)]

        should_fail[0] = True
        res_err = sink.record_security_audit_log(
            {"event_id": "bg-2", "event_type": "filing_10k_section_read", "tickers": ["NVDA"]}
        )
        assert res_err["delivery"] == "bigquery_streaming_async"
        sink.flush_async_writes(timeout=5.0)
        meta = sink.status_metadata()
        assert meta["fallback_counts"]["security_audit_logs"] == 1
        assert "simulated bg failure" in str(meta["last_error"])

        sink.close()
        res_closed = sink.record_security_audit_log(
            {"event_id": "bg-3", "event_type": "filing_10k_section_read", "tickers": ["NVDA"]}
        )
        assert res_closed["delivery"] == "local_buffer_fallback"
    finally:
        sink.close()
