"""Unit/integration tests for full-sector batched map-reduce, quote verification, SSE streaming, and session APIs."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents.context_memory import get_conversation_memory_manager
from edgar_capstone.agents.sector_map_reduce import run_sector_thematic_map_reduce
from edgar_capstone.agents.tools import (
    GcpToolAdapters,
    GcpToolConfig,
    configure_gcp_tool_adapters,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.security.auth import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app
from tests.fake_gcp import preserve_gcp_caches
from tests.structured_report_helpers import ScriptedBaseLlm


class _FakeCreds:
    valid = True
    token = "fake-token"


class _ScriptedMapFlashLlm(ScriptedBaseLlm):
    """Returns one verified quote + one hallucinated/fabricated chunk_id per map batch."""

    model: str = "gemini-3.5-flash-lite"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        req_text = "\n".join(
            p.text for c in (llm_request.contents or []) for p in (c.parts or []) if p.text
        )
        # Extract first real ticker chunk from prompt JSON
        ticker = "NVDA" if "NVDA_2026_item_1a_0001" in req_text else "AAPL"
        batch_index = 0 if ticker == "NVDA" else 1
        valid_cid = f"{ticker}_2026_item_1a_0001"
        valid_quote = (
            "Generative AI regulation and export controls could materially impact demand."
            if ticker == "NVDA"
            else "Deploying generative AI features creates privacy and reputational risk."
        )
        payload = {
            "batch_index": batch_index,
            "tickers_covered": [ticker],
            "dominant_themes": ["generative AI regulation", "export controls"],
            "wording_shifts": [f"{ticker} added generative AI risk disclosures in FY2026."],
            "evidence": [
                {
                    "ticker": ticker,
                    "fiscal_year": 2026,
                    "chunk_id": valid_cid,
                    "verbatim_quote": valid_quote,
                    "takeaway": f"{ticker} expanded generative AI risk factors in FY2026.",
                },
                {
                    "ticker": ticker,
                    "fiscal_year": 2026,
                    "chunk_id": f"{ticker}_2026_item_1a_9999",
                    "verbatim_quote": "This sentence does not exist in any filing.",
                    "takeaway": "Fabricated chunk id that MUST be dropped.",
                },
            ],
        }
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(payload))]
            ),
        )


def _profile(ticker: str, name: str, fy: int) -> FinancialDataProfile:
    return FinancialDataProfile(
        profile_id=f"{ticker}_{fy}",
        cik="0001045810" if ticker == "NVDA" else "0000320193",
        ticker=ticker,
        company_name=name,
        gics_sector="Information Technology",
        gics_sub_industry="Semiconductors",
        fiscal_year=fy,
        period_of_report=f"{fy}-01-25" if ticker == "NVDA" else f"{fy}-09-27",
        filing_date=f"{fy}-02-25",
        accession_number=f"0001045810-{str(fy)[-2:]}-000021",
        sec_filing_url=f"https://www.sec.gov/{ticker.lower()}-{fy}.htm",
        unstructured_blob_path=f"unstructured/{ticker}/{fy}_10k.md",
        structured_blob_path=f"structured/{ticker}/{fy}.json",
        unstructured_gcs_uri=f"gs://bucket/unstructured/{ticker}/{fy}_10k.md",
        structured_gcs_uri=f"gs://bucket/structured/{ticker}/{fy}.json",
        revenue=100e9,
        operating_income=40e9,
        net_income=35e9,
    )


def test_sector_map_reduce_batches_emits_progress_and_verifies_quotes() -> None:
    sql_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if ":predict" in url and "text-embedding-005" in url:
            return httpx.Response(
                200, json={"predictions": [{"embeddings": {"values": [0.1] * 8}}]}
            )
        if "bigquery.googleapis.com" in url:
            sql_queries.append(body["query"])
            rows = [
                (
                    "NVDA_2024_item_1a_0001",
                    "NVDA_2024",
                    "NVDA",
                    "NVIDIA CORP",
                    "2024",
                    "item_1a",
                    "Item 1A. Risk Factors",
                    "1",
                    "Initial generative AI adoption presents competition risks in FY2024.",
                    "0.22",
                ),
                (
                    "NVDA_2026_item_1a_0001",
                    "NVDA_2026",
                    "NVDA",
                    "NVIDIA CORP",
                    "2026",
                    "item_1a",
                    "Item 1A. Risk Factors",
                    "1",
                    "Generative AI regulation and export controls could materially impact demand.",
                    "0.10",
                ),
                (
                    "AAPL_2024_item_1a_0001",
                    "AAPL_2024",
                    "AAPL",
                    "Apple Inc.",
                    "2024",
                    "item_1a",
                    "Item 1A. Risk Factors",
                    "1",
                    "General technology competition could affect future product margins.",
                    "0.28",
                ),
                (
                    "AAPL_2026_item_1a_0001",
                    "AAPL_2026",
                    "AAPL",
                    "Apple Inc.",
                    "2026",
                    "item_1a",
                    "Item 1A. Risk Factors",
                    "1",
                    "Deploying generative AI features creates privacy and reputational risk.",
                    "0.15",
                ),
            ]
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {
                        "fields": [
                            {"name": "chunk_id", "type": "STRING"},
                            {"name": "profile_id", "type": "STRING"},
                            {"name": "ticker", "type": "STRING"},
                            {"name": "company_name", "type": "STRING"},
                            {"name": "gics_sector", "type": "STRING"},
                            {"name": "fiscal_year", "type": "INTEGER"},
                            {"name": "period_of_report", "type": "STRING"},
                            {"name": "filing_date", "type": "STRING"},
                            {"name": "accession_number", "type": "STRING"},
                            {"name": "section_id", "type": "STRING"},
                            {"name": "section_title", "type": "STRING"},
                            {"name": "chunk_index", "type": "INTEGER"},
                            {"name": "chunk_text", "type": "STRING"},
                            {"name": "sec_filing_url", "type": "STRING"},
                            {"name": "unstructured_gcs_uri", "type": "STRING"},
                            {"name": "distance", "type": "FLOAT"},
                        ]
                    },
                    "rows": [
                        {
                            "f": [
                                {"v": cid},
                                {"v": pid},
                                {"v": tk},
                                {"v": cname},
                                {"v": "Information Technology"},
                                {"v": fy},
                                {"v": f"{fy}-01-25"},
                                {"v": f"{fy}-02-25"},
                                {"v": f"0001045810-{fy[-2:]}-000021"},
                                {"v": sec},
                                {"v": stitle},
                                {"v": cidx},
                                {"v": txt},
                                {"v": f"https://www.sec.gov/{tk.lower()}-{fy}.htm"},
                                {"v": f"gs://bucket/{tk}/{fy}.md"},
                                {"v": dist},
                            ]
                        }
                        for cid, pid, tk, cname, fy, sec, stitle, cidx, txt, dist in rows
                    ],
                },
            )
        return httpx.Response(404, text=url)

    progress_events: list[tuple[str, str, dict[str, Any] | None]] = []
    with preserve_gcp_caches():
        configure_gcp_tool_adapters(
            GcpToolAdapters(
                GcpToolConfig(project_id="iron-radio-514217-q7", embedding_dimensions=8),
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                credentials=_FakeCreds(),
            )
        )
        profiles = [
            _profile("NVDA", "NVIDIA CORP", 2024),
            _profile("NVDA", "NVIDIA CORP", 2026),
            _profile("AAPL", "Apple Inc.", 2024),
            _profile("AAPL", "Apple Inc.", 2026),
        ]
        result = run_sector_thematic_map_reduce(
            theme="generative AI risk disclosure",
            gics_sector="Information Technology",
            section_id="item_1a",
            start_year=2024,
            end_year=2026,
            sector_profiles=profiles,
            batch_size=1,
            max_workers=2,
            map_llm=_ScriptedMapFlashLlm(),
            on_progress=lambda st, msg, meta: progress_events.append((st, msg, meta)),
        )

    assert result["status"] == "success"
    assert len(sql_queries) == 1 and "ML.DISTANCE" in sql_queries[0]
    cov = result["sector_coverage"]
    assert cov["issuers_in_sector"] == 2
    assert cov["issuers_analyzed"] == 2
    assert cov["batches_processed"] == 2

    # Quote verifier MUST drop fabricated `_9999` evidence items and keep only verified chunks!
    batches = result["sector_map_summaries"]["batch_summaries"]
    kept_ids = {ev["chunk_id"] for batch in batches for ev in batch.get("evidence", [])}
    assert kept_ids == {"NVDA_2026_item_1a_0001", "AAPL_2026_item_1a_0001"}
    assert sum(int(batch.get("dropped_evidence_count", 0)) for batch in batches) == 2
    assert not any("9999" in cid for cid in kept_ids)
    assert len(progress_events) >= 2
    assert any("companies analyzed" in msg for _, msg, _ in progress_events)


def _auth_headers(email: str = "analyst@fde-workspace.corp") -> dict[str, str]:
    tok, _ = issue_local_dev_workspace_sso_token(
        email=email,
        name="Analyst",
        groups=["sec-edgar-analysts"],
    )
    return {"Authorization": f"Bearer {tok}"}


def _parse_sse(raw_text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw_text.strip().split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if event_name and data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def test_sse_chat_stream_and_user_scoped_sessions_endpoints() -> None:
    get_conversation_memory_manager().reset()
    client = TestClient(app)
    alice_headers = _auth_headers("alice@fde-workspace.corp")
    bob_headers = _auth_headers("bob@fde-workspace.corp")

    def _fake_turn(**kwargs: Any) -> dict[str, Any]:
        cb = kwargs.get("on_progress")
        if cb:
            cb(
                "scope_resolved",
                "Scoped to Information Technology (76 companies)",
                {"total_companies": 76},
            )
            cb(
                "sector_map",
                "76/76 Information Technology companies analyzed (batch 6/6)",
                {"companies_analyzed": 76, "total_companies": 76},
            )
        get_conversation_memory_manager().record_turn(
            session_id=kwargs.get("session_id"),
            user_id=kwargs.get("user_id", "analyst"),
            query=kwargs["query"],
            intent="thematic_trend_tracking",
            tickers=[],
            gics_sector="Information Technology",
            start_year=2024,
            end_year=2026,
            answer_markdown="### Streamed Answer",
        )
        return {
            "status": "success",
            "model_armor_blocked": False,
            "intent": "thematic_trend_tracking",
            "execution_mode": "live_adk",
            "resolved_scope": {"tickers": [], "gics_sector": "Information Technology"},
            "answer_markdown": "### Streamed Answer",
            "answer_verified": True,
            "citations": [],
            "variance_reports": [],
            "total_latency_ms": 125.0,
        }

    with patch("edgar_capstone.server.run_conversational_rag_turn", side_effect=_fake_turn):
        resp = client.post(
            "/api/v1/chat/stream",
            headers=alice_headers,
            json={
                "query": "Trend in generative AI across IT",
                "gics_sector": "Information Technology",
                "session_id": "sse-sess-1",
            },
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"

    events = _parse_sse(resp.text)
    event_names = [e[0] for e in events]
    assert event_names[0] == "progress"
    assert events[0][1]["stage"] == "connected"
    assert "result" in event_names
    result_payload = next(p for name, p in events if name == "result")
    assert result_payload["status"] == "success"
    assert result_payload["authenticated_principal"]["email"] == "alice@fde-workspace.corp"
    assert "trace_id" in result_payload["observability"]

    # SSE error event path
    with patch(
        "edgar_capstone.server.run_conversational_rag_turn",
        side_effect=RuntimeError("live backend error"),
    ):
        err_resp = client.post(
            "/api/v1/chat/stream",
            headers=alice_headers,
            json={"query": "Trigger stream error"},
        )
    assert err_resp.status_code == 200
    err_events = _parse_sse(err_resp.text)
    assert any(name == "error" and "live backend error" in p["message"] for name, p in err_events)

    # GET /health exposes session_memory persistence status
    health = client.get("/health")
    assert health.status_code == 200
    assert "session_memory" in health.json()

    # Session list & delete endpoints are strictly user-scoped
    alice_list = client.get("/api/v1/sessions", headers=alice_headers)
    assert alice_list.status_code == 200
    assert alice_list.json()["session_count"] == 1
    assert alice_list.json()["sessions"][0]["session_id"] == "sse-sess-1"

    bob_list = client.get("/api/v1/sessions", headers=bob_headers)
    assert bob_list.status_code == 200
    assert bob_list.json()["session_count"] == 0

    # Bob cannot delete Alice's session
    bob_del = client.delete("/api/v1/sessions/sse-sess-1", headers=bob_headers)
    assert bob_del.status_code == 404

    # Alice deletes her own session
    alice_del = client.delete("/api/v1/sessions/sse-sess-1", headers=alice_headers)
    assert alice_del.status_code == 200
    assert alice_del.json()["status"] == "deleted"
    assert client.get("/api/v1/sessions", headers=alice_headers).json()["session_count"] == 0


def test_chat_stream_handles_none_trace_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE stream emits a valid `connected` progress event when `current_trace_metadata()` is None."""
    from edgar_capstone.api.routers import chat as chat_router

    monkeypatch.setattr(chat_router, "current_trace_metadata", lambda: None)
    monkeypatch.setattr(
        chat_router,
        "run_conversational_rag_turn",
        lambda **_kw: {
            "status": "success",
            "intent": "longitudinal_single_company",
            "resolved_scope": {"tickers": ["NVDA"]},
            "answer_verified": True,
            "answer_markdown": "ok",
        },
    )
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/stream",
        headers=_auth_headers(),
        json={"query": "Show NVDA revenue"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0][0] == "progress"
    assert events[0][1]["stage"] == "connected"
    assert events[0][1]["meta"]["trace_id"] is None
    assert any(name == "result" and payload["status"] == "success" for name, payload in events)


def test_chat_stream_sanitizes_internal_exception_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE stream must not leak raw internal exception details (e.g., credentials) to clients."""
    from edgar_capstone.api.routers import chat as chat_router

    def _boom(**_kw: Any) -> dict[str, Any]:
        raise ValueError("db connection failed: SECRET_PASSWORD")

    monkeypatch.setattr(chat_router, "run_conversational_rag_turn", _boom)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/stream",
        headers=_auth_headers(),
        json={"query": "Show NVDA revenue"},
    )
    assert resp.status_code == 200
    assert "SECRET_PASSWORD" not in resp.text
    events = _parse_sse(resp.text)
    err_events = [payload for name, payload in events if name == "error"]
    assert len(err_events) == 1
    assert err_events[0]["error"] == "internal_server_error"
    assert "SECRET_PASSWORD" not in err_events[0]["message"]


def test_sector_map_agent_defaults_to_flash_lite(monkeypatch: pytest.MonkeyPatch) -> None:
    """SectorMapSummarizerAgent resolves `gemini-3.5-flash-lite` via `context_compaction`."""
    from google.genai import types as genai_types

    from edgar_capstone.agents import sector_map_reduce as smr
    from edgar_capstone.agents.live_runtime import AdkStreamResult

    monkeypatch.delenv("EDGAR_LITE_MODEL", raising=False)
    captured: dict[str, Any] = {}

    def _fake_stream(**kwargs: Any) -> AdkStreamResult:
        captured["agent"] = kwargs["agent"]
        return AdkStreamResult(
            agent_name=kwargs["agent"].name,
            app_name=kwargs["app_name"],
            session_id=kwargs["session_id"],
            backend="vertex_ai",
            final_text="",
            session_state={
                "sector_batch_summary": {
                    "batch_index": 0,
                    "tickers_covered": ["NVDA"],
                    "dominant_themes": ["AI risk"],
                    "wording_shifts": [],
                    "evidence": [],
                }
            },
        )

    monkeypatch.setattr(smr, "run_adk_agent_stream", _fake_stream)
    out = smr._summarize_batch_via_flash(
        batch_index=0,
        total_batches=1,
        gics_sector="Information Technology",
        theme="AI risk",
        start_year=2024,
        end_year=2026,
        company_payloads=[{"ticker": "NVDA", "passages": []}],
        chunk_lookup={},
    )
    assert out["model_used"] == "gemini-3.5-flash-lite"
    agent = captured["agent"]
    assert agent.name == "SectorMapSummarizerAgent"
    assert agent.model.model == "gemini-3.5-flash-lite"
    assert (
        agent.generate_content_config.thinking_config.thinking_level
        == genai_types.ThinkingLevel.LOW
    )
