"""Unit tests for Firestore-backed multi-instance persistent ConversationMemoryManager & tenant isolation."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from edgar_capstone.agents.runtime.context_memory import ConversationMemoryManager
from edgar_capstone.telemetry import (
    FirestoreSessionStore,
    TelemetryConfig,
    TelemetrySinkError,
    session_firestore_doc_id,
)


class _FakeCreds:
    valid = True
    token = "fake-firestore-token"


def _mock_firestore_transport() -> tuple[
    httpx.MockTransport, dict[str, dict[str, Any]], list[tuple[str, str]]
]:
    docs: dict[str, dict[str, Any]] = {}
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append((request.method, url))
        assert request.headers.get("Authorization") == "Bearer fake-firestore-token"

        if request.method == "POST" and url.endswith(":runQuery"):
            body = json.loads(request.content.decode("utf-8"))
            target_user = body["structuredQuery"]["where"]["fieldFilter"]["value"]["stringValue"]
            matched = []
            for name, raw_fields in docs.items():
                uid = (raw_fields.get("user_id") or {}).get("stringValue")
                if uid == target_user:
                    matched.append({"document": {"name": name, "fields": raw_fields}})
            return httpx.Response(200, json=matched or [{"readTime": "2026-11-17T00:00:00Z"}])

        doc_key = url.split("/documents/", 1)[1]
        if request.method == "PATCH":
            body = json.loads(request.content.decode("utf-8"))
            docs[doc_key] = body["fields"]
            return httpx.Response(200, json={"name": doc_key, "fields": body["fields"]})
        if request.method == "GET":
            if doc_key not in docs:
                return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})
            return httpx.Response(200, json={"name": doc_key, "fields": docs[doc_key]})
        if request.method == "DELETE":
            if doc_key not in docs:
                return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})
            del docs[doc_key]
            return httpx.Response(200, json={})
        return httpx.Response(500, text=f"Unhandled {request.method} {url}")

    return httpx.MockTransport(handler), docs, calls


def test_firestore_session_store_crud_and_multi_instance_hydration() -> None:
    transport, docs, calls = _mock_firestore_transport()
    cfg = TelemetryConfig(
        telemetry_backend="local",
        feedback_backend="gcp",
        project_id="iron-radio-514217-q7",
    )
    store_1 = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    store_2 = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    assert store_1.is_live_enabled() is True

    # Two distinct Cloud Run instances sharing the same Firestore collection
    mgr_1 = ConversationMemoryManager(session_store=store_1, refresh_ttl_seconds=0.0)
    mgr_2 = ConversationMemoryManager(session_store=store_2, refresh_ttl_seconds=0.0)

    persisted = mgr_1.record_turn(
        session_id="shared-session-42",
        user_id="Alice@fde-workspace.corp",
        query="Analyze NVDA FY2024 to FY2026 revenue",
        intent="longitudinal_single_company",
        tickers=["NVDA"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        variance_highlights=["NVDA revenue +254.45%"],
        cited_chunk_ids=["NVDA_2026_item_7_0001"],
        answer_markdown="NVDA grew strongly.",
    )
    assert persisted["persisted"] is True
    assert persisted["persistence_backend"] == "firestore"
    assert len(docs) == 1

    # Instance 2 (cold in-memory cache!) hydrates Alice's session straight from Firestore
    recalled = mgr_2.recall_session_context("shared-session-42", user_id="alice@fde-workspace.corp")
    assert recalled["has_prior_memory"] is True
    assert recalled["persistence_backend"] == "firestore"
    assert recalled["remembered_tickers"] == ["NVDA"]
    assert recalled["turn_count"] == 1
    assert "NVDA revenue +254.45%" in recalled["memory_summary_text"]

    # Tenant isolation: Bob using the EXACT same session_id MUST NOT see Alice's data!
    bob_recall = mgr_2.recall_session_context("shared-session-42", user_id="bob@fde-workspace.corp")
    assert bob_recall["has_prior_memory"] is False
    assert bob_recall["remembered_tickers"] == []
    assert session_firestore_doc_id("alice@fde-workspace.corp", "shared-session-42") != (
        session_firestore_doc_id("bob@fde-workspace.corp", "shared-session-42")
    )

    # Listing sessions is strictly scoped by user_id
    alice_sessions = mgr_2.list_user_sessions("alice@fde-workspace.corp")
    assert len(alice_sessions) == 1
    assert alice_sessions[0]["session_id"] == "shared-session-42"
    assert mgr_2.list_user_sessions("bob@fde-workspace.corp") == []

    # Bob cannot delete Alice's session
    assert mgr_2.delete_session("shared-session-42", user_id="bob@fde-workspace.corp") is False
    assert len(docs) == 1

    # Alice deletes her own session
    assert mgr_2.delete_session("shared-session-42", user_id="alice@fde-workspace.corp") is True
    assert len(docs) == 0
    assert (
        mgr_2.recall_session_context("shared-session-42", user_id="alice@fde-workspace.corp")[
            "has_prior_memory"
        ]
        is False
    )

    meta = mgr_2.persistence_status()
    assert meta["live_firestore_enabled"] is True
    assert meta["reads"] >= 1
    assert any(m == "PATCH" for m, _ in calls)
    assert any(m == "DELETE" for m, _ in calls)


def test_firestore_session_store_local_noop_and_upstream_error_tracking() -> None:
    """Verify local-mode no-ops and error propagation / last_error tracking on Firestore 503."""
    local_store = FirestoreSessionStore(
        TelemetryConfig(telemetry_backend="local", feedback_backend="local")
    )
    assert local_store.is_live_enabled() is False
    assert local_store.load_session("alice@corp", "s1") is None
    local_store.save_session("alice@corp", "s1", {"user_id": "alice@corp"})
    assert local_store.delete_session("alice@corp", "s1") is False
    assert local_store.list_sessions("alice@corp") == []

    fail_client = httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(503, text="unavailable"))
    )
    live_store = FirestoreSessionStore(
        TelemetryConfig(telemetry_backend="local", feedback_backend="gcp"),
        http_client=fail_client,
        credentials=_FakeCreds(),
    )
    try:
        with pytest.raises(TelemetrySinkError):
            live_store.load_session("alice@corp", "s1")
        assert live_store.status_metadata()["last_error"]

        with pytest.raises(TelemetrySinkError):
            live_store.save_session("alice@corp", "s1", {"user_id": "alice@corp"})
        with pytest.raises(TelemetrySinkError):
            live_store.delete_session("alice@corp", "s1")
        with pytest.raises(TelemetrySinkError):
            live_store.list_sessions("alice@corp")
    finally:
        fail_client.close()


def test_record_turn_skips_redundant_firestore_get_after_recall_and_recovers_on_eviction() -> None:
    """Passing `recalled_prior_memory` avoids a second Firestore GET after TTL expiry, while recovering on LRU eviction."""
    transport, docs, calls = _mock_firestore_transport()
    cfg = TelemetryConfig(
        telemetry_backend="local",
        feedback_backend="gcp",
        project_id="iron-radio-514217-q7",
    )
    store = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    # refresh_ttl_seconds=0.0 simulates every live Gemini turn exceeding the 5s L1 refresh TTL.
    mgr = ConversationMemoryManager(
        session_store=store,
        refresh_ttl_seconds=0.0,
        max_local_sessions=1,
    )

    # Turn 1 (brand-new session): recall gets 404 (1 GET), record_turn skips redundant GET -> 1 PATCH.
    recalled_1 = mgr.recall_session_context("sess-1", user_id="alice@corp")
    assert recalled_1["has_prior_memory"] is False
    assert [m for m, _ in calls] == ["GET"]

    calls.clear()
    res_1 = mgr.record_turn(
        session_id="sess-1",
        user_id="alice@corp",
        query="Analyze NVDA FY2024 to FY2026 revenue",
        intent="longitudinal_single_company",
        tickers=["NVDA"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        recalled_prior_memory=bool(recalled_1["has_prior_memory"]),
    )
    assert res_1["turn_count"] == 1
    assert [m for m, _ in calls] == ["PATCH"]

    # Turn 2 (existing session, TTL expired): recall hydrates via 1 GET, record_turn reuses L1 -> 0 GETs, 1 PATCH.
    calls.clear()
    recalled_2 = mgr.recall_session_context("sess-1", user_id="alice@corp")
    assert recalled_2["has_prior_memory"] is True
    assert recalled_2["turn_count"] == 1
    assert [m for m, _ in calls] == ["GET"]

    calls.clear()
    res_2 = mgr.record_turn(
        session_id="sess-1",
        user_id="alice@corp",
        query="Now add AAPL",
        intent="multi_company_peer_comparison",
        tickers=["NVDA", "AAPL"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        recalled_prior_memory=bool(recalled_2["has_prior_memory"]),
    )
    assert res_2["turn_count"] == 2
    assert [m for m, _ in calls] == ["PATCH"]

    # Turn 3 with mid-turn LRU eviction (`max_local_sessions=1`): safely falls back to 1 remote GET so turns 1-2 aren't lost.
    recalled_3 = mgr.recall_session_context("sess-1", user_id="alice@corp")
    assert recalled_3["has_prior_memory"] is True
    assert recalled_3["turn_count"] == 2

    # Evict `sess-1` from L1 by recording a different session.
    mgr.record_turn(
        session_id="other-sess",
        user_id="bob@corp",
        query="Check MSFT",
        intent="longitudinal_single_company",
        tickers=["MSFT"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
    )

    calls.clear()
    res_3 = mgr.record_turn(
        session_id="sess-1",
        user_id="alice@corp",
        query="Drop AAPL",
        intent="longitudinal_single_company",
        tickers=["NVDA"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        recalled_prior_memory=bool(recalled_3["has_prior_memory"]),
    )
    assert res_3["turn_count"] == 3
    assert [m for m, _ in calls] == ["GET", "PATCH"]
    assert len(docs) == 2


def test_chat_turn_issues_only_one_firestore_get_and_one_patch_even_past_refresh_ttl(
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end `run_conversational_rag_turn` avoids a redundant Firestore GET on save."""
    from edgar_capstone.agents.orchestration import chat_engine
    from tests.structured_report_helpers import ScriptedAnalystLlm

    transport, _docs, calls = _mock_firestore_transport()
    cfg = TelemetryConfig(
        telemetry_backend="local",
        feedback_backend="gcp",
        project_id="iron-radio-514217-q7",
    )
    store = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    mgr = ConversationMemoryManager(session_store=store, refresh_ttl_seconds=0.0)
    monkeypatch.setattr(chat_engine, "get_conversation_memory_manager", lambda: mgr)

    for turn_num, query in enumerate(
        (
            "Analyze NVDA FY2024 to FY2026 revenue drivers",
            "How did operating margins trend over the same period?",
        ),
        start=1,
    ):
        calls.clear()
        resp = chat_engine.run_conversational_rag_turn(
            query=query,
            session_id="e2e-no-dup-get",
            user_id="analyst@fde-workspace.corp",
            reasoning_llm=ScriptedAnalystLlm(
                tool_calls=[
                    (
                        "calculate_period_variances",
                        {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
                    )
                ],
                headline=f"Turn {turn_num} NVDA summary.",
                intent="longitudinal_single_company",
            ),
            skip_background_judge=True,
        )
        assert resp["status"] == "success"
        assert resp["context_and_memory"]["session_memory"]["persistence"]["turn_count"] == turn_num
        assert [m for m, _ in calls] == ["GET", "PATCH"]


def test_session_memory_clears_tickers_on_sector_switch_and_isolates_topic_epochs() -> None:
    """Switching from tickers to a sector clears stale tickers, advances topic_epoch, and isolates memory."""
    transport, _docs, _calls = _mock_firestore_transport()
    cfg = TelemetryConfig(
        telemetry_backend="local",
        feedback_backend="gcp",
        project_id="iron-radio-514217-q7",
    )
    store_1 = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    store_2 = FirestoreSessionStore(
        cfg, http_client=httpx.Client(transport=transport), credentials=_FakeCreds()
    )
    mgr_1 = ConversationMemoryManager(session_store=store_1, refresh_ttl_seconds=0.0)
    mgr_2 = ConversationMemoryManager(session_store=store_2, refresh_ttl_seconds=0.0)

    # Turn 1: Single-company Eli Lilly turn (epoch 0)
    mgr_1.record_turn(
        session_id="subject-switch-sess",
        user_id="alice@corp",
        query="Analyze Eli Lilly revenue FY2024-FY2026",
        intent="longitudinal_single_company",
        tickers=["LLY"],
        gics_sector="Health Care",
        start_year=2024,
        end_year=2026,
        variance_highlights=["LLY revenue +40%"],
        topic_epoch=0,
    )
    rec1 = mgr_2.recall_session_context("subject-switch-sess", user_id="alice@corp")
    assert rec1["remembered_tickers"] == ["LLY"]
    assert rec1["last_gics_sector"] == "ALL"
    assert rec1["topic_epoch"] == 0
    assert "LLY" in rec1["memory_summary_text"]

    # Turn 2: Switch to Industrials sector (epoch 1) -> stale LLY ticker is cleared and excluded from summary
    mgr_1.record_turn(
        session_id="subject-switch-sess",
        user_id="alice@corp",
        query="How are Industrials companies discussing supply chain risks?",
        intent="thematic_trend_tracking",
        tickers=[],
        gics_sector="Industrials",
        start_year=2024,
        end_year=2026,
        variance_highlights=["Industrials supply chain risk increased"],
        topic_epoch=1,
    )
    rec2 = mgr_2.recall_session_context("subject-switch-sess", user_id="alice@corp")
    assert rec2["remembered_tickers"] == []
    assert rec2["last_gics_sector"] == "Industrials"
    assert rec2["topic_epoch"] == 1
    assert "Industrials" in rec2["memory_summary_text"]
    assert "LLY" not in rec2["memory_summary_text"]

    # Deleting the logical session evicts both epoch 0 and topic-suffixed ADK sessions
    mgr_1.adk_session_service.sessions["edgar_chat_app"] = {
        "alice@corp": {
            "subject-switch-sess": object(),
            "subject-switch-sess::topic-1": object(),
            "other-session": object(),
        }
    }
    assert mgr_1.delete_session("subject-switch-sess", user_id="alice@corp") is True
    remaining = mgr_1.adk_session_service.sessions["edgar_chat_app"]["alice@corp"]
    assert "subject-switch-sess" not in remaining
    assert "subject-switch-sess::topic-1" not in remaining
    assert "other-session" in remaining


def test_subject_change_isolates_adk_session_and_omits_stale_memory_prompt(
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end subject switch isolates ADK session epoch, omits old memory from prompt, and carries new sector."""
    from edgar_capstone.agents.orchestration import chat_engine
    from edgar_capstone.agents.runtime import live_runtime
    from tests.structured_report_helpers import ScriptedAnalystLlm

    mgr = ConversationMemoryManager(
        session_store=FirestoreSessionStore(
            TelemetryConfig(telemetry_backend="local", feedback_backend="local")
        )
    )
    monkeypatch.setattr(chat_engine, "get_conversation_memory_manager", lambda: mgr)
    monkeypatch.setattr(live_runtime, "get_conversation_memory_manager", lambda: mgr)

    seen_prompts: list[str] = []

    class _CapturingLlm(ScriptedAnalystLlm):
        async def generate_content_async(self, llm_request: Any, stream: bool = False):
            prompt_chunks = [
                getattr(p, "text", "")
                for c in (llm_request.contents or [])
                for p in (getattr(c, "parts", None) or [])
                if getattr(p, "text", None)
            ]
            seen_prompts.append("\n".join(prompt_chunks))
            async for item in super().generate_content_async(llm_request, stream=stream):
                yield item

    # Turn 1: NVDA (epoch 0)
    t1 = chat_engine.run_conversational_rag_turn(
        query="Analyze NVIDIA's revenue from 2024 to 2026",
        session_id="topic-iso-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=_CapturingLlm(
            headline="NVDA revenue summary.", intent="longitudinal_single_company"
        ),
        skip_background_judge=True,
    )
    assert t1["status"] == "success"
    assert t1["resolved_scope"]["tickers"] == ["NVDA"]
    assert t1["resolved_scope"]["subject_changed"] is False
    assert t1["resolved_scope"]["topic_epoch"] == 0

    # Turn 2: Switch to AAPL (disjoint ticker -> subject_changed=True, topic_epoch=1)
    seen_prompts.clear()
    t2 = chat_engine.run_conversational_rag_turn(
        query="Analyze Apple's revenue from 2024 to 2026",
        session_id="topic-iso-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=_CapturingLlm(
            headline="AAPL revenue summary.", intent="longitudinal_single_company"
        ),
        skip_background_judge=True,
    )
    assert t2["status"] == "success"
    assert t2["resolved_scope"]["tickers"] == ["AAPL"]
    assert t2["resolved_scope"]["subject_changed"] is True
    assert t2["resolved_scope"]["topic_epoch"] == 1
    assert seen_prompts
    assert "## Recalled Conversational Session Memory" not in seen_prompts[0]
    assert "NVIDIA" not in seen_prompts[0]

    # Verify ADK created a separate epoch-1 session instead of reusing epoch-0 NVDA history,
    # while LLM telemetry records still carry the logical session_id for BigQuery joins.
    adk_user_sessions = mgr.adk_session_service.sessions["edgar_chat_app"][
        "analyst@fde-workspace.corp"
    ]
    assert "topic-iso-sess" in adk_user_sessions
    assert "topic-iso-sess::topic-1" in adk_user_sessions
    t2_llm_records = t2["context_and_memory"]["model_call_records"]
    assert t2_llm_records and all(r["session_id"] == "topic-iso-sess" for r in t2_llm_records)

    # Turn 3: Switch to Information Technology sector (clears remembered_tickers, epoch 2)
    seen_prompts.clear()
    t3 = chat_engine.run_conversational_rag_turn(
        query="How are tech sector companies discussing AI risks?",
        session_id="topic-iso-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=_CapturingLlm(
            headline="Tech sector AI risk summary.", intent="thematic_trend_tracking"
        ),
        skip_background_judge=True,
    )
    assert t3["status"] == "success"
    assert t3["resolved_scope"]["tickers"] == []
    assert t3["resolved_scope"]["gics_sector"] == "Information Technology"
    assert t3["resolved_scope"]["subject_changed"] is True
    assert t3["resolved_scope"]["topic_epoch"] == 2
    assert "## Recalled Conversational Session Memory" not in seen_prompts[0]

    # Turn 4: Follow-up inherits Information Technology sector (never snaps back to AAPL or NVDA!)
    t4 = chat_engine.run_conversational_rag_turn(
        query="What about export control risks?",
        session_id="topic-iso-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=_CapturingLlm(
            headline="Tech sector export risk summary.", intent="thematic_trend_tracking"
        ),
        skip_background_judge=True,
    )
    assert t4["status"] == "success"
    assert t4["resolved_scope"]["tickers"] == []
    assert t4["resolved_scope"]["gics_sector"] == "Information Technology"
    assert t4["resolved_scope"]["memory_inherited"] is True
    assert t4["resolved_scope"]["subject_changed"] is False
    assert t4["resolved_scope"]["topic_epoch"] == 2
