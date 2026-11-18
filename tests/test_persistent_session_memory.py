"""Unit tests for Firestore-backed multi-instance persistent ConversationMemoryManager & tenant isolation."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from edgar_capstone.agents.context_memory import ConversationMemoryManager
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
