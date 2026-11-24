"""Unit & integration tests for Layer 2: GICS sub-industry catalog, Flash-Lite scope/continuity
resolution, strict EntityCatalog validation, and parallel-with-Model-Armor execution."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents.orchestration import chat_engine
from edgar_capstone.agents.orchestration.scope_continuity import (
    ScopeResolutionDecision,
    classify_scope_and_continuity,
    deterministic_continuity_fallback,
    validate_scope_decision,
)
from edgar_capstone.agents.orchestration.scope_validation import (
    EntityCatalog,
    validate_query_scope_and_coverage,
)
from edgar_capstone.agents.runtime.context_memory import ConversationMemoryManager
from edgar_capstone.telemetry import FirestoreSessionStore, TelemetryConfig


def _defense_and_pharma_profiles() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            ticker="LLY",
            cik="0000059478",
            company_name="Eli Lilly and Company",
            gics_sector="Health Care",
            gics_sub_industry="Pharmaceuticals",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="PFE",
            cik="0000078003",
            company_name="Pfizer Inc.",
            gics_sector="Health Care",
            gics_sub_industry="Pharmaceuticals",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="LMT",
            cik="0000936468",
            company_name="Lockheed Martin Corporation",
            gics_sector="Industrials",
            gics_sub_industry="Aerospace & Defense",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="NOC",
            cik="0001133421",
            company_name="Northrop Grumman Corporation",
            gics_sector="Industrials",
            gics_sub_industry="Aerospace & Defense",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="RTX",
            cik="0000101829",
            company_name="RTX Corporation",
            gics_sector="Industrials",
            gics_sub_industry="Aerospace & Defense",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="GD",
            cik="0000040533",
            company_name="General Dynamics Corporation",
            gics_sector="Industrials",
            gics_sub_industry="Aerospace & Defense",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="DAL",
            cik="0000027904",
            company_name="Delta Air Lines, Inc.",
            gics_sector="Industrials",
            gics_sub_industry="Passenger Airlines",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="UAL",
            cik="0000100517",
            company_name="United Airlines Holdings, Inc.",
            gics_sector="Industrials",
            gics_sub_industry="Passenger Airlines",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="GOOG",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
            gics_sub_industry="Interactive Media & Services",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="GOOGL",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
            gics_sub_industry="Interactive Media & Services",
            fiscal_year=2025,
        ),
    ]


class _ScriptedScopeLlm(BaseLlm):
    """Deterministic ADK `BaseLlm` returning canned `ScopeResolutionDecision` payloads."""

    model: str = "gemini-3.5-flash-lite"
    decision_payload: dict[str, Any] = {}
    delay_seconds: float = 0.0
    raise_error: bool = False
    call_count: int = 0
    seen_prompts: list[str] = []
    started_at: list[float] = []
    finished_at: list[float] = []

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        _ = stream
        self.call_count += 1
        self.started_at.append(time.perf_counter())
        prompt_text = "\n".join(
            getattr(p, "text", "")
            for c in (llm_request.contents or [])
            for p in (getattr(c, "parts", None) or [])
            if getattr(p, "text", None)
        )
        self.seen_prompts.append(prompt_text)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        self.finished_at.append(time.perf_counter())
        if self.raise_error:
            raise RuntimeError("Simulated Flash-Lite outage")
        yield LlmResponse(
            model_version=self.model,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=json.dumps(self.decision_payload))],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=180,
                candidates_token_count=32,
            ),
        )


def test_deterministic_sub_industry_resolves_defense_after_eli_lilly_without_llm() -> None:
    """Eli Lilly -> 'defense industry' switches cleanly to Aerospace & Defense tickers with 0 LLM calls."""
    catalog = EntityCatalog(_defense_and_pharma_profiles())

    # Dual-class CIK deduplication holds inside sub-industries (Alphabet -> GOOGL only)
    canon_im, im_tickers = catalog.tickers_for_sub_industry("Interactive Media & Services")
    assert canon_im == "Interactive Media & Services"
    assert im_tickers == ["GOOGL"]

    # Brand alias resolves Raytheon -> RTX
    found_rtx, _ = catalog.detect_tickers_in_query("How did Raytheon revenue grow in FY2025?")
    assert found_rtx == ["RTX"]

    scope_llm = _ScriptedScopeLlm(
        decision_payload={"action": "continue_prior"},
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    lly_mem = {
        "has_prior_memory": True,
        "remembered_tickers": ["LLY"],
        "last_start_year": 2024,
        "last_end_year": 2026,
        "last_gics_sector": "ALL",
        "topic_epoch": 0,
    }
    defense_scope = chat_engine.resolve_query_scope(
        query="What are the major risks facing the defense industry?",
        memory_context=lly_mem,
        catalog=catalog,
        scope_llm=scope_llm,
    )
    assert scope_llm.call_count == 0  # Resolved deterministically in <1 ms
    assert defense_scope["gics_sub_industry"] == "Aerospace & Defense"
    assert defense_scope["tickers"] == ["GD", "LMT", "NOC", "RTX"]
    assert defense_scope["memory_inherited"] is False
    assert defense_scope["subject_changed"] is True
    assert defense_scope["topic_epoch"] == 1
    assert defense_scope["scope_resolution_source"] == "sub_industry"

    # Follow-up on Aerospace & Defense preserves sub-industry label & tickers
    defense_mem = {
        "has_prior_memory": True,
        "remembered_tickers": defense_scope["tickers"],
        "last_gics_sub_industry": "Aerospace & Defense",
        "last_gics_sector": "ALL",
        "last_start_year": 2024,
        "last_end_year": 2026,
        "topic_epoch": 1,
    }
    followup = chat_engine.resolve_query_scope(
        query="What about their Item 1A supply chain risks?",
        memory_context=defense_mem,
        catalog=catalog,
    )
    assert followup["tickers"] == ["GD", "LMT", "NOC", "RTX"]
    assert followup["gics_sub_industry"] == "Aerospace & Defense"
    assert followup["memory_inherited"] is True
    assert followup["subject_changed"] is False
    assert followup["topic_epoch"] == 1

    # Airlines sub-industry also resolves deterministically
    airlines = chat_engine.resolve_query_scope(
        query="How are airlines discussing fuel costs?",
        memory_context=lly_mem,
        catalog=catalog,
    )
    assert airlines["gics_sub_industry"] == "Passenger Airlines"
    assert airlines["tickers"] == ["DAL", "UAL"]
    assert airlines["subject_changed"] is True


def test_deterministic_continuity_fallback_blocks_false_carryover_on_new_subjects() -> None:
    """Even offline without Flash-Lite, new industries/unknown entities/cross-market queries never stick to LLY."""
    catalog = EntityCatalog(_defense_and_pharma_profiles())
    lly_mem = {
        "has_prior_memory": True,
        "remembered_tickers": ["LLY"],
        "last_start_year": 2024,
        "last_end_year": 2026,
        "last_gics_sector": "ALL",
        "topic_epoch": 0,
    }

    # Unmatched industry phrase -> clarify (never inherits LLY)
    robotics = chat_engine.resolve_query_scope(
        query="What about the robotics industry?",
        memory_context=lly_mem,
        catalog=catalog,
    )
    assert robotics["tickers"] == []
    assert robotics["memory_inherited"] is False
    val_rob = validate_query_scope_and_coverage(
        query="What about the robotics industry?",
        scope=robotics,
        explicit_tickers=[],
        unknown_entities=robotics["unknown_entities"],
        catalog=catalog,
        memory_context=lly_mem,
    )
    assert val_rob["action"] == "clarification"

    # Unrecognized private company -> clarify (never inherits LLY)
    anduril = chat_engine.resolve_query_scope(
        query="Tell me about Anduril",
        memory_context=lly_mem,
        catalog=catalog,
    )
    assert anduril["tickers"] == []
    assert anduril["memory_inherited"] is False

    # Cross-market question after single company -> broad_corpus (never inherits LLY)
    tariffs = chat_engine.resolve_query_scope(
        query="Which companies are most exposed to tariffs?",
        memory_context=lly_mem,
        catalog=catalog,
    )
    assert tariffs["tickers"] == []
    assert tariffs["broad_corpus"] is True
    assert tariffs["subject_changed"] is True
    val_tar = validate_query_scope_and_coverage(
        query="Which companies are most exposed to tariffs?",
        scope=tariffs,
        explicit_tickers=[],
        unknown_entities=[],
        catalog=catalog,
        memory_context=lly_mem,
    )
    assert val_tar["action"] == "proceed"

    # Genuine follow-up still inherits LLY
    followup = chat_engine.resolve_query_scope(
        query="What Item 1A risk factors could impact those margins?",
        memory_context=lly_mem,
        catalog=catalog,
    )
    assert followup["tickers"] == ["LLY"]
    assert followup["memory_inherited"] is True
    assert followup["subject_changed"] is False


def test_flash_lite_scope_classifier_validates_against_catalog_and_handles_outages() -> None:
    """Flash-Lite resolves colloquial industries/unknown entities, drops hallucinated tickers, and falls back on error."""
    catalog = EntityCatalog(_defense_and_pharma_profiles())
    lly_mem = {
        "has_prior_memory": True,
        "remembered_tickers": ["LLY"],
        "last_start_year": 2024,
        "last_end_year": 2026,
        "last_gics_sector": "ALL",
        "topic_epoch": 0,
    }

    # 1. Colloquial phrasing not in deterministic regexes -> Flash-Lite maps to Aerospace & Defense
    llm_sub = _ScriptedScopeLlm(
        decision_payload={
            "action": "new_sub_industry",
            "gics_sub_industry": "Aerospace & Defense",
            "reason": "User switched from Eli Lilly to military contractors",
        },
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    scope_sub = chat_engine.resolve_query_scope(
        query="How are military contractors doing on margins?",
        memory_context=lly_mem,
        catalog=catalog,
        scope_llm=llm_sub,
    )
    assert llm_sub.call_count == 1
    assert scope_sub["gics_sub_industry"] == "Aerospace & Defense"
    assert scope_sub["tickers"] == ["GD", "LMT", "NOC", "RTX"]
    assert scope_sub["subject_changed"] is True
    assert scope_sub["topic_epoch"] == 1
    assert scope_sub["scope_resolution_source"] == "flash_lite"

    # 2. Unknown private entity -> triggers out_of_corpus instead of reporting LLY
    llm_unk = _ScriptedScopeLlm(
        decision_payload={
            "action": "unknown_entity",
            "unknown_entities": ["Anduril"],
            "reason": "Anduril is a private defense company outside the S&P 500",
        },
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    scope_unk = chat_engine.resolve_query_scope(
        query="Tell me about Anduril",
        memory_context=lly_mem,
        catalog=catalog,
        scope_llm=llm_unk,
    )
    assert scope_unk["tickers"] == []
    assert scope_unk["unknown_entities"] == ["Anduril"]
    val_unk = validate_query_scope_and_coverage(
        query="Tell me about Anduril",
        scope=scope_unk,
        explicit_tickers=[],
        unknown_entities=scope_unk["unknown_entities"],
        catalog=catalog,
        memory_context=lly_mem,
    )
    assert val_unk["action"] == "out_of_corpus"
    assert val_unk["reason"] == "non_sp500_issuer"

    # 3. Strict validation strips hallucinated tickers
    validated = validate_scope_decision(
        ScopeResolutionDecision(action="new_tickers", tickers=["FAKE", "LMT"]),
        catalog=catalog,
        has_prior_scope=True,
    )
    assert validated.action == "new_tickers"
    assert validated.tickers == ["LMT"]
    assert validated.unknown_entities == ["FAKE"]

    validated_all_fake = validate_scope_decision(
        ScopeResolutionDecision(action="new_tickers", tickers=["NOPE"]),
        catalog=catalog,
        has_prior_scope=True,
    )
    assert validated_all_fake.action == "unknown_entity"
    assert validated_all_fake.unknown_entities == ["NOPE"]

    # 4. Outage gracefully degrades to deterministic fallback (never crashes)
    llm_err = _ScriptedScopeLlm(raise_error=True, seen_prompts=[], started_at=[], finished_at=[])
    dec, src = classify_scope_and_continuity(
        query="What about those margins?",
        memory_context=lly_mem,
        catalog=catalog,
        scope_llm=llm_err,
    )
    assert src == "deterministic_fallback"
    assert dec.action == "continue_prior"

    # Deterministic fallback helper directly returns clarify on Turn 1 unresolved queries
    t1_fb = deterministic_continuity_fallback(
        "How did margins perform?", memory_context={"has_prior_memory": False}, catalog=catalog
    )
    assert t1_fb.action == "clarify"


def test_parallel_with_model_armor_prepass_overlaps_flash_lite_and_respects_pii_and_stale_l1(
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speculative Flash-Lite runs in parallel with Model Armor when warm, skips PII, and discards stale L1."""
    from edgar_capstone.agents.runtime import live_runtime
    from edgar_capstone.security.model_armor import get_model_armor_guardrail
    from tests.structured_report_helpers import ScriptedAnalystLlm

    mgr = ConversationMemoryManager(
        session_store=FirestoreSessionStore(
            TelemetryConfig(telemetry_backend="local", feedback_backend="local")
        )
    )
    monkeypatch.setattr(chat_engine, "get_conversation_memory_manager", lambda: mgr)
    monkeypatch.setattr(live_runtime, "get_conversation_memory_manager", lambda: mgr)

    # Turn 1: explicit NVDA query -> warms catalog & L1 memory, does NOT invoke scope_llm
    scope_llm_1 = _ScriptedScopeLlm(
        decision_payload={"action": "continue_prior"},
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    t1 = chat_engine.run_conversational_rag_turn(
        query="Analyze NVIDIA's revenue from 2024 to 2026",
        session_id="parallel-prepass-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA summary.", intent="longitudinal_single_company"
        ),
        scope_llm=scope_llm_1,
        skip_background_judge=True,
    )
    assert t1["status"] == "success"
    assert scope_llm_1.call_count == 0

    # Turn 2: ambiguous follow-up with warm L1 memory + warm catalog -> overlaps Model Armor & Flash-Lite!
    guardrail = get_model_armor_guardrail()
    orig_inspect = guardrail.inspect_user_prompt
    guard_times: list[tuple[float, float]] = []
    guard_threads: list[int] = []

    def _slow_inspect(text: str, **kwargs: Any):
        guard_threads.append(threading.get_ident())
        s = time.perf_counter()
        time.sleep(0.06)
        dec = orig_inspect(text, **kwargs)
        e = time.perf_counter()
        guard_times.append((s, e))
        return dec

    monkeypatch.setattr(guardrail, "inspect_user_prompt", _slow_inspect)

    scope_llm_2 = _ScriptedScopeLlm(
        decision_payload={"action": "continue_prior", "reason": "Follow-up on NVDA margins"},
        delay_seconds=0.06,
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    t2 = chat_engine.run_conversational_rag_turn(
        query="What about those margins?",
        session_id="parallel-prepass-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA margin summary.", intent="longitudinal_single_company"
        ),
        scope_llm=scope_llm_2,
        skip_background_judge=True,
    )
    assert t2["status"] == "success"
    assert t2["resolved_scope"]["tickers"] == ["NVDA"]
    assert t2["resolved_scope"]["memory_inherited"] is True
    assert t2["resolved_scope"]["scope_resolution_source"] == "flash_lite_parallel"
    # Invoked ONCE (reused from parallel prepass, NOT re-run in resolve_query_scope)
    assert scope_llm_2.call_count == 1
    # Verify actual concurrent overlap: each started before the other finished
    g_start, g_end = guard_times[-1]
    s_start, s_end = scope_llm_2.started_at[0], scope_llm_2.finished_at[0]
    assert s_start < g_end and g_start < s_end

    # Turn 3: Query with sensitive PII (SSN) MUST NOT run speculatively before SDP redaction
    monkeypatch.setattr(guardrail, "inspect_user_prompt", orig_inspect)
    scope_llm_3 = _ScriptedScopeLlm(
        decision_payload={"action": "continue_prior", "reason": "Follow-up after SDP redaction"},
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    t3 = chat_engine.run_conversational_rag_turn(
        query="My SSN is 123-45-6789, what about those margins?",
        session_id="parallel-prepass-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA margin summary.", intent="longitudinal_single_company"
        ),
        scope_llm=scope_llm_3,
        skip_background_judge=True,
    )
    assert t3["status"] == "success"
    assert t3["resolved_scope"]["scope_resolution_source"] == "flash_lite"
    assert scope_llm_3.call_count == 1
    assert "123-45-6789" not in scope_llm_3.seen_prompts[0]

    # Turn 4: Stale L1 peek mismatch against authoritative recall discards speculative result and re-runs
    orig_peek = mgr.peek_warm_session_context
    monkeypatch.setattr(
        mgr,
        "peek_warm_session_context",
        lambda sid, user_id="analyst": (
            True,
            {
                "has_prior_memory": True,
                "remembered_tickers": ["AAPL"],  # Stale/different from actual L1 NVDA!
                "last_gics_sector": "ALL",
                "last_gics_sub_industry": None,
                "topic_epoch": 0,
                "memory_summary_text": "",
            },
        ),
    )
    scope_llm_4 = _ScriptedScopeLlm(
        decision_payload={"action": "continue_prior", "reason": "Follow-up"},
        seen_prompts=[],
        started_at=[],
        finished_at=[],
    )
    t4 = chat_engine.run_conversational_rag_turn(
        query="What about those margins?",
        session_id="parallel-prepass-sess",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA margin summary.", intent="longitudinal_single_company"
        ),
        scope_llm=scope_llm_4,
        skip_background_judge=True,
    )
    monkeypatch.setattr(mgr, "peek_warm_session_context", orig_peek)
    assert t4["status"] == "success"
    # Speculative call ran on stale AAPL peek, was discarded, and re-ran synchronously on NVDA!
    assert scope_llm_4.call_count == 2
    assert t4["resolved_scope"]["scope_resolution_source"] == "flash_lite"
    assert t4["resolved_scope"]["tickers"] == ["NVDA"]
