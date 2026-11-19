"""Structured ReAct reasoning, dual-model routing, and context/memory latency tracking.

Tools read live GCP (BigQuery, GCS, Vertex AI embeddings) and chat turns call live Gemini. The
only scripted ADK `BaseLlm` drives a fixed 3-step tool sequence over the live tools so the ReAct
phase trace and the context-latency math (which depends on known token counts) can be asserted
exactly; its final answer cites a chunk and quote returned by the live tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.adk.apps import App
from google.adk.memory import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from edgar_capstone.agents import (
    FINANCIAL_REASONING_MODEL,
    FLASH_CODING_AND_EVAL_MODEL,
    FLASH_LITE_MODEL,
    ContextBreakdown,
    ContextLatencyTracker,
    ConversationMemoryManager,
    QueryCached10KTool,
    StructuredReActExecutor,
    after_model_fact_check_callback,
    after_model_telemetry_callback,
    attribute_context_latency,
    before_model_safety_and_latency_callback,
    before_model_telemetry_callback,
    calculate_period_variances,
    create_edgar_adk_app,
    create_financial_analyst_agent,
    create_narrative_evaluator_agent,
    evaluate_calculated_variances,
    extract_llm_request_context_breakdown,
    flush_pending_evaluations,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_model_strategy_metadata,
    query_cached_10k_context,
    reconcile_breakdown_with_usage_metadata,
    resolve_model_for_task,
)
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app
from tests.structured_report_helpers import (
    ScriptedAnalystLlm,
    ScriptedJudgeLlm,
    patch_http_chat_llms,
    report_json,
    verbatim_quote,
)


def _bearer_headers() -> dict[str, str]:
    jwt_token, _ = issue_local_dev_workspace_sso_token(
        email="analyst@fde-workspace.corp",
        name="Lead SEC Analyst",
        groups=["sec-edgar-analysts"],
    )
    return {"Authorization": f"Bearer {jwt_token}"}


class _DeterministicReActProLlm(BaseLlm):
    """Deterministic `BaseLlm` driving a 3-turn ADK `PlanReActPlanner` loop over real 10-K tools."""

    call_log: list[dict[str, Any]] = []

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        _ = stream
        fn_responses: list[tuple[str, dict[str, Any]]] = []
        for content in llm_request.contents or []:
            for part in content.parts or []:
                if part.function_response is not None:
                    fn_responses.append(
                        (
                            str(part.function_response.name),
                            dict(part.function_response.response or {}),
                        )
                    )

        seen_tools = {name for name, _ in fn_responses}
        self.call_log.append(
            {
                "turn": len(self.call_log) + 1,
                "seen_tools": sorted(seen_tools),
                "native_thinking_requested": bool(
                    getattr(
                        getattr(llm_request.config, "thinking_config", None),
                        "include_thoughts",
                        False,
                    )
                ),
            }
        )

        # ReAct Turn 1: Plan & invoke deterministic variance calculation engine
        if "calculate_period_variances" not in seen_tools:
            yield LlmResponse(
                model_version=self.model,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=(
                                "/*PLANNING*/\n"
                                "1. Call calculate_period_variances for NVDA FY2024-FY2026.\n"
                                "2. Query cached 10-K Item 7 MD&A context for causal variance explanations.\n"
                                "3. Format grounded output with inline citations.\n"
                                "/*ACTION*/"
                            ),
                            thought=True,
                        ),
                        genai_types.Part.from_function_call(
                            name="calculate_period_variances",
                            args={"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
                        ),
                    ],
                ),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=920,
                    cached_content_token_count=0,
                    candidates_token_count=68,
                    thoughts_token_count=54,
                ),
            )
            return

        # ReAct Turn 2: Evaluate calculated variances & query cached 10-K context for explanation
        if "query_cached_10k_context" not in seen_tools:
            var_payload = next(
                resp for name, resp in fn_responses if name == "calculate_period_variances"
            )
            nvda_rep = var_payload["variance_reports"][0]
            rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]
            yield LlmResponse(
                model_version=self.model,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=(
                                f"/*REASONING*/\n"
                                f"Evaluated NVDA FY2024-FY2026 revenue delta of +{rev_pct:.2f}%. "
                                f"Now querying cached 10-K Item 7 MD&A context to explain the variance.\n"
                                f"/*ACTION*/"
                            ),
                            thought=True,
                        ),
                        genai_types.Part.from_function_call(
                            name="query_cached_10k_context",
                            args={
                                "ticker": "NVDA",
                                "fiscal_year": 2026,
                                "section_id": "item_7",
                                "variance_query": "revenue operating income margin Blackwell Hopper data center",
                                "max_passages": 2,
                            },
                        ),
                    ],
                ),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1840,
                    cached_content_token_count=0,
                    candidates_token_count=74,
                    thoughts_token_count=62,
                ),
            )
            return

        # ReAct Turn 3: Format final grounded output combining deterministic numbers & cached 10-K citations
        var_payload = next(
            resp for name, resp in fn_responses if name == "calculate_period_variances"
        )
        cache_payload = next(
            resp for name, resp in fn_responses if name == "query_cached_10k_context"
        )
        assert var_payload["variance_reports"][0]["ticker"] == "NVDA"
        passage = cache_payload["passages"][0]
        chunk_id = passage["chunk_id"]
        final_md = report_json(
            headline="NVIDIA CORP (NVDA) Grounded Variance Report",
            key_metrics=[("NVDA", "revenue"), ("NVDA", "operating_income")],
            findings=[
                {
                    "statement": "The cached MD&A passage explains the revenue variance.",
                    "ticker": "NVDA",
                    "fiscal_year": 2026,
                    "chunk_ids": [chunk_id],
                    "evidence_quote": verbatim_quote(passage["text"]),
                }
            ],
        )
        yield LlmResponse(
            model_version=self.model,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text="/*REASONING*/ Synthesizing final grounded response from cached 10-K context.",
                        thought=True,
                    ),
                    genai_types.Part(text=final_md),
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=6400,
                cached_content_token_count=4800,
                candidates_token_count=165,
                thoughts_token_count=40,
            ),
        )


def test_dual_model_specialization_and_adk_app_configuration() -> None:
    """Verify Gemini 3.8 Flash handles reasoning & low-thinking eval while Gemini 3.5 Flash-Lite handles compaction & mapping."""
    assert FINANCIAL_REASONING_MODEL == "gemini-3.8-flash"
    assert FLASH_CODING_AND_EVAL_MODEL == "gemini-3.8-flash"
    assert FLASH_LITE_MODEL == "gemini-3.5-flash-lite"

    # Task routing assertions
    for pro_task in (
        "financial_reasoning",
        "variance_explanation",
        "peer_comparison",
        "thematic_synthesis",
        "orchestration",
    ):
        assert resolve_model_for_task(pro_task) == "gemini-3.8-flash"  # type: ignore[arg-type]

    for eval_task in (
        "evaluation",
        "llm_as_judge",
        "coding_assistance",
    ):
        assert resolve_model_for_task(eval_task) == "gemini-3.8-flash"  # type: ignore[arg-type]

    for lite_task in (
        "context_compaction",
        "context_cache_warmup",
    ):
        assert resolve_model_for_task(lite_task) == "gemini-3.5-flash-lite"  # type: ignore[arg-type]

    # ADK Agent model assignments, planners & callbacks via lazy factories
    analyst = create_financial_analyst_agent()
    evaluator = create_narrative_evaluator_agent()
    assert analyst.model == "gemini-3.8-flash"
    # Native thinking for the analyst: PlanReActPlanner's text "/*ACTION*/" tool snippets made
    # Gemini 3.x write tool calls as prose instead of issuing function calls.
    assert isinstance(analyst.planner, BuiltInPlanner)
    assert analyst.planner.thinking_config.include_thoughts is True
    assert analyst.before_model_callback is before_model_safety_and_latency_callback
    assert analyst.after_model_callback is after_model_fact_check_callback

    assert evaluator.model == "gemini-3.8-flash"
    assert evaluator.generate_content_config is not None
    assert evaluator.generate_content_config.thinking_config is not None
    assert (
        evaluator.generate_content_config.thinking_config.thinking_level
        == genai_types.ThinkingLevel.LOW
    )
    assert evaluator.before_model_callback is before_model_telemetry_callback
    assert evaluator.after_model_callback is after_model_telemetry_callback

    analyst_tool_names = {t.name for t in analyst.tools}
    assert QueryCached10KTool.name == "query_cached_10k_context"
    assert "query_cached_10k_context" in analyst_tool_names

    # ADK App ContextCacheConfig + EventsCompactionConfig (summarized via Gemini 3.5 Flash-Lite)
    custom_app = create_edgar_adk_app(analyst, app_name="edgar_test_app")
    assert isinstance(custom_app, App)
    assert custom_app.context_cache_config is not None
    assert custom_app.context_cache_config.min_tokens == 2048
    assert custom_app.context_cache_config.ttl_seconds == 1800
    assert custom_app.events_compaction_config is not None
    assert custom_app.events_compaction_config.token_threshold == 32000
    assert custom_app.events_compaction_config.summarizer is not None
    assert custom_app.events_compaction_config.summarizer._llm.model == "gemini-3.5-flash-lite"  # type: ignore[attr-defined]

    meta = get_model_strategy_metadata()
    assert meta["financial_reasoning_model"] == "gemini-3.8-flash"
    assert meta["coding_and_evaluation_model"] == "gemini-3.8-flash"
    assert meta["compaction_and_mapping_model"] == "gemini-3.5-flash-lite"
    assert meta["events_compaction_config"]["adk_app_summarizer_model"] == "gemini-3.5-flash-lite"
    assert meta["events_compaction_config"]["compaction_method"] == "deterministic_digest"


def test_adk_runner_executes_structured_react_loop_with_context_latency_tracking(
    fake_gcp_adapters: Any,
) -> None:
    """Run an end-to-end ADK Runner session over FinancialAnalystAgent executing the 3-step ReAct loop."""
    # Pre-warm NVDA 2026 item_7 so the ReAct loop exercises a true KV/section cache hit
    query_cached_10k_context(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        variance_query="Blackwell data center revenue",
        max_passages=1,
    )

    mock_llm = _DeterministicReActProLlm(model="gemini-3.8-flash", call_log=[])
    react_analyst = create_financial_analyst_agent(
        name="FinancialAnalystAgent",
        model=mock_llm,  # type: ignore[arg-type]
    )

    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    runner = Runner(
        app_name="sec_edgar_react_test",
        agent=react_analyst,
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )

    async def _run_turn() -> tuple[str, Any]:
        final_text = ""
        async for event in runner.run_async(
            user_id="lead-analyst",
            session_id="adk-react-sess-01",
            new_message=genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(
                        text="Evaluate NVDA FY2024 to FY2026 variances and explain using cached 10-K context."
                    )
                ],
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    p.text or "" for p in event.content.parts if not getattr(p, "thought", False)
                )
        sess = await session_service.get_session(
            app_name="sec_edgar_react_test",
            user_id="lead-analyst",
            session_id="adk-react-sess-01",
        )
        return final_text, sess

    final_answer, session = asyncio.run(_run_turn())

    # 1. Verify the analyst requested native thinking (BuiltInPlanner) & executed 3 model turns
    assert len(mock_llm.call_log) == 3
    assert mock_llm.call_log[0]["native_thinking_requested"] is True
    assert "NVIDIA CORP (NVDA)" in final_answer
    assert "NVDA_2026_item_7_" in final_answer

    # 2. Verify ReAct phase progression in session state
    assert session is not None
    react_phases = [step["phase"] for step in session.state["react_trace"]]
    assert react_phases == [
        "evaluate_variances",
        "query_cached_10k_context",
        "format_grounded_output",
    ]

    # 3. Verify context-induced latency metrics & KV-cache prefill savings on Turn 3
    latency_metrics = session.state["latency_metrics"]
    assert len(latency_metrics) == 3
    final_call_metric = latency_metrics[-1]
    assert final_call_metric["model_name"] == "gemini-3.8-flash"
    assert final_call_metric["context_tokens"] == 6400
    assert final_call_metric["cached_context_tokens"] == 4800
    assert final_call_metric["uncached_context_tokens"] == 1600
    assert final_call_metric["context_latency_ms"] >= 0.0
    assert final_call_metric["estimated_cache_latency_saved_ms"] > 100.0
    assert final_call_metric["within_ttft_slo"] is True

    # 4. Verify S2S audit trail, numeric fact-check, and after_agent memory duration callback
    s2s_tools = [entry["tool_name"] for entry in session.state["s2s_audit_trail"]]
    assert s2s_tools == ["calculate_period_variances", "query_cached_10k_context"]
    assert session.state["fact_check_report"]["passed"] is True
    assert session.state["last_agent_total_duration_ms"] >= 0.0


def test_context_induced_latency_attribution_and_cache_speedup_math() -> None:
    """Verify context latency scales with input context tokens and reflects cached prefill savings."""
    tracker = ContextLatencyTracker()

    # Case A: Small context (500 tokens) vs Large uncached context (12,000 tokens) with identical 200 output tokens & 800ms inference
    small_ctx = ContextBreakdown(
        system_instruction_tokens=200,
        user_prompt_tokens=300,
        total_context_tokens=500,
        uncached_context_tokens=500,
    )
    large_uncached_ctx = ContextBreakdown(
        system_instruction_tokens=200,
        variance_payload_tokens=1800,
        uncached_10k_and_tool_tokens=9700,
        user_prompt_tokens=300,
        total_context_tokens=12000,
        uncached_context_tokens=12000,
    )
    large_cached_ctx = ContextBreakdown(
        system_instruction_tokens=200,
        variance_payload_tokens=1800,
        cached_10k_tokens=9700,
        user_prompt_tokens=300,
        total_context_tokens=12000,
        cached_context_tokens=9700,
        uncached_context_tokens=2300,
        cache_hit=True,
    )

    rec_small = tracker.record_call(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="evaluate_variances",
        context_breakdown=small_ctx,
        output_tokens=200,
        raw_inference_latency_ms=800.0,
        context_prep_latency_ms=5.0,
    )
    rec_large_uncached = tracker.record_call(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="format_grounded_output",
        context_breakdown=large_uncached_ctx,
        output_tokens=200,
        raw_inference_latency_ms=800.0,
        context_prep_latency_ms=5.0,
    )
    rec_large_cached = tracker.record_call(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="format_grounded_output",
        context_breakdown=large_cached_ctx,
        output_tokens=200,
        raw_inference_latency_ms=800.0,
        context_prep_latency_ms=5.0,
    )

    # Larger context attributes a substantially higher fraction of model latency to context prefill
    assert rec_large_uncached.context_latency_ms > rec_small.context_latency_ms
    assert rec_large_uncached.context_latency_ratio > rec_small.context_latency_ratio

    # Caching 9,700 10-K tokens reduces prefill share and records positive KV-cache latency savings
    assert (
        rec_large_cached.context_prefill_latency_ms < rec_large_uncached.context_prefill_latency_ms
    )
    assert rec_large_cached.estimated_cache_latency_saved_ms > 1000.0

    # Explicit observed TTFT above 3000ms SLO marks within_ttft_slo=False
    slo_breach = attribute_context_latency(
        agent_name="FinancialAnalystAgent",
        model_name="gemini-2.5-pro",
        task_category="financial_reasoning",
        react_phase="format_grounded_output",
        context_breakdown=large_uncached_ctx,
        output_tokens=150,
        raw_inference_latency_ms=4200.0,
        context_prep_latency_ms=25.0,
        observed_ttft_ms=3400.0,
    )
    assert slo_breach.within_ttft_slo is False
    assert slo_breach.ttft_ms == 3425.0

    # Record a Gemini 3.5 Flash evaluation call and verify per-model summary aggregation
    tracker.record_call(
        agent_name="NarrativeEvaluatorAgent",
        model_name="gemini-3.5-flash",
        task_category="evaluation",
        react_phase="fact_check_and_eval",
        context_breakdown=small_ctx,
        output_tokens=80,
        raw_inference_latency_ms=110.0,
        context_prep_latency_ms=2.0,
    )
    summary = tracker.summarize()
    assert summary.total_calls == 4
    assert "gemini-2.5-pro" in summary.by_model
    assert "gemini-3.5-flash" in summary.by_model
    assert summary.by_model["gemini-2.5-pro"]["call_count"] == 3
    assert summary.by_model["gemini-3.5-flash"]["call_count"] == 1

    # Verify LlmRequest breakdown extraction + usage_metadata reconciliation
    req = LlmRequest(
        model="gemini-2.5-pro",
        config=genai_types.GenerateContentConfig(system_instruction="System grounding rules."),
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text="Explain NVDA FY2026 revenue variance.")],
            )
        ],
    )
    est_bd = extract_llm_request_context_breakdown(req, state={})
    assert est_bd.system_instruction_tokens > 0
    assert est_bd.user_prompt_tokens > 0
    resp_with_usage = LlmResponse(
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text="Grounded output.")]
        ),
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=4096,
            cached_content_token_count=3072,
            candidates_token_count=95,
            thoughts_token_count=30,
        ),
    )
    reconciled_bd, out_toks, th_toks = reconcile_breakdown_with_usage_metadata(
        est_bd, resp_with_usage
    )
    assert reconciled_bd.total_context_tokens == 4096
    assert reconciled_bd.cached_context_tokens == 3072
    assert reconciled_bd.uncached_context_tokens == 1024
    assert reconciled_bd.cache_hit is True
    assert out_toks == 95
    assert th_toks == 30


def _is_model(name: str, family: str) -> bool:
    """Live responses may report a versioned model name (e.g. `gemini-3.5-flash-001`)."""
    return str(name).split("/")[-1].startswith(family)


def test_structured_react_executor_peer_comparison_with_sync_judge(
    fake_gcp_adapters: Any,
) -> None:
    """Variance evaluation, pre-retrieved citations (no legacy cached-10K pre-pass), synthesis + sync judge."""
    from edgar_capstone.agents.tools import search_10k_chunks_rag

    var_res = calculate_period_variances(tickers="NVDA,AAPL", start_year=2024, end_year=2026)
    reports = var_res["variance_reports"]
    assert {r["ticker"] for r in reports} == {"AAPL", "NVDA"}

    evals = evaluate_calculated_variances(reports, query="AI compute and Services margin")
    assert len(evals) == 2
    for ev in evals:
        rep = next(r for r in reports if r["ticker"] == ev.ticker)
        assert ev.revenue_pct_delta == rep["revenue_variance"]["percentage_delta"]
        assert ev.targeted_10k_queries
        assert {q["section_id"] for q in ev.targeted_10k_queries} <= {"item_7", "item_1a"}
        assert all(q["ticker"] == ev.ticker for q in ev.targeted_10k_queries)

    rag_res = search_10k_chunks_rag(
        query="Compare NVIDIA and Apple revenue and margin drivers from 2024 to 2026",
        tickers="NVDA,AAPL",
        gics_sector="Information Technology",
        section_id="ALL",
        start_year=2024,
        end_year=2026,
        top_k=8,
    )
    prepass_citations = list(rag_res["citations"])
    assert {c["ticker"] for c in prepass_citations} >= {"AAPL", "NVDA"}

    executor = StructuredReActExecutor(
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA and AAPL revenue drivers from FY2024 to FY2026.",
            intent="multi_company_peer_comparison",
            key_metrics=[("NVDA", "revenue"), ("AAPL", "revenue")],
        ),
        judge_llm=ScriptedJudgeLlm(),
    )
    out = executor.execute_post_retrieval_reasoning(
        query="Compare NVIDIA and Apple revenue and margin drivers from 2024 to 2026",
        intent="multi_company_peer_comparison",
        scope={
            "tickers": ["AAPL", "NVDA"],
            "tickers_csv": "AAPL,NVDA",
            "gics_sector": "Information Technology",
            "section_id": "ALL",
            "start_year": 2024,
            "end_year": 2026,
        },
        variance_reports=reports,
        citations=prepass_citations,
        sector_benchmarks=[],
        chunk_health=None,
        memory_tokens=48,
        memory_prep_latency_ms=0.8,
        run_judge_sync=True,
    )
    assert out["execution_mode"] == "live_adk"
    assert out["reasoning_model"] == "gemini-3.8-flash"
    assert _is_model(out["evaluation_model"], "gemini-3.8-flash")
    assert out["adk_runtime"]["executed_via_adk_runner"] is True
    assert out["answer_verified"] is True
    assert out["adk_runtime"]["answer_withheld_reason"] is None

    # No legacy pre-synthesis cached-10K pass; pre-retrieved citations cover both peers directly.
    assert out["cached_explanations"] == []
    assert len(out["citations"]) >= 2
    assert {c["ticker"] for c in out["citations"]} >= {"AAPL", "NVDA"}

    step_phases = [s.phase for s in out["react_steps"]]
    assert step_phases == ["evaluate_variances", "format_grounded_output", "fact_check_and_eval"]

    eval_report = out["evaluation_report"]
    assert _is_model(eval_report.evaluator_model, "gemini-3.8-flash")
    assert eval_report.grade == "pass"
    assert eval_report.overall_score is not None and 0.90 <= eval_report.overall_score <= 1.0
    assert eval_report.context_tokens > 0
    assert out["adk_runtime"]["agent_retrieval"]["unresolved_cited_chunk_ids"] == []


def test_conversation_memory_token_threshold_compaction() -> None:
    """Older turns are compacted into a digest once session memory crosses the token threshold."""
    compact_mgr = ConversationMemoryManager(token_threshold=80, retention_turns=1)
    tracker = get_context_latency_tracker()
    before_all_count = len(tracker.recent_records(limit=500))

    compact_mgr.record_turn(
        session_id="compact-sess",
        query="Analyze NVDA FY2024-2026 revenue variance",
        intent="longitudinal_single_company",
        tickers=["NVDA"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        variance_highlights=["Revenue variance highlight"],
        answer_markdown="Detailed grounded synthesis paragraph " * 12,
    )
    res2 = compact_mgr.record_turn(
        session_id="compact-sess",
        query="Compare with AAPL Services growth",
        intent="multi_company_peer_comparison",
        tickers=["NVDA", "AAPL"],
        gics_sector="Information Technology",
        start_year=2024,
        end_year=2026,
        variance_highlights=["AAPL Services margin highlight"],
        answer_markdown="Second detailed grounded synthesis paragraph " * 12,
    )
    assert res2["compaction_triggered"] is True
    assert res2["compaction_method"] == "deterministic_digest"
    assert res2["compaction_count"] == 1
    recalled = compact_mgr.recall_session_context("compact-sess")
    assert "[Compacted Prior Context]:" in recalled["memory_summary_text"]
    assert "T1(NVDA FY2024-2026)" in recalled["memory_summary_text"]

    # Deterministic string-digest compaction MUST NOT fabricate synthetic LLM billing records
    assert len(tracker.recent_records(limit=500)) == before_all_count


def test_http_chat_multi_turn_memory_recall_and_telemetry_endpoints(
    fake_gcp_adapters: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two scripted /api/v1/chat turns verify follow-up memory inheritance, async judge, and telemetry."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    from edgar_capstone.agents import react_loop as react_loop_mod

    judge_llm = ScriptedJudgeLlm()
    orig_eval = react_loop_mod.evaluate_variance_explanation

    def _scripted_bg_eval(**kwargs: Any):
        kwargs["llm"] = judge_llm
        return orig_eval(**kwargs)

    monkeypatch.setattr(react_loop_mod, "evaluate_variance_explanation", _scripted_bg_eval)
    patch_http_chat_llms(
        monkeypatch,
        reasoning_llm=ScriptedAnalystLlm(
            headline="NVDA FY2024-FY2026 revenue and risk disclosures.",
            intent="longitudinal_single_company",
        ),
        skip_background_judge=False,
    )

    client = TestClient(app)
    headers = _bearer_headers()

    # Turn 1 explicitly names NVIDIA (2024-2026)
    resp1 = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "query": "Analyze NVIDIA's revenue and operating margin from 2024 to 2026",
            "session_id": "http-multi-turn-session",
        },
    )
    assert resp1.status_code == 200, resp1.text
    turn1 = resp1.json()
    assert turn1["status"] == "success"
    assert turn1["execution_mode"] == "live_adk"
    assert turn1["answer_verified"] is True
    assert turn1["resolved_scope"]["tickers"] == ["NVDA"]
    assert turn1["resolved_scope"]["memory_inherited"] is False
    assert turn1["context_and_memory"]["session_memory"]["persistence"]["persisted"] is True
    assert turn1["react_execution"]["strategy"] == "structured_react_loop"
    assert turn1["react_execution"]["iterations"] >= 3
    assert turn1["react_execution"]["cached_10k_explanations"] == []
    assert _is_model(turn1["evaluation"]["evaluator_model"], "gemini-3.8-flash")
    assert turn1["context_and_memory"]["models"]["financial_reasoning_model"] == "gemini-3.8-flash"
    assert turn1["context_and_memory"]["turn_context_latency"]["total_calls"] >= 1
    assert turn1["react_execution"]["judge_mode"] == "async_background"
    assert turn1["evaluation"]["grade"] == "pending"
    assert turn1["evaluation"]["overall_score"] is None

    # Turn 2 omits the company ("those margins") and inherits NVDA from session memory
    resp2 = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "query": "What Item 1A risk factors could impact those margins?",
            "session_id": "http-multi-turn-session",
        },
    )
    assert resp2.status_code == 200, resp2.text
    turn2 = resp2.json()
    assert turn2["status"] == "success"
    assert turn2["resolved_scope"]["tickers"] == ["NVDA"]
    assert turn2["resolved_scope"]["memory_inherited"] is True
    assert turn2["resolved_scope"]["section_id"] == "item_1a"
    assert turn2["context_and_memory"]["session_memory"]["has_prior_memory"] is True
    assert turn2["context_and_memory"]["session_memory"]["turn_count"] == 1

    # Flush async scripted judge tasks so telemetry records reflect both analyst & judge calls
    flush_pending_evaluations(timeout=10.0)
    assert judge_llm.call_count == 2

    sess_tel_resp = client.get(
        "/api/v1/telemetry/sessions",
        headers=headers,
        params={"session_id": "http-multi-turn-session", "limit": 20},
    )
    assert sess_tel_resp.status_code == 200, sess_tel_resp.text
    st_body = sess_tel_resp.json()
    assert st_body["status"] == "ok"
    assert st_body["event_count"] >= 2

    tracker_summary = tracker.summarize()
    assert tracker_summary.total_calls >= 4
    by_model = list(tracker_summary.by_model)
    assert any(_is_model(m, "gemini-3.8-flash") for m in by_model), by_model


def test_react_async_judge_pending_and_background_telemetry_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify async background judge updates local telemetry sink from pending/None to scored."""
    from edgar_capstone.agents import react_loop
    from edgar_capstone.agents.live_runtime import AdkStreamResult
    from edgar_capstone.agents.schemas import NarrativeEvaluationReport
    from edgar_capstone.telemetry import get_telemetry_sink

    excerpt = "Data center demand drove strong revenue expansion across hyperscale cloud customers."
    citation = {
        "chunk_id": "NVDA_2026_item_7_0001",
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "gics_sector": "Information Technology",
        "accession_number": "0001045810-26-000021",
        "sec_filing_url": "https://www.sec.gov/example",
        "excerpt": excerpt,
        "score": 0.95,
    }
    report_payload = {
        "headline": "NVIDIA revenue expanded on data center demand.",
        "key_metrics": [],
        "findings": [
            {
                "statement": "Data center demand drove strong revenue expansion.",
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "chunk_ids": ["NVDA_2026_item_7_0001"],
                "evidence_quote": excerpt,
            }
        ],
    }
    stub_stream = AdkStreamResult(
        agent_name="FinancialAnalystAgent",
        app_name="edgar_chat_app",
        session_id="sess-async-judge-1",
        backend="custom_llm",
        final_text=json.dumps(report_payload),
        observed_ttft_ms=112.5,
        session_state={"financial_analyst_report": report_payload},
    )
    monkeypatch.setattr(
        react_loop.StructuredReActExecutor,
        "_invoke_financial_reasoning",
        lambda self, **_kw: stub_stream,
    )
    monkeypatch.setattr(
        react_loop,
        "evaluate_variance_explanation",
        lambda **_kw: NarrativeEvaluationReport(
            evaluator_model="gemini-3.8-flash",
            grade="pass",
            overall_score=0.93,
            numeric_fidelity_score=1.0,
            citation_grounding_score=0.95,
            citation_precision_score=0.95,
            explanation_quality_score=0.90,
            thematic_consistency_score=0.88,
            rationale="Verified async judge result.",
        ),
    )

    sink = get_telemetry_sink()
    sink.reset_local_buffers()
    session_id = "sess-async-judge-1"
    turn_row = sink.record_session_turn_telemetry(
        session_id=session_id,
        user_id="analyst@fde-workspace.corp",
        intent="longitudinal_single_company",
        execution_mode="live_adk",
        status_outcome="success",
        tickers_analyzed=["NVDA"],
        gics_sector="Information Technology",
        start_year=2025,
        end_year=2026,
        report_generated=True,
        variance_reports_count=0,
        citations_count=1,
        agent_trace=[],
        turn_latency_records=[],
        total_latency_ms=150.0,
        evaluation_grade="pending",
        evaluation_score=None,
    )["row"]

    result = react_loop.StructuredReActExecutor().execute_post_retrieval_reasoning(
        query="Why did NVDA revenue grow in FY2026?",
        intent="longitudinal_single_company",
        scope={
            "tickers": ["NVDA"],
            "gics_sector": "ALL",
            "section_id": "item_7",
            "start_year": 2025,
            "end_year": 2026,
        },
        variance_reports=[],
        citations=[citation],
        sector_benchmarks=[],
        chunk_health=None,
        session_id=session_id,
        telemetry_session_id=session_id,
        turn_id=turn_row["turn_id"],
        user_id="analyst@fde-workspace.corp",
        run_judge_sync=False,
        skip_background_judge=False,
    )
    eval_rep = result["evaluation_report"]
    assert result["judge_mode"] == "async_background"
    assert eval_rep.grade == "pending"
    assert eval_rep.overall_score is None

    completed = react_loop.flush_pending_evaluations(timeout=5.0)
    assert completed >= 1
    async_evals = sink.recent_async_evaluations(session_id=session_id)
    assert len(async_evals) == 1
    assert async_evals[0]["evaluation_grade"] == "pass"
    assert async_evals[0]["evaluation_score"] == 0.93
    updated_turns = sink.query_session_telemetry(session_id=session_id)
    assert updated_turns[0]["evaluation_grade"] == "pass"
    assert updated_turns[0]["evaluation_score"] == 0.93
    sink.reset_local_buffers()


def test_react_guided_regeneration_retry_recovers_and_handles_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When attempt 1 fails claim verification, one guided retry receives rejection reasons and can recover."""
    from edgar_capstone.agents import react_loop
    from edgar_capstone.agents.live_runtime import AdkStreamResult

    excerpt = "Blackwell GPUs drove strong Data Center revenue expansion across cloud customers."
    citation = {
        "chunk_id": "NVDA_2026_item_7_0001",
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "gics_sector": "Information Technology",
        "accession_number": "0001045810-26-000021",
        "sec_filing_url": "https://www.sec.gov/example",
        "excerpt": excerpt,
        "score": 0.95,
    }
    bad_payload = {
        "headline": "Hallucinated first attempt",
        "key_metrics": [],
        "findings": [
            {
                "statement": "Revenue fell -99.99% due to imaginary shock.",
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "chunk_ids": ["NVDA_2026_item_7_0001"],
                "evidence_quote": excerpt,
            }
        ],
    }
    good_payload = {
        "headline": "Blackwell drove Data Center revenue expansion.",
        "intent": "longitudinal_single_company",
        "key_metrics": [],
        "findings": [
            {
                "statement": "Blackwell GPUs drove Data Center revenue expansion.",
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "chunk_ids": ["NVDA_2026_item_7_0001"],
                "evidence_quote": excerpt,
            }
        ],
    }
    # Case 1: Retry recovers a verified answer, receives verification_feedback, and runs tool-less + low-thinking
    captured_calls: list[dict[str, Any]] = []

    def _invoke_recover(self, **kwargs: Any) -> AdkStreamResult:
        captured_calls.append(dict(kwargs))
        payload = bad_payload if len(captured_calls) == 1 else good_payload
        return AdkStreamResult(
            agent_name="FinancialAnalystAgent",
            app_name="edgar_chat_app",
            session_id="sess-retry-ok",
            backend="vertex_ai",
            final_text=json.dumps(payload),
            observed_ttft_ms=50.0,
            session_state={"financial_analyst_report": payload},
        )

    monkeypatch.setattr(
        react_loop.StructuredReActExecutor, "_invoke_financial_reasoning", _invoke_recover
    )
    recovered = react_loop.StructuredReActExecutor().execute_post_retrieval_reasoning(
        query="Why did NVDA revenue expand in FY2026?",
        intent="longitudinal_single_company",
        scope={
            "tickers": ["NVDA"],
            "gics_sector": "ALL",
            "section_id": "item_7",
            "start_year": 2025,
            "end_year": 2026,
        },
        variance_reports=[],
        citations=[citation],
        sector_benchmarks=[],
        chunk_health=None,
        skip_background_judge=True,
    )
    assert len(captured_calls) == 2
    assert captured_calls[0].get("verification_feedback") is None
    assert captured_calls[0].get("enable_tools", True) is True
    assert captured_calls[0].get("low_thinking", False) is False
    assert (
        captured_calls[1].get("verification_feedback") is not None
        and "99.99%" in captured_calls[1]["verification_feedback"]
    )
    assert captured_calls[1].get("enable_tools") is False
    assert captured_calls[1].get("low_thinking") is True
    assert recovered["answer_verified"] is True
    assert "Blackwell" in recovered["answer_markdown"]

    # Verify create_financial_analyst_agent honors enable_tools=False, low_thinking=True
    from edgar_capstone.agents.factories import create_financial_analyst_agent
    from edgar_capstone.agents.prompt_builders import build_financial_reasoning_prompt

    repair_agent = create_financial_analyst_agent(enable_tools=False, low_thinking=True)
    assert list(repair_agent.tools or []) == []
    assert repair_agent.before_tool_callback is None
    assert repair_agent.after_tool_callback is None
    assert repair_agent.planner is not None
    assert repair_agent.planner.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
    assert repair_agent.planner.thinking_config.include_thoughts is True

    repair_prompt = build_financial_reasoning_prompt(
        query="Why did NVDA revenue expand?",
        intent="longitudinal_single_company",
        scope={
            "tickers": ["NVDA"],
            "tickers_csv": "NVDA",
            "gics_sector": "ALL",
            "section_id": "item_7",
            "start_year": 2025,
            "end_year": 2026,
        },
        variance_reports=[],
        variance_evaluations=[],
        cached_explanations=[],
        citations=[citation],
        sector_benchmarks=[],
        chunk_health=None,
        verification_feedback="- Finding 0: figure mismatch",
    )
    assert "## Verification Correction Required" in repair_prompt
    assert "## Deterministic Tool Selection" not in repair_prompt
    assert "## 10-K Retrieval Guidance" not in repair_prompt

    # Case 2: Retry raises an upstream error -> safely withholds response without crashing
    calls = [0]

    def _invoke_boom(self, **kwargs: Any) -> AdkStreamResult:
        calls[0] += 1
        if calls[0] == 1:
            return AdkStreamResult(
                agent_name="FinancialAnalystAgent",
                app_name="edgar_chat_app",
                session_id="sess-retry-fail",
                backend="vertex_ai",
                final_text=json.dumps(bad_payload),
                observed_ttft_ms=50.0,
                session_state={"financial_analyst_report": bad_payload},
            )
        raise RuntimeError("transient vertex 503 during retry")

    monkeypatch.setattr(
        react_loop.StructuredReActExecutor, "_invoke_financial_reasoning", _invoke_boom
    )
    withheld = react_loop.StructuredReActExecutor().execute_post_retrieval_reasoning(
        query="Why did NVDA revenue expand in FY2026?",
        intent="longitudinal_single_company",
        scope={
            "tickers": ["NVDA"],
            "gics_sector": "ALL",
            "section_id": "item_7",
            "start_year": 2025,
            "end_year": 2026,
        },
        variance_reports=[],
        citations=[citation],
        sector_benchmarks=[],
        chunk_health=None,
        skip_background_judge=True,
    )
    assert calls[0] == 2
    assert withheld["answer_verified"] is False
    assert withheld["answer_markdown"].startswith("### Answer withheld")


def test_background_judge_manager_shutdown_and_error_isolation() -> None:
    """Exercise `_BackgroundJudgeManager` error handling and `shutdown()` cleanly."""
    from edgar_capstone.agents.react_loop import _BackgroundJudgeManager

    mgr = _BackgroundJudgeManager(max_workers=1)
    ran: list[str] = []

    def _ok() -> None:
        ran.append("ok")

    def _fail() -> None:
        raise RuntimeError("judge boom")

    mgr.submit(_ok)
    mgr.submit(_fail)
    completed = mgr.flush(timeout=5.0)
    assert completed >= 0
    assert ran == ["ok"]
    mgr.shutdown(timeout=2.0)
    # Submitting after shutdown is safely ignored / handled without crashing
    mgr.shutdown(timeout=1.0)


def test_financial_reasoning_model_pricing_and_runtime_prompt_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemini-3.8-flash has its own cost profile and agents render the configured model names."""
    from types import SimpleNamespace

    from edgar_capstone.agents.context_memory import _get_profile_for_model
    from edgar_capstone.agents.prompt_store import make_adk_instruction_provider

    prof = _get_profile_for_model("gemini-3.8-flash")
    assert prof.model_name == "gemini-3.8-flash"
    assert prof.tier == "complex_financial_reasoning"
    assert (prof.uncached_input_usd_per_1m, prof.cached_input_usd_per_1m) == (0.75, 0.075)
    assert prof.output_usd_per_1m == 3.75
    assert _get_profile_for_model("publishers/google/models/gemini-3.8-flash-001") is prof
    lite_prof = _get_profile_for_model("gemini-3.5-flash-lite")
    assert lite_prof.model_name == "gemini-3.5-flash-lite"
    assert lite_prof.tier == "fast_compaction_and_mapping"
    assert (lite_prof.uncached_input_usd_per_1m, lite_prof.cached_input_usd_per_1m) == (0.30, 0.03)
    assert lite_prof.output_usd_per_1m == 2.50
    # Longest-prefix match ensures versioned flash-lite never collides with gemini-3.5-flash.
    assert _get_profile_for_model("publishers/google/models/gemini-3.5-flash-lite-001") is lite_prof
    assert _get_profile_for_model("gemini-3.5-flash").model_name == "gemini-3.5-flash"
    assert _get_profile_for_model("gemini-2.5-pro").model_name == "gemini-2.5-pro"
    # No silent pricing fallback for unlisted models (previously defaulted to Gemini 2.5 Pro).
    with pytest.raises(ValueError, match="No performance/pricing profile"):
        _get_profile_for_model("gemini-unlisted-model")
    with pytest.raises(ValueError):
        _get_profile_for_model("some-other-flash")

    ctx = SimpleNamespace(state={})
    analyst_text = make_adk_instruction_provider("financial_analyst")(ctx)
    assert "FinancialAnalystAgent (`gemini-3.8-flash`)" in analyst_text
    assert "gemini-2.5-pro" not in analyst_text

    eval_text = make_adk_instruction_provider("narrative_evaluator")(ctx)
    assert "NarrativeEvaluatorAgent" in eval_text
    assert "(`gemini-3.8-flash`)" in eval_text

    monkeypatch.setenv("EDGAR_ANALYST_MODEL", "gemini-next-pro")
    assert "FinancialAnalystAgent (`gemini-next-pro`)" in make_adk_instruction_provider(
        "financial_analyst"
    )(ctx)
    monkeypatch.setenv("EDGAR_FLASH_MODEL", "gemini-next-flash")
    assert "(`gemini-next-flash`)" in make_adk_instruction_provider("narrative_evaluator")(ctx)
    monkeypatch.setenv("EDGAR_LITE_MODEL", "gemini-next-lite")
    assert resolve_model_for_task("context_compaction") == "gemini-next-lite"
