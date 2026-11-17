"""Structured ReAct reasoning, dual-model routing, and context/memory latency tracking.

Tools read live GCP (BigQuery, GCS, Vertex AI embeddings) and chat turns call live Gemini. The
only scripted ADK `BaseLlm` drives a fixed 3-step tool sequence over the live tools so the ReAct
phase trace and the context-latency math (which depends on known token counts) can be asserted
exactly; its final answer cites a chunk and quote returned by the live tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.adk.apps import App
from google.adk.memory import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from edgar_capstone.agents import (
    FINANCIAL_REASONING_MODEL,
    FLASH_CODING_AND_EVAL_MODEL,
    CodingAssistantAgent,
    ContextBreakdown,
    ContextCacheAgent,
    ContextLatencyTracker,
    ConversationMemoryManager,
    FinancialAnalystAgent,
    NarrativeEvaluatorAgent,
    QueryCached10KTool,
    RootOrchestrator,
    StructuredReActExecutor,
    attribute_context_latency,
    calculate_period_variances,
    create_edgar_adk_app,
    create_financial_analyst_agent,
    edgar_adk_app,
    evaluate_calculated_variances,
    extract_llm_request_context_breakdown,
    generate_financial_coding_assistance,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_model_strategy_metadata,
    query_cached_10k_context,
    reconcile_breakdown_with_usage_metadata,
    resolve_model_for_task,
)
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app
from tests.structured_report_helpers import report_json, verbatim_quote


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
                model_version="gemini-2.5-pro",
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
                model_version="gemini-2.5-pro",
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
            model_version="gemini-2.5-pro",
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
    """Verify Gemini 3.8 Flash handles complex financial reasoning while Gemini 3.5 Flash handles coding, eval, & compaction."""
    assert FINANCIAL_REASONING_MODEL == "gemini-3.8-flash"
    assert FLASH_CODING_AND_EVAL_MODEL == "gemini-3.5-flash"

    # Task routing assertions
    for pro_task in (
        "financial_reasoning",
        "variance_explanation",
        "peer_comparison",
        "thematic_synthesis",
        "orchestration",
    ):
        assert resolve_model_for_task(pro_task) == "gemini-3.8-flash"  # type: ignore[arg-type]

    for flash_task in (
        "evaluation",
        "llm_as_judge",
        "coding_assistance",
        "context_compaction",
        "context_cache_warmup",
    ):
        assert resolve_model_for_task(flash_task) == "gemini-3.5-flash"  # type: ignore[arg-type]

    # ADK Agent model assignments & PlanReActPlanner registration
    assert RootOrchestrator.model == "gemini-3.8-flash"
    assert FinancialAnalystAgent.model == "gemini-3.8-flash"
    assert isinstance(RootOrchestrator.planner, PlanReActPlanner)
    # Native thinking for the analyst: PlanReActPlanner's text "/*ACTION*/" tool snippets made
    # Gemini 3.x write tool calls as prose instead of issuing function calls.
    assert isinstance(FinancialAnalystAgent.planner, BuiltInPlanner)
    assert FinancialAnalystAgent.planner.thinking_config.include_thoughts is True

    assert NarrativeEvaluatorAgent.model == "gemini-3.5-flash"
    assert CodingAssistantAgent.model == "gemini-3.5-flash"
    assert ContextCacheAgent.model == "gemini-3.5-flash"

    sub_agent_models = {sa.name: sa.model for sa in RootOrchestrator.sub_agents}
    assert sub_agent_models["FinancialAnalystAgent"] == "gemini-3.8-flash"
    assert sub_agent_models["ContextCacheAgent"] == "gemini-3.5-flash"
    assert sub_agent_models["NarrativeEvaluatorAgent"] == "gemini-3.5-flash"
    assert sub_agent_models["CodingAssistantAgent"] == "gemini-3.5-flash"

    analyst_tool_names = {t.name for t in FinancialAnalystAgent.tools}
    assert QueryCached10KTool.name == "query_cached_10k_context"
    assert "query_cached_10k_context" in analyst_tool_names

    # ADK App ContextCacheConfig + EventsCompactionConfig (summarized via Gemini 3.5 Flash)
    assert isinstance(edgar_adk_app, App)
    custom_app = create_edgar_adk_app(RootOrchestrator, app_name="edgar_test_app")
    assert custom_app.context_cache_config is not None
    assert custom_app.context_cache_config.min_tokens == 2048
    assert custom_app.context_cache_config.ttl_seconds == 1800
    assert custom_app.events_compaction_config is not None
    assert custom_app.events_compaction_config.token_threshold == 32000
    assert custom_app.events_compaction_config.summarizer is not None
    assert custom_app.events_compaction_config.summarizer._llm.model == "gemini-3.5-flash"  # type: ignore[attr-defined]

    meta = get_model_strategy_metadata()
    assert meta["financial_reasoning_model"] == "gemini-3.8-flash"
    assert meta["coding_and_evaluation_model"] == "gemini-3.5-flash"


def test_adk_runner_executes_structured_react_loop_with_context_latency_tracking() -> None:
    """Run an end-to-end ADK Runner session over FinancialAnalystAgent executing the 3-step ReAct loop."""
    # Pre-warm NVDA 2026 item_7 so the ReAct loop exercises a true KV/section cache hit
    query_cached_10k_context(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        variance_query="Blackwell data center revenue",
        max_passages=1,
    )

    mock_llm = _DeterministicReActProLlm(model="gemini-2.5-pro", call_log=[])
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
    assert final_call_metric["model_name"] == "gemini-2.5-pro"
    assert final_call_metric["context_tokens"] == 6400
    assert final_call_metric["cached_context_tokens"] == 4800
    assert final_call_metric["uncached_context_tokens"] == 1600
    assert final_call_metric["context_latency_ms"] >= 0.0
    assert final_call_metric["estimated_cache_latency_saved_ms"] > 500.0
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


def test_structured_react_executor_live_peer_comparison() -> None:
    """Variance pre-pass, cached 10-K querying with adaptive peer fill, live synthesis + judge."""
    var_res = calculate_period_variances(tickers="NVDA,AAPL", start_year=2024, end_year=2026)
    reports = var_res["variance_reports"]
    assert {r["ticker"] for r in reports} == {"AAPL", "NVDA"}

    evals = evaluate_calculated_variances(reports, query="AI compute and Services margin")
    assert len(evals) == 2
    for ev in evals:
        rep = next(r for r in reports if r["ticker"] == ev.ticker)
        # The pre-pass reads the deterministic engine's figures, never recomputes them
        assert ev.revenue_pct_delta == rep["revenue_variance"]["percentage_delta"]
        assert ev.targeted_10k_queries
        assert {q["section_id"] for q in ev.targeted_10k_queries} <= {"item_7", "item_1a"}
        assert all(q["ticker"] == ev.ticker for q in ev.targeted_10k_queries)

    executor = StructuredReActExecutor()
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
        citations=[],
        sector_benchmarks=[],
        chunk_health=None,
        memory_tokens=48,
        memory_prep_latency_ms=0.8,
    )
    assert out["execution_mode"] == "live_adk"
    assert out["reasoning_model"] == "gemini-3.8-flash"
    assert _is_model(out["evaluation_model"], "gemini-3.5-flash")
    assert out["adk_runtime"]["backend"] == "vertex_ai"
    assert out["adk_runtime"]["executed_via_adk_runner"] is True

    # Both peers got cached 10-K explanations, linked to live chunk citations
    assert {e.ticker for e in out["cached_explanations"]} == {"AAPL", "NVDA"}
    assert len(out["citations"]) >= 2
    assert {c["ticker"] for c in out["citations"]} >= {"AAPL", "NVDA"}
    for exp in out["cached_explanations"]:
        assert exp.extracted_explanations
        assert all(cid.startswith(f"{exp.ticker}_") for cid in exp.matched_chunk_ids)

    step_phases = [s.phase for s in out["react_steps"]]
    assert step_phases[0] == "evaluate_variances"
    assert "query_cached_10k_context" in step_phases
    assert "format_grounded_output" in step_phases
    assert step_phases[-1] == "fact_check_and_eval"

    eval_report = out["evaluation_report"]
    assert _is_model(eval_report.evaluator_model, "gemini-3.5-flash")
    assert eval_report.grade in {"pass", "needs_revision", "fail"}
    assert eval_report.context_tokens > 0

    if out["answer_verified"]:
        assert out["adk_runtime"]["answer_withheld_reason"] is None
        retrieval = out["adk_runtime"]["agent_retrieval"]
        assert retrieval["unresolved_cited_chunk_ids"] == []
    else:
        assert out["answer_markdown"].startswith("### Answer withheld")
        assert out["adk_runtime"]["answer_withheld_reason"]


def test_flash_coding_assistance_python_and_lookml() -> None:
    """Live `gemini-3.5-flash` CodingAssistantAgent across non-SQL languages."""
    for lang in ("python", "lookml"):
        code_rep = generate_financial_coding_assistance(
            prompt="Audit NVDA and MSFT YoY revenue and operating margin variances",
            language=lang,  # type: ignore[arg-type]
            tickers="NVDA,MSFT",
            start_year=2024,
            end_year=2026,
        )
        assert _is_model(code_rep.assistant_model, "gemini-3.5-flash")
        assert code_rep.task_category == "coding_assistance"
        assert code_rep.language == lang
        assert code_rep.code.strip()
        assert code_rep.context_tokens > 0


def test_conversation_memory_token_threshold_compaction() -> None:
    """Older turns are compacted into a digest once session memory crosses the token threshold."""
    compact_mgr = ConversationMemoryManager(token_threshold=80, retention_turns=1)
    tracker = get_context_latency_tracker()
    before_flash_count = len(tracker.recent_records(limit=200, model_name="gemini-3.5-flash"))

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
    assert res2["compaction_count"] == 1
    recalled = compact_mgr.recall_session_context("compact-sess")
    assert "[Compacted Prior Context]:" in recalled["memory_summary_text"]
    assert "T1(NVDA FY2024-2026)" in recalled["memory_summary_text"]

    after_flash_records = tracker.recent_records(limit=200, model_name="gemini-3.5-flash")
    assert len(after_flash_records) > before_flash_count
    assert any(r.task_category == "context_compaction" for r in after_flash_records)


def test_http_chat_multi_turn_memory_recall_and_telemetry_endpoints() -> None:
    """Two live /api/v1/chat turns (follow-up inherits scope), then telemetry, evaluate, coding."""
    get_conversation_memory_manager().reset()
    client = TestClient(app)
    headers = _bearer_headers()

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_body = health_resp.json()
    assert h_body["reasoning_strategy"] == "structured_react_loop"
    assert h_body["models"] == {
        "financial_reasoning": "gemini-3.8-flash",
        "coding_and_evaluation": "gemini-3.5-flash",
    }

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
    assert turn1["resolved_scope"]["tickers"] == ["NVDA"]
    assert turn1["resolved_scope"]["memory_inherited"] is False
    assert turn1["context_and_memory"]["session_memory"]["persistence"]["persisted"] is True
    assert turn1["react_execution"]["strategy"] == "structured_react_loop"
    assert turn1["react_execution"]["iterations"] >= 4
    assert len(turn1["react_execution"]["cached_10k_explanations"]) >= 1
    assert _is_model(turn1["evaluation"]["evaluator_model"], "gemini-3.5-flash")
    assert turn1["context_and_memory"]["models"]["financial_reasoning_model"] == "gemini-3.8-flash"
    assert turn1["context_and_memory"]["turn_context_latency"]["total_calls"] >= 2
    assert isinstance(turn1["answer_verified"], bool)

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

    # Context-induced latency telemetry endpoint reflects the live model calls
    tel_resp = client.get(
        "/api/v1/telemetry/context-latency",
        headers=headers,
        params={"limit": 10},
    )
    assert tel_resp.status_code == 200
    t_body = tel_resp.json()
    assert t_body["status"] == "ok"
    assert t_body["summary"]["total_calls"] >= 4
    assert len(t_body["recent_records"]) >= 1
    by_model = list(t_body["summary"]["by_model"])
    assert any(_is_model(m, "gemini-3.8-flash") for m in by_model), by_model
    assert any(_is_model(m, "gemini-3.5-flash") for m in by_model), by_model

    # Gemini 3.5 Flash Evaluation endpoint (live judge)
    eval_resp = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "query": "NVDA FY2024-2026 revenue and operating margin",
            "response_markdown": turn1["answer_markdown"],
            "variance_reports": turn1["variance_reports"],
            "citations": turn1["citations"],
            "cached_explanations_count": len(turn1["react_execution"]["cached_10k_explanations"]),
        },
    )
    assert eval_resp.status_code == 200, eval_resp.text
    evaluation = eval_resp.json()["evaluation"]
    assert _is_model(evaluation["evaluator_model"], "gemini-3.5-flash")
    assert evaluation["grade"] in {"pass", "needs_revision", "fail"}

    # Gemini 3.5 Flash Coding Assistance endpoint (live)
    code_resp = client.post(
        "/api/v1/coding-assist",
        headers=headers,
        json={
            "prompt": "Generate a BigQuery SQL query for YoY revenue and operating margin variances",
            "language": "sql",
            "tickers": "NVDA,MSFT",
            "start_year": 2024,
            "end_year": 2026,
        },
    )
    assert code_resp.status_code == 200, code_resp.text
    code_body = code_resp.json()["coding_assistance"]
    assert _is_model(code_body["assistant_model"], "gemini-3.5-flash")
    assert "SELECT" in code_body["code"].upper()
    assert "sec_edgar." in code_body["code"]


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
    root_text = make_adk_instruction_provider("root_orchestrator")(ctx)
    assert "RootOrchestrator (`gemini-3.8-flash`)" in root_text
    assert "gemini-2.5-pro" not in root_text

    monkeypatch.setenv("EDGAR_ANALYST_MODEL", "gemini-next-pro")
    assert "FinancialAnalystAgent (`gemini-next-pro`)" in make_adk_instruction_provider(
        "financial_analyst"
    )(ctx)
