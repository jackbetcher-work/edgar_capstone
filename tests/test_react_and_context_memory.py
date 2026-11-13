"""Automated tests for Structured ReAct Reasoning, Dual-Model Routing, and Context/Memory Latency Tracking."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.testclient import TestClient
from google.adk.apps import App
from google.adk.memory import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import PlanReActPlanner
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
    evaluate_variance_explanation,
    extract_llm_request_context_breakdown,
    generate_financial_coding_assistance,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_model_strategy_metadata,
    query_cached_10k_context,
    reconcile_breakdown_with_usage_metadata,
    resolve_model_for_task,
)
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app


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
                "has_plan_react_instruction": "/*PLANNING*/"
                in str(getattr(llm_request.config, "system_instruction", "")),
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
        nvda_rep = var_payload["variance_reports"][0]
        rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]
        op_pct = nvda_rep["operating_income_variance"]["percentage_delta"]
        chunk_id = (cache_payload.get("matched_chunk_ids") or ["NVDA_2026_item_7_0001"])[0]
        filing_url = cache_payload.get("sec_filing_url") or "https://www.sec.gov/nvda"
        explanation_snippet = (cache_payload.get("explanations") or ["Data center compute growth."])[0][:180]

        final_md = (
            "/*FINAL_ANSWER*/\n"
            f"### NVIDIA CORP (NVDA) Grounded Variance Report\n"
            f"- **Evaluated Variances (FY2024→FY2026):** Revenue grew **+{rev_pct:.2f}%** and "
            f"Operating Income grew **+{op_pct:.2f}%**.\n"
            f"- **Cached 10-K Explanation:** {explanation_snippet} "
            f"[NVDA FY2026 | Item 7 MD&A | {chunk_id}]({filing_url})"
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
    """Verify Gemini 2.5 Pro handles complex financial reasoning while Gemini 3.5 Flash handles coding, eval, & compaction."""
    assert FINANCIAL_REASONING_MODEL == "gemini-2.5-pro"
    assert FLASH_CODING_AND_EVAL_MODEL == "gemini-3.5-flash"

    # Task routing assertions
    for pro_task in (
        "financial_reasoning",
        "variance_explanation",
        "peer_comparison",
        "thematic_synthesis",
        "orchestration",
    ):
        assert resolve_model_for_task(pro_task) == "gemini-2.5-pro"  # type: ignore[arg-type]

    for flash_task in (
        "evaluation",
        "llm_as_judge",
        "coding_assistance",
        "context_compaction",
        "context_cache_warmup",
    ):
        assert resolve_model_for_task(flash_task) == "gemini-3.5-flash"  # type: ignore[arg-type]

    # ADK Agent model assignments & PlanReActPlanner registration
    assert RootOrchestrator.model == "gemini-2.5-pro"
    assert FinancialAnalystAgent.model == "gemini-2.5-pro"
    assert isinstance(RootOrchestrator.planner, PlanReActPlanner)
    assert isinstance(FinancialAnalystAgent.planner, PlanReActPlanner)

    assert NarrativeEvaluatorAgent.model == "gemini-3.5-flash"
    assert CodingAssistantAgent.model == "gemini-3.5-flash"
    assert ContextCacheAgent.model == "gemini-3.5-flash"

    sub_agent_models = {sa.name: sa.model for sa in RootOrchestrator.sub_agents}
    assert sub_agent_models["FinancialAnalystAgent"] == "gemini-2.5-pro"
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
    assert meta["financial_reasoning_model"] == "gemini-2.5-pro"
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
                final_text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
        sess = await session_service.get_session(
            app_name="sec_edgar_react_test",
            user_id="lead-analyst",
            session_id="adk-react-sess-01",
        )
        return final_text, sess

    final_answer, session = asyncio.run(_run_turn())

    # 1. Verify PlanReActPlanner injected structured planning instructions & executed 3 model turns
    assert len(mock_llm.call_log) == 3
    assert mock_llm.call_log[0]["has_plan_react_instruction"] is True
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
    assert rec_large_cached.context_prefill_latency_ms < rec_large_uncached.context_prefill_latency_ms
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
        content=genai_types.Content(role="model", parts=[genai_types.Part(text="Grounded output.")]),
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


def test_structured_react_executor_and_gemini_3_5_flash_tasks() -> None:
    """Verify ReAct variance evaluation, cached 10-K querying, adaptive peer fill, and Flash coding/eval tasks."""
    var_res = calculate_period_variances(tickers="NVDA,AAPL", start_year=2024, end_year=2026)
    reports = var_res["variance_reports"]

    evals = evaluate_calculated_variances(reports, query="AI compute and Services margin")
    assert len(evals) == 2
    nvda_eval = next(e for e in evals if e.ticker == "NVDA")
    assert nvda_eval.revenue_pct_delta is not None and nvda_eval.revenue_pct_delta > 200.0
    assert len(nvda_eval.divergences) >= 1
    assert {q["section_id"] for q in nvda_eval.targeted_10k_queries} == {"item_7", "item_1a"}

    # Execute full StructuredReActExecutor post-retrieval reasoning
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
    assert out["reasoning_model"] == "gemini-2.5-pro"
    assert out["evaluation_model"] == "gemini-3.5-flash"
    assert len(out["cached_explanations"]) >= 2
    assert len(out["citations"]) >= 2
    assert out["evaluation_report"].evaluator_model == "gemini-3.5-flash"
    assert out["evaluation_report"].grade == "pass"

    step_phases = [s.phase for s in out["react_steps"]]
    assert step_phases[0] == "evaluate_variances"
    assert "query_cached_10k_context" in step_phases
    assert "format_grounded_output" in step_phases
    assert step_phases[-1] == "fact_check_and_eval"

    # Gemini 3.5 Flash Coding Assistance across SQL, Python, and LookML
    for lang in ("sql", "python", "lookml"):
        code_rep = generate_financial_coding_assistance(
            prompt="Audit NVDA and MSFT YoY revenue and operating margin variances",
            language=lang,  # type: ignore[arg-type]
            tickers="NVDA,MSFT",
            start_year=2024,
            end_year=2026,
        )
        assert code_rep.assistant_model == "gemini-3.5-flash"
        assert code_rep.task_category == "coding_assistance"
        assert code_rep.language == lang
        assert "NVDA" in code_rep.code
        assert code_rep.context_tokens > 0

    # Direct Gemini 3.5 Flash evaluation check
    eval_rep = evaluate_variance_explanation(
        query="NVDA FY2024-2026 variance",
        response_markdown=out["answer_markdown"],
        variance_reports=reports,
        citations=out["citations"],
        cached_explanations_count=len(out["cached_explanations"]),
    )
    assert eval_rep.evaluator_model == "gemini-3.5-flash"
    assert eval_rep.numeric_fidelity_score == 1.0


def test_session_memory_multi_turn_recall_and_flash_compaction() -> None:
    """Verify multi-turn entity/timeframe recall across follow-up turns and token-threshold compaction."""
    mem_mgr = get_conversation_memory_manager()
    mem_mgr.reset()

    # Turn 1 explicitly names NVIDIA (2024-2026)
    turn1 = run_conversational_rag_turn(
        query="Analyze NVIDIA's revenue and operating margin from 2024 to 2026",
        session_id="analyst-multi-turn-session",
    )
    assert turn1["status"] == "success"
    assert turn1["resolved_scope"]["tickers"] == ["NVDA"]
    assert turn1["resolved_scope"]["memory_inherited"] is False
    assert turn1["context_and_memory"]["session_memory"]["persistence"]["persisted"] is True

    # Turn 2 omits ticker name ("What Item 1A risk factors could impact those margins?")
    turn2 = run_conversational_rag_turn(
        query="What Item 1A risk factors could impact those margins?",
        session_id="analyst-multi-turn-session",
    )
    assert turn2["status"] == "success"
    assert turn2["resolved_scope"]["tickers"] == ["NVDA"]
    assert turn2["resolved_scope"]["memory_inherited"] is True
    assert turn2["resolved_scope"]["section_id"] == "item_1a"
    assert turn2["context_and_memory"]["session_memory"]["has_prior_memory"] is True
    assert turn2["context_and_memory"]["session_memory"]["turn_count"] == 1

    # Verify token-threshold compaction using a tight threshold ConversationMemoryManager
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
        variance_highlights=["Revenue increased +254.45%"],
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
        variance_highlights=["AAPL Services margin expanded"],
        answer_markdown="Second detailed grounded synthesis paragraph " * 12,
    )
    assert res2["compaction_triggered"] is True
    assert res2["compaction_count"] == 1
    recalled = compact_mgr.recall_session_context("compact-sess")
    assert "[Compacted Prior Context]:" in recalled["memory_summary_text"]

    after_flash_records = tracker.recent_records(limit=200, model_name="gemini-3.5-flash")
    assert len(after_flash_records) > before_flash_count
    assert any(r.task_category == "context_compaction" for r in after_flash_records)


def test_fastapi_react_telemetry_evaluation_and_coding_endpoints() -> None:
    """Verify /health, /api/v1/chat, /api/v1/telemetry/context-latency, /api/v1/evaluate, and /api/v1/coding-assist."""
    client = TestClient(app)
    headers = _bearer_headers()

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_body = health_resp.json()
    assert h_body["reasoning_strategy"] == "structured_react_loop"
    assert h_body["models"] == {
        "financial_reasoning": "gemini-2.5-pro",
        "coding_and_evaluation": "gemini-3.5-flash",
    }

    chat_resp = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "query": "Explain Microsoft's FY2024 to FY2026 revenue and cloud operating income variances",
            "session_id": "http-sess-msft",
        },
    )
    assert chat_resp.status_code == 200
    c_body = chat_resp.json()
    assert c_body["status"] == "success"
    assert c_body["react_execution"]["strategy"] == "structured_react_loop"
    assert c_body["react_execution"]["iterations"] >= 4
    assert len(c_body["react_execution"]["cached_10k_explanations"]) >= 1
    assert c_body["evaluation"]["evaluator_model"] == "gemini-3.5-flash"
    assert (
        c_body["context_and_memory"]["models"]["financial_reasoning_model"]
        == "gemini-2.5-pro"
    )
    assert (
        c_body["context_and_memory"]["turn_context_latency"]["total_calls"] >= 3
    )

    # Context-induced latency telemetry endpoint
    tel_resp = client.get(
        "/api/v1/telemetry/context-latency",
        headers=headers,
        params={"limit": 10},
    )
    assert tel_resp.status_code == 200
    t_body = tel_resp.json()
    assert t_body["status"] == "ok"
    assert t_body["summary"]["total_calls"] >= 3
    assert len(t_body["recent_records"]) >= 1
    assert "gemini-2.5-pro" in t_body["summary"]["by_model"]
    assert "gemini-3.5-flash" in t_body["summary"]["by_model"]

    # Gemini 3.5 Flash Evaluation endpoint
    eval_resp = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "query": "MSFT FY2024-2026 variance",
            "response_markdown": c_body["answer_markdown"],
            "variance_reports": c_body["variance_reports"],
            "citations": c_body["citations"],
            "cached_explanations_count": len(
                c_body["react_execution"]["cached_10k_explanations"]
            ),
        },
    )
    assert eval_resp.status_code == 200
    assert eval_resp.json()["evaluation"]["evaluator_model"] == "gemini-3.5-flash"
    assert eval_resp.json()["evaluation"]["grade"] == "pass"

    # Gemini 3.5 Flash Coding Assistance endpoint
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
    assert code_resp.status_code == 200
    code_body = code_resp.json()["coding_assistance"]
    assert code_body["assistant_model"] == "gemini-3.5-flash"
    assert "v_financial_variances_yoy" in code_body["code"]
