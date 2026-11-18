"""Google ADK `Runner` SSE execution, runtime wiring, and the flagship live Gemini E2E smoke test.

- `test_conversational_rag_turn_executes_live_gemini_end_to_end` (`@pytest.mark.live`): the single
  flagship end-to-end live Gemini `gemini-3.8-flash` + async `gemini-3.5-flash` judge + live
  Model Armor + live BigQuery / Vertex AI Search / GCS turn.
- All other tests in this module are fast, deterministic unit/integration tests using scripted
  ADK `BaseLlm`s and `fake_gcp_adapters`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.agents.run_config import ToolThreadPoolConfig
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents import (
    build_adk_llm_for_task,
    calculate_period_variances,
    create_financial_analyst_agent,
    evaluate_variance_explanation,
    flush_pending_evaluations,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_live_runtime_status_metadata,
    resolve_live_adk_runtime,
    run_adk_agent_stream,
)
from edgar_capstone.agents import live_runtime as live_runtime_mod
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.tools import search_10k_chunks_rag
from edgar_capstone.telemetry import get_telemetry_sink
from tests.structured_report_helpers import (
    ScriptedBaseLlm,
    ScriptedJudgeLlm,
    evidence_leads,
    finding_from_citation,
    report_json,
)

TEST_PROJECT_ID = "iron-radio-514217-q7"

_REQUIRED_ADK_RUNTIME_KEYS = {
    "backend",
    "vertex_project_id",
    "vertex_location",
    "gemini_api_key_configured",
    "streaming_mode",
    "executed_via_adk_runner",
    "observed_ttft_ms",
    "streamed_chunks_count",
    "adk_events_count",
    "adk_tool_calls",
    "answer_verified",
    "answer_withheld_reason",
    "judge_mode",
    "structured_report",
    "report_verification",
    "agent_retrieval",
    "agent_invoked_tools",
    "agent_intent",
    "key_metric_backfill_tickers",
}


def _is_model(name: str, family: str) -> bool:
    """Live responses may report a versioned model name (e.g. `gemini-3.5-flash-001`)."""
    return str(name).split("/")[-1].startswith(family)


class StreamingProAnalystLlm(ScriptedBaseLlm):
    """Scripted SSE stream with a controlled prefill (TTFT) delay and decode delay."""

    model: str = "gemini-3.8-flash"
    ttft_delay_s: float = 0.018
    decode_delay_s: float = 0.030
    override_statement: str | None = None

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True, "ADK Runner must invoke Gemini with stream=True (StreamingMode.SSE)"

        leads = evidence_leads(llm_request)
        if self.override_statement is not None:
            final_md = report_json(
                headline="Hallucinated Report",
                findings=[finding_from_citation(leads[0], self.override_statement)],
            )
        else:
            final_md = report_json(headline="Streaming timing probe", findings=[])

        await asyncio.sleep(self.ttft_delay_s)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="Reviewing the variance payload...", thought=True)],
            ),
        )

        midpoint = len(final_md) // 2
        await asyncio.sleep(0.010)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=final_md[:midpoint])],
            ),
        )

        await asyncio.sleep(self.decode_delay_s)
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(text="Reviewing the variance payload...", thought=True),
                    genai_types.Part(text=final_md),
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=3400,
                cached_content_token_count=2100,
                candidates_token_count=165,
                thoughts_token_count=38,
            ),
        )


def test_live_runtime_resolution_and_gemini_client_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vertex AI by default (Gemini on `global`), project precedence, API key mode, and custom LLM."""
    for env_key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "EDGAR_GEMINI_LOCATION",
        "EDGAR_ANALYST_MODEL",
        "EDGAR_FLASH_MODEL",
        "GCP_PROJECT_ID",
    ):
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", TEST_PROJECT_ID)

    # 1. Default: Vertex AI with ADC, Gemini served from `global`, HTTP 429 retries
    vertex_res = resolve_live_adk_runtime()
    assert vertex_res.backend == "vertex_ai"
    assert vertex_res.project_id == TEST_PROJECT_ID
    assert vertex_res.location == "global"
    assert vertex_res.api_key_configured is False

    # GCP_PROJECT_ID takes precedence over GOOGLE_CLOUD_PROJECT
    monkeypatch.setenv("GCP_PROJECT_ID", "preferred-gcp-proj")
    assert resolve_live_adk_runtime().project_id == "preferred-gcp-proj"
    monkeypatch.delenv("GCP_PROJECT_ID")

    pro_gemini = build_adk_llm_for_task("financial_reasoning", resolution=vertex_res)
    flash_gemini = build_adk_llm_for_task("evaluation", resolution=vertex_res)
    assert isinstance(pro_gemini, Gemini)
    assert isinstance(flash_gemini, Gemini)
    assert pro_gemini.model == "gemini-3.8-flash"
    assert flash_gemini.model == "gemini-3.5-flash"
    assert pro_gemini.client_kwargs == {
        "vertexai": True,
        "project": TEST_PROJECT_ID,
        "location": "global",
    }
    assert pro_gemini.retry_options is not None
    assert pro_gemini.retry_options.attempts == 3
    assert 429 in (pro_gemini.retry_options.http_status_codes or [])

    monkeypatch.setenv("EDGAR_GEMINI_LOCATION", "us-central1")
    regional_res = resolve_live_adk_runtime()
    assert regional_res.location == "us-central1"
    regional_gemini = build_adk_llm_for_task("financial_reasoning", resolution=regional_res)
    assert regional_gemini.client_kwargs["location"] == "us-central1"
    monkeypatch.delenv("EDGAR_GEMINI_LOCATION")

    # 2. Gemini API key mode -> secret redacted from repr & public status metadata
    synthetic_key = "test-ephemeral-" + "credential-value"
    monkeypatch.setenv("GEMINI_API_KEY", synthetic_key)
    api_res = resolve_live_adk_runtime()
    assert api_res.backend == "gemini_api"
    assert api_res.api_key_configured is True
    assert synthetic_key not in repr(api_res)
    api_gemini = build_adk_llm_for_task("financial_reasoning", resolution=api_res)
    assert isinstance(api_gemini, Gemini)
    assert api_gemini.client_kwargs == {"api_key": synthetic_key}
    pub_meta = get_live_runtime_status_metadata()
    assert pub_meta["backend"] == "gemini_api"
    assert pub_meta["gemini_api_key_configured"] is True
    assert synthetic_key not in json.dumps(pub_meta)

    # 3. GOOGLE_GENAI_USE_VERTEXAI wins over a configured key (and the key is not retained)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    both_res = resolve_live_adk_runtime()
    assert both_res.backend == "vertex_ai"
    assert both_res.api_key_configured is True
    assert both_res._api_key is None
    assert get_live_runtime_status_metadata()["backend"] == "vertex_ai"

    # 4. An injected ADK BaseLlm is used as-is
    custom = StreamingProAnalystLlm()
    custom_res = resolve_live_adk_runtime(custom_llm=custom)
    assert custom_res.backend == "custom_llm"
    assert build_adk_llm_for_task("financial_reasoning", custom_llm=custom) is custom


def test_adk_runner_sse_streaming_captures_ttft_and_tool_thread_pool_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StreamingMode.SSE captures first-chunk TTFT, sets ToolThreadPoolConfig, and deduplicates telemetry."""
    from google.adk.runners import Runner

    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    captured_run_configs: list[Any] = []
    orig_run_async = Runner.run_async

    def _spy_run_async(self, *args: Any, **kwargs: Any):
        captured_run_configs.append(kwargs.get("run_config"))
        return orig_run_async(self, *args, **kwargs)

    monkeypatch.setattr(Runner, "run_async", _spy_run_async)

    pro_llm = StreamingProAnalystLlm(ttft_delay_s=0.020, decode_delay_s=0.045)
    agent = create_financial_analyst_agent(model=pro_llm)

    stream_res = run_adk_agent_stream(
        agent=agent,
        prompt_text="Summarize the streaming timing probe.",
        session_id="sse-ttft-test-1",
        use_adk_app=True,
        summarizer_llm=pro_llm,
        backend="custom_llm",
    )

    assert len(captured_run_configs) == 1
    assert isinstance(captured_run_configs[0].tool_thread_pool_config, ToolThreadPoolConfig)

    assert stream_res.partial_chunks_count == 2
    assert stream_res.total_events_count >= 3
    assert stream_res.observed_ttft_ms is not None
    assert 8.0 <= stream_res.observed_ttft_ms < (stream_res.total_stream_latency_ms - 15.0)
    assert len(stream_res.thoughts) >= 1

    # Exactly 1 latency record (partial SSE chunks must NOT double-record!)
    all_records = tracker.recent_records(limit=10)
    assert len(all_records) == 1
    rec = all_records[0]
    assert rec.model_name == "gemini-3.8-flash"
    assert rec.context_breakdown.total_context_tokens == 3400
    assert rec.context_breakdown.cached_context_tokens == 2100
    assert rec.output_tokens == 165
    assert rec.thoughts_tokens == 38
    assert rec.context_prefill_latency_ms == pytest.approx(stream_res.observed_ttft_ms, abs=1.5)
    assert rec.generation_latency_ms >= 15.0


def test_extract_raw_json_dict_and_background_loop_lifecycle() -> None:
    """Cover `_extract_raw_json_dict` fallbacks and `_BackgroundAsyncLoop` stop/restart."""
    extract = live_runtime_mod._extract_raw_json_dict

    assert extract("") is None
    assert extract("   ") is None
    assert extract('{"a": 1}') == {"a": 1}
    assert extract('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract('```json\n{"a": 3}') == {"a": 3}
    assert extract('Some prose {"nested": {"k": "v"}} trailing') == {"nested": {"k": "v"}}
    assert extract("[1, 2, 3]") is None
    assert extract("prose {not: json} end") is None
    assert extract("no braces at all") is None

    # Exercise error propagation and thread-loop execution of `_run_coro_sync`
    async def _boom() -> None:
        raise ValueError("nested async error")

    async def _invoke_from_running_loop() -> None:
        with pytest.raises(ValueError, match="nested async error"):
            live_runtime_mod._run_coro_sync(_boom())

    asyncio.run(_invoke_from_running_loop())

    # Dedicated persistent loop thread starts, runs a coro, and stops cleanly
    loop_runner = live_runtime_mod._PersistentAdkLoopManager()
    try:

        async def _double(x: int) -> int:
            return x * 2

        loop = loop_runner.get_loop()
        assert asyncio.run_coroutine_threadsafe(_double(21), loop).result(timeout=5.0) == 42
        assert loop_runner.is_loop_thread() is False
    finally:
        loop_runner.stop()
        # Idempotent second stop
        loop_runner.stop()


@pytest.mark.live
def test_conversational_rag_turn_executes_live_gemini_end_to_end(
    live_model_armor: None,
) -> None:
    """Flagship live E2E smoke test: live Gemini + async Flash judge + live Model Armor + live GCP."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue and operating income drivers",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="live-chat-sess-01",
        user_id="analyst@fde-workspace.corp",
    )

    try:
        assert turn["status"] == "success"
        assert turn["execution_mode"] == "live_adk"
        runtime = turn["adk_runtime"]
        assert _REQUIRED_ADK_RUNTIME_KEYS <= set(runtime)
        assert runtime["backend"] == "vertex_ai"
        assert runtime["vertex_project_id"] == TEST_PROJECT_ID
        assert runtime["executed_via_adk_runner"] is True
        assert runtime["streaming_mode"] == "sse"
        assert runtime["observed_ttft_ms"] is not None and runtime["observed_ttft_ms"] > 0.0
        assert runtime["adk_events_count"] >= 1

        assert turn["answer_verified"] is True, runtime["answer_withheld_reason"]
        assert runtime["answer_verified"] is True
        assert runtime["answer_withheld_reason"] is None
        assert turn["fact_check"]["passed"] is True
        assert turn["structured_report"] is not None
        assert turn["report_verification"]["usable"] is True
        assert not turn["answer_markdown"].startswith("### Answer withheld")

        assert "calculate_period_variances" in runtime["agent_invoked_tools"] or runtime[
            "key_metric_backfill_tickers"
        ] == ["NVDA"], runtime["adk_tool_calls"]
        assert turn["intent"] == "longitudinal_single_company"
        direct = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
        nvda = direct["variance_reports"][0]
        turn_nvda = next(r for r in turn["variance_reports"] if r["ticker"] == "NVDA")
        assert turn_nvda["revenue_variance"] == nvda["revenue_variance"]
        assert turn_nvda["operating_income_variance"] == nvda["operating_income_variance"]
        assert "#### Key Metrics (deterministic engine)" in turn["answer_markdown"]

        evaluation = turn["evaluation"]
        assert _is_model(evaluation["evaluator_model"], "gemini-3.5-flash")
        assert evaluation["grade"] == "pending"
        assert evaluation["overall_score"] is None
        assert evaluation["numeric_fidelity_score"] is None
        assert evaluation["citation_grounding_score"] is None

        models_called = [r["model_name"] for r in turn["context_and_memory"]["model_call_records"]]
        assert any(_is_model(m, "gemini-3.8-flash") for m in models_called), models_called
        assert runtime["judge_mode"] == "async_background"
        flush_pending_evaluations(timeout=30.0)
        all_models_called = [r.model_name for r in tracker.recent_records(limit=20)]
        assert any(_is_model(m, "gemini-3.5-flash") for m in all_models_called), all_models_called

        async_evals = get_telemetry_sink().recent_async_evaluations(session_id="live-chat-sess-01")
        assert len(async_evals) >= 1
        assert any(
            _is_model(ev.get("evaluator_model", ""), "gemini-3.5-flash")
            and ev.get("evaluation_grade") in {"pass", "needs_revision", "fail"}
            and ev.get("evaluation_score") is not None
            and 0.0 <= float(ev["evaluation_score"]) <= 1.0
            for ev in async_evals
        ), async_evals

        assert len(turn["agent_trace"]) >= 4
        for step in turn["agent_trace"]:
            assert step["s2s_verified"] is True
            assert step["agent_service_account"].endswith(".iam.gserviceaccount.com")
            assert step["spiffe_id"].startswith("spiffe://")
    finally:
        # Never leak an in-flight background judge thread into subsequent test modules
        flush_pending_evaluations(timeout=15.0)


def test_unverifiable_live_answer_is_withheld_not_replaced(fake_gcp_adapters: Any) -> None:
    """A finding whose figure isn't in its quote is rejected and the narrative is withheld."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()

    hallucinating_llm = StreamingProAnalystLlm(
        ttft_delay_s=0.001,
        decode_delay_s=0.001,
        override_statement="NVIDIA revenue declined -99.99% in FY2026.",
    )

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue variance",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        reasoning_llm=hallucinating_llm,
        skip_background_judge=True,
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["answer_verified"] is False
    assert turn["adk_runtime"]["answer_verified"] is False
    assert "could be verified" in str(turn["adk_runtime"]["answer_withheld_reason"])
    assert turn["answer_markdown"].startswith("### Answer withheld")
    assert "Hallucinated Report" not in turn["answer_markdown"]
    assert "-99.99%" not in turn["answer_markdown"]

    verification = turn["report_verification"]
    assert verification["usable"] is False
    assert (verification["findings_total"], verification["findings_verified"]) == (1, 0)
    reasons = " ".join(verification["findings"][0]["reasons"])
    assert "figures not in the quote" in reasons and "99.99%" in reasons
    assert "too short" not in reasons and "not found" not in reasons
    assert turn["variance_reports"] == [] and turn["citations"]


def test_narrative_evaluator_deterministic_cap_overrides_generous_judge(
    fake_gcp_adapters: Any,
) -> None:
    """Even when the LLM judge returns 1.0/pass, a wrong figure caps numeric fidelity <= 0.45."""
    get_context_latency_tracker().reset()

    nvda_rep = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)[
        "variance_reports"
    ][0]
    rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]
    search = search_10k_chunks_rag(
        query="data center revenue growth drivers",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="item_7",
        start_year=2026,
        end_year=2026,
        top_k=1,
    )
    cit = search["citations"][0]
    tag = (
        f"[{cit['ticker']} FY{cit['fiscal_year']} | {cit['section_title']} | {cit['chunk_id']}]"
        f"({cit['sec_filing_url']})"
    )

    overly_generous_judge = ScriptedJudgeLlm(
        overall_score=0.99,
        numeric_fidelity_score=1.0,
        citation_grounding_score=1.0,
        explanation_quality_score=0.98,
        grade="pass",
    )

    grounded = evaluate_variance_explanation(
        query="Evaluate NVDA FY2024-FY2026 revenue variance summary",
        response_markdown=f"NVIDIA revenue changed {rev_pct:+.2f}% from FY2024 to FY2026 {tag}.",
        variance_reports=[nvda_rep],
        citations=[cit],
        llm=overly_generous_judge,
    )
    assert _is_model(grounded.evaluator_model, "gemini-3.5-flash")
    assert grounded.grade == "pass"
    assert grounded.overall_score is not None and 0.95 <= grounded.overall_score <= 1.0
    assert grounded.context_tokens > 0
    assert grounded.context_latency_ms >= 0.0

    # Contradicting percentage MUST cap numeric fidelity even though the judge scored 1.0/pass!
    wrong = evaluate_variance_explanation(
        query="Evaluate NVDA FY2024-FY2026 revenue variance summary",
        response_markdown=(
            f"NVIDIA revenue changed {rev_pct + 37.0:+.2f}% from FY2024 to FY2026 {tag}."
        ),
        variance_reports=[nvda_rep],
        citations=[cit],
        llm=overly_generous_judge,
    )
    assert wrong.numeric_fidelity_score is not None
    assert wrong.numeric_fidelity_score <= 0.45
    assert wrong.grade != "pass"
